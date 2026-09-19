// Command loadgen is a closed-loop streaming load generator for any
// OpenAI-compatible chat endpoint (the gateway, a mock worker, or a real SGLang
// server). It records one CSV row per request and a JSON summary.
//
// A request counts as successful only if it returned 200, streamed at least one
// token, ended with "data: [DONE]" and carried no error event.
package main

import (
	"bufio"
	"context"
	"encoding/csv"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"log"
	"math/rand"
	"net"
	"net/http"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/unicoregpu/radixgates/gateway/internal/stats"
)

type record struct {
	tS       float64 // request start, seconds since the run began
	latMS    float64
	ttftMS   float64
	status   int
	ok       bool
	attempts string
	backend  string
	errMsg   string
}

type summary struct {
	Label        string         `json:"label"`
	URL          string         `json:"url"`
	Clients      int            `json:"clients"`
	Prefixes     int            `json:"prefixes"`
	MaxTokens    int            `json:"max_tokens"`
	DurationS    float64        `json:"duration_s"`
	Total        int            `json:"total"`
	OK           int            `json:"ok"`
	Failed       int            `json:"failed"`
	SuccessRate  float64        `json:"success_rate"`
	Retried      int            `json:"retried_requests"`
	RPS          float64        `json:"requests_per_second"`
	LatencyMS    stats.Summary  `json:"latency_ms_ok_only"`
	TTFTMS       stats.Summary  `json:"ttft_ms_ok_only"`
	StatusCounts map[string]int `json:"status_counts"`
}

func systemPrompt(i int) string {
	return fmt.Sprintf("You are assistant persona #%d for department %d. Answer concisely and follow the house style guide. %s",
		i, i, strings.Repeat("Context line. ", 12))
}

func main() {
	url := flag.String("url", "http://127.0.0.1:18080/v1/chat", "chat completions URL")
	clients := flag.Int("clients", 16, "concurrent closed-loop clients")
	duration := flag.Duration("duration", 30*time.Second, "test duration")
	prefixes := flag.Int("prefixes", 12, "distinct system prompts (prefix-affinity groups)")
	maxTokens := flag.Int("max-tokens", 16, "max_tokens per request")
	timeout := flag.Duration("timeout", 20*time.Second, "per-request client timeout")
	out := flag.String("out", "", "output directory for requests.csv and summary.json")
	label := flag.String("label", "run", "label stored in the summary")
	seed := flag.Int64("seed", 1, "random seed for prefix selection")
	flag.Parse()

	client := &http.Client{Transport: &http.Transport{MaxIdleConnsPerHost: *clients, DialContext: (&net.Dialer{Timeout: 3 * time.Second}).DialContext}}
	begin := time.Now()
	deadline := begin.Add(*duration)

	var mu sync.Mutex
	var recs []record
	var wg sync.WaitGroup
	for c := 0; c < *clients; c++ {
		wg.Add(1)
		go func(c int) {
			defer wg.Done()
			rng := rand.New(rand.NewSource(*seed + int64(c)))
			for time.Now().Before(deadline) {
				r := oneRequest(client, *url, systemPrompt(rng.Intn(*prefixes)), *maxTokens, *timeout, begin)
				mu.Lock()
				recs = append(recs, r)
				mu.Unlock()
			}
		}(c)
	}
	wg.Wait()
	elapsed := time.Since(begin)

	sort.Slice(recs, func(i, j int) bool { return recs[i].tS < recs[j].tS })
	sum := summary{Label: *label, URL: *url, Clients: *clients, Prefixes: *prefixes, MaxTokens: *maxTokens,
		DurationS: elapsed.Seconds(), Total: len(recs), StatusCounts: map[string]int{}}
	var lat, ttft []float64
	for _, r := range recs {
		sum.StatusCounts[strconv.Itoa(r.status)]++
		if r.attempts != "" && r.attempts != "1" {
			sum.Retried++
		}
		if r.ok {
			sum.OK++
			lat = append(lat, r.latMS)
			ttft = append(ttft, r.ttftMS)
		}
	}
	sum.Failed = sum.Total - sum.OK
	if sum.Total > 0 {
		sum.SuccessRate = float64(sum.OK) / float64(sum.Total)
	}
	sum.RPS = float64(sum.Total) / elapsed.Seconds()
	sum.LatencyMS = stats.Summarize(lat)
	sum.TTFTMS = stats.Summarize(ttft)

	js, _ := json.MarshalIndent(sum, "", "  ")
	fmt.Println(string(js))
	if *out == "" {
		return
	}
	if err := os.MkdirAll(*out, 0o755); err != nil {
		log.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(*out, "summary.json"), append(js, '\n'), 0o644); err != nil {
		log.Fatal(err)
	}
	f, err := os.Create(filepath.Join(*out, "requests.csv"))
	if err != nil {
		log.Fatal(err)
	}
	defer f.Close()
	w := csv.NewWriter(f)
	_ = w.Write([]string{"t_s", "latency_ms", "ttft_ms", "status", "ok", "attempts", "backend", "error"})
	for _, r := range recs {
		_ = w.Write([]string{
			strconv.FormatFloat(r.tS, 'f', 3, 64), strconv.FormatFloat(r.latMS, 'f', 1, 64),
			strconv.FormatFloat(r.ttftMS, 'f', 1, 64), strconv.Itoa(r.status), strconv.FormatBool(r.ok),
			r.attempts, r.backend, r.errMsg,
		})
	}
	w.Flush()
	if err := w.Error(); err != nil {
		log.Fatal(err)
	}
}

func oneRequest(client *http.Client, url, system string, maxTokens int, timeout time.Duration, begin time.Time) record {
	start := time.Now()
	rec := record{tS: start.Sub(begin).Seconds()}
	body := fmt.Sprintf(`{"model":"default","stream":true,"max_tokens":%d,"messages":[{"role":"system","content":%q},{"role":"user","content":"Summarise the plan."}]}`, maxTokens, system)

	ctx, cancel := context.WithTimeout(context.Background(), timeout)
	defer cancel()
	req, _ := http.NewRequestWithContext(ctx, http.MethodPost, url, strings.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	resp, err := client.Do(req)
	if err != nil {
		rec.latMS = float64(time.Since(start).Microseconds()) / 1000
		rec.errMsg = shorten(err.Error())
		return rec
	}
	defer resp.Body.Close()
	rec.status = resp.StatusCode
	rec.attempts = resp.Header.Get("X-Gateway-Attempts")
	rec.backend = resp.Header.Get("X-Gateway-Node")

	if resp.StatusCode != http.StatusOK {
		_, _ = io.Copy(io.Discard, resp.Body)
		rec.latMS = float64(time.Since(start).Microseconds()) / 1000
		rec.errMsg = fmt.Sprintf("http %d", resp.StatusCode)
		return rec
	}
	rd := bufio.NewReader(resp.Body)
	sawToken, sawDone, sawErr := false, false, false
	for {
		line, err := rd.ReadString('\n')
		line = strings.TrimSpace(line)
		switch {
		case strings.HasPrefix(line, "data: [DONE]"):
			sawDone = true
		case strings.Contains(line, `"error"`):
			sawErr = true
		case strings.HasPrefix(line, "data:"):
			if !sawToken {
				sawToken = true
				rec.ttftMS = float64(time.Since(start).Microseconds()) / 1000
			}
		}
		if err != nil {
			if err != io.EOF {
				rec.errMsg = shorten(err.Error())
			}
			break
		}
	}
	rec.latMS = float64(time.Since(start).Microseconds()) / 1000
	rec.ok = sawToken && sawDone && !sawErr
	if !rec.ok && rec.errMsg == "" {
		switch {
		case sawErr:
			rec.errMsg = "stream error event"
		case !sawDone:
			rec.errMsg = "stream ended without [DONE]"
		default:
			rec.errMsg = "no tokens"
		}
	}
	return rec
}

func shorten(s string) string {
	s = strings.ReplaceAll(s, "\n", " ")
	if len(s) > 80 {
		s = s[:80]
	}
	return s
}
