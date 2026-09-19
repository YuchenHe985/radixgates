// Package mock is a simulated SGLang-compatible inference worker with runtime
// fault injection. It exists so the gateway's failure handling can be tested and
// benchmarked without a GPU. Latencies are a model, not a measurement of any GPU:
// cold/warm TTFT and TPOT are parameters, calibrated in the benchmark configs to
// the ranges observed in the real 4x RTX 4090 runs (see docs/real-gpu-results.md).
//
// Endpoints: POST /v1/chat/completions (SSE or JSON), GET /health,
// POST /admin/fault, GET /admin/stats.
package mock

import (
	"container/list"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"net/http"
	"strings"
	"sync"
	"time"
)

// Fault modes.
const (
	FaultNone       = "none"
	FaultError503   = "error503"    // answer 503 immediately
	FaultHang       = "hang"        // accept the request and never answer
	FaultSlow       = "slow"        // add DelayMS to time-to-first-token (gray failure)
	FaultReset      = "reset"       // close the connection without a response
	FaultMidstream  = "midstream"   // close the connection after AfterTokens tokens
	FaultHealthDown = "health_down" // only /health fails; inference still works
)

type Fault struct {
	Mode        string `json:"mode"`
	DelayMS     int    `json:"delay_ms,omitempty"`
	AfterTokens int    `json:"after_tokens,omitempty"`
}

type Options struct {
	Name      string
	TTFTCold  time.Duration // time to first token when the prefix is not cached
	TTFTWarm  time.Duration // time to first token on a prefix-cache hit
	TPOT      time.Duration // time per output token
	Tokens    int           // default output tokens when the request sets no max_tokens
	Capacity  int           // concurrent requests served at full speed; above it latency scales linearly
	CacheSize int           // distinct prefixes kept in the simulated KV/radix cache (LRU)
}

func (o *Options) defaults() {
	if o.TTFTCold <= 0 {
		o.TTFTCold = 100 * time.Millisecond
	}
	if o.TTFTWarm <= 0 {
		o.TTFTWarm = 50 * time.Millisecond
	}
	if o.TPOT <= 0 {
		o.TPOT = 15 * time.Millisecond
	}
	if o.Tokens <= 0 {
		o.Tokens = 16
	}
	if o.Capacity <= 0 {
		o.Capacity = 4
	}
	if o.CacheSize <= 0 {
		o.CacheSize = 4
	}
}

type Worker struct {
	opts Options

	mu       sync.Mutex
	fault    Fault
	inflight int
	requests int
	warm     int
	lru      *list.List
	cached   map[string]*list.Element
}

func New(o Options) *Worker {
	o.defaults()
	return &Worker{opts: o, fault: Fault{Mode: FaultNone}, lru: list.New(), cached: map[string]*list.Element{}}
}

func (w *Worker) SetFault(f Fault) {
	if f.Mode == "" {
		f.Mode = FaultNone
	}
	w.mu.Lock()
	w.fault = f
	w.mu.Unlock()
}

func (w *Worker) currentFault() Fault {
	w.mu.Lock()
	defer w.mu.Unlock()
	return w.fault
}

// Stats is the worker's own view of what it served.
type Stats struct {
	Name     string `json:"name"`
	Requests int    `json:"requests"`
	WarmHits int    `json:"warm_hits"`
	Inflight int    `json:"inflight"`
}

func (w *Worker) Stats() Stats {
	w.mu.Lock()
	defer w.mu.Unlock()
	return Stats{Name: w.opts.Name, Requests: w.requests, WarmHits: w.warm, Inflight: w.inflight}
}

func (w *Worker) Handler() http.Handler {
	mux := http.NewServeMux()
	mux.HandleFunc("GET /health", w.health)
	mux.HandleFunc("POST /v1/chat/completions", w.chat)
	mux.HandleFunc("POST /admin/fault", w.adminFault)
	mux.HandleFunc("GET /admin/stats", func(rw http.ResponseWriter, _ *http.Request) {
		rw.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(rw).Encode(w.Stats())
	})
	return mux
}

func (w *Worker) health(rw http.ResponseWriter, _ *http.Request) {
	switch w.currentFault().Mode {
	case FaultHealthDown, FaultError503:
		http.Error(rw, `{"status":"unhealthy"}`, http.StatusServiceUnavailable)
	default:
		rw.Header().Set("Content-Type", "application/json")
		fmt.Fprint(rw, `{"status":"ok"}`)
	}
}

func (w *Worker) adminFault(rw http.ResponseWriter, r *http.Request) {
	var f Fault
	if err := json.NewDecoder(r.Body).Decode(&f); err != nil {
		http.Error(rw, err.Error(), http.StatusBadRequest)
		return
	}
	w.SetFault(f)
	rw.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(rw).Encode(w.currentFault())
}

type chatReq struct {
	Messages []struct {
		Role    string          `json:"role"`
		Content json.RawMessage `json:"content"`
	} `json:"messages"`
	Stream    bool `json:"stream"`
	MaxTokens int  `json:"max_tokens"`
}

func prefixKey(r chatReq) string {
	var sb strings.Builder
	for _, m := range r.Messages {
		var s string
		if json.Unmarshal(m.Content, &s) != nil {
			s = string(m.Content)
		}
		if m.Role == "system" {
			return s
		}
		if sb.Len() < 256 {
			sb.WriteString(s)
		}
	}
	return sb.String()
}

// touch reports whether key was cached and marks it most recently used.
func (w *Worker) touch(key string) bool {
	w.mu.Lock()
	defer w.mu.Unlock()
	if el, ok := w.cached[key]; ok {
		w.lru.MoveToFront(el)
		return true
	}
	w.cached[key] = w.lru.PushFront(key)
	for w.lru.Len() > w.opts.CacheSize {
		last := w.lru.Back()
		w.lru.Remove(last)
		delete(w.cached, last.Value.(string))
	}
	return false
}

func sleepCtx(ctx context.Context, d time.Duration) bool {
	if d <= 0 {
		return ctx.Err() == nil
	}
	t := time.NewTimer(d)
	defer t.Stop()
	select {
	case <-t.C:
		return true
	case <-ctx.Done():
		return false
	}
}

func hangUp(rw http.ResponseWriter) {
	if hj, ok := rw.(http.Hijacker); ok {
		if conn, _, err := hj.Hijack(); err == nil {
			_ = conn.Close()
		}
	}
}

func (w *Worker) chat(rw http.ResponseWriter, r *http.Request) {
	// Read the whole body first: net/http only watches the connection for a client
	// disconnect (and so cancels r.Context()) once the body has been consumed. A
	// hang fault that returned early would otherwise never notice the gateway giving up.
	raw, _ := io.ReadAll(http.MaxBytesReader(rw, r.Body, 4<<20))

	f := w.currentFault()
	switch f.Mode {
	case FaultError503:
		http.Error(rw, `{"error":{"message":"worker overloaded"}}`, http.StatusServiceUnavailable)
		return
	case FaultReset:
		hangUp(rw)
		return
	case FaultHang:
		<-r.Context().Done()
		return
	}

	var req chatReq
	if err := json.Unmarshal(raw, &req); err != nil || len(req.Messages) == 0 {
		http.Error(rw, `{"error":{"message":"bad request"}}`, http.StatusBadRequest)
		return
	}
	warm := w.touch(prefixKey(req))

	w.mu.Lock()
	w.inflight++
	w.requests++
	if warm {
		w.warm++
	}
	inflight := w.inflight
	w.mu.Unlock()
	defer func() {
		w.mu.Lock()
		w.inflight--
		w.mu.Unlock()
	}()

	// GPU contention model: beyond Capacity, everything slows down proportionally.
	factor := math.Max(1, float64(inflight)/float64(w.opts.Capacity))
	scale := func(d time.Duration) time.Duration { return time.Duration(float64(d) * factor) }

	ttft := w.opts.TTFTCold
	if warm {
		ttft = w.opts.TTFTWarm
	}
	if f.Mode == FaultSlow {
		ttft += time.Duration(f.DelayMS) * time.Millisecond
	}
	tokens := w.opts.Tokens
	if req.MaxTokens > 0 {
		tokens = req.MaxTokens
	}
	tpot := scale(w.opts.TPOT)

	if !sleepCtx(r.Context(), scale(ttft)) {
		return
	}

	if !req.Stream {
		if !sleepCtx(r.Context(), time.Duration(tokens)*tpot) {
			return
		}
		rw.Header().Set("Content-Type", "application/json")
		fmt.Fprintf(rw, `{"id":"mock","object":"chat.completion","model":"mock","choices":[{"index":0,"message":{"role":"assistant","content":%q},"finish_reason":"stop"}],"usage":{"completion_tokens":%d}}`,
			strings.Repeat("tok ", tokens), tokens)
		return
	}

	rw.Header().Set("Content-Type", "text/event-stream")
	rw.Header().Set("Cache-Control", "no-cache")
	rw.WriteHeader(http.StatusOK)
	fl, _ := rw.(http.Flusher)
	for i := 0; i < tokens; i++ {
		if f.Mode == FaultMidstream && i == f.AfterTokens {
			hangUp(rw)
			return
		}
		if _, err := fmt.Fprintf(rw, "data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"tok%d \"}}]}\n\n", i); err != nil {
			return
		}
		if fl != nil {
			fl.Flush()
		}
		if !sleepCtx(r.Context(), tpot) {
			return
		}
	}
	fmt.Fprint(rw, "data: [DONE]\n\n")
	if fl != nil {
		fl.Flush()
	}
}
