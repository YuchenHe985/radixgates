package handler

import (
	"bufio"
	"context"
	"fmt"
	"io"
	"net"
	"net/http"
	"net/http/httptest"
	"strconv"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/unicoregpu/radixgates/gateway/breaker"
	"github.com/unicoregpu/radixgates/gateway/config"
	"github.com/unicoregpu/radixgates/gateway/internal/mock"
	"github.com/unicoregpu/radixgates/gateway/router"
)

// ---- harness: real router + handler in front of simulated SGLang workers ----------------------

type rig struct {
	t       *testing.T
	workers []*mock.Worker
	servers []*httptest.Server
	router  *router.Router
	front   *httptest.Server
}

type rigOpts struct {
	n           int
	maxConc     int
	breaker     bool
	maxAttempts int
	maxQueue    int
	queueWait   time.Duration
	health      bool
}

func newRig(t *testing.T, o rigOpts) *rig {
	t.Helper()
	if o.maxConc == 0 {
		o.maxConc = 16
	}
	if o.maxAttempts == 0 {
		o.maxAttempts = 3
	}
	if o.maxQueue == 0 {
		o.maxQueue = 64
	}
	if o.queueWait == 0 {
		o.queueWait = 500 * time.Millisecond
	}
	r := &rig{t: t}
	var inst []config.SGLangInstance
	for i := 0; i < o.n; i++ {
		w := mock.New(mock.Options{Name: fmt.Sprintf("w%d", i), TTFTCold: 2 * time.Millisecond, TTFTWarm: time.Millisecond,
			TPOT: time.Millisecond, Tokens: 4, Capacity: 64})
		s := httptest.NewServer(w.Handler())
		r.workers, r.servers = append(r.workers, w), append(r.servers, s)
		host, port, _ := net.SplitHostPort(strings.TrimPrefix(s.URL, "http://"))
		p, _ := strconv.Atoi(port)
		inst = append(inst, config.SGLangInstance{Group: "local", Host: host, Port: p, MaxConcurrent: o.maxConc})
	}
	ro := router.Options{BoundedLoadFactor: 1.25, AffinityFloor: 4, MaxQueue: o.maxQueue, QueueTimeout: o.queueWait}
	if o.breaker {
		ro.Breaker = &breaker.Config{FailureThreshold: 2, OpenDuration: 150 * time.Millisecond, MaxOpenDuration: time.Second, HalfOpenMax: 1}
	}
	r.router = router.New(inst, o.maxConc, ro)
	rel := config.DefaultReliability()
	rel.MaxAttempts = o.maxAttempts
	rel.AttemptTimeout = config.Duration(400 * time.Millisecond)
	rel.StreamIdleTimeout = config.Duration(500 * time.Millisecond)
	rel.RetryBackoff = config.Duration(time.Millisecond)
	h := &DirectChatHandler{Router: r.router, DefaultModel: "m", Reliability: rel}
	mux := http.NewServeMux()
	mux.Handle("POST /v1/chat", h)
	r.front = httptest.NewServer(mux)
	t.Cleanup(func() {
		r.front.Close()
		for _, s := range r.servers {
			s.Close()
		}
	})
	return r
}

type result struct {
	status   int
	node     string
	attempts string
	body     string
	err      error
}

func (r *rig) chat(system string, stream bool) result {
	return r.chatCtx(context.Background(), system, stream)
}

func (r *rig) chatCtx(ctx context.Context, system string, stream bool) result {
	// max_tokens is explicit: the handler defaults it to 512 (original behaviour) and the mock honours it.
	body := fmt.Sprintf(`{"stream":%t,"max_tokens":4,"messages":[{"role":"system","content":%q},{"role":"user","content":"hi"}]}`, stream, system)
	req, _ := http.NewRequestWithContext(ctx, http.MethodPost, r.front.URL+"/v1/chat", strings.NewReader(body))
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return result{err: err}
	}
	defer resp.Body.Close()
	b, err := io.ReadAll(resp.Body)
	return result{status: resp.StatusCode, node: resp.Header.Get(HeaderNode), attempts: resp.Header.Get(HeaderAttempts), body: string(b), err: err}
}

// keyOwnedBy finds a system prompt whose preferred node is worker i.
func (r *rig) keyOwnedBy(i int) string {
	want := r.router.Nodes()[i].Name
	for k := 0; k < 1000; k++ {
		key := fmt.Sprintf("prompt-%d", k)
		if r.router.Ranking(computePrefixHash(key))[0] == want {
			return key
		}
	}
	r.t.Fatalf("no key for worker %d", i)
	return ""
}

func (r *rig) waitIdle() {
	r.t.Helper()
	deadline := time.Now().Add(2 * time.Second)
	for time.Now().Before(deadline) {
		idle := true
		for _, s := range r.router.Snapshot() {
			if s.Inflight != 0 {
				idle = false
			}
		}
		if idle {
			return
		}
		time.Sleep(5 * time.Millisecond)
	}
	r.t.Fatalf("in-flight leaked: %+v", r.router.Snapshot())
}

// ---- tests ---------------------------------------------------------------------------------------

func TestStreamingPassthroughAndAffinity(t *testing.T) {
	r := newRig(t, rigOpts{n: 4, breaker: true})
	first := r.chat("finance assistant", true)
	if first.status != 200 || first.err != nil || !strings.Contains(first.body, "data: [DONE]") || strings.Count(first.body, "delta") != 4 {
		t.Fatalf("stream not proxied intact: %+v", first)
	}
	for i := 0; i < 8; i++ {
		if res := r.chat("finance assistant", true); res.node != first.node {
			t.Fatalf("prefix affinity broken: %s then %s", first.node, res.node)
		}
	}
	r.waitIdle()
}

func TestNonStreamingResponseAndBadRequest(t *testing.T) {
	r := newRig(t, rigOpts{n: 2, breaker: true})
	if res := r.chat("sys", false); res.status != 200 || !strings.Contains(res.body, `"chat.completion"`) {
		t.Fatalf("non-stream: %+v", res)
	}
	resp, err := http.Post(r.front.URL+"/v1/chat", "application/json", strings.NewReader(`{"messages":[]}`))
	if err != nil {
		t.Fatal(err)
	}
	resp.Body.Close()
	if resp.StatusCode != http.StatusBadRequest {
		t.Fatalf("want 400, got %d", resp.StatusCode)
	}
}

func TestDeadNodeIsRetriedThenSkipped(t *testing.T) {
	r := newRig(t, rigOpts{n: 4, breaker: true})
	key := r.keyOwnedBy(1)
	r.servers[1].Close() // crash: connection refused

	res := r.chat(key, true)
	if res.status != 200 || res.attempts != "2" || res.node == r.router.Nodes()[1].Name {
		t.Fatalf("request must survive a dead preferred node via one retry: %+v", res)
	}
	_ = r.chat(key, true) // second failure trips the breaker (threshold 2)
	if st := r.router.Snapshot()[1].Breaker; st != "open" {
		t.Fatalf("breaker should be open, is %s", st)
	}
	for i := 0; i < 5; i++ {
		if res := r.chat(key, true); res.status != 200 || res.attempts != "1" {
			t.Fatalf("with the breaker open requests must skip the dead node: %+v", res)
		}
	}
	r.waitIdle()
}

// This is the failure mode of the original gateway: one attempt, so a dead node means a 502.
func TestSingleAttemptReproducesTheOriginalFailure(t *testing.T) {
	r := newRig(t, rigOpts{n: 4, maxAttempts: 1})
	key := r.keyOwnedBy(2)
	r.servers[2].Close()
	if res := r.chat(key, true); res.status != http.StatusBadGateway {
		t.Fatalf("with max_attempts=1 a dead node must surface as 502, got %d", res.status)
	}
}

func TestHTTP503AndHangFailOver(t *testing.T) {
	r := newRig(t, rigOpts{n: 3, breaker: true})
	k0 := r.keyOwnedBy(0)
	r.workers[0].SetFault(mock.Fault{Mode: mock.FaultError503})
	for i := 0; i < 4; i++ {
		if res := r.chat(k0, true); res.status != 200 {
			t.Fatalf("503 storm request %d failed: %d", i, res.status)
		}
	}
	if r.router.Snapshot()[0].Breaker != "open" {
		t.Fatal("breaker should open after repeated 5xx")
	}

	// separate rig: after 150ms the breaker above would half-open and probe the still-failing node
	r = newRig(t, rigOpts{n: 3, breaker: true})
	k1 := r.keyOwnedBy(1)
	r.workers[1].SetFault(mock.Fault{Mode: mock.FaultHang})
	start := time.Now()
	res := r.chat(k1, true)
	el := time.Since(start)
	if res.status != 200 || res.attempts != "2" || el < 350*time.Millisecond || el > 1500*time.Millisecond {
		t.Fatalf("hang must time out (~400ms) and fail over: %+v after %v", res, el)
	}
}

func TestMidStreamFailureIsNotRetried(t *testing.T) {
	r := newRig(t, rigOpts{n: 2, breaker: true})
	key := r.keyOwnedBy(0)
	r.workers[0].SetFault(mock.Fault{Mode: mock.FaultMidstream, AfterTokens: 2})
	res := r.chat(key, true)
	if res.attempts != "1" || strings.Count(res.body, "delta") != 2 ||
		!strings.Contains(res.body, "upstream_interrupted") || strings.Contains(res.body, "[DONE]") {
		t.Fatalf("output already delivered must not be retried; client needs the tokens then an error event: %+v", res)
	}
}

func TestBreakerRecoversAndAffinityReturns(t *testing.T) {
	r := newRig(t, rigOpts{n: 3, breaker: true})
	key := r.keyOwnedBy(0)
	r.workers[0].SetFault(mock.Fault{Mode: mock.FaultError503})
	for i := 0; i < 3; i++ {
		_ = r.chat(key, true)
	}
	r.workers[0].SetFault(mock.Fault{Mode: mock.FaultNone})
	time.Sleep(200 * time.Millisecond) // > open_duration: the next request is the half-open probe
	res := r.chat(key, true)
	if res.status != 200 || res.node != r.router.Nodes()[0].Name || r.router.Snapshot()[0].Breaker != "closed" {
		t.Fatalf("probe should reach the preferred node and close the breaker: %+v %+v", res, r.router.Snapshot()[0])
	}
}

func TestHealthCheckRemovesAndRestoresNode(t *testing.T) {
	r := newRig(t, rigOpts{n: 3, breaker: true})
	key := r.keyOwnedBy(1)
	client := &http.Client{}
	state := router.NewProbeState()
	h := router.HealthOptions{Timeout: 200 * time.Millisecond, Path: "/health", UnhealthyThreshold: 2, HealthyThreshold: 1}
	r.workers[1].SetFault(mock.Fault{Mode: mock.FaultHealthDown})
	r.router.ProbeAll(context.Background(), h, client, state)
	if !r.router.Snapshot()[1].Up {
		t.Fatal("one failed probe must not mark the node down (threshold 2)")
	}
	r.router.ProbeAll(context.Background(), h, client, state)
	if r.router.Snapshot()[1].Up {
		t.Fatal("two failed probes must mark it down")
	}
	if res := r.chat(key, true); res.node == r.router.Nodes()[1].Name || res.attempts != "1" {
		t.Fatalf("a node marked down must be avoided proactively, without a failed attempt: %+v", res)
	}
	r.workers[1].SetFault(mock.Fault{Mode: mock.FaultNone})
	r.router.ProbeAll(context.Background(), h, client, state)
	if res := r.chat(key, true); res.node != r.router.Nodes()[1].Name {
		t.Fatalf("affinity must return once healthy: %+v", res)
	}
}

func TestAllNodesDown(t *testing.T) {
	r := newRig(t, rigOpts{n: 2, breaker: true})
	for _, s := range r.servers {
		s.Close()
	}
	if res := r.chat("x", true); res.status != http.StatusBadGateway {
		t.Fatalf("every attempt failed: want 502, got %d", res.status)
	}
	for i := 0; i < 3; i++ {
		_ = r.chat("x", true)
	}
	if res := r.chat("x", true); res.status != http.StatusServiceUnavailable {
		t.Fatalf("all breakers open: want a fast 503, got %d", res.status)
	}
	if r.router.AnyAvailable() {
		t.Fatal("AnyAvailable must be false when every breaker is open")
	}
}

func TestQueueLosesNothingAndOverflowGets429(t *testing.T) {
	r := newRig(t, rigOpts{n: 1, maxConc: 2, maxQueue: 64, queueWait: 5 * time.Second, breaker: true})
	var wg sync.WaitGroup
	oks := make(chan bool, 24)
	for i := 0; i < 24; i++ {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			res := r.chat(fmt.Sprintf("k%d", i%3), true)
			oks <- res.status == 200 && strings.Contains(res.body, "[DONE]")
		}(i)
	}
	wg.Wait()
	close(oks)
	for ok := range oks {
		if !ok {
			t.Fatal("a queued request was dropped or truncated")
		}
	}
	r.waitIdle()

	r2 := newRig(t, rigOpts{n: 1, maxConc: 1, maxQueue: 1, queueWait: 2 * time.Second})
	r2.workers[0].SetFault(mock.Fault{Mode: mock.FaultSlow, DelayMS: 300})
	statuses := make(chan int, 6)
	var wg2 sync.WaitGroup
	for i := 0; i < 6; i++ {
		wg2.Add(1)
		go func() { defer wg2.Done(); statuses <- r2.chat("k", true).status }()
		time.Sleep(20 * time.Millisecond)
	}
	wg2.Wait()
	close(statuses)
	saw429 := false
	for s := range statuses {
		if s == http.StatusTooManyRequests {
			saw429 = true
		}
	}
	if !saw429 {
		t.Fatal("expected a 429 when the admission queue overflows")
	}
}

func TestClientCancelReleasesSlotWithoutBlamingTheNode(t *testing.T) {
	r := newRig(t, rigOpts{n: 1, breaker: true})
	r.workers[0].SetFault(mock.Fault{Mode: mock.FaultSlow, DelayMS: 2000})
	ctx, cancel := context.WithCancel(context.Background())
	done := make(chan result, 1)
	go func() { done <- r.chatCtx(ctx, "k", true) }()
	time.Sleep(80 * time.Millisecond)
	cancel()
	<-done
	r.waitIdle()
	if st := r.router.Snapshot()[0].Breaker; st != "closed" {
		t.Fatalf("a client cancel must not count against the node, breaker=%s", st)
	}
}

func TestSSEIsForwardedIncrementally(t *testing.T) {
	r := newRig(t, rigOpts{n: 1})
	r.workers[0] = nil
	w := mock.New(mock.Options{Name: "slow", TTFTCold: time.Millisecond, TTFTWarm: time.Millisecond, TPOT: 120 * time.Millisecond, Tokens: 5, Capacity: 8})
	s := httptest.NewServer(w.Handler())
	defer s.Close()
	host, port, _ := net.SplitHostPort(strings.TrimPrefix(s.URL, "http://"))
	p, _ := strconv.Atoi(port)
	rt := router.New([]config.SGLangInstance{{Host: host, Port: p, MaxConcurrent: 4}}, 4,
		router.Options{BoundedLoadFactor: 1.25, AffinityFloor: 4, MaxQueue: 4, QueueTimeout: time.Second})
	mux := http.NewServeMux()
	mux.Handle("POST /v1/chat", &DirectChatHandler{Router: rt, DefaultModel: "m"})
	front := httptest.NewServer(mux)
	defer front.Close()

	resp, err := http.Post(front.URL+"/v1/chat", "application/json", strings.NewReader(`{"stream":true,"max_tokens":5,"messages":[{"role":"user","content":"hi"}]}`))
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	start := time.Now()
	sc := bufio.NewScanner(resp.Body)
	var first time.Duration
	for sc.Scan() {
		if strings.HasPrefix(sc.Text(), "data:") && first == 0 {
			first = time.Since(start)
		}
	}
	if total := time.Since(start); first > total/2 {
		t.Fatalf("first token at %v of %v: the handler is buffering the stream", first, total)
	}
}
