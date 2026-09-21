// Package handler — DirectChatHandler routes POST /v1/chat directly to SGLang,
// streaming the response back to the client via Server-Sent Events (SSE).
//
// Compared with the original single-shot forwarder it adds failover: the request
// is retried on another node while it is still safe to do so.
//
// Retry safety rule ("first-byte commit"): a request is retried only while no byte
// of the response has been sent to the client. The handler therefore reads the
// first chunk of the upstream body before it writes anything downstream. A node
// that dies, hangs or answers 5xx before producing its first token is retried
// transparently; a stream that breaks after tokens were delivered is ended with an
// explicit error event and never silently retried (that would duplicate output).
package handler

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log"
	"math/rand"
	"net"
	"net/http"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/unicoregpu/radixgates/gateway/config"
	"github.com/unicoregpu/radixgates/gateway/metrics"
	"github.com/unicoregpu/radixgates/gateway/router"
)

const (
	HeaderNode     = "X-Gateway-Node"
	HeaderAttempts = "X-Gateway-Attempts"
)

// DirectChatHandler handles POST /v1/chat.
// Flow: validate -> prefix-hash -> acquire a node lease (queue if full) -> forward -> stream back,
// retrying on another node before the first byte is committed.
type DirectChatHandler struct {
	Router       *router.Router
	DefaultModel string
	Timeout      time.Duration      // deprecated: overall timeout; Reliability.OverallTimeout wins when set
	Reliability  config.Reliability // zero fields fall back to config.DefaultReliability()

	once   sync.Once
	client *http.Client
	rel    config.Reliability
}

func (h *DirectChatHandler) init() {
	h.once.Do(func() {
		def := config.DefaultReliability()
		r := h.Reliability
		if r.MaxAttempts == 0 {
			r.MaxAttempts = def.MaxAttempts
		}
		if r.AttemptTimeout == 0 {
			r.AttemptTimeout = def.AttemptTimeout
		}
		if r.StreamIdleTimeout == 0 {
			r.StreamIdleTimeout = def.StreamIdleTimeout
		}
		if r.RetryBackoff == 0 {
			r.RetryBackoff = def.RetryBackoff
		}
		if r.OverallTimeout == 0 {
			r.OverallTimeout = def.OverallTimeout
			if h.Timeout > 0 {
				r.OverallTimeout = config.Duration(h.Timeout)
			}
		}
		h.rel = r
		// One shared client: the original built a new http.Client (and connection pool) per request.
		h.client = &http.Client{Transport: &http.Transport{
			MaxIdleConns:        256,
			MaxIdleConnsPerHost: 64,
			IdleConnTimeout:     90 * time.Second,
			DialContext:         (&net.Dialer{Timeout: 2 * time.Second}).DialContext,
		}}
	})
}

// Client exposes the shared upstream client (used by the health checker).
func (h *DirectChatHandler) Client() *http.Client { h.init(); return h.client }

type trackedWriter struct {
	http.ResponseWriter
	wrote bool
}

func (t *trackedWriter) WriteHeader(code int) { t.wrote = true; t.ResponseWriter.WriteHeader(code) }
func (t *trackedWriter) Write(p []byte) (int, error) {
	t.wrote = true
	return t.ResponseWriter.Write(p)
}
func (t *trackedWriter) Flush() {
	if f, ok := t.ResponseWriter.(http.Flusher); ok {
		f.Flush()
	}
}

type outcome int

const (
	outDone  outcome = iota // response finished (fully or ended with an error event)
	outRetry                // nothing was sent to the client; try another node
	outAbort                // client went away or the deadline passed
)

func (h *DirectChatHandler) ServeHTTP(rw http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(rw, "method not allowed", http.StatusMethodNotAllowed)
		return
	}
	h.init()

	start := time.Now()
	w := &trackedWriter{ResponseWriter: rw}
	metrics.ActiveRequests.Inc()
	defer func() {
		metrics.ActiveRequests.Dec()
		metrics.RequestDuration.Observe(time.Since(start).Seconds())
	}()
	finish := func(o string) { metrics.RequestsTotal.WithLabelValues(o).Inc() }

	// 1. Parse the routing fields, while retaining the complete request so an
	// OpenAI-compatible client does not lose tools, response_format, top_p,
	// multimodal content, or future upstream fields.
	raw, err := io.ReadAll(http.MaxBytesReader(rw, r.Body, 4<<20))
	if err != nil {
		finish("bad_request")
		writeJSON(w, http.StatusBadRequest, map[string]string{"error": "request body is invalid or exceeds 4 MiB"})
		return
	}
	var req ChatRequest
	var fields map[string]json.RawMessage
	if err := json.Unmarshal(raw, &req); err != nil || len(req.Messages) == 0 || json.Unmarshal(raw, &fields) != nil {
		finish("bad_request")
		writeJSON(w, http.StatusBadRequest, map[string]string{"error": "missing or invalid 'messages' field"})
		return
	}

	// 2. Defaults
	model := h.DefaultModel
	if req.Model != "" && req.Model != "default" {
		model = req.Model
	}
	streamMode := true
	if req.Stream != nil {
		streamMode = *req.Stream
	}
	temperature := 0.7
	if req.Temperature != nil {
		temperature = *req.Temperature
	}
	maxTokens := 512
	if req.MaxTokens != nil {
		maxTokens = *req.MaxTokens
	}

	// 3. Prefix hash for consistent routing
	systemContent := ""
	for _, m := range req.Messages {
		if m.Role == "system" {
			systemContent = messageContentForRouting(m.Content)
			break
		}
	}
	prefixHash := computePrefixHash(systemContent)

	fields["model"], _ = json.Marshal(model)
	fields["stream"], _ = json.Marshal(streamMode)
	if _, ok := fields["temperature"]; !ok {
		fields["temperature"], _ = json.Marshal(temperature)
	}
	if _, ok := fields["max_tokens"]; !ok {
		fields["max_tokens"], _ = json.Marshal(maxTokens)
	}
	delete(fields, "target") // gateway-only routing hint; SGLang should not see it
	body, err := json.Marshal(fields)
	if err != nil {
		finish("bad_request")
		writeJSON(w, http.StatusBadRequest, map[string]string{"error": "request body could not be forwarded"})
		return
	}

	ctx, cancel := context.WithTimeout(r.Context(), h.rel.OverallTimeout.Std())
	defer cancel()

	// 4. Acquire a node, forward, retry on another node while that is still safe
	exclude := map[string]struct{}{}
	lastReason := ""
	for attempt := 1; attempt <= h.rel.MaxAttempts; attempt++ {
		qStart := time.Now()
		lease, err := h.Router.Acquire(ctx, prefixHash, req.Target, nil, exclude)
		metrics.QueueWait.Observe(time.Since(qStart).Seconds())
		if err != nil {
			h.rejectAcquire(w, err, attempt, finish)
			return
		}
		if attempt > 1 {
			metrics.Retries.WithLabelValues(lastReason).Inc()
		}
		o, reason := h.attempt(ctx, w, lease, body, streamMode, attempt, start, prefixHash, finish)
		switch o {
		case outDone:
			return
		case outAbort:
			if !w.wrote && errors.Is(ctx.Err(), context.DeadlineExceeded) {
				finish("timeout")
				writeJSON(w, http.StatusGatewayTimeout, map[string]string{"error": "request deadline exceeded"})
			} else if !w.wrote {
				finish("cancelled")
			}
			return
		case outRetry:
			lastReason = reason
			exclude[lease.N.Name] = struct{}{}
			if attempt < h.rel.MaxAttempts && !sleepCtx(ctx, h.backoff(attempt)) {
				finish("cancelled")
				return
			}
		}
	}
	finish("upstream_failed")
	w.Header().Set("Retry-After", "1")
	writeJSON(w, http.StatusBadGateway, map[string]string{
		"error": fmt.Sprintf("inference engine error: all %d attempts failed (last: %s)", h.rel.MaxAttempts, lastReason)})
}

func (h *DirectChatHandler) rejectAcquire(w *trackedWriter, err error, attempt int, finish func(string)) {
	w.Header().Set("Retry-After", "1")
	switch {
	case errors.Is(err, router.ErrQueueFull):
		finish("queue_full")
		writeJSON(w, http.StatusTooManyRequests, map[string]string{"error": "gateway admission queue is full"})
	case errors.Is(err, router.ErrQueueTimeout):
		finish("queue_timeout")
		writeJSON(w, http.StatusServiceUnavailable, map[string]string{"error": "timed out waiting for node capacity"})
	case errors.Is(err, router.ErrNoNode):
		if attempt > 1 {
			finish("upstream_failed")
			writeJSON(w, http.StatusBadGateway, map[string]string{"error": "inference engine error: every available node failed"})
			return
		}
		finish("no_node")
		writeJSON(w, http.StatusServiceUnavailable, map[string]string{"error": "no SGLang instance available for target group"})
	case errors.Is(err, context.DeadlineExceeded):
		finish("timeout")
		writeJSON(w, http.StatusGatewayTimeout, map[string]string{"error": "request deadline exceeded while waiting for capacity"})
	default:
		finish("cancelled") // client disconnected while waiting
	}
}

func (h *DirectChatHandler) backoff(attempt int) time.Duration {
	d := h.rel.RetryBackoff.Std() << (attempt - 1)
	if d > time.Second {
		d = time.Second
	}
	return d/2 + time.Duration(rand.Int63n(int64(d/2)+1)) // 50-100% jitter
}

func sleepCtx(ctx context.Context, d time.Duration) bool {
	t := time.NewTimer(d)
	defer t.Stop()
	select {
	case <-t.C:
		return true
	case <-ctx.Done():
		return false
	}
}

// attempt makes one upstream try. On outRetry the lease was already reported to the
// breaker and nothing has been written to the client.
func (h *DirectChatHandler) attempt(ctx context.Context, w *trackedWriter, lease *router.Lease, body []byte,
	stream bool, attempt int, start time.Time, prefixHash string, finish func(string)) (outcome, string) {

	n := lease.N
	actx, acancel := context.WithCancel(ctx)
	defer acancel()

	// Non-streaming answers only appear once generation ends, so they get the
	// overall budget instead of the short first-token timeout.
	firstByte := h.rel.AttemptTimeout.Std()
	if !stream {
		firstByte = h.rel.OverallTimeout.Std()
	}
	var stalled atomic.Bool
	fb := time.AfterFunc(firstByte, func() { stalled.Store(true); acancel() })
	defer fb.Stop()

	fail := func(result string) (outcome, string) {
		lease.Done(false)
		metrics.UpstreamAttempts.WithLabelValues(n.Name, result).Inc()
		return outRetry, result
	}
	abort := func() (outcome, string) {
		lease.Release()
		metrics.UpstreamAttempts.WithLabelValues(n.Name, "canceled").Inc()
		return outAbort, "canceled"
	}

	req, err := http.NewRequestWithContext(actx, http.MethodPost, n.URL+"/v1/chat/completions", bytes.NewReader(body))
	if err != nil {
		lease.Release()
		return outRetry, "internal_error"
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := h.client.Do(req)
	if err != nil {
		fb.Stop()
		if ctx.Err() != nil {
			return abort()
		}
		log.Printf("[DirectHandler] node=%s attempt=%d failed: %v", n.Name, attempt, err)
		if stalled.Load() {
			return fail("timeout")
		}
		return fail("conn_error")
	}
	defer resp.Body.Close()

	switch {
	case resp.StatusCode == http.StatusTooManyRequests:
		// Node is busy, not broken: try elsewhere without charging the breaker.
		fb.Stop()
		_, _ = io.Copy(io.Discard, io.LimitReader(resp.Body, 4096))
		lease.Release()
		metrics.UpstreamAttempts.WithLabelValues(n.Name, "busy").Inc()
		return outRetry, "busy"
	case resp.StatusCode >= 500 && resp.StatusCode != http.StatusNotImplemented:
		fb.Stop()
		_, _ = io.Copy(io.Discard, io.LimitReader(resp.Body, 4096))
		return fail("http_5xx")
	}

	// Read the first chunk BEFORE committing anything to the client.
	buf := make([]byte, 32<<10)
	nb, rerr := resp.Body.Read(buf)
	fb.Stop()
	if nb == 0 && rerr != nil && rerr != io.EOF {
		if ctx.Err() != nil {
			return abort()
		}
		if stalled.Load() {
			return fail("timeout")
		}
		return fail("read_error")
	}

	// ---- commit point: from here on this request is never retried ----
	hd := w.Header()
	if ct := resp.Header.Get("Content-Type"); ct != "" {
		hd.Set("Content-Type", ct)
	}
	isSSE := strings.HasPrefix(resp.Header.Get("Content-Type"), "text/event-stream")
	if isSSE {
		hd.Set("Cache-Control", "no-cache")
		hd.Set("X-Accel-Buffering", "no")
	}
	hd.Set(HeaderNode, n.Name)
	hd.Set(HeaderAttempts, fmt.Sprint(attempt))
	w.WriteHeader(resp.StatusCode)
	if nb > 0 {
		if _, err := w.Write(buf[:nb]); err != nil {
			lease.Release()
			metrics.UpstreamAttempts.WithLabelValues(n.Name, "canceled").Inc()
			finish("cancelled")
			return outAbort, "canceled"
		}
	}
	w.Flush()
	metrics.TimeToFirstByte.Observe(time.Since(start).Seconds())
	log.Printf("[DirectHandler] prefix=%s node=%s stream=%v attempt=%d", prefixHash[:8], n.Name, stream, attempt)

	idle := time.AfterFunc(h.rel.StreamIdleTimeout.Std(), func() { stalled.Store(true); acancel() })
	defer idle.Stop()
	for rerr == nil {
		nb, rerr = resp.Body.Read(buf)
		idle.Reset(h.rel.StreamIdleTimeout.Std())
		if nb > 0 {
			if _, err := w.Write(buf[:nb]); err != nil {
				lease.Release()
				metrics.UpstreamAttempts.WithLabelValues(n.Name, "canceled").Inc()
				finish("cancelled")
				return outAbort, "canceled"
			}
			w.Flush()
		}
	}

	if rerr == io.EOF {
		lease.Done(true)
		metrics.UpstreamAttempts.WithLabelValues(n.Name, "ok").Inc()
		finish("direct_ok")
		return outDone, "ok"
	}

	// The stream broke after output had started.
	if ctx.Err() != nil {
		lease.Release()
		metrics.UpstreamAttempts.WithLabelValues(n.Name, "canceled").Inc()
		finish("cancelled")
		return outAbort, "canceled"
	}
	lease.Done(false)
	metrics.UpstreamAttempts.WithLabelValues(n.Name, "midstream_error").Inc()
	metrics.MidstreamFailures.WithLabelValues(n.Name).Inc()
	finish("midstream_failed")
	log.Printf("[DirectHandler] node=%s stream interrupted after output started; not retrying: %v", n.Name, rerr)
	if isSSE {
		_, _ = w.Write([]byte(`data: {"error":{"type":"upstream_interrupted","message":"node stream interrupted after output started; the request was not retried"}}` + "\n\n"))
		w.Flush()
		return outDone, "midstream_error"
	}
	panic(http.ErrAbortHandler) // a truncated non-SSE body must not look complete
}
