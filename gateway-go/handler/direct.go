// Package handler — DirectChatHandler routes POST /v1/chat directly to SGLang,
// streaming the response back to the client via Server-Sent Events (SSE).
// The gateway forwards synchronously; no message broker or state store is involved.
package handler

import (
	"bufio"
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"strings"
	"time"

	"github.com/unicoregpu/radixgates/gateway/metrics"
	"github.com/unicoregpu/radixgates/gateway/router"
)

// DirectChatHandler handles POST /v1/chat synchronously.
// Flow: validate → prefix-hash → acquire semaphore slot → forward to SGLang → SSE stream back.
type DirectChatHandler struct {
	Router       *router.Router
	DefaultModel string
	Timeout      time.Duration // per-request inference timeout; 0 → 10 min
}

func (h *DirectChatHandler) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}

	start := time.Now()
	metrics.ActiveRequests.Inc()
	defer func() {
		metrics.ActiveRequests.Dec()
		metrics.RequestDuration.Observe(time.Since(start).Seconds())
	}()

	// 1. Parse body
	var req ChatRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil || len(req.Messages) == 0 {
		metrics.RequestsTotal.WithLabelValues("bad_request").Inc()
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
			systemContent = m.Content
			break
		}
	}
	prefixHash := computePrefixHash(systemContent)

	// 4. Select SGLang node
	node, ok := h.Router.Select(prefixHash, req.Target)
	if !ok {
		metrics.RequestsTotal.WithLabelValues("no_node").Inc()
		writeJSON(w, http.StatusServiceUnavailable, map[string]string{"error": "no SGLang instance available for target group"})
		return
	}

	// 5. Acquire concurrency slot (blocks until free or client disconnects)
	if err := node.Acquire(r.Context()); err != nil {
		metrics.RequestsTotal.WithLabelValues("cancelled").Inc()
		return // client disconnected while waiting
	}
	defer node.Release()

	// 6. Build SGLang request
	sgPayload := map[string]interface{}{
		"model":       model,
		"messages":    req.Messages,
		"stream":      streamMode,
		"temperature": temperature,
		"max_tokens":  maxTokens,
	}
	body, _ := json.Marshal(sgPayload)

	timeout := h.Timeout
	if timeout == 0 {
		timeout = 10 * time.Minute
	}
	sgReq, err := http.NewRequestWithContext(r.Context(), http.MethodPost,
		node.URL+"/v1/chat/completions", bytes.NewReader(body))
	if err != nil {
		writeJSON(w, http.StatusInternalServerError, map[string]string{"error": err.Error()})
		return
	}
	sgReq.Header.Set("Content-Type", "application/json")

	client := &http.Client{Timeout: timeout}
	resp, err := client.Do(sgReq)
	if err != nil {
		metrics.RequestsTotal.WithLabelValues("sglang_error").Inc()
		log.Printf("[DirectHandler] SGLang request failed: %v", err)
		writeJSON(w, http.StatusBadGateway, map[string]string{"error": fmt.Sprintf("inference engine error: %v", err)})
		return
	}
	defer resp.Body.Close()

	metrics.RequestsTotal.WithLabelValues("direct_ok").Inc()
	log.Printf("[DirectHandler] prefix=%s node=%s stream=%v", prefixHash[:8], node.URL, streamMode)

	if streamMode {
		// 7a. SSE streaming — proxy SGLang's token stream directly to the client
		w.Header().Set("Content-Type", "text/event-stream")
		w.Header().Set("Cache-Control", "no-cache")
		w.Header().Set("X-Accel-Buffering", "no")
		w.WriteHeader(http.StatusOK)

		flusher, canFlush := w.(http.Flusher)
		scanner := bufio.NewScanner(resp.Body)
		for scanner.Scan() {
			line := scanner.Text()
			if strings.TrimSpace(line) == "" {
				continue
			}
			// SGLang returns "data: {...}" lines; forward as-is
			fmt.Fprintf(w, "%s\n\n", line)
			if canFlush {
				flusher.Flush()
			}
			if strings.Contains(line, "[DONE]") {
				break
			}
		}
	} else {
		// 7b. Non-streaming — read entire response, return as JSON
		respBody, err := io.ReadAll(resp.Body)
		if err != nil {
			writeJSON(w, http.StatusBadGateway, map[string]string{"error": "failed to read response"})
			return
		}
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(resp.StatusCode)
		_, _ = w.Write(respBody)
	}
}
