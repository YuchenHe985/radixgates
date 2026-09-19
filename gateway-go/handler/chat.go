// Package handler implements the HTTP handlers for the gateway.
package handler

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"net/http"
)

// ── Request / Response types ─────────────────────────────────────────────────

type Message struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

type ChatRequest struct {
	Messages    []Message `json:"messages"`
	Model       string    `json:"model,omitempty"`
	Stream      *bool     `json:"stream,omitempty"`
	Temperature *float64  `json:"temperature,omitempty"`
	MaxTokens   *int      `json:"max_tokens,omitempty"`
	// Target controls which GPU group handles this request.
	//   "local" / "remote" → prefer that group
	//   omit / ""          → no preference, any group
	Target string `json:"target,omitempty"`
}

// computePrefixHash returns the SHA-256 hex of the system prompt content.
// This is the routing key that keeps same-context requests on the same GPU.
func computePrefixHash(systemContent string) string {
	input := systemContent
	if input == "" {
		input = "__no_system__"
	}
	h := sha256.Sum256([]byte(input))
	return hex.EncodeToString(h[:])
}

// writeJSON writes v as a JSON response with the given status code.
func writeJSON(w http.ResponseWriter, status int, v interface{}) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(v)
}
