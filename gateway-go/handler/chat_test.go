package handler

import "testing"

func TestComputePrefixHash(t *testing.T) {
	// Empty string → consistent fallback key
	h1 := computePrefixHash("")
	h2 := computePrefixHash("")
	if h1 != h2 {
		t.Error("hash should be deterministic for empty input")
	}

	// Different inputs → different hashes
	ha := computePrefixHash("context A")
	hb := computePrefixHash("context B")
	if ha == hb {
		t.Error("different inputs should produce different hashes")
	}

	// Valid hex string, length = 64 (SHA-256 = 32 bytes = 64 hex chars)
	if len(h1) != 64 {
		t.Errorf("expected 64-char hex, got %d chars", len(h1))
	}
}
