package main

import (
	"net/http"
	"net/http/httptest"
	"testing"
)

func TestChatRoutesExposeLegacyAndOpenAIPaths(t *testing.T) {
	mux := http.NewServeMux()
	registerChatRoutes(mux, http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusNoContent)
	}))

	for _, path := range []string{"/v1/chat", "/v1/chat/completions"} {
		req := httptest.NewRequest(http.MethodPost, path, nil)
		rr := httptest.NewRecorder()
		mux.ServeHTTP(rr, req)
		if rr.Code != http.StatusNoContent {
			t.Fatalf("POST %s: got %d, want %d", path, rr.Code, http.StatusNoContent)
		}
	}
}
