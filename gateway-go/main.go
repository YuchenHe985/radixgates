package main

import (
	"context"
	"fmt"
	"log"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/prometheus/client_golang/prometheus/promhttp"
	"github.com/unicoregpu/radixgates/gateway/config"
	"github.com/unicoregpu/radixgates/gateway/handler"
	_ "github.com/unicoregpu/radixgates/gateway/metrics" // register metrics
	"github.com/unicoregpu/radixgates/gateway/router"
)

func main() {
	// ── Load config ──────────────────────────────────────────────────────────
	cfgPath := "config/config.json"
	if p := os.Getenv("CONFIG_PATH"); p != "" {
		cfgPath = p
	}
	cfg, err := config.Load(cfgPath)
	if err != nil {
		log.Fatalf("[Gateway] Failed to load config: %v", err)
	}

	// ── Register routes ──────────────────────────────────────────────────────
	mux := http.NewServeMux()
	setupDirectRoutes(mux, cfg)

	// Health + metrics
	mux.HandleFunc("GET /health", func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte(`{"status":"ok","mode":"direct"}`))
	})
	mux.Handle("GET /metrics", promhttp.Handler())

	// ── Start HTTP server ────────────────────────────────────────────────────
	srv := &http.Server{
		Addr:         fmt.Sprintf("0.0.0.0:%d", cfg.Port),
		Handler:      mux,
		ReadTimeout:  30 * time.Second,
		WriteTimeout: 15 * time.Minute, // long for streaming responses
		IdleTimeout:  60 * time.Second,
	}

	go func() {
		log.Printf("[Gateway] Listening on :%d", cfg.Port)
		if err := srv.ListenAndServe(); err != nil && err != http.ErrServerClosed {
			log.Fatalf("[Gateway] Server error: %v", err)
		}
	}()

	// ── Graceful shutdown ────────────────────────────────────────────────────
	quit := make(chan os.Signal, 1)
	signal.Notify(quit, syscall.SIGINT, syscall.SIGTERM)
	<-quit

	log.Println("[Gateway] Shutting down...")
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	if err := srv.Shutdown(ctx); err != nil {
		log.Printf("[Gateway] Shutdown error: %v", err)
	}
	log.Println("[Gateway] Stopped cleanly.")
}

// setupDirectRoutes wires POST /v1/chat to the direct SGLang handler:
// Gateway forwards synchronously to a selected SGLang node and streams the
// response back to the client via SSE.
func setupDirectRoutes(mux *http.ServeMux, cfg *config.Config) {
	if len(cfg.SGLangInstances) == 0 {
		log.Fatal("[Gateway] requires at least one entry in sglang_instances")
	}
	r := router.New(cfg.SGLangInstances, cfg.MaxConcurrentPerNode)
	mux.Handle("POST /v1/chat", &handler.DirectChatHandler{
		Router:       r,
		DefaultModel: cfg.DefaultModel,
	})
	log.Printf("[Gateway] Direct mode: %d SGLang node(s)", len(cfg.SGLangInstances))
}
