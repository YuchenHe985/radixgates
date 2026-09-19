package main

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/prometheus/client_golang/prometheus/promhttp"
	"github.com/unicoregpu/radixgates/gateway/breaker"
	"github.com/unicoregpu/radixgates/gateway/config"
	"github.com/unicoregpu/radixgates/gateway/handler"
	"github.com/unicoregpu/radixgates/gateway/metrics"
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
	ctx, stop := signal.NotifyContext(context.Background(), syscall.SIGINT, syscall.SIGTERM)
	defer stop()

	mux := http.NewServeMux()
	rt := setupDirectRoutes(ctx, mux, cfg)

	// Health + metrics
	mux.HandleFunc("GET /health", func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte(`{"status":"ok","mode":"direct"}`))
	})
	mux.HandleFunc("GET /readyz", func(w http.ResponseWriter, r *http.Request) {
		if rt.AnyAvailable() {
			_, _ = w.Write([]byte("ready"))
			return
		}
		http.Error(w, "no SGLang node available", http.StatusServiceUnavailable)
	})
	mux.HandleFunc("GET /admin/nodes", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(rt.Snapshot())
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
	<-ctx.Done()

	log.Println("[Gateway] Shutting down...")
	shutdownCtx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	if err := srv.Shutdown(shutdownCtx); err != nil {
		log.Printf("[Gateway] Shutdown error: %v", err)
	}
	log.Println("[Gateway] Stopped cleanly.")
}

// setupDirectRoutes wires POST /v1/chat to the direct SGLang handler: the gateway
// forwards to a selected SGLang node (failing over to another before the first byte
// is committed) and streams the response back to the client via SSE.
func setupDirectRoutes(ctx context.Context, mux *http.ServeMux, cfg *config.Config) *router.Router {
	if len(cfg.SGLangInstances) == 0 {
		log.Fatal("[Gateway] requires at least one entry in sglang_instances")
	}
	opts := router.Options{
		BoundedLoadFactor: cfg.Routing.BoundedLoadFactor,
		AffinityFloor:     cfg.Routing.AffinityFloor,
		MaxQueue:          cfg.Admission.MaxQueue,
		QueueTimeout:      cfg.Admission.QueueTimeout.Std(),
		OnBreakerChange: func(node string, from, to breaker.State) {
			metrics.BreakerTransitions.WithLabelValues(node, to.String()).Inc()
			log.Printf("[Router] circuit breaker %s: %s -> %s", node, from, to)
		},
	}
	if b := cfg.Reliability.Breaker; b.Enabled {
		opts.Breaker = &breaker.Config{FailureThreshold: b.FailureThreshold, OpenDuration: b.OpenDuration.Std(),
			MaxOpenDuration: b.MaxOpenDuration.Std(), HalfOpenMax: b.HalfOpenMax}
	}
	r := router.New(cfg.SGLangInstances, cfg.MaxConcurrentPerNode, opts)
	metrics.RegisterRouter(r)

	h := &handler.DirectChatHandler{Router: r, DefaultModel: cfg.DefaultModel, Reliability: cfg.Reliability}
	mux.Handle("POST /v1/chat", h)
	if hc := cfg.Reliability.Health; hc.Enabled {
		r.StartHealthChecks(ctx, router.HealthOptions{Interval: hc.Interval.Std(), Timeout: hc.Timeout.Std(),
			Path: hc.Path, UnhealthyThreshold: hc.UnhealthyThreshold, HealthyThreshold: hc.HealthyThreshold}, h.Client())
	}
	log.Printf("[Gateway] Direct mode: %d SGLang node(s), max_attempts=%d breaker=%v health_checks=%v",
		len(cfg.SGLangInstances), cfg.Reliability.MaxAttempts, cfg.Reliability.Breaker.Enabled, cfg.Reliability.Health.Enabled)
	return r
}
