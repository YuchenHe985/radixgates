// Command mockworker runs a simulated SGLang-compatible worker with fault
// injection (see internal/mock). Its latencies are a model, not GPU measurements.
package main

import (
	"errors"
	"flag"
	"log"
	"net/http"
	"time"

	"github.com/unicoregpu/radixgates/gateway/internal/mock"
)

func main() {
	addr := flag.String("addr", ":31000", "listen address")
	name := flag.String("name", "w0", "worker name (reported by /admin/stats)")
	cold := flag.Duration("ttft-cold", 100*time.Millisecond, "time to first token on a prefix-cache miss")
	warm := flag.Duration("ttft-warm", 50*time.Millisecond, "time to first token on a prefix-cache hit")
	tpot := flag.Duration("tpot", 15*time.Millisecond, "time per output token")
	tokens := flag.Int("tokens", 16, "default output tokens")
	capacity := flag.Int("capacity", 4, "concurrent requests served at full speed")
	cache := flag.Int("cache", 4, "distinct prefixes kept in the simulated prefix cache")
	flag.Parse()

	w := mock.New(mock.Options{
		Name: *name, TTFTCold: *cold, TTFTWarm: *warm, TPOT: *tpot,
		Tokens: *tokens, Capacity: *capacity, CacheSize: *cache,
	})
	srv := &http.Server{Addr: *addr, Handler: w.Handler(), ReadHeaderTimeout: 5 * time.Second}
	log.Printf("mockworker %s listening on %s", *name, *addr)
	if err := srv.ListenAndServe(); err != nil && !errors.Is(err, http.ErrServerClosed) {
		log.Fatal(err)
	}
}
