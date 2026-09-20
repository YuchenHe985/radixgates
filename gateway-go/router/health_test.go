package router

import (
	"context"
	"net/http"
	"net/http/httptest"
	"net/url"
	"strconv"
	"sync/atomic"
	"testing"
	"time"

	"github.com/unicoregpu/radixgates/gateway/config"
)

// healthRig is a router whose nodes are real HTTP servers, so probes go over the wire.
type healthRig struct {
	r       *Router
	servers []*httptest.Server
	healthy []*atomic.Bool
	hits    []*atomic.Int64
}

func newHealthRig(t *testing.T, n int, hang time.Duration) *healthRig {
	t.Helper()
	rig := &healthRig{}
	var inst []config.SGLangInstance
	for i := 0; i < n; i++ {
		ok, hits := new(atomic.Bool), new(atomic.Int64)
		ok.Store(true)
		srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
			hits.Add(1)
			if hang > 0 {
				select {
				case <-time.After(hang):
				case <-req.Context().Done():
					return
				}
			}
			if !ok.Load() {
				http.Error(w, "unhealthy", http.StatusServiceUnavailable)
				return
			}
			w.WriteHeader(http.StatusOK)
		}))
		t.Cleanup(srv.Close)
		u, _ := url.Parse(srv.URL)
		port, _ := strconv.Atoi(u.Port())
		inst = append(inst, config.SGLangInstance{Group: "local", Host: u.Hostname(), Port: port})
		rig.servers = append(rig.servers, srv)
		rig.healthy = append(rig.healthy, ok)
		rig.hits = append(rig.hits, hits)
	}
	rig.r = New(inst, 8, opts())
	return rig
}

func (h *healthRig) up() []bool {
	var out []bool
	for _, s := range h.r.Snapshot() {
		out = append(out, s.Up)
	}
	return out
}

func healthOpts() HealthOptions {
	return HealthOptions{Interval: 20 * time.Millisecond, Timeout: 200 * time.Millisecond,
		Path: "/health", UnhealthyThreshold: 2, HealthyThreshold: 2}
}

func TestProbeThresholdsMarkNodesDownAndUp(t *testing.T) {
	rig := newHealthRig(t, 3, 0)
	ctx, st, ho := context.Background(), NewProbeState(), healthOpts()
	client := &http.Client{}

	rig.r.ProbeAll(ctx, ho, client, st)
	if got := rig.up(); !got[0] || !got[1] || !got[2] {
		t.Fatalf("all nodes should start up, got %v", got)
	}

	rig.healthy[1].Store(false)
	rig.r.ProbeAll(ctx, ho, client, st)
	if !rig.up()[1] {
		t.Fatal("one failed probe must not mark the node down (threshold is 2)")
	}
	rig.r.ProbeAll(ctx, ho, client, st)
	if got := rig.up(); got[1] || !got[0] || !got[2] {
		t.Fatalf("node 1 should be down after 2 consecutive failures and the others up, got %v", got)
	}

	// While the node is down, no key is routed to it.
	down := rig.r.Nodes()[1].Name
	for i := 0; i < 60; i++ {
		l := acquire(t, rig.r, "prompt-"+strconv.Itoa(i), nil)
		if l.N.Name == down {
			t.Fatalf("key %d was routed to the node the health check marked down", i)
		}
		l.Release()
	}

	rig.healthy[1].Store(true)
	rig.r.ProbeAll(ctx, ho, client, st)
	if rig.up()[1] {
		t.Fatal("one successful probe must not bring the node back (recovery threshold is 2)")
	}
	rig.r.ProbeAll(ctx, ho, client, st)
	if !rig.up()[1] {
		t.Fatal("node should be up again after 2 consecutive successes")
	}
}

func TestProbeSuccessInterruptsFailureStreak(t *testing.T) {
	rig := newHealthRig(t, 1, 0)
	ctx, st, ho := context.Background(), NewProbeState(), healthOpts()
	client := &http.Client{}

	rig.healthy[0].Store(false)
	rig.r.ProbeAll(ctx, ho, client, st) // 1 failure
	rig.healthy[0].Store(true)
	rig.r.ProbeAll(ctx, ho, client, st) // resets the streak
	rig.healthy[0].Store(false)
	rig.r.ProbeAll(ctx, ho, client, st) // 1 failure again, not 2 in a row
	if !rig.up()[0] {
		t.Fatal("failures that are not consecutive must not mark the node down")
	}
}

func TestProbeTimeoutCountsAsFailure(t *testing.T) {
	rig := newHealthRig(t, 1, 2*time.Second) // /health never answers within the timeout
	ho := healthOpts()
	ho.Timeout, ho.UnhealthyThreshold = 50*time.Millisecond, 1

	start := time.Now()
	rig.r.ProbeAll(context.Background(), ho, &http.Client{}, NewProbeState())
	if rig.up()[0] {
		t.Fatal("a hung /health must count as a failed probe")
	}
	if d := time.Since(start); d > time.Second {
		t.Fatalf("probe should give up at the timeout, took %v", d)
	}
}

func TestProbeUnreachableNodeIsDown(t *testing.T) {
	rig := newHealthRig(t, 2, 0)
	rig.servers[0].Close()
	ho := healthOpts()
	ho.UnhealthyThreshold = 1

	rig.r.ProbeAll(context.Background(), ho, &http.Client{}, NewProbeState())
	if got := rig.up(); got[0] || !got[1] {
		t.Fatalf("closed server should be down and the other up, got %v", got)
	}
}

func TestProbeAbortedByShutdownDoesNotCountAgainstNodes(t *testing.T) {
	rig := newHealthRig(t, 2, 0)
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	ho := healthOpts()
	ho.UnhealthyThreshold = 1

	rig.r.ProbeAll(ctx, ho, &http.Client{}, NewProbeState())
	if got := rig.up(); !got[0] || !got[1] {
		t.Fatalf("probes aborted by shutdown must not mark healthy nodes down: %v", got)
	}
}

func TestStartHealthChecksMarksFailingNodeDown(t *testing.T) {
	rig := newHealthRig(t, 1, 0)
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	rig.healthy[0].Store(false)
	rig.r.StartHealthChecks(ctx, healthOpts(), &http.Client{})

	deadline := time.Now().Add(3 * time.Second)
	for rig.up()[0] && time.Now().Before(deadline) {
		time.Sleep(10 * time.Millisecond)
	}
	if rig.up()[0] {
		t.Fatal("background prober never marked the failing node down")
	}
	if rig.hits[0].Load() < 2 {
		t.Fatalf("expected repeated probes, saw %d", rig.hits[0].Load())
	}
}

// Shutting the gateway down cancels the context; probes issued with a cancelled context fail, so a
// prober that kept looping would mark every healthy node down.
func TestStartHealthChecksStopsWithoutFlappingNodes(t *testing.T) {
	rig := newHealthRig(t, 2, 0)
	ctx, cancel := context.WithCancel(context.Background())
	rig.r.StartHealthChecks(ctx, healthOpts(), &http.Client{})

	deadline := time.Now().Add(3 * time.Second)
	for rig.hits[0].Load() < 3 && time.Now().Before(deadline) {
		time.Sleep(10 * time.Millisecond)
	}
	if rig.hits[0].Load() < 3 {
		t.Fatalf("prober did not run periodically, saw %d probes", rig.hits[0].Load())
	}

	cancel()
	time.Sleep(200 * time.Millisecond) // ten probe intervals
	if got := rig.up(); !got[0] || !got[1] {
		t.Fatalf("healthy nodes were marked down after the context ended: %v", got)
	}
}
