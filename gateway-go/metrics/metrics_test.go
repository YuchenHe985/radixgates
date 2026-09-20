package metrics

import (
	"context"
	"testing"
	"time"

	"github.com/prometheus/client_golang/prometheus"

	"github.com/unicoregpu/radixgates/gateway/breaker"
	"github.com/unicoregpu/radixgates/gateway/config"
	"github.com/unicoregpu/radixgates/gateway/router"
)

// gauge reads one series from the registry, the same data /metrics serves.
func gauge(t *testing.T, reg *prometheus.Registry, name, node string) float64 {
	t.Helper()
	mfs, err := reg.Gather()
	if err != nil {
		t.Fatal(err)
	}
	for _, mf := range mfs {
		if mf.GetName() != name {
			continue
		}
		for _, m := range mf.GetMetric() {
			if node == "" && len(m.GetLabel()) == 0 {
				return m.GetGauge().GetValue()
			}
			for _, l := range m.GetLabel() {
				if l.GetName() == "node" && l.GetValue() == node {
					return m.GetGauge().GetValue()
				}
			}
		}
	}
	t.Fatalf("series %s{node=%q} not found", name, node)
	return 0
}

func TestRegisterRouterExposesLiveNodeState(t *testing.T) {
	inst := []config.SGLangInstance{
		{Group: "g", Host: "127.0.0.1", Port: 31001},
		{Group: "g", Host: "127.0.0.1", Port: 31002},
	}
	bc := breaker.Config{FailureThreshold: 1, OpenDuration: time.Hour}
	r := router.New(inst, 2, router.Options{BoundedLoadFactor: 1.25, AffinityFloor: 4, MaxQueue: 4,
		QueueTimeout: 50 * time.Millisecond, Breaker: &bc})
	reg := prometheus.NewRegistry()
	RegisterRouterWith(reg, r)
	a, b := r.Nodes()[0].Name, r.Nodes()[1].Name

	if gauge(t, reg, "radixgates_node_up", a) != 1 || gauge(t, reg, "radixgates_node_up", b) != 1 {
		t.Fatal("both nodes should report up")
	}
	if gauge(t, reg, "radixgates_node_breaker_open", a) != 0 {
		t.Fatal("breaker should start closed")
	}

	r.Nodes()[1].SetUp(false)
	if gauge(t, reg, "radixgates_node_up", b) != 0 || gauge(t, reg, "radixgates_node_up", a) != 1 {
		t.Fatal("node_up must follow the health check result per node")
	}

	// One failed lease opens the breaker (threshold 1) and one held lease shows as in flight.
	l, err := r.Acquire(context.Background(), "key", "", nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	if gauge(t, reg, "radixgates_node_inflight", l.N.Name) != 1 {
		t.Fatal("inflight gauge should count the held lease")
	}
	l.Done(false)
	if gauge(t, reg, "radixgates_node_inflight", a) != 0 {
		t.Fatal("inflight gauge should drop when the lease is released")
	}
	if gauge(t, reg, "radixgates_node_breaker_open", a) != 1 {
		t.Fatal("breaker gauge should show the tripped breaker")
	}
	if gauge(t, reg, "radixgates_queue_depth", "") != 0 {
		t.Fatal("queue should be empty")
	}
}
