// Package metrics defines and registers Prometheus metrics for the gateway.
package metrics

import (
	"github.com/prometheus/client_golang/prometheus"

	"github.com/unicoregpu/radixgates/gateway/router"
)

var latencyBuckets = []float64{.005, .01, .025, .05, .1, .25, .5, 1, 2.5, 5, 10, 30, 60}

var (
	// RequestsTotal counts every POST /v1/chat, labelled by final outcome:
	// "direct_ok" | "bad_request" | "no_node" | "queue_full" | "queue_timeout" |
	// "upstream_failed" | "midstream_failed" | "timeout" | "cancelled".
	// (The "sglang_error" outcome of the original is now "upstream_failed", after retries.)
	RequestsTotal = prometheus.NewCounterVec(
		prometheus.CounterOpts{
			Name: "radixgates_requests_total",
			Help: "Total number of inference requests received by the gateway.",
		},
		[]string{"status"},
	)

	// RequestDuration measures end-to-end gateway-side latency of POST /v1/chat.
	// The original buckets stopped at 1 s although the handler covers whole streams.
	RequestDuration = prometheus.NewHistogram(
		prometheus.HistogramOpts{
			Name:    "radixgates_request_duration_seconds",
			Help:    "End-to-end latency of the POST /v1/chat handler (includes streaming).",
			Buckets: latencyBuckets,
		},
	)

	// ActiveRequests tracks requests currently being processed by the gateway.
	ActiveRequests = prometheus.NewGauge(
		prometheus.GaugeOpts{
			Name: "radixgates_active_requests",
			Help: "Number of requests currently in-flight at the gateway.",
		},
	)

	UpstreamAttempts = prometheus.NewCounterVec(
		prometheus.CounterOpts{
			Name: "radixgates_upstream_attempts_total",
			Help: "Upstream attempts by node and result (ok, conn_error, timeout, http_5xx, busy, read_error, midstream_error, canceled).",
		},
		[]string{"node", "result"},
	)

	Retries = prometheus.NewCounterVec(
		prometheus.CounterOpts{
			Name: "radixgates_retries_total",
			Help: "Extra upstream attempts, by the reason the previous attempt failed.",
		},
		[]string{"reason"},
	)

	BreakerTransitions = prometheus.NewCounterVec(
		prometheus.CounterOpts{
			Name: "radixgates_breaker_transitions_total",
			Help: "Circuit breaker state transitions per node.",
		},
		[]string{"node", "to"},
	)

	MidstreamFailures = prometheus.NewCounterVec(
		prometheus.CounterOpts{
			Name: "radixgates_midstream_failures_total",
			Help: "Streams that broke after output started (never retried).",
		},
		[]string{"node"},
	)

	TimeToFirstByte = prometheus.NewHistogram(
		prometheus.HistogramOpts{
			Name:    "radixgates_time_to_first_byte_seconds",
			Help:    "Time from request arrival to the first response byte sent to the client.",
			Buckets: latencyBuckets,
		},
	)

	QueueWait = prometheus.NewHistogram(
		prometheus.HistogramOpts{
			Name:    "radixgates_queue_wait_seconds",
			Help:    "Time spent waiting for a node slot.",
			Buckets: latencyBuckets,
		},
	)
)

func init() {
	prometheus.MustRegister(
		RequestsTotal, RequestDuration, ActiveRequests,
		UpstreamAttempts, Retries, BreakerTransitions, MidstreamFailures, TimeToFirstByte, QueueWait,
	)
}

// RegisterRouter exposes per-node state (up, breaker, in-flight) and queue depth on the
// default registry. Call it once, from main, with the router built from the loaded config.
func RegisterRouter(r *router.Router) { RegisterRouterWith(prometheus.DefaultRegisterer, r) }

// RegisterRouterWith is RegisterRouter for an explicit registry.
func RegisterRouterWith(reg prometheus.Registerer, r *router.Router) {
	snap := func(f func(router.Status) float64, i int) func() float64 {
		return func() float64 { return f(r.Snapshot()[i]) }
	}
	for i, n := range r.Nodes() {
		lbl := prometheus.Labels{"node": n.Name}
		reg.MustRegister(
			prometheus.NewGaugeFunc(prometheus.GaugeOpts{Name: "radixgates_node_up",
				Help: "1 if the active health check considers the node up.", ConstLabels: lbl},
				snap(func(s router.Status) float64 { return b2f(s.Up) }, i)),
			prometheus.NewGaugeFunc(prometheus.GaugeOpts{Name: "radixgates_node_breaker_open",
				Help: "1 if the node's circuit breaker is not closed.", ConstLabels: lbl},
				snap(func(s router.Status) float64 { return b2f(s.Breaker != "closed") }, i)),
			prometheus.NewGaugeFunc(prometheus.GaugeOpts{Name: "radixgates_node_inflight",
				Help: "Requests currently leased on the node.", ConstLabels: lbl},
				snap(func(s router.Status) float64 { return float64(s.Inflight) }, i)),
		)
	}
	reg.MustRegister(prometheus.NewGaugeFunc(prometheus.GaugeOpts{Name: "radixgates_queue_depth",
		Help: "Requests waiting for a node slot."}, func() float64 { return float64(r.QueueDepth()) }))
}

func b2f(b bool) float64 {
	if b {
		return 1
	}
	return 0
}
