// Package metrics defines and registers Prometheus metrics for the gateway.
package metrics

import "github.com/prometheus/client_golang/prometheus"

var (
	// RequestsTotal counts every POST /v1/chat, labelled by outcome.
	RequestsTotal = prometheus.NewCounterVec(
		prometheus.CounterOpts{
			Name: "radixgates_requests_total",
			Help: "Total number of inference requests received by the gateway.",
		},
		[]string{"status"}, // "direct_ok" | "bad_request" | "no_node" | "sglang_error" | "cancelled"
	)

	// RequestDuration measures end-to-end gateway-side latency of POST /v1/chat.
	RequestDuration = prometheus.NewHistogram(
		prometheus.HistogramOpts{
			Name:    "radixgates_request_duration_seconds",
			Help:    "Latency of the POST /v1/chat handler (gateway side only, not inference).",
			Buckets: []float64{.005, .01, .025, .05, .1, .25, .5, 1.0},
		},
	)

	// ActiveRequests tracks requests currently being processed by the gateway.
	ActiveRequests = prometheus.NewGauge(
		prometheus.GaugeOpts{
			Name: "radixgates_active_requests",
			Help: "Number of requests currently in-flight at the gateway.",
		},
	)
)

func init() {
	prometheus.MustRegister(
		RequestsTotal,
		RequestDuration,
		ActiveRequests,
	)
}
