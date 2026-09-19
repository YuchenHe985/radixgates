// Package stats has the small amount of statistics the load generator needs.
package stats

import (
	"math"
	"sort"
)

// Percentile returns the nearest-rank percentile (p in 0..100) of an ascending slice.
func Percentile(sorted []float64, p float64) float64 {
	n := len(sorted)
	if n == 0 {
		return 0
	}
	idx := int(math.Ceil(p/100*float64(n))) - 1
	idx = max(0, min(n-1, idx))
	return sorted[idx]
}

// Summary is a latency distribution summary.
type Summary struct {
	Count int     `json:"count"`
	Mean  float64 `json:"mean"`
	P50   float64 `json:"p50"`
	P95   float64 `json:"p95"`
	P99   float64 `json:"p99"`
	Max   float64 `json:"max"`
}

// Summarize does not modify vals.
func Summarize(vals []float64) Summary {
	if len(vals) == 0 {
		return Summary{}
	}
	s := append([]float64(nil), vals...)
	sort.Float64s(s)
	var sum float64
	for _, v := range s {
		sum += v
	}
	return Summary{
		Count: len(s), Mean: sum / float64(len(s)),
		P50: Percentile(s, 50), P95: Percentile(s, 95), P99: Percentile(s, 99), Max: s[len(s)-1],
	}
}
