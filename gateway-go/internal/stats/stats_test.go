package stats

import "testing"

func TestPercentileNearestRank(t *testing.T) {
	v := make([]float64, 100)
	for i := range v {
		v[i] = float64(i + 1) // 1..100
	}
	for _, c := range []struct{ p, want float64 }{{50, 50}, {95, 95}, {99, 99}, {100, 100}, {0, 1}, {1, 1}} {
		if got := Percentile(v, c.p); got != c.want {
			t.Errorf("P%v = %v, want %v", c.p, got, c.want)
		}
	}
	if Percentile(nil, 50) != 0 {
		t.Error("empty input must return 0")
	}
	if got := Percentile([]float64{7}, 99); got != 7 {
		t.Errorf("single value: %v", got)
	}
}

func TestSummarizeDoesNotMutate(t *testing.T) {
	in := []float64{5, 1, 3}
	s := Summarize(in)
	if in[0] != 5 || in[1] != 1 || in[2] != 3 {
		t.Fatalf("input was reordered: %v", in)
	}
	if s.Count != 3 || s.P50 != 3 || s.Max != 5 || s.Mean != 3 {
		t.Fatalf("unexpected summary %+v", s)
	}
}
