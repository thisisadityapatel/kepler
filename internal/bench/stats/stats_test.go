package stats

import "testing"

func TestMedianMatchesV1(t *testing.T) {
	if got := Median([]float64{4, 1, 3, 2}); got != 3 {
		t.Errorf("even median = %v, want upper-middle 3", got)
	}
	if got := Median([]float64{5, 1, 3}); got != 3 {
		t.Errorf("odd median = %v", got)
	}
	if Median(nil) != 0 {
		t.Error("empty median")
	}
}

func TestPercentileNearestRank(t *testing.T) {
	xs := []float64{10, 20, 30, 40, 50, 60, 70, 80, 90, 100}
	for p, want := range map[float64]float64{10: 10, 50: 50, 90: 90, 99: 100, 0: 10} {
		if got := Percentile(xs, p); got != want {
			t.Errorf("p%v = %v, want %v", p, got, want)
		}
	}
	s := Summarize([]float64{3, 1, 2})
	if s.P10 != 1 || s.Median != 2 || s.P90 != 3 {
		t.Errorf("summary = %+v", s)
	}
}
