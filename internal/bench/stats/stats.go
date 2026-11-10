// Package stats computes the summary statistics used in results.
package stats

import (
	"math"
	"sort"
)

// Median returns the upper-middle element for even-length input, matching Kepler v1
// (legacy benchmark.cpp vec_median) so medians stay comparable across versions.
func Median(xs []float64) float64 {
	if len(xs) == 0 {
		return 0
	}
	s := sorted(xs)
	return s[len(s)/2]
}

// Percentile returns the nearest-rank percentile (p in 0..100).
func Percentile(xs []float64, p float64) float64 {
	if len(xs) == 0 {
		return 0
	}
	s := sorted(xs)
	rank := int(math.Ceil(p / 100 * float64(len(s))))
	if rank < 1 {
		rank = 1
	}
	if rank > len(s) {
		rank = len(s)
	}
	return s[rank-1]
}

// Max returns the largest element (0 for empty input).
func Max(xs []float64) float64 {
	m := 0.0
	for i, x := range xs {
		if i == 0 || x > m {
			m = x
		}
	}
	return m
}

func sorted(xs []float64) []float64 {
	s := append([]float64(nil), xs...)
	sort.Float64s(s)
	return s
}

// Summary is the p10 / median / p90 triple reported for every metric.
type Summary struct {
	P10    float64 `json:"p10"`
	Median float64 `json:"median"`
	P90    float64 `json:"p90"`
}

// Summarize computes a Summary.
func Summarize(xs []float64) Summary {
	return Summary{P10: Percentile(xs, 10), Median: Median(xs), P90: Percentile(xs, 90)}
}
