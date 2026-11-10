package results

import (
	"fmt"
	"strings"
)

// Table is a rendered report: header plus rows, with best cells marked.
type Table struct {
	Title  string
	Header []string
	Rows   [][]string
	Best   [][]bool // Best[row][col]
	Legend []string
}

type column struct {
	header       string
	value        func(*Result) (float64, bool)
	format       func(float64) string
	higherBetter bool
	degraded     func(*Result) bool
}

// Report renders results as one comparison table. Values are medians.
func Report(rs []*Result) Table {
	t := Table{}
	if len(rs) == 0 {
		return t
	}
	first := rs[0]
	chip := first.SystemInfo.Chip
	t.Title = fmt.Sprintf("%s · %s · %s (%s) · %s", first.Model.Repo, first.Config.Preset, chip,
		bytesH(first.SystemInfo.RAMBytes), first.Timestamp.Local().Format("2006-01-02 15:04"))

	present := map[string]bool{}
	for _, r := range rs {
		for _, s := range r.Scenarios {
			if len(s.Iterations) > 0 {
				present[s.Name] = true
			}
		}
	}
	median := func(name string, pick func(Summary) float64) func(*Result) (float64, bool) {
		return func(r *Result) (float64, bool) {
			s := r.Scenario(name)
			if s == nil || len(s.Iterations) == 0 {
				return 0, false
			}
			return pick(s.Summary), true
		}
	}
	flagged := func(name string) func(*Result) bool {
		return func(r *Result) bool {
			s := r.Scenario(name)
			return s != nil && len(s.Flags) > 0
		}
	}
	cols := []column{{header: "load", value: func(r *Result) (float64, bool) { return r.Load.LoadS, r.Load.LoadS > 0 },
		format: func(v float64) string { return fmt.Sprintf("%.1fs", v) }}}
	if present["chat"] {
		cols = append(cols,
			column{header: "decode tok/s", value: median("chat", func(s Summary) float64 { return s.DecodeTokPerS.Median }),
				format: func(v float64) string { return fmt.Sprintf("%.1f", v) }, higherBetter: true, degraded: flagged("chat")},
			column{header: "ITL p99", value: median("chat", func(s Summary) float64 { return s.ITLP99Ms.Median }),
				format: func(v float64) string { return fmt.Sprintf("%.0fms", v) }})
	}
	for _, n := range []string{"prefill-2k", "prefill-8k"} {
		if present[n] {
			cols = append(cols, column{header: "TTFT " + strings.TrimPrefix(n, "prefill-"),
				value:  median(n, func(s Summary) float64 { return s.TTFTMs.Median }),
				format: func(v float64) string { return fmt.Sprintf("%.2fs", v/1000) }, degraded: flagged(n)})
		}
	}
	for _, n := range []string{"depth-4k", "depth-16k", "depth-32k"} {
		if present[n] {
			cols = append(cols, column{header: "decode@" + strings.TrimPrefix(n, "depth-"),
				value:  median(n, func(s Summary) float64 { return s.DecodeTokPerS.Median }),
				format: func(v float64) string { return fmt.Sprintf("%.1f", v) }, higherBetter: true, degraded: flagged(n)})
		}
	}
	cols = append(cols, column{header: "peak mem", value: func(r *Result) (float64, bool) { return float64(r.PeakMem()), r.PeakMem() > 0 },
		format: func(v float64) string { return bytesH(int64(v)) }})

	t.Header = []string{"engine", "format", "quant"}
	for _, c := range cols {
		t.Header = append(t.Header, c.header)
	}
	best := make([]int, len(cols))
	for ci, c := range cols {
		best[ci] = -1
		var bv float64
		for ri, r := range rs {
			v, ok := c.value(r)
			if !ok || r.Status == StatusFailed {
				continue
			}
			if best[ci] < 0 || (c.higherBetter && v > bv) || (!c.higherBetter && v < bv) {
				best[ci], bv = ri, v
			}
		}
	}
	anyDegraded := false
	for ri, r := range rs {
		name := r.Engine.Name
		if r.Status != StatusComplete {
			name += " (" + r.Status + ")"
		}
		row := []string{name, r.Model.Format, r.Model.Quant}
		marks := []bool{false, false, false}
		for ci, c := range cols {
			v, ok := c.value(r)
			cell := "—"
			if ok {
				cell = c.format(v)
				if c.degraded != nil && c.degraded(r) {
					cell += " ~"
					anyDegraded = true
				}
			}
			row = append(row, cell)
			marks = append(marks, ok && best[ci] == ri && len(rs) > 1)
		}
		t.Rows = append(t.Rows, row)
		t.Best = append(t.Best, marks)
	}
	t.Legend = append(t.Legend, "medians · best per column highlighted")
	if anyDegraded {
		t.Legend = append(t.Legend, "~ engine did not honor the full spec (e.g. stopped before max_tokens); see the result file")
	}
	return t
}

func bytesH(n int64) string {
	switch {
	case n >= 1<<30:
		return fmt.Sprintf("%.1f GB", float64(n)/(1<<30))
	case n >= 1<<20:
		return fmt.Sprintf("%.0f MB", float64(n)/(1<<20))
	}
	return fmt.Sprintf("%d B", n)
}
