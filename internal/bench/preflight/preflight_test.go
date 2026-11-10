package preflight

import (
	"errors"
	"strings"
	"testing"

	"github.com/thisisadityapatel/kepler/internal/bench/scenario"
	"github.com/thisisadityapatel/kepler/internal/hf"
	"github.com/thisisadityapatel/kepler/internal/probe"
)

const gib = int64(1 << 30)

// Llama-3-8B-like: 32 layers, 8 KV heads, head dim 128 → 128 KiB of KV per token.
var llama8b = hf.ArchInfo{Layers: 32, KVHeads: 8, HeadDim: 128, ContextMax: 131072}

func host(gpu int64, disk int64) probe.HostFacts {
	return probe.HostFacts{RAMBytes: 16 * gib, FreeDiskBytes: disk,
		Accelerators: []probe.Accelerator{{Kind: probe.AccelMetal, MemBytes: gpu}, {Kind: probe.AccelCPU}}}
}

func TestKVBytes(t *testing.T) {
	if got := KVBytes(llama8b, 8192); got != 1*gib {
		t.Errorf("8k context KV = %d, want 1 GiB", got)
	}
}

func TestPicksPreferredQuantWhenItFits(t *testing.T) {
	std, _ := scenario.Build("standard", nil, 0)
	p, err := Run(Input{
		Candidates: []Candidate{{"Q4_K_M", 49 * gib / 10}, {"Q8_0", 85 * gib / 10}},
		Arch:       llama8b, Scenarios: std.Scenarios, Facts: host(gib*107/10, 100*gib), DiskCopies: 2,
	})
	if err != nil {
		t.Fatal(err)
	}
	if p.Quant != "Q4_K_M" || len(p.Scenarios) != 4 || len(p.Skipped) != 0 || p.NeedDisk < 2*49*gib/10 {
		t.Fatalf("plan = %+v", p)
	}
}

func TestFallsBackToSmallerQuantThenDropsScenarios(t *testing.T) {
	std, _ := scenario.Build("standard", nil, 0)
	cands := []Candidate{{"Q4_K_M", 49 * gib / 10}, {"Q3_K_M", 40 * gib / 10}}

	// 7 GiB budget: Q4_K_M + 16.9k ctx KV (~2.1 GiB) + 1 GiB = 8 GiB → no; Q3_K_M = 7.1 → no;
	// dropping depth-16k leaves ~9k ctx: Q4_K_M 4.9 + 1.1 + 1 = 7.0 → fits.
	p, err := Run(Input{Candidates: cands, Arch: llama8b, Scenarios: std.Scenarios, Facts: host(7*gib+gib/10, 100*gib)})
	if err != nil {
		t.Fatal(err)
	}
	if p.Quant != "Q4_K_M" || len(p.Skipped) != 1 || p.Skipped[0].Name != "depth-16k" {
		t.Fatalf("plan = quant %s skipped %+v", p.Quant, p.Skipped)
	}

	// Very large context need on a big budget with a too-large preferred quant → smaller quant, all scenarios.
	p, err = Run(Input{Candidates: []Candidate{{"Q8_0", 9 * gib}, {"Q4_K_M", 5 * gib}}, Arch: llama8b, Scenarios: std.Scenarios, Facts: host(9*gib, 100*gib)})
	if err != nil || p.Quant != "Q4_K_M" || len(p.Skipped) != 0 || len(p.Notes) == 0 {
		t.Fatalf("fallback plan = %+v, %v", p, err)
	}
}

func TestFitErrors(t *testing.T) {
	q, _ := scenario.Build("quick", nil, 0)
	_, err := Run(Input{Candidates: []Candidate{{"Q4_K_M", 40 * gib}}, Arch: llama8b, Scenarios: q.Scenarios, Facts: host(10*gib, 500*gib)})
	var fe FitError
	if !errors.As(err, &fe) || !strings.Contains(err.Error(), "does not fit") {
		t.Errorf("memory err = %v", err)
	}
	_, err = Run(Input{Candidates: []Candidate{{"Q4_K_M", 4 * gib}}, Arch: llama8b, Scenarios: q.Scenarios, Facts: host(10*gib, 3*gib)})
	if !errors.As(err, &fe) || !strings.Contains(err.Error(), "not enough disk") {
		t.Errorf("disk err = %v", err)
	}
	small := llama8b
	small.ContextMax = 256
	_, err = Run(Input{Candidates: []Candidate{{"Q4_K_M", gib}}, Arch: small, Scenarios: q.Scenarios, Facts: host(10*gib, 100*gib)})
	if !errors.As(err, &fe) {
		t.Errorf("context err = %v", err)
	}
}

func TestSkipsScenariosBeyondModelContext(t *testing.T) {
	full, _ := scenario.Build("full", nil, 0)
	a := llama8b
	a.ContextMax = 8192
	p, err := Run(Input{Candidates: []Candidate{{"Q4_K_M", gib / 10}}, Arch: a, Scenarios: full.Scenarios, Facts: host(10*gib, 100*gib)})
	if err != nil {
		t.Fatal(err)
	}
	if len(p.Skipped) != 3 || !strings.Contains(p.Skipped[0].Reason, "model context is 8192") {
		t.Fatalf("skipped = %+v", p.Skipped)
	}
}
