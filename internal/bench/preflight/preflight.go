// Package preflight decides whether a model fits this machine before anything is
// downloaded: it picks the quant, prunes scenarios that cannot fit, and checks disk.
package preflight

import (
	"context"
	"fmt"
	"os/exec"
	"sort"
	"strings"

	"github.com/thisisadityapatel/kepler/internal/bench/scenario"
	"github.com/thisisadityapatel/kepler/internal/hf"
	"github.com/thisisadityapatel/kepler/internal/probe"
)

// Overhead covers compute buffers, the engine runtime, and allocator slack.
const Overhead = 1 << 30

// Candidate is one way to run the model (a GGUF variant, or the single MLX build).
type Candidate struct {
	Quant string
	Size  int64
}

// Input describes what is being planned.
type Input struct {
	Candidates []Candidate // in preference order
	Arch       hf.ArchInfo
	Scenarios  []scenario.Scenario
	Facts      probe.HostFacts
	DiskCopies int // how many on-disk copies the plan needs (Ollama imports a second copy)
}

// Skipped records a scenario that was dropped and why.
type Skipped struct {
	Name   string `json:"name"`
	Reason string `json:"reason"`
}

// Plan is the outcome.
type Plan struct {
	Choice    int // index into Input.Candidates
	Quant     string
	Scenarios []scenario.Scenario
	Skipped   []Skipped
	Ctx       int
	NeedMem   int64
	BudgetMem int64
	NeedDisk  int64
	FreeDisk  int64
	Notes     []string // e.g. "Q4_K_M did not fit; using Q3_K_M"
}

// FitError means nothing fits; Exit code 3.
type FitError struct{ Msg string }

func (e FitError) Error() string { return e.Msg }

// KVBytes estimates an f16 KV cache for ctx tokens.
func KVBytes(a hf.ArchInfo, ctx int) int64 {
	if !a.Known() {
		return 0
	}
	return 2 * int64(a.Layers) * int64(a.KVHeads) * int64(a.HeadDim) * int64(ctx) * 2
}

// NeedMem estimates resident memory for weights plus KV cache at ctx.
func NeedMem(weights int64, a hf.ArchInfo, ctx int) int64 {
	kv := KVBytes(a, ctx)
	if kv == 0 {
		kv = weights / 5 // architecture unknown: assume a generous cache
	}
	return weights + kv + Overhead
}

// Budget is the memory engines can use on this host.
func Budget(f probe.HostFacts) int64 {
	for _, a := range f.Accelerators {
		if a.Kind != probe.AccelCPU && a.MemBytes > 0 {
			return a.MemBytes
		}
	}
	return f.RAMBytes * 3 / 4
}

// Run computes the plan. It prefers keeping every scenario and falling back to a
// smaller quant; only if no quant fits does it drop the largest scenarios.
func Run(in Input) (Plan, error) {
	p := Plan{BudgetMem: Budget(in.Facts), FreeDisk: in.Facts.FreeDiskBytes}
	if len(in.Candidates) == 0 {
		return p, fmt.Errorf("no model variants to choose from")
	}
	if !in.Arch.Known() {
		p.Notes = append(p.Notes, "model architecture unknown; memory estimate is approximate")
	}

	scen, tooLong := scenario.Fit(in.Scenarios, in.Arch.ContextMax)
	for _, s := range tooLong {
		p.Skipped = append(p.Skipped, Skipped{s.Name, fmt.Sprintf("needs %d tokens; model context is %d", s.Need(), in.Arch.ContextMax)})
	}
	if len(scen) == 0 {
		return p, FitError{fmt.Sprintf("no selected scenario fits the model's %d-token context", in.Arch.ContextMax)}
	}

	// Try the full scenario set across all quants, then progressively drop the
	// scenario with the largest context need.
	sets := [][]scenario.Scenario{scen}
	bySize := append([]scenario.Scenario(nil), scen...)
	sort.SliceStable(bySize, func(i, j int) bool { return bySize[i].Need() > bySize[j].Need() })
	for k := 1; k < len(bySize); k++ {
		drop := map[string]bool{}
		for _, s := range bySize[:k] {
			drop[s.Name] = true
		}
		var set []scenario.Scenario
		for _, s := range scen {
			if !drop[s.Name] {
				set = append(set, s)
			}
		}
		sets = append(sets, set)
	}

	for _, set := range sets {
		ctx := scenario.Ctx(set)
		for i, c := range in.Candidates {
			need := NeedMem(c.Size, in.Arch, ctx)
			if need > p.BudgetMem {
				continue
			}
			p.Choice, p.Quant, p.Scenarios, p.Ctx, p.NeedMem = i, c.Quant, set, ctx, need
			if i > 0 {
				p.Notes = append(p.Notes, fmt.Sprintf("%s does not fit in %s of GPU memory; using %s",
					in.Candidates[0].Quant, gb(p.BudgetMem), c.Quant))
			}
			kept := map[string]bool{}
			for _, s := range set {
				kept[s.Name] = true
			}
			for _, s := range scen {
				if !kept[s.Name] {
					p.Skipped = append(p.Skipped, Skipped{s.Name, fmt.Sprintf("needs ~%s of memory at %d tokens; budget is %s",
						gb(NeedMem(c.Size, in.Arch, scenario.Ctx([]scenario.Scenario{s}))), s.Need(), gb(p.BudgetMem))})
				}
			}
			copies := in.DiskCopies
			if copies < 1 {
				copies = 1
			}
			p.NeedDisk = c.Size*int64(copies) + c.Size/10
			if p.FreeDisk > 0 && p.NeedDisk > p.FreeDisk {
				return p, FitError{fmt.Sprintf("not enough disk: need %s, %s free", gb(p.NeedDisk), gb(p.FreeDisk))}
			}
			return p, nil
		}
	}
	smallest := in.Candidates[0]
	for _, c := range in.Candidates {
		if c.Size < smallest.Size {
			smallest = c
		}
	}
	return p, FitError{fmt.Sprintf("model does not fit: smallest option (%s, %s weights) needs ~%s; this machine allows %s of GPU memory",
		orName(smallest.Quant), gb(smallest.Size), gb(NeedMem(smallest.Size, in.Arch, scenario.Ctx(sets[len(sets)-1]))), gb(p.BudgetMem))}
}

func orName(q string) string {
	if q == "" {
		return "unnamed variant"
	}
	return q
}

func gb(b int64) string { return fmt.Sprintf("%.1f GB", float64(b)/(1<<30)) }

// Warnings returns host conditions that would distort results.
func Warnings(ctx context.Context, c probe.Conditions) []string {
	var w []string
	if c.PowerSource == "battery" {
		w = append(w, "running on battery power; results may be lower and less consistent")
	}
	if c.LowPowerMode {
		w = append(w, "Low Power Mode is on; it throttles CPU and GPU")
	}
	if c.Thermal == probe.ThermalElevated {
		w = append(w, "machine is thermally throttled ("+c.ThermalDetail+"); let it cool down")
	}
	if c.MemoryFreePct >= 0 && c.MemoryFreePct < 25 {
		w = append(w, fmt.Sprintf("only %d%% of memory is free; close other apps", c.MemoryFreePct))
	}
	if others := OtherEngines(ctx); len(others) > 0 {
		w = append(w, "other inference engines are running and using memory: "+strings.Join(others, ", "))
	}
	return w
}

// OtherEngines lists engine processes already running outside Kepler.
func OtherEngines(ctx context.Context) []string {
	var found []string
	for _, pat := range []struct{ label, pattern string }{
		{"llama-server", "llama-server"},
		{"mlx_lm.server", "mlx_lm.server"},
		{"Ollama with a model loaded", "ollama runner"},
		{"LM Studio", "LM Studio"},
	} {
		if err := exec.CommandContext(ctx, "pgrep", "-f", pat.pattern).Run(); err == nil {
			found = append(found, pat.label)
		}
	}
	return found
}
