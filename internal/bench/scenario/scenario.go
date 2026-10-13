// Package scenario defines benchmark scenarios, presets, and the deterministic
// synthetic prompts used to drive them.
package scenario

import (
	"fmt"
	"sort"
	"strings"
)

// Kinds of scenario.
const (
	KindChat    = "chat"    // short prompt, long output: decode speed and smoothness
	KindPrefill = "prefill" // long prompt, short output: TTFT and prefill speed
	KindDepth   = "depth"   // long prompt, medium output: decode speed with context filled
)

// Scenario is one request shape.
type Scenario struct {
	Name         string `json:"name"`
	Kind         string `json:"kind"`
	PromptTokens int    `json:"target_prompt_tokens"`
	MaxTokens    int    `json:"max_tokens"`
}

// Need is the context this scenario requires, with headroom for tokenizer variance
// (prompts are sized by an estimate) and the chat template.
func (s Scenario) Need() int { return s.PromptTokens + s.PromptTokens/10 + s.MaxTokens + 64 }

// Catalog is every scenario Kepler knows.
var Catalog = map[string]Scenario{
	"chat":       {Name: "chat", Kind: KindChat, PromptTokens: 64, MaxTokens: 512},
	"prefill-2k": {Name: "prefill-2k", Kind: KindPrefill, PromptTokens: 2048, MaxTokens: 32},
	"prefill-8k": {Name: "prefill-8k", Kind: KindPrefill, PromptTokens: 8192, MaxTokens: 32},
	"depth-4k":   {Name: "depth-4k", Kind: KindDepth, PromptTokens: 4096, MaxTokens: 128},
	"depth-16k":  {Name: "depth-16k", Kind: KindDepth, PromptTokens: 16384, MaxTokens: 128},
	"depth-32k":  {Name: "depth-32k", Kind: KindDepth, PromptTokens: 32768, MaxTokens: 128},
}

// Order is the fixed execution and display order.
var Order = []string{"chat", "prefill-2k", "prefill-8k", "depth-4k", "depth-16k", "depth-32k"}

// Preset bundles scenarios and iteration counts.
type Preset struct {
	Name       string
	Scenarios  []Scenario
	Iterations int
	Warmup     int
}

// Presets, per DESIGN.md §9.4.
var Presets = map[string]Preset{
	"quick": {Name: "quick", Iterations: 3, Warmup: 1, Scenarios: []Scenario{
		withMax(Catalog["chat"], 256), Catalog["prefill-2k"],
	}},
	"standard": {Name: "standard", Iterations: 5, Warmup: 1, Scenarios: []Scenario{
		Catalog["chat"], Catalog["prefill-2k"], Catalog["prefill-8k"], Catalog["depth-16k"],
	}},
	"full": {Name: "full", Iterations: 10, Warmup: 1, Scenarios: []Scenario{
		Catalog["chat"], Catalog["prefill-2k"], Catalog["prefill-8k"],
		Catalog["depth-4k"], Catalog["depth-16k"], Catalog["depth-32k"],
	}},
}

func withMax(s Scenario, max int) Scenario { s.MaxTokens = max; return s }

// PresetNames lists presets in display order.
func PresetNames() []string { return []string{"quick", "standard", "full"} }

// Build returns the preset with optional overrides: an explicit scenario list and
// iteration count (0 = keep preset value).
func Build(preset string, scenarios []string, iterations int) (Preset, error) {
	p, ok := Presets[preset]
	if !ok {
		return p, fmt.Errorf("unknown preset %q (use %s)", preset, strings.Join(PresetNames(), ", "))
	}
	if len(scenarios) > 0 {
		var list []Scenario
		for _, n := range scenarios {
			s, ok := Catalog[n]
			if !ok {
				return p, fmt.Errorf("unknown scenario %q (known: %s)", n, strings.Join(Order, ", "))
			}
			list = append(list, s)
		}
		sort.SliceStable(list, func(i, j int) bool { return indexOf(list[i].Name) < indexOf(list[j].Name) })
		p.Scenarios = list
	}
	if iterations > 0 {
		p.Iterations = iterations
	}
	return p, nil
}

func indexOf(name string) int {
	for i, n := range Order {
		if n == name {
			return i
		}
	}
	return len(Order)
}

// Ctx is the context length engines are launched with: the largest scenario need,
// rounded up to a multiple of 512.
func Ctx(ss []Scenario) int {
	need := 2048
	for _, s := range ss {
		if n := s.Need(); n > need {
			need = n
		}
	}
	return (need + 511) / 512 * 512
}

// Fit drops scenarios that need more than maxCtx tokens, returning kept and skipped.
func Fit(ss []Scenario, maxCtx int) (kept, skipped []Scenario) {
	for _, s := range ss {
		if maxCtx > 0 && s.Need() > maxCtx {
			skipped = append(skipped, s)
		} else {
			kept = append(kept, s)
		}
	}
	return kept, skipped
}
