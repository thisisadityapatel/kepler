// Package results defines the result file schema (v2), writes and reads result and
// comparison files, and renders the terminal report.
package results

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strings"
	"time"

	"github.com/thisisadityapatel/kepler/internal/bench/preflight"
	"github.com/thisisadityapatel/kepler/internal/bench/stats"
	"github.com/thisisadityapatel/kepler/internal/hf"
	"github.com/thisisadityapatel/kepler/internal/probe"
)

// SchemaVersion of result and comparison files.
const SchemaVersion = 2

// Result statuses.
const (
	StatusComplete    = "complete"
	StatusPartial     = "partial"
	StatusFailed      = "failed"
	StatusInterrupted = "interrupted"
)

// Comparability statuses.
const (
	ComparabilityFull     = "full"
	ComparabilityDegraded = "degraded"
)

// Result is one engine's benchmark of one model (one JSON file).
type Result struct {
	SchemaVersion    int                 `json:"schema_version"`
	KeplerVersion    string              `json:"kepler_version"`
	Timestamp        time.Time           `json:"timestamp"`
	RunID            string              `json:"run_id"`
	ComparisonID     string              `json:"comparison_id,omitempty"`
	Status           string              `json:"status"`
	Error            string              `json:"error,omitempty"`
	Model            hf.ModelProfile     `json:"model"`
	Engine           EngineInfo          `json:"engine"`
	SystemInfo       SystemInfo          `json:"system_info"`
	Config           Config              `json:"config"`
	Load             Load                `json:"load"`
	Scenarios        []Scenario          `json:"scenarios"`
	SkippedScenarios []preflight.Skipped `json:"skipped_scenarios"`
	Comparability    Comparability       `json:"comparability"`
	Conditions       Conditions          `json:"conditions"`
	SampleOutput     string              `json:"sample_output,omitempty"`
	LogPath          string              `json:"log_path,omitempty"`
}

// EngineInfo identifies the engine build that ran.
type EngineInfo struct {
	Name        string   `json:"name"`
	Display     string   `json:"display"`
	Version     string   `json:"version"`
	Target      string   `json:"target"`
	InstallKind string   `json:"install_kind"`
	LaunchArgs  []string `json:"launch_args"`
}

// SystemInfo is the host, flattened for readability.
type SystemInfo struct {
	OS               string `json:"os"`
	OSVersion        string `json:"os_version"`
	Arch             string `json:"arch"`
	Chip             string `json:"chip"`
	ModelID          string `json:"model_id,omitempty"`
	CPUCores         Cores  `json:"cpu_cores"`
	GPU              string `json:"gpu,omitempty"`
	GPUCores         int    `json:"gpu_cores,omitempty"`
	RAMBytes         int64  `json:"ram_bytes"`
	GPUMemLimitBytes int64  `json:"gpu_mem_limit_bytes"`
	PowerSource      string `json:"power_source"`
	LowPowerMode     bool   `json:"low_power_mode"`
}

// Cores is the CPU core split.
type Cores struct {
	Performance int `json:"performance"`
	Efficiency  int `json:"efficiency"`
}

// NewSystemInfo flattens HostFacts.
func NewSystemInfo(h probe.HostFacts) SystemInfo {
	s := SystemInfo{OS: h.OS, OSVersion: h.OSVersion, Arch: h.Arch, Chip: h.CPU.Brand, ModelID: h.ModelID,
		CPUCores: Cores{h.CPU.PerformanceCores, h.CPU.EfficiencyCores}, RAMBytes: h.RAMBytes,
		PowerSource: h.Power.Source, LowPowerMode: h.Power.LowPowerMode}
	for _, a := range h.Accelerators {
		if a.Kind != probe.AccelCPU {
			s.GPU, s.GPUCores, s.GPUMemLimitBytes = a.Kind, a.Cores, a.MemBytes
			break
		}
	}
	return s
}

// Config is the benchmark spec that was applied.
type Config struct {
	Preset           string  `json:"preset"`
	Temperature      float64 `json:"temperature"`
	Seed             int     `json:"seed"`
	Ctx              int     `json:"ctx"`
	WarmupIterations int     `json:"warmup_iterations"`
	Iterations       int     `json:"iterations"`
	CooldownS        float64 `json:"cooldown_s"`
	CharsPerToken    float64 `json:"chars_per_token"`
	MemoryMetric     string  `json:"memory_metric"`
}

// Load describes model load.
type Load struct {
	LoadS             float64 `json:"load_s"`
	MaterializeS      float64 `json:"materialize_s,omitempty"` // engine-specific import (Ollama create), excluded from load_s
	MemAfterLoadBytes int64   `json:"mem_after_load_bytes"`
	MappedModelBytes  int64   `json:"mapped_model_bytes"` // part of mem_after_load: mmapped weights

	PeakMemBytes int64 `json:"peak_mem_bytes"`
}

// Scenario holds every iteration of one scenario plus its summary.
type Scenario struct {
	Name               string      `json:"name"`
	Kind               string      `json:"kind"`
	TargetPromptTokens int         `json:"target_prompt_tokens"`
	MaxTokens          int         `json:"max_tokens"`
	Iterations         []Iteration `json:"iterations"`
	Summary            Summary     `json:"summary"`
	Flags              []string    `json:"flags"`
	Error              string      `json:"error,omitempty"`
}

// Iteration is one measured request.
type Iteration struct {
	PromptTokens    int             `json:"prompt_tokens"`
	GeneratedTokens int             `json:"generated_tokens"`
	TokenEvents     int             `json:"token_events"`
	TTFTMs          float64         `json:"ttft_ms"`
	PrefillTokPerS  float64         `json:"prefill_tok_per_s"`
	DecodeTokPerS   float64         `json:"decode_tok_per_s"`
	E2ETokPerS      float64         `json:"e2e_tok_per_s"`
	ITLP50Ms        float64         `json:"itl_p50_ms"`
	ITLP99Ms        float64         `json:"itl_p99_ms"`
	WallS           float64         `json:"wall_s"`
	PeakMemBytes    int64           `json:"peak_mem_bytes"`
	FinishReason    string          `json:"finish_reason,omitempty"`
	EngineReported  json.RawMessage `json:"engine_reported,omitempty"`
}

// Summary aggregates a scenario's iterations.
type Summary struct {
	TTFTMs         stats.Summary `json:"ttft_ms"`
	PrefillTokPerS stats.Summary `json:"prefill_tok_per_s"`
	DecodeTokPerS  stats.Summary `json:"decode_tok_per_s"`
	E2ETokPerS     stats.Summary `json:"e2e_tok_per_s"`
	ITLP99Ms       stats.Summary `json:"itl_p99_ms"`
	PeakMemBytes   MaxValue      `json:"peak_mem_bytes"`
}

// MaxValue wraps a maximum.
type MaxValue struct {
	Max int64 `json:"max"`
}

// Summarize fills s.Summary from its iterations.
func (s *Scenario) Summarize() {
	var ttft, pre, dec, e2e, itl []float64
	var peak int64
	for _, it := range s.Iterations {
		ttft = append(ttft, it.TTFTMs)
		pre = append(pre, it.PrefillTokPerS)
		dec = append(dec, it.DecodeTokPerS)
		e2e = append(e2e, it.E2ETokPerS)
		itl = append(itl, it.ITLP99Ms)
		if it.PeakMemBytes > peak {
			peak = it.PeakMemBytes
		}
	}
	s.Summary = Summary{TTFTMs: stats.Summarize(ttft), PrefillTokPerS: stats.Summarize(pre), DecodeTokPerS: stats.Summarize(dec),
		E2ETokPerS: stats.Summarize(e2e), ITLP99Ms: stats.Summarize(itl), PeakMemBytes: MaxValue{peak}}
}

// Comparability says whether the fairness rules were honored.
type Comparability struct {
	Status string   `json:"status"`
	Notes  []string `json:"notes"` // reasons the run is degraded
	Info   []string `json:"info"`  // non-degrading observations
}

// Conditions are host conditions sampled before and after the engine ran.
type Conditions struct {
	Before probe.Conditions `json:"before"`
	After  probe.Conditions `json:"after"`
}

// Scenario returns the named scenario result.
func (r *Result) Scenario(name string) *Scenario {
	for i := range r.Scenarios {
		if r.Scenarios[i].Name == name {
			return &r.Scenarios[i]
		}
	}
	return nil
}

// PeakMem is the largest memory sample of the run.
func (r *Result) PeakMem() int64 {
	peak := r.Load.PeakMemBytes
	for _, s := range r.Scenarios {
		if s.Summary.PeakMemBytes.Max > peak {
			peak = s.Summary.PeakMemBytes.Max
		}
	}
	return peak
}

var unsafeRe = regexp.MustCompile(`[^A-Za-z0-9._-]+`)

// SanitizeRepo makes a repo id filename-safe (same rule as Kepler v1).
func SanitizeRepo(repo string) string { return strings.Trim(unsafeRe.ReplaceAllString(repo, "-"), "-") }

// Filename is {date}T{HHMMSS}_{repo}_{engine}_{preset}.json in local time.
func Filename(r *Result) string {
	return fmt.Sprintf("%s_%s_%s_%s.json", r.Timestamp.Local().Format("2006-01-02T150405"),
		SanitizeRepo(r.Model.Repo), r.Engine.Name, r.Config.Preset)
}

// Write saves r into dir and returns the path. Existing files are never overwritten.
func Write(dir string, r *Result) (string, error) {
	return writeJSON(dir, Filename(r), r)
}

func writeJSON(dir, name string, v any) (string, error) {
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return "", err
	}
	b, err := json.MarshalIndent(v, "", "  ")
	if err != nil {
		return "", err
	}
	path := filepath.Join(dir, name)
	for i := 2; ; i++ {
		f, err := os.OpenFile(path, os.O_WRONLY|os.O_CREATE|os.O_EXCL, 0o644)
		if os.IsExist(err) {
			path = filepath.Join(dir, strings.TrimSuffix(name, ".json")+fmt.Sprintf("_%d.json", i))
			continue
		}
		if err != nil {
			return "", err
		}
		_, werr := f.Write(append(b, '\n'))
		cerr := f.Close()
		if werr != nil {
			return "", werr
		}
		return path, cerr
	}
}

// Read loads a result file.
func Read(path string) (*Result, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var r Result
	if err := json.Unmarshal(b, &r); err != nil {
		return nil, fmt.Errorf("%s: %w", path, err)
	}
	if r.SchemaVersion != SchemaVersion {
		return nil, fmt.Errorf("%s: schema version %d is not supported (expected %d)", path, r.SchemaVersion, SchemaVersion)
	}
	return &r, nil
}

// ---------------------------------------------------------------- comparison

// Comparison links the per-engine results of one `kepler bench` run.
type Comparison struct {
	SchemaVersion int                 `json:"schema_version"`
	ComparisonID  string              `json:"comparison_id"`
	Timestamp     time.Time           `json:"timestamp"`
	RequestedAs   string              `json:"requested_as"`
	Preset        string              `json:"preset"`
	SystemInfo    SystemInfo          `json:"system_info"`
	Entries       []ComparisonEntry   `json:"entries"`
	Ranking       map[string][]string `json:"ranking"`
	Notes         []string            `json:"notes"`
}

// ComparisonEntry is one engine's result reference.
type ComparisonEntry struct {
	Engine        string `json:"engine"`
	Format        string `json:"format"`
	Quant         string `json:"quant"`
	Status        string `json:"status"`
	Comparability string `json:"comparability"`
	ResultFile    string `json:"result_file"`
}

// ComparisonFilename is comparison_{id}.json.
func ComparisonFilename(id string) string { return "comparison_" + id + ".json" }

// BuildComparison ranks results (best first) by the headline metrics.
func BuildComparison(id, requestedAs, preset string, sys SystemInfo, rs []*Result, files []string) Comparison {
	c := Comparison{SchemaVersion: SchemaVersion, ComparisonID: id, Timestamp: time.Now().UTC(), RequestedAs: requestedAs,
		Preset: preset, SystemInfo: sys, Ranking: map[string][]string{}, Notes: []string{}}
	formats, quants := map[string]bool{}, map[string]bool{}
	for i, r := range rs {
		c.Entries = append(c.Entries, ComparisonEntry{Engine: r.Engine.Name, Format: r.Model.Format, Quant: r.Model.Quant,
			Status: r.Status, Comparability: r.Comparability.Status, ResultFile: filepath.Base(files[i])})
		formats[r.Model.Format], quants[r.Model.Quant] = true, true
	}
	if len(formats) > 1 || len(quants) > 1 {
		c.Notes = append(c.Notes, "engines ran different model files (formats/quantizations); comparison is approximate")
	}
	rank := func(key string, higherBetter bool, value func(*Result) (float64, bool)) {
		type kv struct {
			engine string
			v      float64
		}
		var xs []kv
		for _, r := range rs {
			if v, ok := value(r); ok {
				xs = append(xs, kv{r.Engine.Name, v})
			}
		}
		if len(xs) < 2 {
			return
		}
		sort.SliceStable(xs, func(i, j int) bool {
			if higherBetter {
				return xs[i].v > xs[j].v
			}
			return xs[i].v < xs[j].v
		})
		for _, x := range xs {
			c.Ranking[key] = append(c.Ranking[key], x.engine)
		}
	}
	scen := func(name string, pick func(Summary) float64) func(*Result) (float64, bool) {
		return func(r *Result) (float64, bool) {
			s := r.Scenario(name)
			if s == nil || len(s.Iterations) == 0 {
				return 0, false
			}
			return pick(s.Summary), true
		}
	}
	rank("decode_tok_per_s", true, scen("chat", func(s Summary) float64 { return s.DecodeTokPerS.Median }))
	for _, n := range []string{"prefill-2k", "prefill-8k"} {
		rank("ttft_ms_"+n, false, scen(n, func(s Summary) float64 { return s.TTFTMs.Median }))
	}
	for _, n := range []string{"depth-4k", "depth-16k", "depth-32k"} {
		rank("decode_tok_per_s_"+n, true, scen(n, func(s Summary) float64 { return s.DecodeTokPerS.Median }))
	}
	rank("load_s", false, func(r *Result) (float64, bool) { return r.Load.LoadS, r.Load.LoadS > 0 })
	rank("peak_mem_bytes", false, func(r *Result) (float64, bool) { return float64(r.PeakMem()), r.PeakMem() > 0 })
	return c
}

// WriteComparison saves c into dir.
func WriteComparison(dir string, c Comparison) (string, error) {
	return writeJSON(dir, ComparisonFilename(c.ComparisonID), c)
}

// ReadComparison loads a comparison file.
func ReadComparison(path string) (*Comparison, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var c Comparison
	if err := json.Unmarshal(b, &c); err != nil {
		return nil, fmt.Errorf("%s: %w", path, err)
	}
	return &c, nil
}
