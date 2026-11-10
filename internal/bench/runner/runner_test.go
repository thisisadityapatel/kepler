package runner

import (
	"context"
	"encoding/json"
	"os"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/thisisadityapatel/kepler/internal/bench/results"
	"github.com/thisisadityapatel/kepler/internal/bench/scenario"
	"github.com/thisisadityapatel/kepler/internal/engines/adapter"
	"github.com/thisisadityapatel/kepler/internal/engines/def"
	"github.com/thisisadityapatel/kepler/internal/events"
	"github.com/thisisadityapatel/kepler/internal/fakeengine"
	"github.com/thisisadityapatel/kepler/internal/hf"
	"github.com/thisisadityapatel/kepler/internal/home"
	"github.com/thisisadityapatel/kepler/internal/ledger"
	"github.com/thisisadityapatel/kepler/internal/openai"
	"github.com/thisisadityapatel/kepler/internal/probe"
	"github.com/thisisadityapatel/kepler/internal/registry"
)

func TestMain(m *testing.M) {
	fakeengine.MaybeServe()
	os.Exit(m.Run())
}

const fakeDef = `
schema: 1
name: fake
display: Fake
version: "1"
formats: [gguf]
tier: default
adapter: generic
launch:
  cmd: "{bin}"
  args: ["-test.run=^$", "--", "-m", "{model_path}", "--port", "{port}", "-c", "{ctx}"]
  env: { KEPLER_FAKE_ENGINE: "1", EXTRA }
  ready: { http_get: /health, expect: 200, timeout: 10s }
request: { api: openai, extra_body: { ignore_eos: true } }
caps: { ignore_eos: true, disable_prompt_cache: true }
targets:
  - id: any
    match: {}
    install: [{ kind: adopt, bin_name: fake }]
`

func setup(t *testing.T, env string) (Spec, Engine, *ledger.Ledger) {
	t.Helper()
	if !strings.Contains(env, "FAKE_TOKEN_DELAY") {
		env = strings.TrimPrefix(env+`, FAKE_TOKEN_DELAY: "1ms"`, ", ")
	}
	d, err := def.Parse([]byte(strings.Replace(fakeDef, "EXTRA", env, 1)), "fake.yaml", def.ValidateOptions{})
	if err != nil {
		t.Fatal(err)
	}
	p := home.New(t.TempDir())
	if err := p.Ensure(); err != nil {
		t.Fatal(err)
	}
	l, err := ledger.Open(p)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { l.Close() })
	preset, _ := scenario.Build("quick", nil, 2)
	preset.Scenarios[0].MaxTokens = 40
	spec := Spec{
		RunID: "r_test", KeplerVersion: "test", Preset: preset, Ctx: scenario.Ctx(preset.Scenarios),
		Model:      hf.ModelProfile{Repo: "org/m", Format: "gguf", Quant: "Q4_K_M", Files: []hf.File{{Path: "m.gguf", Size: 1}}},
		ModelFiles: adapter.Model{Path: "/models/m.gguf"},
		Facts:      probe.HostFacts{OS: "darwin", Arch: "arm64", CPU: probe.CPUInfo{Brand: "Apple Test"}},
		Paths:      p, Ledger: l, Bus: &events.Bus{}, SampleEvery: 20 * time.Millisecond,
		Conditions: func(context.Context) probe.Conditions {
			return probe.Conditions{Thermal: probe.ThermalNominal, PowerSource: "ac", MemoryFreePct: 60}
		},
	}
	eng := Engine{RT: d.Resolve(&d.Targets[0]), Entry: &registry.Engine{Version: "1", Install: &registry.Install{Kind: "adopt", Path: os.Args[0]}}}
	return spec, eng, l
}

func assertCleanedUp(t *testing.T, l *ledger.Ledger) {
	t.Helper()
	if rows, _ := l.List(context.Background(), ledger.StateTemp); len(rows) != 0 {
		t.Errorf("temp ledger rows remain (engine not stopped?): %+v", rows)
	}
}

func TestRunCompleteAndFair(t *testing.T) {
	spec, eng, l := setup(t, "")
	r := Run(context.Background(), spec, eng)
	if r.Status != results.StatusComplete {
		t.Fatalf("status = %s: %s", r.Status, r.Error)
	}
	if len(r.Scenarios) != 2 || len(r.Scenarios[0].Iterations) != 2 {
		t.Fatalf("scenarios = %+v", r.Scenarios)
	}
	chat := r.Scenario("chat")
	it := chat.Iterations[0]
	if it.GeneratedTokens != 40 || it.DecodeTokPerS <= 0 || it.TTFTMs <= 0 || it.ITLP99Ms <= 0 {
		t.Errorf("chat iteration = %+v", it)
	}
	if chat.Summary.DecodeTokPerS.Median <= 0 {
		t.Error("summary not computed")
	}
	// Calibration: the fake tokenizes at 4 chars/token.
	if c := r.Config.CharsPerToken; c < 3.9 || c > 4.1 {
		t.Errorf("chars/token = %.2f, want ~4", c)
	}
	pre := r.Scenario("prefill-2k").Iterations[0]
	if pre.PromptTokens < 1900 || pre.PromptTokens > 2300 {
		t.Errorf("prefill-2k prompt tokens = %d, want ~2048", pre.PromptTokens)
	}
	if r.Load.LoadS <= 0 || r.Load.PeakMemBytes <= 0 || it.PeakMemBytes <= 0 {
		t.Errorf("load/mem = %+v, iteration peak %d", r.Load, it.PeakMemBytes)
	}
	if r.Comparability.Status != results.ComparabilityFull || len(r.Comparability.Notes) != 0 {
		t.Errorf("comparability = %+v", r.Comparability)
	}
	if !strings.Contains(strings.Join(r.Engine.LaunchArgs, " "), "-c "+strconv.Itoa(spec.Ctx)) {
		t.Errorf("launch args = %v", r.Engine.LaunchArgs)
	}
	if r.SampleOutput == "" || r.LogPath == "" {
		t.Error("sample output / log path missing")
	}
	assertCleanedUp(t, l)
}

func TestRunFlagsEarlyStopAndMissingUsage(t *testing.T) {
	spec, eng, l := setup(t, `FAKE_STOP_AT: "10", FAKE_NO_USAGE: "1"`)
	r := Run(context.Background(), spec, eng)
	if r.Status != results.StatusComplete {
		t.Fatalf("status = %s: %s", r.Status, r.Error)
	}
	if r.Comparability.Status != results.ComparabilityDegraded {
		t.Fatalf("comparability = %+v", r.Comparability)
	}
	joined := strings.Join(r.Comparability.Notes, "\n")
	for _, want := range []string{"chat: stopped before 40 tokens in 2/2", "token counts from stream events"} {
		if !strings.Contains(joined, want) {
			t.Errorf("notes missing %q:\n%s", want, joined)
		}
	}
	if !strings.Contains(strings.Join(r.Comparability.Info, "\n"), "prompt sizes estimated") {
		t.Errorf("info = %v", r.Comparability.Info)
	}
	assertCleanedUp(t, l)
}

func TestRunEngineCrashIsPartial(t *testing.T) {
	// load probe + 2 calibration + 1 warmup + 1 chat iteration, then the engine dies.
	spec, eng, l := setup(t, `FAKE_DIE_AFTER: "5"`)
	r := Run(context.Background(), spec, eng)
	if r.Status != results.StatusPartial || !strings.Contains(r.Error, "engine exited during chat") {
		t.Fatalf("status = %s err = %s", r.Status, r.Error)
	}
	if got := len(r.Scenario("chat").Iterations); got != 1 {
		t.Errorf("chat iterations = %d, want 1", got)
	}
	assertCleanedUp(t, l)
}

func TestRunInterrupted(t *testing.T) {
	spec, eng, l := setup(t, `FAKE_TOKEN_DELAY: "20ms"`)
	ctx, cancel := context.WithCancel(context.Background())
	spec.Bus.Subscribe(func(e events.Event) {
		if e.Type == EventIteration && e.Message == "chat" {
			go func() { time.Sleep(50 * time.Millisecond); cancel() }()
		}
	})
	r := Run(ctx, spec, eng)
	if r.Status != results.StatusInterrupted {
		t.Fatalf("status = %s err = %s", r.Status, r.Error)
	}
	b, _ := json.Marshal(r)
	if strings.Contains(string(b), "null") {
		t.Errorf("result JSON contains null (use empty arrays): %s", b)
	}
	assertCleanedUp(t, l)
}

func TestMeasure(t *testing.T) {
	t0 := time.Unix(0, 0)
	res := openai.StreamResult{Sent: t0, Closed: t0.Add(1100 * time.Millisecond),
		Usage: &openai.Usage{PromptTokens: 100, CompletionTokens: 11}}
	for i := 0; i < 11; i++ {
		res.TokenTimes = append(res.TokenTimes, t0.Add(100*time.Millisecond+time.Duration(i)*100*time.Millisecond))
	}
	it, obs := Measure(res, 11, 0)
	if it.TTFTMs != 100 || it.PrefillTokPerS != 1000 || it.DecodeTokPerS != 10 || it.ITLP50Ms != 100 || obs.StoppedEarly || obs.UsageMissing {
		t.Fatalf("iteration = %+v obs = %+v", it, obs)
	}
}
