// Package runner benchmarks one engine on one downloaded model: launch, measure load,
// warm up, calibrate prompt sizing, run every scenario, stop, and assemble the result.
package runner

import (
	"context"
	"errors"
	"fmt"
	"path/filepath"
	"strings"
	"time"

	"github.com/thisisadityapatel/kepler/internal/bench/preflight"
	"github.com/thisisadityapatel/kepler/internal/bench/results"
	"github.com/thisisadityapatel/kepler/internal/bench/scenario"
	"github.com/thisisadityapatel/kepler/internal/engines/adapter"
	"github.com/thisisadityapatel/kepler/internal/engines/def"
	"github.com/thisisadityapatel/kepler/internal/events"
	"github.com/thisisadityapatel/kepler/internal/hf"
	"github.com/thisisadityapatel/kepler/internal/home"
	"github.com/thisisadityapatel/kepler/internal/ledger"
	"github.com/thisisadityapatel/kepler/internal/openai"
	"github.com/thisisadityapatel/kepler/internal/probe"
	"github.com/thisisadityapatel/kepler/internal/proc"
	"github.com/thisisadityapatel/kepler/internal/registry"
)

// Seed is sent with every request.
const Seed = 42

// Progress event types published by the runner (rendered by the CLI).
const (
	EventPhase     = events.BenchPhase
	EventIteration = events.BenchIteration
)

// Spec is everything shared by all engines in one bench run.
type Spec struct {
	RunID          string
	ComparisonID   string
	KeplerVersion  string
	Preset         scenario.Preset // scenarios already pruned by preflight
	Ctx            int
	Skipped        []preflight.Skipped
	Cooldown       time.Duration
	Model          hf.ModelProfile
	ModelFiles     adapter.Model // on-disk location
	Facts          probe.HostFacts
	Paths          home.Paths
	Ledger         *ledger.Ledger
	Bus            *events.Bus
	SampleEvery    time.Duration                          // memory sampling interval (default 100ms)
	RequestTimeout time.Duration                          // per request (default 10m)
	Conditions     func(context.Context) probe.Conditions // nil = probe.SampleConditions
}

// Engine is one engine to run, as resolved and installed.
type Engine struct {
	RT    def.ResolvedTarget
	Entry *registry.Engine
}

// Run benchmarks one engine. It always returns a result (status failed/partial/
// interrupted on problems) and never leaves the engine running.
func Run(ctx context.Context, spec Spec, eng Engine) *results.Result {
	name := eng.RT.Def.Name
	r := &results.Result{
		SchemaVersion: results.SchemaVersion, KeplerVersion: spec.KeplerVersion, Timestamp: time.Now().UTC(),
		RunID: spec.RunID, ComparisonID: spec.ComparisonID, Status: results.StatusComplete, Model: spec.Model,
		Engine: results.EngineInfo{Name: name, Display: eng.RT.Def.Display, Version: eng.Entry.Version,
			Target: eng.RT.Target.ID, InstallKind: eng.Entry.Install.Kind},
		SystemInfo: results.NewSystemInfo(spec.Facts),
		Config: results.Config{Preset: spec.Preset.Name, Temperature: 0, Seed: Seed, Ctx: spec.Ctx,
			WarmupIterations: spec.Preset.Warmup, Iterations: spec.Preset.Iterations, CooldownS: spec.Cooldown.Seconds(),
			CharsPerToken: scenario.DefaultCharsPerToken, MemoryMetric: proc.FootprintMethod},
		Scenarios:        []results.Scenario{},
		SkippedScenarios: append([]preflight.Skipped{}, spec.Skipped...),
		Comparability:    results.Comparability{Status: results.ComparabilityFull, Notes: []string{}, Info: []string{}},
	}
	if r.SkippedScenarios == nil {
		r.SkippedScenarios = []preflight.Skipped{}
	}
	conditions := spec.Conditions
	if conditions == nil {
		conditions = func(ctx context.Context) probe.Conditions { return probe.SampleConditions(ctx, nil) }
	}
	r.Conditions.Before = conditions(ctx)
	phase := func(format string, a ...any) {
		spec.Bus.Publish(events.Event{Type: EventPhase, Engine: name, Message: fmt.Sprintf(format, a...)})
	}
	fail := func(status string, err error) *results.Result {
		r.Status, r.Error = status, err.Error()
		if ctx.Err() != nil {
			r.Status, r.Error = results.StatusInterrupted, "interrupted"
		}
		return r
	}

	a, err := adapter.New(eng.RT, adapter.Installed{Bin: eng.Entry.Install.Path, Venv: eng.Entry.Install.Venv, Version: eng.Entry.Version}, spec.Paths)
	if err != nil {
		return fail(results.StatusFailed, err)
	}
	logDir := filepath.Join(spec.Paths.Logs, spec.RunID)
	r.LogPath = filepath.Join(logDir, name+".log")
	run := adapter.RunEnv{RunID: spec.RunID, Ctx: spec.Ctx, LogDir: logDir, Ledger: spec.Ledger, Bus: spec.Bus}

	// ---- launch and load
	phase("loading model")
	em, err := a.Materialize(ctx, spec.ModelFiles, run)
	if err != nil {
		return fail(results.StatusFailed, err)
	}
	running, err := a.Launch(ctx, em, run)
	if err != nil {
		return fail(results.StatusFailed, err)
	}
	r.Engine.LaunchArgs = running.Proc.Cmd.Args[1:]
	every := spec.SampleEvery
	if every == 0 {
		every = 100 * time.Millisecond
	}
	sampler := NewSampler(running.PGIDOrPID(), every, spec.Paths.Tmp)
	sampler.Start()
	defer func() {
		sampler.Stop()
		_ = adapter.StopAndRelease(context.WithoutCancel(ctx), a, running, run)
		r.Load.PeakMemBytes = max(r.Load.PeakMemBytes, sampler.Peak())
		r.Conditions.After = conditions(context.WithoutCancel(ctx))
		finalizeComparability(r, eng.RT, r.Conditions)
	}()

	if err := a.WaitReady(ctx, running); err != nil {
		return fail(results.StatusFailed, err)
	}
	readyAt := time.Now()
	em, err = a.PostReady(ctx, running, em)
	if err != nil {
		return fail(results.StatusFailed, err)
	}
	postReady := time.Now()
	r.Load.MaterializeS = postReady.Sub(readyAt).Seconds()
	ep := a.Endpoint(running, em)
	client := openai.New(ep.BaseURL)
	timeout := spec.RequestTimeout
	if timeout == 0 {
		timeout = 10 * time.Minute
	}
	send := func(prompt string, maxTokens int) (openai.StreamResult, error) {
		cctx, cancel := context.WithTimeout(ctx, timeout)
		defer cancel()
		seed := Seed
		return client.Stream(cctx, openai.ChatRequest{Model: ep.Model, MaxTokens: maxTokens, Temperature: 0, Seed: &seed,
			Extra: ep.ExtraBody, Messages: []openai.Message{{Role: "user", Content: prompt}}})
	}

	first, err := send("Hi", 1)
	if err != nil {
		return fail(results.StatusFailed, fmt.Errorf("first request: %w", err))
	}
	// load_s = spawn → first token, excluding engine-specific import time (Ollama create).
	r.Load.LoadS = first.TokenTimes[0].Sub(running.Proc.Started).Seconds() - r.Load.MaterializeS
	r.Load.MemAfterLoadBytes = sampler.SampleFull()
	r.Load.MappedModelBytes = sampler.Mapped()
	phase("loaded in %.1fs", r.Load.LoadS)

	// ---- calibrate prompt sizing to this model's tokenizer
	cpt, calErr := calibrate(send, spec.RunID+"-"+name)
	if calErr != nil {
		r.Comparability.Info = append(r.Comparability.Info, "prompt sizes estimated at 4 chars/token: "+calErr.Error())
	} else {
		r.Config.CharsPerToken = cpt
	}

	// ---- warmup (discarded)
	if len(spec.Preset.Scenarios) > 0 {
		warm := spec.Preset.Scenarios[0]
		for i := 0; i < spec.Preset.Warmup; i++ {
			phase("warming up")
			if _, err := send(scenario.Prompt(warm, fmt.Sprintf("%s-%s-warmup-%d", spec.RunID, name, i), r.Config.CharsPerToken), min(warm.MaxTokens, 128)); err != nil {
				return fail(results.StatusFailed, fmt.Errorf("warmup: %w", err))
			}
		}
	}

	// ---- scenarios
	for _, s := range spec.Preset.Scenarios {
		sr := results.Scenario{Name: s.Name, Kind: s.Kind, TargetPromptTokens: s.PromptTokens, MaxTokens: s.MaxTokens,
			Iterations: []results.Iteration{}, Flags: []string{}}
		var obs []Observation
		for i := 0; i < spec.Preset.Iterations; i++ {
			prompt := scenario.Prompt(s, fmt.Sprintf("%s-%s-%s-%d", spec.RunID, name, s.Name, i), r.Config.CharsPerToken)
			sampler.ResetWindow()
			res, err := send(prompt, s.MaxTokens)
			if err != nil {
				sr.Error = err.Error()
				break
			}
			it, o := Measure(res, s.MaxTokens, int(float64(len(prompt))/r.Config.CharsPerToken))
			it.PeakMemBytes = sampler.WindowPeak()
			sr.Iterations = append(sr.Iterations, it)
			obs = append(obs, o)
			spec.Bus.Publish(events.Event{Type: EventIteration, Engine: name, Message: s.Name, Done: int64(i + 1), Total: int64(spec.Preset.Iterations)})
			if r.SampleOutput == "" && s.Kind == scenario.KindChat {
				r.SampleOutput = truncate(res.Text, 500)
			}
		}
		sr.Flags = append(sr.Flags, scenarioFlags(obs, s.MaxTokens)...)
		sr.Summarize()
		r.Scenarios = append(r.Scenarios, sr)
		if sr.Error != "" {
			if ctx.Err() != nil {
				return fail(results.StatusInterrupted, ctx.Err())
			}
			r.Status = results.StatusPartial
			select { // a failed request often means the engine crashed; give it a moment to exit
			case <-running.Proc.Done():
			case <-time.After(time.Second):
			}
			if exited, _ := running.Proc.Exited(); exited {
				r.Error = fmt.Sprintf("engine exited during %s; last log lines:\n%s", s.Name, running.Proc.LogTail(10))
				return r
			}
		}
	}
	if !sampler.Working() {
		r.Comparability.Info = append(r.Comparability.Info, "memory sampling unavailable on this platform")
	}
	return r
}

// calibrate measures characters per token for this model by sending a large and a
// small prompt (1 output token each); the difference cancels chat-template overhead.
func calibrate(send func(string, int) (openai.StreamResult, error), nonce string) (float64, error) {
	big := scenario.Scenario{Name: "calibrate", PromptTokens: 512}
	pBig := scenario.Prompt(big, nonce+"-cal-a", scenario.DefaultCharsPerToken)
	pSmall := "[session " + nonce + "-cal-b]\nHi"
	a, err := send(pBig, 1)
	if err != nil {
		return 0, err
	}
	b, err := send(pSmall, 1)
	if err != nil {
		return 0, err
	}
	if a.Usage == nil || b.Usage == nil || a.Usage.PromptTokens <= b.Usage.PromptTokens {
		return 0, errors.New("engine did not report prompt token usage")
	}
	cpt := float64(len(pBig)-len(pSmall)) / float64(a.Usage.PromptTokens-b.Usage.PromptTokens)
	if cpt < 1.5 || cpt > 10 {
		return 0, fmt.Errorf("implausible tokenizer ratio %.2f chars/token", cpt)
	}
	return cpt, nil
}

// finalizeComparability folds scenario flags and host conditions into the verdict.
func finalizeComparability(r *results.Result, rt def.ResolvedTarget, c results.Conditions) {
	for _, s := range r.Scenarios {
		for _, f := range s.Flags {
			if strings.HasPrefix(f, "inter-token latency approximate") {
				r.Comparability.Info = append(r.Comparability.Info, s.Name+": "+f)
				continue
			}
			r.Comparability.Notes = append(r.Comparability.Notes, s.Name+": "+f)
		}
	}
	for _, cond := range []probe.Conditions{c.Before, c.After} {
		if cond.Thermal == probe.ThermalElevated {
			r.Comparability.Notes = appendOnce(r.Comparability.Notes, "machine was thermally throttled ("+cond.ThermalDetail+")")
		}
		if cond.PowerSource == "battery" {
			r.Comparability.Notes = appendOnce(r.Comparability.Notes, "ran on battery power")
		}
		if cond.LowPowerMode {
			r.Comparability.Notes = appendOnce(r.Comparability.Notes, "Low Power Mode was on")
		}
	}
	if !rt.Caps["disable_prompt_cache"] {
		r.Comparability.Info = appendOnce(r.Comparability.Info, "engine prompt cache cannot be disabled; a unique per-iteration prefix prevents cache hits")
	}
	if r.Model.Vision {
		r.Comparability.Info = appendOnce(r.Comparability.Info, "vision projector not loaded (text-only benchmark)")
	}
	if len(r.Comparability.Notes) > 0 {
		r.Comparability.Status = results.ComparabilityDegraded
	}
}

func appendOnce(xs []string, s string) []string {
	for _, x := range xs {
		if x == s {
			return xs
		}
	}
	return append(xs, s)
}

func truncate(s string, n int) string {
	r := []rune(s)
	if len(r) <= n {
		return s
	}
	return string(r[:n])
}
