// Package bench orchestrates `kepler bench`: resolve the model, pick engines, run
// preflight, download, benchmark each engine one at a time, write results, and delete
// the model. It depends on the shared core; nothing in the shared core depends on it.
package bench

import (
	"context"
	"errors"
	"fmt"
	"path"
	"path/filepath"
	"strings"
	"time"

	"github.com/thisisadityapatel/kepler/internal/bench/preflight"
	"github.com/thisisadityapatel/kepler/internal/bench/results"
	"github.com/thisisadityapatel/kepler/internal/bench/runner"
	"github.com/thisisadityapatel/kepler/internal/bench/scenario"
	"github.com/thisisadityapatel/kepler/internal/download"
	"github.com/thisisadityapatel/kepler/internal/engines/adapter"
	"github.com/thisisadityapatel/kepler/internal/engines/resolve"
	"github.com/thisisadityapatel/kepler/internal/events"
	"github.com/thisisadityapatel/kepler/internal/hf"
	"github.com/thisisadityapatel/kepler/internal/ledger"
	"github.com/thisisadityapatel/kepler/internal/probe"
	"github.com/thisisadityapatel/kepler/internal/setup"
)

// Options are the `kepler bench` flags.
type Options struct {
	Model      string
	Preset     string
	Scenarios  []string
	Iterations int
	Engines    []string
	Skip       []string
	Quant      string
	Cooldown   time.Duration
	Keep       bool
	Yes        bool
}

// UI is what the orchestrator needs from the terminal.
type UI interface {
	Printf(format string, a ...any)
	Confirm(question string, def bool) bool
	Table(header []string, rows [][]string)
	Interactive() bool
}

// Env is the wired-up shared core.
type Env struct {
	Setup      *setup.Setup
	HF         *hf.Client
	UI         UI
	Version    string
	Conditions func(context.Context) probe.Conditions // nil = probe.SampleConditions
}

// Outcome is what a run produced.
type Outcome struct {
	Results        []*results.Result
	Files          []string
	ComparisonFile string
	Failed         int
	FreedBytes     int64
	KeptPath       string
}

// Errors the CLI maps to exit codes.
var (
	ErrNoEngine = errors.New("no ready engine can run this model")
	ErrDeclined = errors.New("cancelled")
	ErrNeedYes  = errors.New("not a terminal; re-run with --yes to start the benchmark")
	ErrInput    = errors.New("invalid input")
)

// Run executes a benchmark.
func Run(ctx context.Context, env Env, opt Options) (*Outcome, error) {
	s, u := env.Setup, env.UI
	conditions := env.Conditions
	if conditions == nil {
		conditions = func(ctx context.Context) probe.Conditions { return probe.SampleConditions(ctx, nil) }
	}

	ref, err := hf.ParseRef(opt.Model)
	if err != nil {
		return nil, wrapped{ErrInput, err.Error()}
	}
	if opt.Quant != "" {
		ref.Quant = strings.ToUpper(opt.Quant)
	}
	preset, err := scenario.Build(opt.Preset, opt.Scenarios, opt.Iterations)
	if err != nil {
		return nil, wrapped{ErrInput, err.Error()}
	}

	u.Printf("Resolving %s …\n", ref)
	res, err := hf.Resolve(ctx, env.HF, ref)
	if err != nil {
		return nil, err
	}

	// ---- engines that can run this format
	var engines []runner.Engine
	var hints []string
	for _, d := range s.Resolve(&resolve.Query{Formats: []string{res.Format}, Allow: opt.Engines, Deny: opt.Skip}) {
		if d.Status != resolve.Eligible {
			continue
		}
		if d.Install != resolve.InstallReady {
			hints = append(hints, fmt.Sprintf("%s is %s (kepler engines install %s)", d.Engine, strings.ReplaceAll(d.Install, "_", " "), d.Engine))
			continue
		}
		engines = append(engines, runner.Engine{RT: d.Def.Resolve(d.Target), Entry: s.Registry.Engines[d.Engine]})
	}
	if len(engines) == 0 {
		msg := fmt.Sprintf("%s (model format: %s)", ErrNoEngine, res.Format)
		if len(hints) > 0 {
			msg += "; " + strings.Join(hints, "; ")
		} else {
			msg += "; run `kepler engines list` to see why"
		}
		return nil, wrapped{ErrNoEngine, msg}
	}

	// ---- profile + preflight
	var profile hf.ModelProfile
	var cands []preflight.Candidate
	var variants []hf.Variant
	switch res.Format {
	case hf.FormatGGUF:
		if variants, err = res.Candidates(); err != nil {
			return nil, wrapped{ErrInput, err.Error()}
		}
		if profile, err = res.GGUFProfile(ctx, env.HF, variants[0]); err != nil {
			return nil, err
		}
		for _, v := range variants {
			cands = append(cands, preflight.Candidate{Quant: v.Quant, Size: v.Size})
		}
	case hf.FormatMLX:
		if profile, err = res.MLXProfile(ctx, env.HF); err != nil {
			return nil, err
		}
		cands = []preflight.Candidate{{Quant: profile.Quant, Size: profile.SizeBytes}}
	}
	copies := 1
	for _, e := range engines {
		if e.RT.Def.Materialize != nil {
			copies++ // the engine imports its own copy (Ollama)
			break
		}
	}
	plan, err := preflight.Run(preflight.Input{Candidates: cands, Arch: profile.Arch, Scenarios: preset.Scenarios, Facts: s.Facts, DiskCopies: copies})
	if err != nil {
		return nil, err
	}
	if res.Format == hf.FormatGGUF && plan.Choice > 0 {
		v := variants[plan.Choice]
		profile.Quant, profile.Files, profile.SizeBytes, profile.EntryFile = v.Quant, v.Files, v.Size, v.Entry().Path
	}
	preset.Scenarios = plan.Scenarios

	// ---- show the plan
	runID := setup.NewRunID()
	compID := ""
	if len(engines) > 1 {
		compID = "c" + strings.TrimPrefix(runID, "r")
	}
	printPlan(u, profile, res, engines, preset, plan)
	cond := conditions(ctx)
	for _, w := range preflight.Warnings(ctx, cond) {
		u.Printf("  ! %s\n", w)
	}
	if !opt.Yes {
		if !u.Interactive() {
			return nil, ErrNeedYes
		}
		if !u.Confirm("\nStart benchmark?", true) {
			return nil, ErrDeclined
		}
	}

	// ---- download
	out := &Outcome{}
	modelDir := filepath.Join(s.Paths.Models, runID)
	dirID, err := s.Ledger.Record(ctx, ledger.Artifact{RunID: runID, Kind: ledger.KindDir, Path: modelDir, Note: "model " + profile.Repo})
	if err != nil {
		return nil, err
	}
	kept := false
	defer func() {
		if kept {
			return
		}
		_ = s.Ledger.Dispose(context.WithoutCancel(ctx), dirID)
	}()
	if err := downloadModel(ctx, env, runID, modelDir, profile); err != nil {
		return nil, err
	}
	files := adapter.Model{Path: modelDir}
	if profile.EntryFile != "" {
		files.Path = filepath.Join(modelDir, profile.EntryFile)
	}

	// ---- benchmark each engine
	spec := runner.Spec{
		RunID: runID, ComparisonID: compID, KeplerVersion: env.Version, Preset: preset, Ctx: plan.Ctx,
		Skipped: plan.Skipped, Cooldown: opt.Cooldown, Model: profile, ModelFiles: files, Facts: s.Facts,
		Paths: s.Paths, Ledger: s.Ledger, Bus: s.Bus, Conditions: env.Conditions,
	}
	for i, e := range engines {
		if i > 0 {
			if err := cooldown(ctx, opt.Cooldown, conditions, u); err != nil {
				break
			}
		}
		u.Printf("\n%s (%s)\n", e.RT.Def.Name, e.Entry.Version)
		r := runner.Run(ctx, spec, e)
		file, err := results.Write(s.Paths.Results, r)
		if err != nil {
			return out, fmt.Errorf("write result: %w", err)
		}
		out.Results, out.Files = append(out.Results, r), append(out.Files, file)
		publishSummary(s.Bus, r)
		if r.Status == results.StatusFailed {
			out.Failed++
		}
		if ctx.Err() != nil {
			break
		}
	}

	// ---- comparison
	if compID != "" && len(out.Results) > 1 {
		c := results.BuildComparison(compID, profile.RequestedAs, preset.Name, results.NewSystemInfo(s.Facts), out.Results, out.Files)
		if out.ComparisonFile, err = results.WriteComparison(s.Paths.Results, c); err != nil {
			return out, err
		}
	}

	// ---- keep or delete the model
	if opt.Keep && ctx.Err() == nil {
		kept = true
		out.KeptPath = modelDir
		_ = s.Ledger.Keep(context.WithoutCancel(ctx), dirID)
	} else {
		out.FreedBytes = profile.SizeBytes
	}
	return out, ctx.Err()
}

type wrapped struct {
	err error
	msg string
}

func (w wrapped) Error() string { return w.msg }
func (w wrapped) Unwrap() error { return w.err }

func printPlan(u UI, p hf.ModelProfile, res *hf.Resolution, engines []runner.Engine, preset scenario.Preset, plan preflight.Plan) {
	var names, scen []string
	for _, e := range engines {
		names = append(names, fmt.Sprintf("%s (%s)", e.RT.Def.Name, e.Entry.Version))
	}
	for _, s := range preset.Scenarios {
		scen = append(scen, s.Name)
	}
	files := fmt.Sprintf("%d files", len(p.Files))
	if len(p.Files) == 1 {
		files = path.Base(p.Files[0].Path)
	}
	arch := "architecture unknown"
	if p.Arch.Known() {
		arch = fmt.Sprintf("%s · %d layers · %d-token context", orDash(p.Arch.Architecture), p.Arch.Layers, p.Arch.ContextMax)
	}
	rows := [][]string{
		{"model", fmt.Sprintf("%s @ %s", p.Repo, p.Revision[:min(7, len(p.Revision))])},
		{"weights", fmt.Sprintf("%s %s · %s · %s", strings.ToUpper(p.Format), p.Quant, gb(p.SizeBytes), files)},
		{"arch", arch},
		{"engines", strings.Join(names, ", ")},
		{"preset", fmt.Sprintf("%s: %s · %d iterations · context %d", preset.Name, strings.Join(scen, ", "), preset.Iterations, plan.Ctx)},
		{"memory", fmt.Sprintf("~%s needed of %s GPU memory", gb(plan.NeedMem), gb(plan.BudgetMem))},
		{"disk", fmt.Sprintf("%s needed of %s free; deleted after the run", gb(plan.NeedDisk), gb(plan.FreeDisk))},
	}
	if res.Format == hf.FormatGGUF && res.Ref.Quant == "" && res.Ref.File == "" && len(res.Variants) > 1 {
		rows = append(rows, []string{"quants", "also available: " + strings.Join(otherQuants(res, p.Quant), ", ") + " (use --quant)"})
	}
	u.Printf("\n")
	u.Table(nil, rows)
	for _, n := range plan.Notes {
		u.Printf("  · %s\n", n)
	}
	for _, sk := range plan.Skipped {
		u.Printf("  · skipping %s: %s\n", sk.Name, sk.Reason)
	}
}

func otherQuants(res *hf.Resolution, chosen string) []string {
	var out []string
	for _, q := range res.QuantNames() {
		if q != chosen && len(out) < 8 {
			out = append(out, q)
		}
	}
	return out
}

func downloadModel(ctx context.Context, env Env, runID, dir string, p hf.ModelProfile) error {
	s := env.Setup
	var done int64
	for _, f := range p.Files {
		req := download.Request{
			URL: env.HF.FileURL(p.Repo, p.Revision, f.Path), Dest: filepath.Join(dir, f.Path), Size: f.Size,
			Digest: download.Digest{SHA256: f.SHA256, GitOID: f.GitOID}, Header: env.HF.AuthHeader(),
		}
		// The partial file is ledgered too: an interrupted multi-GB download is deleted,
		// not left behind.
		partID, err := s.Ledger.Record(ctx, ledger.Artifact{RunID: runID, Kind: ledger.KindFile, Path: s.Downloader.PartPath(req), Note: "partial download"})
		if err != nil {
			return err
		}
		base := done
		label := path.Base(f.Path)
		err = s.Downloader.Fetch(ctx, req, func(d, _ int64) {
			s.Bus.Publish(events.Event{Type: events.DownloadProgress, Message: label, Done: base + d, Total: p.SizeBytes})
		})
		_ = s.Ledger.Dispose(context.WithoutCancel(ctx), partID)
		if err != nil {
			return fmt.Errorf("download %s: %w", f.Path, err)
		}
		done += f.Size
	}
	return nil
}

func cooldown(ctx context.Context, d time.Duration, conditions func(context.Context) probe.Conditions, u UI) error {
	if d <= 0 {
		return ctx.Err()
	}
	u.Printf("\ncooling down %s …\n", d)
	select {
	case <-ctx.Done():
		return ctx.Err()
	case <-time.After(d):
	}
	deadline := time.Now().Add(60 * time.Second)
	for conditions(ctx).Thermal == probe.ThermalElevated && time.Now().Before(deadline) {
		select {
		case <-ctx.Done():
			return ctx.Err()
		case <-time.After(5 * time.Second):
		}
	}
	return nil
}

func publishSummary(bus *events.Bus, r *results.Result) {
	if r.Status == results.StatusFailed || r.Status == results.StatusInterrupted {
		bus.Publish(events.Event{Type: events.BenchEngineDone, Engine: r.Engine.Name, Err: errors.New(firstLine(r.Error))})
		return
	}
	var parts []string
	if c := r.Scenario("chat"); c != nil && len(c.Iterations) > 0 {
		parts = append(parts, fmt.Sprintf("decode %.1f tok/s", c.Summary.DecodeTokPerS.Median))
	}
	for _, n := range []string{"prefill-2k", "prefill-8k"} {
		if c := r.Scenario(n); c != nil && len(c.Iterations) > 0 {
			parts = append(parts, fmt.Sprintf("TTFT %s %.2fs", strings.TrimPrefix(n, "prefill-"), c.Summary.TTFTMs.Median/1000))
		}
	}
	parts = append(parts, "peak "+gb(r.PeakMem()))
	msg := strings.Join(parts, " · ")
	if r.Status == results.StatusPartial {
		msg += " (partial: " + firstLine(r.Error) + ")"
	}
	if r.Comparability.Status == results.ComparabilityDegraded {
		msg += " ~"
	}
	bus.Publish(events.Event{Type: events.BenchEngineDone, Engine: r.Engine.Name, Message: msg})
}

func firstLine(s string) string {
	if i := strings.IndexByte(s, '\n'); i >= 0 {
		return s[:i]
	}
	return s
}

func orDash(s string) string {
	if s == "" {
		return "—"
	}
	return s
}

func gb(b int64) string {
	if b < 1<<30 {
		return fmt.Sprintf("%.0f MB", float64(b)/(1<<20))
	}
	return fmt.Sprintf("%.1f GB", float64(b)/(1<<30))
}
