// Package setup installs and health-checks engines: the logic behind `kepler init`,
// `kepler engines install`, and `kepler doctor`. Shared core; no benchmark knowledge.
package setup

import (
	"context"
	"crypto/rand"
	"encoding/hex"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"regexp"
	"time"

	"github.com/thisisadityapatel/kepler/internal/download"
	"github.com/thisisadityapatel/kepler/internal/engines/adapter"
	"github.com/thisisadityapatel/kepler/internal/engines/def"
	"github.com/thisisadityapatel/kepler/internal/engines/resolve"
	"github.com/thisisadityapatel/kepler/internal/events"
	"github.com/thisisadityapatel/kepler/internal/home"
	"github.com/thisisadityapatel/kepler/internal/install"
	"github.com/thisisadityapatel/kepler/internal/ledger"
	"github.com/thisisadityapatel/kepler/internal/openai"
	"github.com/thisisadityapatel/kepler/internal/probe"
	"github.com/thisisadityapatel/kepler/internal/registry"
)

// Setup holds everything needed to install and check engines.
type Setup struct {
	Paths      home.Paths
	Defs       def.Set
	Canaries   def.Canaries
	Facts      probe.HostFacts
	Registry   *registry.Registry
	Ledger     *ledger.Ledger
	Downloader *download.Client
	Bus        *events.Bus
	Version    string // kepler version, written to the registry
	HFEndpoint string // Hugging Face base URL; "" = $HF_ENDPOINT or https://huggingface.co
}

// hf returns the Hugging Face base URL.
func (s *Setup) hf() string {
	if s.HFEndpoint != "" {
		return s.HFEndpoint
	}
	if e := os.Getenv("HF_ENDPOINT"); e != "" {
		return e
	}
	return "https://huggingface.co"
}

// NewRunID returns a sortable unique id such as r_20261012T153000_3fa9c1.
func NewRunID() string {
	b := make([]byte, 3)
	_, _ = rand.Read(b)
	return "r_" + time.Now().UTC().Format("20060102T150405") + "_" + hex.EncodeToString(b)
}

// Resolve runs the engine resolver against current facts and registry state.
func (s *Setup) Resolve(q *resolve.Query) []resolve.Decision {
	return resolve.Resolve(s.Facts, s.Defs, s.Registry.Installed(), q)
}

// SaveRegistry stamps and writes the registry.
func (s *Setup) SaveRegistry() error {
	s.Registry.KeplerVersion = s.Version
	s.Registry.DefinitionsSchema = def.SchemaVersion
	s.Registry.ProbedAt = time.Now().UTC()
	s.Registry.Host = s.Facts
	return s.Registry.Save(s.Paths.Registry)
}

// RecordDecisions writes ineligible/skipped decisions into the registry so `engines
// list` and future runs can show reasons. Installed engines are left untouched.
func (s *Setup) RecordDecisions(ds []resolve.Decision) {
	for _, d := range ds {
		e := s.Registry.Engines[d.Engine]
		installed := e != nil && (e.Status == registry.StatusReady || e.Status == registry.StatusUnhealthy)
		switch {
		case d.Status == resolve.Ineligible && !installed:
			s.Registry.Engines[d.Engine] = &registry.Engine{Status: registry.StatusIneligible, Reason: d.Reason, Source: d.Def.Source}
		case d.Status == resolve.Eligible && e == nil:
			s.Registry.Engines[d.Engine] = &registry.Engine{Status: registry.StatusNotInstalled, Source: d.Def.Source, Target: d.Target.ID, Formats: d.Def.Formats}
		}
	}
}

// MarkSkipped records that the user declined an engine.
func (s *Setup) MarkSkipped(d resolve.Decision) {
	s.Registry.Engines[d.Engine] = &registry.Engine{Status: registry.StatusSkipped, Source: d.Def.Source}
}

// Install tries each install method of the decision's target in order. A method
// succeeds only if its health check passes; otherwise its files are rolled back and
// the next method is tried. The registry is updated either way.
func (s *Setup) Install(ctx context.Context, d resolve.Decision) error {
	if d.Status != resolve.Eligible {
		return fmt.Errorf("%s is %s: %s", d.Engine, d.Status, d.Reason)
	}
	runID := NewRunID()
	logDir := filepath.Join(s.Paths.Logs, runID)
	if err := os.MkdirAll(logDir, 0o755); err != nil {
		return err
	}
	s.Bus.Publish(events.Event{Type: events.InstallStarted, Engine: d.Engine})

	var errs []error
	for i, m := range d.Target.Install {
		inst, ok := install.Get(m.Kind)
		if !ok {
			errs = append(errs, fmt.Errorf("method %d: unknown kind %s", i+1, m.Kind))
			continue
		}
		env := install.Env{
			Paths: s.Paths, Ledger: s.Ledger, Downloader: s.Downloader, Bus: s.Bus, Facts: s.Facts,
			Engine: d.Engine, Version: d.Def.Version, RunID: runID, LogPath: filepath.Join(logDir, "install.log"),
		}
		res, err := inst.Install(ctx, env, m)
		if err != nil {
			s.rollback(ctx, res)
			errs = append(errs, fmt.Errorf("%s: %w", m.Kind, err))
			if ctx.Err() != nil {
				break
			}
			continue
		}
		entry := &registry.Engine{
			Status: registry.StatusReady, Source: d.Def.Source, Target: d.Target.ID,
			Version: res.Version, Formats: d.Def.Formats,
			Install: &registry.Install{Kind: res.Kind, Path: res.Bin, Dir: res.Dir, Venv: res.Venv, At: time.Now().UTC().Format(time.RFC3339)},
		}
		if res.Kind != "adopt" {
			entry.Version = d.Def.Version
		}
		h := s.HealthCheck(ctx, d, entry, runID)
		entry.Health = h
		if !h.OK {
			s.rollback(ctx, res)
			errs = append(errs, fmt.Errorf("%s: installed but health check failed: %s (log: %s)", m.Kind, h.Error, h.LogPath))
			if ctx.Err() != nil {
				break
			}
			continue
		}
		if res.LedgerID != 0 {
			if err := s.Ledger.Release(ctx, res.LedgerID); err != nil {
				return err
			}
		}
		s.Registry.Engines[d.Engine] = entry
		s.Bus.Publish(events.Event{Type: events.InstallFinished, Engine: d.Engine,
			Message: fmt.Sprintf("%s via %s · health ok (load %.1fs, %.0f tok/s)", entry.Version, res.Kind, h.LoadS, h.DecodeTokPerS)})
		return s.SaveRegistry()
	}

	if ctx.Err() != nil {
		return ctx.Err() // interrupted: files were rolled back; don't record a failure
	}
	err := errors.Join(errs...)
	prev := s.Registry.Engines[d.Engine]
	if prev == nil || prev.Status != registry.StatusReady {
		s.Registry.Engines[d.Engine] = &registry.Engine{Status: registry.StatusUnhealthy, Reason: err.Error(), Source: d.Def.Source, Target: d.Target.ID, Formats: d.Def.Formats}
	}
	s.Bus.Publish(events.Event{Type: events.InstallFinished, Engine: d.Engine, Err: err})
	if saveErr := s.SaveRegistry(); saveErr != nil {
		return errors.Join(err, saveErr)
	}
	return err
}

func (s *Setup) rollback(ctx context.Context, res install.Result) {
	if res.LedgerID != 0 {
		_ = s.Ledger.Dispose(context.WithoutCancel(ctx), res.LedgerID)
	}
}

// Remove deletes a Kepler-owned install (adopted binaries are never touched) and
// forgets the engine.
func (s *Setup) Remove(name string) error {
	e := s.Registry.Engines[name]
	if e == nil {
		return fmt.Errorf("%s is not installed", name)
	}
	if e.Install != nil && e.Install.Dir != "" {
		if err := s.Paths.SafeRemoveAll(e.Install.Dir); err != nil {
			return err
		}
		_ = os.Remove(filepath.Dir(e.Install.Dir)) // engine dir, if now empty
	}
	delete(s.Registry.Engines, name)
	return s.SaveRegistry()
}

// ---------------------------------------------------------------- canaries

var unsafeChars = regexp.MustCompile(`[^A-Za-z0-9._-]+`)

// CanaryDir is where a format's canary lives.
func (s *Setup) CanaryDir(format string) string {
	c := s.Canaries[format]
	return filepath.Join(s.Paths.Canary, format, unsafeChars.ReplaceAllString(c.Repo, "-")+"@"+c.Revision[:12])
}

// EnsureCanary downloads (or verifies) the canary for format and returns the model.
// Canaries are kept permanently; a partially downloaded canary is cleaned by the reaper.
func (s *Setup) EnsureCanary(ctx context.Context, format, runID string) (adapter.Model, error) {
	c, ok := s.Canaries[format]
	if !ok {
		return adapter.Model{}, fmt.Errorf("no canary for format %q", format)
	}
	dir := s.CanaryDir(format)
	complete := filepath.Join(dir, ".complete")
	model := adapter.Model{Path: dir}
	if c.Entry != "" {
		model.Path = filepath.Join(dir, c.Entry)
	}
	if _, err := os.Stat(complete); err == nil {
		return model, nil
	}
	id, err := s.Ledger.Record(ctx, ledger.Artifact{RunID: runID, Kind: ledger.KindDir, Path: dir, Note: "canary " + format})
	if err != nil {
		return model, err
	}
	total := c.SizeBytes()
	var base int64
	for _, f := range c.Files {
		url := fmt.Sprintf("%s/%s/resolve/%s/%s", s.hf(), c.Repo, c.Revision, f.Path)
		req := download.Request{URL: url, Dest: filepath.Join(dir, f.Path), Size: f.Size,
			Digest: download.Digest{SHA256: f.SHA256, GitOID: f.GitOID}, Header: HFHeader()}
		offset := base
		err := s.Downloader.Fetch(ctx, req, func(done, _ int64) {
			s.Bus.Publish(events.Event{Type: events.DownloadProgress, Message: "canary " + format, Done: offset + done, Total: total})
		})
		if err != nil {
			_ = s.Ledger.Dispose(context.WithoutCancel(ctx), id)
			return model, fmt.Errorf("download canary %s: %w", format, err)
		}
		base += f.Size
	}
	if err := os.WriteFile(complete, nil, 0o644); err != nil {
		return model, err
	}
	return model, s.Ledger.Release(ctx, id)
}

// HFHeader returns auth headers for Hugging Face if HF_TOKEN is set.
func HFHeader() map[string][]string {
	if tok := os.Getenv("HF_TOKEN"); tok != "" {
		return map[string][]string{"Authorization": {"Bearer " + tok}}
	}
	return nil
}

// ---------------------------------------------------------------- health check

// HealthCheck launches the engine on its format's canary, waits for ready, runs one
// short completion, and stops it. It never returns an error; failures are in Health.
func (s *Setup) HealthCheck(ctx context.Context, d resolve.Decision, e *registry.Engine, runID string) *registry.Health {
	h := &registry.Health{At: time.Now().UTC().Format(time.RFC3339)}
	logDir := filepath.Join(s.Paths.Logs, runID)
	h.LogPath = filepath.Join(logDir, d.Engine+".log")
	fail := func(err error) *registry.Health {
		h.Error = err.Error()
		s.Bus.Publish(events.Event{Type: events.HealthFinished, Engine: d.Engine, Err: err})
		return h
	}

	format := d.Def.Formats[0]
	model, err := s.EnsureCanary(ctx, format, runID)
	if err != nil {
		return fail(err)
	}
	a, err := adapter.New(d.Def.Resolve(d.Target), adapter.Installed{Bin: e.Install.Path, Venv: e.Install.Venv, Version: e.Version}, s.Paths)
	if err != nil {
		return fail(err)
	}
	run := adapter.RunEnv{RunID: runID, Ctx: 2048, LogDir: logDir, Ledger: s.Ledger, Bus: s.Bus}
	em, err := a.Materialize(ctx, model, run)
	if err != nil {
		return fail(err)
	}
	r, err := a.Launch(ctx, em, run)
	if err != nil {
		return fail(err)
	}
	defer adapter.StopAndRelease(ctx, a, r, run)

	if err := a.WaitReady(ctx, r); err != nil {
		return fail(err)
	}
	s.Bus.Publish(events.Event{Type: events.EngineReady, Engine: d.Engine})
	if em, err = a.PostReady(ctx, r, em); err != nil {
		return fail(err)
	}
	ep := a.Endpoint(r, em)
	seed := 42
	cctx, cancel := context.WithTimeout(ctx, 2*time.Minute)
	defer cancel()
	res, err := openai.New(ep.BaseURL).Stream(cctx, openai.ChatRequest{
		Model: ep.Model, MaxTokens: 32, Temperature: 0, Seed: &seed, Extra: ep.ExtraBody,
		Messages: []openai.Message{{Role: "user", Content: "Name three planets in our solar system."}},
	})
	if err != nil {
		return fail(fmt.Errorf("completion: %w", err))
	}
	h.OK = true
	h.LoadS = res.TokenTimes[0].Sub(r.Proc.Started).Seconds()
	if n := res.Events(); n > 1 {
		h.DecodeTokPerS = float64(n-1) / res.TokenTimes[n-1].Sub(res.TokenTimes[0]).Seconds()
	}
	h.ReportedVersion = e.Version
	s.Bus.Publish(events.Event{Type: events.HealthFinished, Engine: d.Engine})
	return h
}
