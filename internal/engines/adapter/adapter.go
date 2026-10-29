// Package adapter runs engines described by YAML definitions. GenericAdapter is the only
// Adapter implementation; engines needing custom steps register Hooks under the name
// used in their definition's `adapter:` field. Nothing here knows about benchmarking.
package adapter

import (
	"context"
	"fmt"
	"net/http"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"time"

	"github.com/thisisadityapatel/kepler/internal/engines/def"
	"github.com/thisisadityapatel/kepler/internal/events"
	"github.com/thisisadityapatel/kepler/internal/home"
	"github.com/thisisadityapatel/kepler/internal/ledger"
	"github.com/thisisadityapatel/kepler/internal/proc"
)

// Generic is the adapter name for YAML-only engines.
const Generic = "generic"

// Adapter is what modules (bench today, inference later) use to drive an engine.
type Adapter interface {
	Name() string
	Formats() []string
	Caps() map[string]bool

	// Materialize prepares engine-specific model artifacts before launch.
	Materialize(ctx context.Context, m Model, run RunEnv) (EngineModel, error)
	// Launch spawns the server in its own process group. It does not wait for ready.
	Launch(ctx context.Context, em EngineModel, run RunEnv) (*Running, error)
	// WaitReady polls the readiness probe until success, timeout, or process exit.
	WaitReady(ctx context.Context, r *Running) error
	// PostReady runs steps that need the live server (e.g. `ollama create`).
	PostReady(ctx context.Context, r *Running, em EngineModel) (EngineModel, error)
	// Endpoint returns the base URL and per-engine request defaults.
	Endpoint(r *Running, em EngineModel) Endpoint
	// Stop kills the process group and waits until it is gone.
	Stop(r *Running) error
}

// Model is a model on disk in a format the engine supports.
type Model struct {
	Path   string // file (GGUF entry) or directory (MLX)
	MMProj string // optional vision projector
}

// EngineModel is the model as the engine will see it, plus template variables.
type EngineModel struct {
	Model
	Vars map[string]string
}

// RunEnv is per-run context shared by all steps.
type RunEnv struct {
	RunID  string
	Ctx    int    // context length passed to the engine
	LogDir string // engine logs go to LogDir/<engine>.log
	Ledger *ledger.Ledger
	Bus    *events.Bus
}

// Running is a launched engine.
type Running struct {
	Proc     *proc.Proc
	Port     int
	BaseURL  string // http://127.0.0.1:<port>
	LedgerID int64  // the server's process-group row
	Vars     map[string]string
	// Owned are extra ledger rows (e.g. a private model store) disposed after Stop.
	Owned []int64
}

// PGIDOrPID is the process group id of the running engine (its leader pid).
func (r *Running) PGIDOrPID() int { return r.Proc.PGID }

// Endpoint is how a client talks to a running engine.
type Endpoint struct {
	BaseURL   string // OpenAI-compatible base, e.g. http://127.0.0.1:51234/v1
	Model     string
	ExtraBody map[string]any
}

// Installed is where an engine binary lives (from the registry).
type Installed struct {
	Bin     string
	Venv    string
	Version string
}

// GenericAdapter implements Adapter entirely from a resolved definition. Hooks may
// override individual steps.
type GenericAdapter struct {
	RT    def.ResolvedTarget
	Inst  Installed
	Paths home.Paths
	hooks Hooks
}

// New builds the adapter for a resolved definition, attaching registered hooks.
func New(rt def.ResolvedTarget, inst Installed, paths home.Paths) (*GenericAdapter, error) {
	g := &GenericAdapter{RT: rt, Inst: inst, Paths: paths}
	if rt.Def.Adapter != Generic {
		h, ok := hooksRegistry[rt.Def.Adapter]
		if !ok {
			return nil, fmt.Errorf("engine %s: no adapter registered as %q", rt.Def.Name, rt.Def.Adapter)
		}
		g.hooks = h
	}
	return g, nil
}

func (g *GenericAdapter) Name() string          { return g.RT.Def.Name }
func (g *GenericAdapter) Formats() []string     { return g.RT.Def.Formats }
func (g *GenericAdapter) Caps() map[string]bool { return g.RT.Caps }

// BaseVars are the template variables known before launch.
func (g *GenericAdapter) BaseVars(m Model, run RunEnv) map[string]string {
	v := map[string]string{
		"bin":         g.Inst.Bin,
		"venv":        g.Inst.Venv,
		"version":     g.RT.Def.Version,
		"kepler_home": g.Paths.Root,
		"run_id":      run.RunID,
		"ctx":         strconv.Itoa(run.Ctx),
		"model_path":  m.Path,
	}
	if m.MMProj != "" {
		v["mmproj_path"] = m.MMProj
	}
	return v
}

func (g *GenericAdapter) Materialize(ctx context.Context, m Model, run RunEnv) (EngineModel, error) {
	if h, ok := g.hooks.(MaterializeHook); ok {
		return h.Materialize(ctx, g, m, run)
	}
	return EngineModel{Model: m, Vars: g.BaseVars(m, run)}, nil
}

func (g *GenericAdapter) Launch(ctx context.Context, em EngineModel, run RunEnv) (*Running, error) {
	if h, ok := g.hooks.(LaunchHook); ok {
		return h.Launch(ctx, g, em, run)
	}
	return g.LaunchGeneric(ctx, em, run)
}

// LaunchGeneric is the default launch; hooks may wrap it.
func (g *GenericAdapter) LaunchGeneric(ctx context.Context, em EngineModel, run RunEnv) (*Running, error) {
	port, err := proc.FreePort()
	if err != nil {
		return nil, err
	}
	vars := map[string]string{}
	for k, v := range em.Vars {
		vars[k] = v
	}
	vars["port"] = strconv.Itoa(port)

	l := g.RT.Launch
	cmd, err := def.Expand(l.Cmd, vars)
	if err != nil {
		return nil, err
	}
	args, err := def.ExpandAll(l.Args, vars)
	if err != nil {
		return nil, err
	}
	conds := make([]string, 0, len(l.ArgsIf))
	for c := range l.ArgsIf {
		conds = append(conds, c)
	}
	sort.Strings(conds)
	for _, c := range conds {
		if vars[c+"_path"] == "" {
			continue
		}
		extra, err := def.ExpandAll(l.ArgsIf[c], vars)
		if err != nil {
			return nil, err
		}
		args = append(args, extra...)
	}
	env := map[string]string{}
	for k, v := range l.Env {
		if env[k], err = def.Expand(v, vars); err != nil {
			return nil, err
		}
	}

	if err := os.MkdirAll(run.LogDir, 0o755); err != nil {
		return nil, err
	}
	id, err := run.Ledger.Record(ctx, ledger.Artifact{RunID: run.RunID, Kind: ledger.KindProcessGroup, Note: g.Name() + " server"})
	if err != nil {
		return nil, err
	}
	p, err := proc.Start(proc.Spec{Path: cmd, Args: args, Env: env, LogPath: filepath.Join(run.LogDir, g.Name()+".log")})
	if err != nil {
		_ = run.Ledger.Dispose(context.WithoutCancel(ctx), id)
		return nil, err
	}
	_ = run.Ledger.SetPGID(ctx, id, p.PGID)
	r := &Running{Proc: p, Port: port, BaseURL: "http://127.0.0.1:" + strconv.Itoa(port), LedgerID: id, Vars: vars}
	run.Bus.Publish(events.Event{Type: events.EngineLaunched, Engine: g.Name(), Message: fmt.Sprintf("pid %d port %d", p.PGID, port)})
	return r, nil
}

func (g *GenericAdapter) WaitReady(ctx context.Context, r *Running) error {
	ready := g.RT.Launch.Ready
	timeout := time.Duration(ready.Timeout)
	if timeout == 0 {
		timeout = 300 * time.Second
	}
	expect := ready.Expect
	if expect == 0 {
		expect = http.StatusOK
	}
	ctx, cancel := context.WithTimeout(ctx, timeout)
	defer cancel()
	client := &http.Client{Timeout: 2 * time.Second}
	url := r.BaseURL + ready.HTTPGet
	tick := time.NewTicker(250 * time.Millisecond)
	defer tick.Stop()
	for {
		if exited, err := r.Proc.Exited(); exited {
			return fmt.Errorf("%s exited before becoming ready (%v); last log lines:\n%s", g.Name(), err, r.Proc.LogTail(15))
		}
		req, _ := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
		if resp, err := client.Do(req); err == nil {
			resp.Body.Close()
			if resp.StatusCode == expect {
				return nil
			}
		}
		select {
		case <-ctx.Done():
			return fmt.Errorf("%s not ready after %s (GET %s); last log lines:\n%s", g.Name(), timeout, ready.HTTPGet, r.Proc.LogTail(15))
		case <-tick.C:
		}
	}
}

func (g *GenericAdapter) PostReady(ctx context.Context, r *Running, em EngineModel) (EngineModel, error) {
	if h, ok := g.hooks.(PostReadyHook); ok {
		return h.PostReady(ctx, g, r, em)
	}
	return em, nil
}

func (g *GenericAdapter) Endpoint(r *Running, em EngineModel) Endpoint {
	if h, ok := g.hooks.(EndpointHook); ok {
		return h.Endpoint(g, r, em)
	}
	model := g.RT.Def.Request.Model
	if expanded, err := def.Expand(model, r.Vars); err == nil {
		model = expanded
	}
	if model == "" {
		model = "kepler"
	}
	return Endpoint{BaseURL: r.BaseURL + "/v1", Model: model, ExtraBody: g.RT.Def.Request.ExtraBody}
}

// StopGrace is how long engines get to exit after SIGTERM.
var StopGrace = 10 * time.Second

func (g *GenericAdapter) Stop(r *Running) error { return r.Proc.Stop(StopGrace) }

// StopAndRelease stops r and marks its ledger row deleted.
func StopAndRelease(ctx context.Context, a Adapter, r *Running, run RunEnv) error {
	err := a.Stop(r)
	for _, id := range append([]int64{r.LedgerID}, r.Owned...) {
		if id != 0 {
			_ = run.Ledger.Dispose(context.WithoutCancel(ctx), id)
		}
	}
	run.Bus.Publish(events.Event{Type: events.EngineStopped, Engine: a.Name()})
	return err
}
