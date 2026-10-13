// Package ollama holds the Ollama-specific adapter steps. Ollama runs as a private
// `ollama serve` with its own port and model store inside the run's temp dir, and a
// GGUF is imported with `ollama create` once the server is up.
package ollama

import (
	"context"
	"fmt"
	"os"
	"path/filepath"

	"github.com/thisisadityapatel/kepler/internal/engines/adapter"
	"github.com/thisisadityapatel/kepler/internal/engines/def"
	"github.com/thisisadityapatel/kepler/internal/ledger"
	"github.com/thisisadityapatel/kepler/internal/proc"
)

// Name is the `adapter:` value that selects these hooks.
const Name = "ollama"

// Hooks implements adapter.LaunchHook and adapter.PostReadyHook.
type Hooks struct{}

// Register makes the hooks available to definitions.
func Register() { adapter.RegisterHooks(Name, Hooks{}) }

// Launch creates the private model store (ledgered, so it is always deleted) and then
// launches generically.
func (Hooks) Launch(ctx context.Context, g *adapter.GenericAdapter, em adapter.EngineModel, run adapter.RunEnv) (*adapter.Running, error) {
	store := filepath.Join(g.Paths.Ollama, run.RunID)
	id, err := run.Ledger.Record(ctx, ledger.Artifact{RunID: run.RunID, Kind: ledger.KindDir, Path: store, Note: "ollama model store"})
	if err != nil {
		return nil, err
	}
	if err := os.MkdirAll(store, 0o755); err != nil {
		return nil, err
	}
	r, err := g.LaunchGeneric(ctx, em, run)
	if err != nil {
		_ = run.Ledger.Dispose(context.WithoutCancel(ctx), id)
		return nil, err
	}
	r.Owned = append(r.Owned, id) // deleted after the server stops
	return r, nil
}

// PostReady writes the Modelfile and runs `ollama create` against the private server.
func (Hooks) PostReady(ctx context.Context, g *adapter.GenericAdapter, r *adapter.Running, em adapter.EngineModel) (adapter.EngineModel, error) {
	m := g.RT.Def.Materialize
	if m == nil {
		return em, fmt.Errorf("ollama definition has no materialize block")
	}
	vars := map[string]string{}
	for k, v := range r.Vars {
		vars[k] = v
	}
	modelfile := filepath.Join(filepath.Dir(r.Proc.LogPath), "Modelfile."+vars["run_id"])
	vars["modelfile_path"] = modelfile
	content, err := def.Expand(m.Modelfile, vars)
	if err != nil {
		return em, err
	}
	if err := os.WriteFile(modelfile, []byte(content), 0o644); err != nil {
		return em, err
	}
	defer os.Remove(modelfile)
	args, err := def.ExpandAll(m.Cmd, vars)
	if err != nil {
		return em, err
	}
	err = proc.Run(ctx, proc.Spec{
		Path:    g.Inst.Bin,
		Args:    args,
		Env:     map[string]string{"OLLAMA_HOST": fmt.Sprintf("127.0.0.1:%d", r.Port)},
		LogPath: r.Proc.LogPath,
	})
	if err != nil {
		return em, fmt.Errorf("ollama create: %w; last log lines:\n%s", err, r.Proc.LogTail(10))
	}
	return em, nil
}
