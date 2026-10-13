// Package install implements engine install methods. Each method kind is an Installer
// registered by name; engine definitions select kinds per target. Adding a kind (e.g.
// docker, winget) means adding an Installer — existing definitions are unaffected.
package install

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"sort"

	"github.com/thisisadityapatel/kepler/internal/download"
	"github.com/thisisadityapatel/kepler/internal/engines/def"
	"github.com/thisisadityapatel/kepler/internal/events"
	"github.com/thisisadityapatel/kepler/internal/home"
	"github.com/thisisadityapatel/kepler/internal/ledger"
	"github.com/thisisadityapatel/kepler/internal/probe"
)

// Env is what an installer may use.
type Env struct {
	Paths      home.Paths
	Ledger     *ledger.Ledger
	Downloader *download.Client
	Bus        *events.Bus
	Facts      probe.HostFacts
	Engine     string
	Version    string
	RunID      string
	LogPath    string // build/install command output is appended here
}

// Vars are the template variables available to install fields.
func (e Env) Vars() map[string]string {
	return map[string]string{"version": e.Version, "kepler_home": e.Paths.Root}
}

// Dir is the Kepler-owned install directory for this engine version.
func (e Env) Dir() string { return filepath.Join(e.Paths.Engines, e.Engine, e.Version) }

// Result describes a completed install.
type Result struct {
	Kind     string
	Bin      string // absolute path of the server binary
	Dir      string // Kepler-owned install root ("" for adopted installs)
	Venv     string
	Version  string
	LedgerID int64 // engine_install ledger row; 0 for adopted installs
}

// Installer is one install method kind.
type Installer interface {
	Kind() string
	Validate(m def.InstallMethod) error
	Install(ctx context.Context, env Env, m def.InstallMethod) (Result, error)
}

var installers = map[string]Installer{}

func register(i Installer) { installers[i.Kind()] = i }

func init() {
	register(prebuilt{})
	register(source{})
	register(uvVenv{})
	register(adopt{})
}

// Get returns the installer for kind.
func Get(kind string) (Installer, bool) {
	i, ok := installers[kind]
	return i, ok
}

// Kinds returns kind → validator, for engine definition validation.
func Kinds() map[string]func(def.InstallMethod) error {
	out := map[string]func(def.InstallMethod) error{}
	for k, i := range installers {
		out[k] = i.Validate
	}
	return out
}

// KindNames returns registered kinds, sorted.
func KindNames() []string {
	var out []string
	for k := range installers {
		out = append(out, k)
	}
	sort.Strings(out)
	return out
}

// claimDir records dir in the ledger (write-ahead) and recreates it empty. The row stays
// temp until the caller releases it after a passing health check, so an interrupted
// install is rolled back by the reaper.
func claimDir(ctx context.Context, env Env, dir, note string) (int64, error) {
	id, err := env.Ledger.Record(ctx, ledger.Artifact{RunID: env.RunID, Kind: ledger.KindEngineInstall, Path: dir, Note: note})
	if err != nil {
		return 0, err
	}
	if err := env.Paths.SafeRemoveAll(dir); err != nil {
		return id, err
	}
	return id, os.MkdirAll(dir, 0o755)
}

// scratch returns a temp working directory for this run, recorded in the ledger.
func scratch(ctx context.Context, env Env, name string) (string, int64, error) {
	dir := filepath.Join(env.Paths.Tmp, "install", env.RunID+"-"+name)
	id, err := env.Ledger.Record(ctx, ledger.Artifact{RunID: env.RunID, Kind: ledger.KindDir, Path: dir, Note: "install scratch"})
	if err != nil {
		return "", 0, err
	}
	return dir, id, os.MkdirAll(dir, 0o755)
}

func step(env Env, format string, a ...any) {
	env.Bus.Publish(events.Event{Type: events.InstallStep, Engine: env.Engine, Message: fmt.Sprintf(format, a...)})
}

// checkExecutable verifies bin exists and is executable.
func checkExecutable(bin string) error {
	st, err := os.Stat(bin)
	if err != nil {
		return fmt.Errorf("expected binary not found: %s", bin)
	}
	if st.IsDir() || st.Mode()&0o111 == 0 {
		return fmt.Errorf("not an executable file: %s", bin)
	}
	return nil
}
