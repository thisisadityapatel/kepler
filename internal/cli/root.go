// Package cli is the composition root: it wires shared-core packages together and
// defines the cobra commands.
package cli

import (
	"context"
	"errors"
	"fmt"
	"os"
	"os/signal"
	"sync"
	"syscall"

	"github.com/spf13/cobra"

	"github.com/thisisadityapatel/kepler/definitions"
	"github.com/thisisadityapatel/kepler/internal/download"
	"github.com/thisisadityapatel/kepler/internal/engines/adapter"
	"github.com/thisisadityapatel/kepler/internal/engines/def"
	"github.com/thisisadityapatel/kepler/internal/engines/hooks/ollama"
	"github.com/thisisadityapatel/kepler/internal/events"
	"github.com/thisisadityapatel/kepler/internal/home"
	"github.com/thisisadityapatel/kepler/internal/install"
	"github.com/thisisadityapatel/kepler/internal/ledger"
	"github.com/thisisadityapatel/kepler/internal/probe"
	"github.com/thisisadityapatel/kepler/internal/registry"
	"github.com/thisisadityapatel/kepler/internal/setup"
	"github.com/thisisadityapatel/kepler/internal/ui"
)

// Version is set at build time with -ldflags "-X .../internal/cli.Version=...".
var Version = "2.0.0-dev"

// Exit codes (DESIGN.md §3).
const (
	ExitOK          = 0
	ExitError       = 1
	ExitUsage       = 2
	ExitPreflight   = 3
	ExitNoEngine    = 4
	ExitPartial     = 5
	ExitInterrupted = 130
)

type exitError struct {
	code int
	err  error
}

func (e exitError) Error() string { return e.err.Error() }
func (e exitError) Unwrap() error { return e.err }

func withCode(code int, err error) error { return exitError{code: code, err: err} }

type globalFlags struct {
	home    string
	verbose bool
}

// Execute runs the CLI and returns the process exit code.
func Execute() int { return run(os.Args[1:]) }

var registerOnce sync.Once

func run(args []string) int {
	registerOnce.Do(ollama.Register)

	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()

	g := &globalFlags{}
	root := &cobra.Command{
		Use:               "kepler",
		Short:             "Benchmark LLMs across every inference engine your machine can run",
		SilenceUsage:      true,
		SilenceErrors:     true,
		CompletionOptions: cobra.CompletionOptions{DisableDefaultCmd: true},
	}
	root.PersistentFlags().StringVar(&g.home, "home", "", "Kepler home directory (env KEPLER_HOME, default ~/.kepler)")
	root.PersistentFlags().BoolVarP(&g.verbose, "verbose", "v", false, "verbose output")
	root.AddCommand(newInitCmd(g), newDoctorCmd(g), newEnginesCmd(g), newCleanCmd(g), newBenchCmd(g), newResultsCmd(g), newVersionCmd())

	root.SetArgs(args)
	err := root.ExecuteContext(ctx)
	if err == nil {
		return ExitOK
	}
	if ctx.Err() != nil {
		fmt.Fprintln(os.Stderr, "\ninterrupted; temporary files were cleaned up")
		return ExitInterrupted
	}
	fmt.Fprintln(os.Stderr, "error:", err)
	var ee exitError
	if errors.As(err, &ee) {
		return ee.code
	}
	return ExitError
}

// app is the wired-up shared core for one command invocation.
type app struct {
	paths home.Paths
	ui    *ui.UI
	setup *setup.Setup
	lock  *home.Lock
}

type appOpts struct {
	lock bool // take the global lock (any command that mutates engines or runs benches)
}

func newApp(ctx context.Context, g *globalFlags, opt appOpts) (*app, error) {
	root, err := home.Resolve(g.home)
	if err != nil {
		return nil, err
	}
	paths := home.New(root)
	if err := paths.Ensure(); err != nil {
		return nil, err
	}
	a := &app{paths: paths, ui: ui.New(g.verbose)}
	if opt.lock {
		if a.lock, err = paths.TryLock(); err != nil {
			return nil, err
		}
	}

	led, err := ledger.Open(paths)
	if err != nil {
		a.close()
		return nil, err
	}
	// Every command first cleans up after any run that crashed.
	if rep, err := led.Reap(ctx, false); err == nil && len(rep.Reaped) > 0 {
		a.ui.Printf("Cleaned up %d leftover artifact(s) from an interrupted run.\n", len(rep.Reaped))
	}

	canaries, err := def.LoadCanaries(definitions.Root(), definitions.CanariesFile)
	if err != nil {
		led.Close()
		a.close()
		return nil, err
	}
	defs, err := def.Load(definitions.Engines(), paths.EnginesD, def.ValidateOptions{
		InstallKinds:  install.Kinds(),
		Adapters:      adapter.Names(),
		CanaryFormats: canaries.Formats(),
	})
	if err != nil {
		led.Close()
		a.close()
		return nil, fmt.Errorf("engine definitions:\n%w", err)
	}
	facts, err := probe.Probe(ctx, probe.Options{DiskPath: paths.Root})
	if err != nil {
		led.Close()
		a.close()
		return nil, fmt.Errorf("hardware probe: %w", err)
	}
	reg, err := registry.Load(paths.Registry)
	if err != nil {
		led.Close()
		a.close()
		return nil, err
	}
	bus := &events.Bus{}
	bus.Subscribe(a.ui.OnEvent)
	a.setup = &setup.Setup{
		Paths: paths, Defs: defs, Canaries: canaries, Facts: facts, Registry: reg,
		Ledger: led, Downloader: download.New(paths.Downloads), Bus: bus, Version: Version,
	}
	return a, nil
}

func (a *app) close() {
	if a.setup != nil {
		a.setup.Ledger.Close()
	}
	_ = a.lock.Unlock()
}
