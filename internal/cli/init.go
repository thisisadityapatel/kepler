package cli

import (
	"errors"
	"fmt"
	"strings"

	"github.com/spf13/cobra"

	"github.com/thisisadityapatel/kepler/internal/engines/def"
	"github.com/thisisadityapatel/kepler/internal/engines/resolve"
	"github.com/thisisadityapatel/kepler/internal/ui"
)

func newInitCmd(g *globalFlags) *cobra.Command {
	var yes bool
	var only, skip []string
	cmd := &cobra.Command{
		Use:   "init",
		Short: "Detect hardware and install every engine this machine can run",
		Long: `Probes the machine, matches every engine definition against it, and installs the
eligible engines (default tier automatically, optional tier on request). Each install
is verified by running a tiny model. Safe to re-run: ready engines are skipped.`,
		Args: cobra.NoArgs,
		RunE: func(cmd *cobra.Command, _ []string) error {
			a, err := newApp(cmd.Context(), g, appOpts{lock: true})
			if err != nil {
				return err
			}
			defer a.close()
			return runInit(cmd, a, yes, only, skip)
		},
	}
	cmd.Flags().BoolVarP(&yes, "yes", "y", false, "install without prompting")
	cmd.Flags().StringSliceVar(&only, "engines", nil, "only consider these engines (comma-separated)")
	cmd.Flags().StringSliceVar(&skip, "skip", nil, "skip these engines (comma-separated)")
	return cmd
}

func runInit(cmd *cobra.Command, a *app, yes bool, only, skip []string) error {
	s, u := a.setup, a.ui
	if err := checkEngineNames(s.Defs, append(append([]string{}, only...), skip...)); err != nil {
		return withCode(ExitUsage, err)
	}
	u.Printf("Kepler %s · %s\n", Version, hostLine(s.Facts))
	for _, w := range hostWarnings(s.Facts) {
		u.Printf("  ! %s\n", w)
	}

	decisions := s.Resolve(&resolve.Query{Allow: only})
	s.RecordDecisions(decisions)

	var plan []resolve.Decision
	var rows [][]string
	for _, d := range decisions {
		if d.Status == resolve.Excluded {
			continue
		}
		action := ""
		switch {
		case d.Status != resolve.Eligible:
			action = d.Reason
		case contains(skip, d.Engine):
			action = "skipped (--skip)"
			s.MarkSkipped(d)
		case d.Install == resolve.InstallReady:
			action = "ready (" + s.Registry.Engines[d.Engine].Version + ")"
		case d.Def.Tier == def.TierLazy && !contains(only, d.Engine):
			action = "available on demand: kepler engines install " + d.Engine
		case d.Def.Tier == def.TierOptional && !contains(only, d.Engine):
			if yes || !u.TTY {
				action = "optional; add with --engines " + d.Engine
				break
			}
			if u.Confirm(fmt.Sprintf("Also install optional engine %s?", d.Def.Display), false) {
				action = "install: " + methodChain(d.Target)
				plan = append(plan, d)
			} else {
				action = "skipped"
				s.MarkSkipped(d)
			}
		default:
			verb := "install"
			if d.Install == resolve.InstallOutdated {
				verb = "upgrade"
			} else if d.Install == resolve.InstallUnhealthy {
				verb = "reinstall"
			}
			action = verb + ": " + methodChain(d.Target)
			plan = append(plan, d)
		}
		rows = append(rows, []string{d.Engine, d.Def.Display, orDash(d.Def.Version), ui.Truncate(action, 80)})
	}
	u.Printf("\nEngines\n")
	u.Table(nil, rows)

	if len(plan) == 0 {
		if err := s.SaveRegistry(); err != nil {
			return err
		}
		if anyReady(s) {
			u.Printf("\nNothing to install; all eligible engines are ready.\n")
			return nil
		}
		return withCode(ExitNoEngine, errors.New("no engine can be installed on this machine"))
	}

	formats := map[string]bool{}
	for _, d := range plan {
		formats[d.Def.Formats[0]] = true
	}
	var canaries []string
	for f := range formats {
		canaries = append(canaries, fmt.Sprintf("%s %s", f, ui.Bytes(s.Canaries[f].SizeBytes())))
	}
	u.Printf("\nHealth-check models (kept in %s): %s\n", s.Paths.Canary, strings.Join(canaries, ", "))

	if !yes {
		if !u.TTY {
			return withCode(ExitUsage, errors.New("not a terminal; re-run with --yes to install"))
		}
		if !u.Confirm(fmt.Sprintf("\nInstall %d engine(s)?", len(plan)), true) {
			return s.SaveRegistry()
		}
	}

	var failed []string
	for _, d := range plan {
		if err := s.Install(cmd.Context(), d); err != nil {
			if cmd.Context().Err() != nil {
				return cmd.Context().Err()
			}
			failed = append(failed, d.Engine)
		}
	}
	if err := s.SaveRegistry(); err != nil {
		return err
	}

	u.Printf("\n")
	printEngineTable(a, nil)
	switch {
	case len(failed) == len(plan):
		return withCode(ExitError, fmt.Errorf("no engine installed successfully (logs in %s)", s.Paths.Logs))
	case len(failed) > 0:
		return withCode(ExitPartial, fmt.Errorf("%s failed to install; the other engines are ready", strings.Join(failed, ", ")))
	}
	u.Printf("\nReady. Benchmark a model with: kepler bench <huggingface-repo>\n")
	return nil
}

func anyReady(s interface {
	Resolve(*resolve.Query) []resolve.Decision
}) bool {
	for _, d := range s.Resolve(nil) {
		if d.Status == resolve.Eligible && d.Install == resolve.InstallReady {
			return true
		}
	}
	return false
}

func checkEngineNames(set def.Set, names []string) error {
	for _, n := range names {
		if _, ok := set.Defs[n]; !ok {
			var known []string
			for _, d := range set.Sorted() {
				known = append(known, d.Name)
			}
			return fmt.Errorf("unknown engine %q (known: %s)", n, strings.Join(known, ", "))
		}
	}
	return nil
}

func contains(ss []string, v string) bool {
	for _, s := range ss {
		if s == v {
			return true
		}
	}
	return false
}
