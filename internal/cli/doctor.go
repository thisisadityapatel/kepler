package cli

import (
	"fmt"
	"sort"
	"strings"

	"github.com/spf13/cobra"

	"github.com/thisisadityapatel/kepler/internal/engines/resolve"
	"github.com/thisisadityapatel/kepler/internal/probe"
	"github.com/thisisadityapatel/kepler/internal/registry"
	"github.com/thisisadityapatel/kepler/internal/setup"
	"github.com/thisisadityapatel/kepler/internal/ui"
)

func newDoctorCmd(g *globalFlags) *cobra.Command {
	var noHealth bool
	cmd := &cobra.Command{
		Use:   "doctor",
		Short: "Show detected hardware, engine status, and re-run health checks",
		Args:  cobra.NoArgs,
		RunE: func(cmd *cobra.Command, _ []string) error {
			a, err := newApp(cmd.Context(), g, appOpts{lock: !noHealth})
			if err != nil {
				return err
			}
			defer a.close()
			return runDoctor(cmd, a, noHealth)
		},
	}
	cmd.Flags().BoolVar(&noHealth, "no-health", false, "skip health checks (read-only)")
	return cmd
}

func runDoctor(cmd *cobra.Command, a *app, noHealth bool) error {
	s, u := a.setup, a.ui
	h := s.Facts
	u.Printf("Kepler %s\n\nHost\n", Version)
	rows := [][]string{
		{"os", fmt.Sprintf("%s %s (%s)", h.OS, h.OSVersion, h.Arch)},
		{"model", orDash(h.ModelID)},
		{"cpu", fmt.Sprintf("%s · %dP + %dE cores", orDash(h.CPU.Brand), h.CPU.PerformanceCores, h.CPU.EfficiencyCores)},
		{"memory", ui.Bytes(h.RAMBytes)},
	}
	for _, acc := range h.Accelerators {
		desc := fmt.Sprintf("%s · %s usable", orDash(acc.Name), ui.Bytes(acc.MemBytes))
		if acc.Cores > 0 {
			desc += fmt.Sprintf(" · %d cores", acc.Cores)
		}
		if acc.Unified && acc.Kind != probe.AccelCPU {
			desc += " · unified memory"
		}
		rows = append(rows, []string{"accel/" + acc.Kind, desc})
	}
	rows = append(rows,
		[]string{"free disk", ui.Bytes(h.FreeDiskBytes) + " (" + s.Paths.Root + ")"},
		[]string{"power", fmt.Sprintf("%s · low power mode %v", h.Power.Source, h.Power.LowPowerMode)},
		[]string{"tools", toolsLine(h.Tools)},
	)
	u.Table(nil, rows)
	var problems []string
	warnings := hostWarnings(h)

	u.Printf("\nEngines\n")
	decisions := s.Resolve(nil)
	s.RecordDecisions(decisions)
	if !noHealth {
		for _, d := range decisions {
			e := s.Registry.Engines[d.Engine]
			if d.Status != resolve.Eligible || e == nil || e.Install == nil ||
				(e.Status != registry.StatusReady && e.Status != registry.StatusUnhealthy) {
				continue
			}
			hc := s.HealthCheck(cmd.Context(), d, e, setup.NewRunID())
			e.Health = hc
			if hc.OK {
				e.Status, e.Reason = registry.StatusReady, ""
				u.Printf("  ✓ %s healthy (load %.1fs, %.0f tok/s)\n", d.Engine, hc.LoadS, hc.DecodeTokPerS)
			} else {
				e.Status, e.Reason = registry.StatusUnhealthy, hc.Error
				problems = append(problems, fmt.Sprintf("%s failed its health check: %s (log: %s)", d.Engine, firstLine(hc.Error), hc.LogPath))
			}
			if cmd.Context().Err() != nil {
				return cmd.Context().Err()
			}
		}
		u.Printf("\n")
	}
	printEngineTable(a, nil)
	if err := s.SaveRegistry(); err != nil {
		return err
	}

	ready := 0
	for _, d := range s.Resolve(nil) {
		if d.Status == resolve.Eligible && d.Install == resolve.InstallReady {
			ready++
		}
		if d.Status == resolve.Eligible && d.Install == resolve.InstallOutdated {
			problems = append(problems, fmt.Sprintf("%s is outdated; run `kepler engines install %s`", d.Engine, d.Engine))
		}
	}
	if ready == 0 {
		problems = append(problems, "no engine is ready; run `kepler init`")
	}
	if len(warnings) > 0 {
		u.Printf("\nWarnings\n")
		for _, w := range warnings {
			u.Printf("  ! %s\n", w)
		}
	}
	if len(problems) == 0 {
		u.Printf("\nNo problems found.\n")
		return nil
	}
	u.Printf("\nProblems\n")
	for _, p := range problems {
		u.Printf("  ! %s\n", p)
	}
	return withCode(ExitError, fmt.Errorf("%d problem(s) found", len(problems)))
}

func toolsLine(tools map[string]string) string {
	var parts []string
	for name, v := range tools {
		if v != "" {
			name += " " + v
		}
		parts = append(parts, name)
	}
	sort.Strings(parts)
	if len(parts) == 0 {
		return "—"
	}
	return strings.Join(parts, ", ")
}
