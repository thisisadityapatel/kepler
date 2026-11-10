package cli

import (
	"fmt"

	"github.com/spf13/cobra"

	"github.com/thisisadityapatel/kepler/internal/engines/resolve"
)

func newEnginesCmd(g *globalFlags) *cobra.Command {
	cmd := &cobra.Command{Use: "engines", Short: "List, install, or remove inference engines"}

	list := &cobra.Command{
		Use:   "list",
		Short: "Show every engine, whether this machine can run it, and why",
		Args:  cobra.NoArgs,
		RunE: func(cmd *cobra.Command, _ []string) error {
			a, err := newApp(cmd.Context(), g, appOpts{})
			if err != nil {
				return err
			}
			defer a.close()
			a.ui.Printf("%s\n\n", hostLine(a.setup.Facts))
			printEngineTable(a, nil)
			return nil
		},
	}

	var force bool
	inst := &cobra.Command{
		Use:   "install <engine>",
		Short: "Install (or reinstall) one engine",
		Args:  cobra.ExactArgs(1),
		RunE: func(cmd *cobra.Command, args []string) error {
			a, err := newApp(cmd.Context(), g, appOpts{lock: true})
			if err != nil {
				return err
			}
			defer a.close()
			if err := checkEngineNames(a.setup.Defs, args); err != nil {
				return withCode(ExitUsage, err)
			}
			d := a.setup.Resolve(&resolve.Query{Allow: args})
			var dec resolve.Decision
			for _, x := range d {
				if x.Engine == args[0] {
					dec = x
				}
			}
			if dec.Status != resolve.Eligible {
				return withCode(ExitNoEngine, fmt.Errorf("%s cannot run on this machine: %s", dec.Engine, dec.Reason))
			}
			if dec.Install == resolve.InstallReady && !force {
				a.ui.Printf("%s is already ready (%s). Use --force to reinstall.\n", dec.Engine, a.setup.Registry.Engines[dec.Engine].Version)
				return nil
			}
			return a.setup.Install(cmd.Context(), dec)
		},
	}
	inst.Flags().BoolVar(&force, "force", false, "reinstall even if ready")

	remove := &cobra.Command{
		Use:   "remove <engine>",
		Short: "Remove an engine Kepler installed (adopted system installs are left alone)",
		Args:  cobra.ExactArgs(1),
		RunE: func(cmd *cobra.Command, args []string) error {
			a, err := newApp(cmd.Context(), g, appOpts{lock: true})
			if err != nil {
				return err
			}
			defer a.close()
			if err := a.setup.Remove(args[0]); err != nil {
				return err
			}
			a.ui.Printf("Removed %s.\n", args[0])
			return nil
		},
	}

	cmd.AddCommand(list, inst, remove)
	return cmd
}

// printEngineTable prints the resolver decision for every engine with registry state.
func printEngineTable(a *app, q *resolve.Query) {
	var rows [][]string
	for _, d := range a.setup.Resolve(q) {
		rows = append(rows, decisionRow(d, a.setup.Registry.Engines[d.Engine]))
	}
	a.ui.Table([]string{"ENGINE", "STATUS", "VERSION", "TARGET", "DETAIL"}, rows)
}
