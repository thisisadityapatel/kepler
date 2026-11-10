package cli

import (
	"errors"
	"fmt"
	"io/fs"
	"path/filepath"
	"runtime"

	"github.com/spf13/cobra"

	"github.com/thisisadityapatel/kepler/internal/ui"
)

func newCleanCmd(g *globalFlags) *cobra.Command {
	var all bool
	cmd := &cobra.Command{
		Use:   "clean",
		Short: "Delete leftovers from interrupted runs (--all: also kept models, canaries, logs)",
		Args:  cobra.NoArgs,
		RunE: func(cmd *cobra.Command, _ []string) error {
			a, err := newApp(cmd.Context(), g, appOpts{lock: true})
			if err != nil {
				return err
			}
			defer a.close()
			before := dirSize(a.paths.Root)
			rep, err := a.setup.Ledger.Reap(cmd.Context(), all)
			if err != nil {
				return err
			}
			if all {
				for _, d := range []string{a.paths.Canary, a.paths.Logs, a.paths.Downloads, a.paths.Models, a.paths.Ollama} {
					if err := a.paths.SafeRemoveAll(d); err != nil {
						return err
					}
				}
				if err := a.paths.Ensure(); err != nil {
					return err
				}
			}
			for _, e := range rep.Errors {
				a.ui.Printf("  ! %v\n", e)
			}
			freed := before - dirSize(a.paths.Root)
			if freed < 0 {
				freed = 0
			}
			a.ui.Printf("Removed %d tracked artifact(s); freed %s.\n", len(rep.Reaped), ui.Bytes(freed))
			if len(rep.Errors) > 0 {
				return errors.New("some artifacts could not be removed")
			}
			return nil
		},
	}
	cmd.Flags().BoolVar(&all, "all", false, "also remove kept models, health-check models, logs, and partial downloads")
	return cmd
}

func newVersionCmd() *cobra.Command {
	return &cobra.Command{
		Use:   "version",
		Short: "Print the Kepler version",
		Args:  cobra.NoArgs,
		Run: func(cmd *cobra.Command, _ []string) {
			fmt.Printf("kepler %s (%s, %s/%s)\n", Version, runtime.Version(), runtime.GOOS, runtime.GOARCH)
		},
	}
}

func dirSize(root string) int64 {
	var n int64
	_ = filepath.WalkDir(root, func(_ string, d fs.DirEntry, err error) error {
		if err == nil && d.Type().IsRegular() {
			if info, err := d.Info(); err == nil {
				n += info.Size()
			}
		}
		return nil
	})
	return n
}
