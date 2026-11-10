package cli

import (
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/spf13/cobra"

	"github.com/thisisadityapatel/kepler/internal/bench"
	"github.com/thisisadityapatel/kepler/internal/bench/preflight"
	"github.com/thisisadityapatel/kepler/internal/bench/results"
	"github.com/thisisadityapatel/kepler/internal/bench/scenario"
	"github.com/thisisadityapatel/kepler/internal/hf"
	"github.com/thisisadityapatel/kepler/internal/ui"
)

func newBenchCmd(g *globalFlags) *cobra.Command {
	var opt bench.Options
	var jsonOut bool
	cmd := &cobra.Command{
		Use:   "bench <model>",
		Short: "Benchmark a Hugging Face model across every ready engine",
		Long: `Resolves a Hugging Face model (org/repo, org/repo:QUANT, or a huggingface.co link),
checks it fits in memory and on disk, downloads it, benchmarks every ready engine that
supports its format one at a time, writes one JSON result per engine to
~/.kepler/results, and deletes the model afterwards (unless --keep).

Presets:
  quick     chat + prefill-2k, 3 iterations
  standard  chat + prefill-2k + prefill-8k + depth-16k, 5 iterations
  full      all six scenarios, 10 iterations`,
		Example: `  kepler bench Qwen/Qwen2.5-0.5B-Instruct-GGUF
  kepler bench unsloth/gemma-3-4b-it-GGUF:Q8_0 --preset quick
  kepler bench https://huggingface.co/mlx-community/Qwen2.5-0.5B-Instruct-4bit`,
		Args: cobra.ExactArgs(1),
		RunE: func(cmd *cobra.Command, args []string) error {
			opt.Model = args[0]
			a, err := newApp(cmd.Context(), g, appOpts{lock: true})
			if err != nil {
				return err
			}
			defer a.close()
			if jsonOut {
				a.ui.Out = os.Stderr // stdout is reserved for the JSON summary
			}
			if err := checkEngineNames(a.setup.Defs, append(append([]string{}, opt.Engines...), opt.Skip...)); err != nil {
				return withCode(ExitUsage, err)
			}
			return runBench(cmd, a, opt, jsonOut)
		},
	}
	f := cmd.Flags()
	f.StringVar(&opt.Preset, "preset", "standard", "quick | standard | full")
	f.StringSliceVar(&opt.Engines, "engines", nil, "only run these engines (comma-separated)")
	f.StringSliceVar(&opt.Skip, "skip", nil, "skip these engines (comma-separated)")
	f.StringVar(&opt.Quant, "quant", "", "GGUF quantization to use, e.g. Q4_K_M, Q8_0 (default: Q4_K_M or the best that fits)")
	f.StringSliceVar(&opt.Scenarios, "scenarios", nil, "override the preset's scenarios: "+strings.Join(scenario.Order, ","))
	f.IntVar(&opt.Iterations, "iterations", 0, "override the preset's iteration count")
	f.DurationVar(&opt.Cooldown, "cooldown", 10*time.Second, "pause between engines for thermal recovery")
	f.BoolVar(&opt.Keep, "keep", false, "keep the downloaded model instead of deleting it")
	f.BoolVarP(&opt.Yes, "yes", "y", false, "start without confirmation")
	f.BoolVar(&jsonOut, "json", false, "print a JSON summary (result file paths) on stdout")
	return cmd
}

func runBench(cmd *cobra.Command, a *app, opt bench.Options, jsonOut bool) error {
	u := a.ui
	u.Printf("Kepler %s · %s\n", Version, hostLine(a.setup.Facts))
	out, err := bench.Run(cmd.Context(), bench.Env{Setup: a.setup, HF: hf.NewClient(""), UI: u, Version: Version}, opt)

	if out != nil && len(out.Results) > 0 {
		u.Printf("\n")
		printReport(u, out.Results)
		u.Printf("\nResults:\n")
		for _, f := range out.Files {
			u.Printf("  %s\n", f)
		}
		if out.ComparisonFile != "" {
			u.Printf("  %s\n", out.ComparisonFile)
		}
		switch {
		case out.KeptPath != "":
			u.Printf("Model kept at %s (remove with `kepler clean --all`).\n", out.KeptPath)
		case out.FreedBytes > 0:
			u.Printf("Deleted %s of model files.\n", ui.Bytes(out.FreedBytes))
		}
		if jsonOut {
			printJSONSummary(out)
		}
	}

	var fit preflight.FitError
	switch {
	case cmd.Context().Err() != nil:
		return cmd.Context().Err()
	case errors.Is(err, bench.ErrDeclined):
		u.Printf("Cancelled; nothing was downloaded.\n")
		return nil
	case errors.Is(err, bench.ErrNeedYes), errors.Is(err, bench.ErrInput):
		return withCode(ExitUsage, err)
	case errors.As(err, &fit):
		return withCode(ExitPreflight, err)
	case errors.Is(err, bench.ErrNoEngine):
		return withCode(ExitNoEngine, err)
	case errors.Is(err, hf.ErrNotFound), errors.Is(err, hf.ErrNoUsableFormat):
		return withCode(ExitUsage, err)
	case err != nil:
		return err
	}
	switch {
	case out.Failed == len(out.Results):
		return withCode(ExitError, fmt.Errorf("every engine failed; see the log paths in the result files"))
	case out.Failed > 0 || anyPartial(out.Results):
		return withCode(ExitPartial, fmt.Errorf("some engines failed or did not finish"))
	}
	return nil
}

func anyPartial(rs []*results.Result) bool {
	for _, r := range rs {
		if r.Status != results.StatusComplete {
			return true
		}
	}
	return false
}

func printReport(u *ui.UI, rs []*results.Result) {
	t := results.Report(rs)
	u.Printf("%s\n\n", t.Title)
	u.HighlightTable(t.Header, t.Rows, t.Best)
	for _, l := range t.Legend {
		u.Printf("  %s\n", l)
	}
}

func printJSONSummary(out *bench.Outcome) {
	type entry struct {
		Engine        string `json:"engine"`
		Status        string `json:"status"`
		Comparability string `json:"comparability"`
		File          string `json:"file"`
	}
	s := struct {
		Comparison string  `json:"comparison_file,omitempty"`
		Results    []entry `json:"results"`
	}{Comparison: out.ComparisonFile}
	for i, r := range out.Results {
		s.Results = append(s.Results, entry{r.Engine.Name, r.Status, r.Comparability.Status, out.Files[i]})
	}
	b, _ := json.MarshalIndent(s, "", "  ")
	fmt.Println(string(b))
}

// ---------------------------------------------------------------- results

func newResultsCmd(g *globalFlags) *cobra.Command {
	cmd := &cobra.Command{Use: "results", Short: "List and show saved benchmark results"}
	list := &cobra.Command{
		Use:   "list",
		Short: "List saved results, newest first",
		Args:  cobra.NoArgs,
		RunE: func(cmd *cobra.Command, _ []string) error {
			a, err := newApp(cmd.Context(), g, appOpts{})
			if err != nil {
				return err
			}
			defer a.close()
			files, _ := filepath.Glob(filepath.Join(a.paths.Results, "*.json"))
			var rows [][]string
			for i := len(files) - 1; i >= 0; i-- {
				base := filepath.Base(files[i])
				if strings.HasPrefix(base, "comparison_") {
					continue
				}
				r, err := results.Read(files[i])
				if err != nil {
					continue
				}
				decode := "—"
				if c := r.Scenario("chat"); c != nil && len(c.Iterations) > 0 {
					decode = fmt.Sprintf("%.1f", c.Summary.DecodeTokPerS.Median)
				}
				rows = append(rows, []string{r.Timestamp.Local().Format("2006-01-02 15:04"), ui.Truncate(r.Model.Repo, 40), r.Model.Quant,
					r.Engine.Name, r.Config.Preset, decode, r.Status, base})
			}
			if len(rows) == 0 {
				a.ui.Printf("No results yet. Run `kepler bench <model>`.\n")
				return nil
			}
			a.ui.Table([]string{"DATE", "MODEL", "QUANT", "ENGINE", "PRESET", "DECODE tok/s", "STATUS", "FILE"}, rows)
			return nil
		},
	}
	show := &cobra.Command{
		Use:   "show <file | comparison file | run id>",
		Short: "Render a saved result or comparison as a report",
		Args:  cobra.ExactArgs(1),
		RunE: func(cmd *cobra.Command, args []string) error {
			a, err := newApp(cmd.Context(), g, appOpts{})
			if err != nil {
				return err
			}
			defer a.close()
			rs, err := loadForShow(a.paths.Results, args[0])
			if err != nil {
				return err
			}
			printReport(a.ui, rs)
			for _, r := range rs {
				if len(r.Comparability.Notes) == 0 && r.Error == "" {
					continue
				}
				a.ui.Printf("\n%s (%s)\n", r.Engine.Name, r.Status)
				if r.Error != "" {
					a.ui.Printf("  error: %s\n", firstLine(r.Error))
				}
				for _, n := range r.Comparability.Notes {
					a.ui.Printf("  ~ %s\n", n)
				}
			}
			return nil
		},
	}
	cmd.AddCommand(list, show)
	return cmd
}

// loadForShow accepts a result path/filename, a comparison path/filename, or a run id.
func loadForShow(dir, arg string) ([]*results.Result, error) {
	candidates := []string{arg, filepath.Join(dir, arg), filepath.Join(dir, arg+".json")}
	for _, p := range candidates {
		if _, err := os.Stat(p); err != nil {
			continue
		}
		if strings.HasPrefix(filepath.Base(p), "comparison_") {
			c, err := results.ReadComparison(p)
			if err != nil {
				return nil, err
			}
			var rs []*results.Result
			for _, e := range c.Entries {
				r, err := results.Read(filepath.Join(filepath.Dir(p), e.ResultFile))
				if err != nil {
					return nil, err
				}
				rs = append(rs, r)
			}
			return rs, nil
		}
		r, err := results.Read(p)
		if err != nil {
			return nil, err
		}
		return []*results.Result{r}, nil
	}
	// run id: every result with that run_id
	files, _ := filepath.Glob(filepath.Join(dir, "*.json"))
	var rs []*results.Result
	for _, f := range files {
		if r, err := results.Read(f); err == nil && (r.RunID == arg || r.ComparisonID == arg) {
			rs = append(rs, r)
		}
	}
	if len(rs) == 0 {
		return nil, fmt.Errorf("no result matches %q (see `kepler results list`)", arg)
	}
	return rs, nil
}
