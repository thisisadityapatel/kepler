// Package ui renders terminal output. It degrades to plain lines when stdout is not
// a terminal.
package ui

import (
	"bufio"
	"fmt"
	"io"
	"os"
	"strings"
	"sync"
	"text/tabwriter"
	"time"

	"github.com/thisisadityapatel/kepler/internal/events"
)

// UI writes to Out and reads confirmations from In.
type UI struct {
	Out     io.Writer
	In      io.Reader
	TTY     bool
	Verbose bool

	mu           sync.Mutex
	progressLine bool
	keepProgress bool // finish the in-place line with a newline instead of erasing it
	lastProgress time.Time
	lastIter     string
}

// New returns a UI on stdout/stdin.
func New(verbose bool) *UI {
	return &UI{Out: os.Stdout, In: os.Stdin, TTY: isTerminal(os.Stdout) && isTerminal(os.Stdin), Verbose: verbose}
}

func isTerminal(f *os.File) bool {
	st, err := f.Stat()
	return err == nil && st.Mode()&os.ModeCharDevice != 0
}

// Printf writes a line, clearing any in-place progress line first.
func (u *UI) Printf(format string, a ...any) {
	u.mu.Lock()
	defer u.mu.Unlock()
	u.clearProgress()
	fmt.Fprintf(u.Out, format, a...)
}

func (u *UI) clearProgress() {
	if u.progressLine {
		if u.keepProgress {
			fmt.Fprintln(u.Out)
		} else {
			fmt.Fprint(u.Out, "\r\033[K")
		}
		u.progressLine, u.keepProgress = false, false
	}
}

// Interactive reports whether prompts can be answered.
func (u *UI) Interactive() bool { return u.TTY }

// Confirm asks a yes/no question. def is returned on empty input.
func (u *UI) Confirm(question string, def bool) bool {
	hint := "[y/N]"
	if def {
		hint = "[Y/n]"
	}
	u.Printf("%s %s ", question, hint)
	line, _ := bufio.NewReader(u.In).ReadString('\n')
	switch strings.ToLower(strings.TrimSpace(line)) {
	case "y", "yes":
		return true
	case "n", "no":
		return false
	}
	return def
}

// Table renders rows with aligned columns.
func (u *UI) Table(header []string, rows [][]string) {
	u.mu.Lock()
	defer u.mu.Unlock()
	u.clearProgress()
	tw := tabwriter.NewWriter(u.Out, 0, 2, 2, ' ', 0)
	if header != nil {
		fmt.Fprintln(tw, "  "+strings.Join(header, "\t"))
	}
	for _, r := range rows {
		fmt.Fprintln(tw, "  "+strings.Join(r, "\t"))
	}
	tw.Flush()
}

// HighlightTable renders rows, bolding cells where best[row][col] is true (TTY only;
// plain output marks them with a trailing *).
func (u *UI) HighlightTable(header []string, rows [][]string, best [][]bool) {
	styled := make([][]string, len(rows))
	for i, r := range rows {
		styled[i] = make([]string, len(r))
		for j, c := range r {
			if i < len(best) && j < len(best[i]) && best[i][j] {
				if u.TTY {
					// Pad inside the escape so tabwriter alignment is unaffected.
					c = "\033[1m" + c + "\033[0m"
				} else {
					c += " *"
				}
			}
			styled[i][j] = c
		}
	}
	if u.TTY {
		// tabwriter counts escape bytes; neutralize by padding every non-bold cell equally.
		for i := range styled {
			for j := range styled[i] {
				if !(i < len(best) && j < len(best[i]) && best[i][j]) {
					styled[i][j] = "\033[0m" + styled[i][j] + "\033[0m"
				}
			}
		}
		for j := range header {
			header[j] = "\033[0m" + header[j] + "\033[0m"
		}
	}
	u.Table(header, styled)
}

// OnEvent renders shared-core events. Subscribe it to the bus.
func (u *UI) OnEvent(e events.Event) {
	switch e.Type {
	case events.InstallStarted:
		u.Printf("\n%s\n", e.Engine)
	case events.InstallStep:
		u.Printf("  · %s\n", e.Message)
	case events.DownloadProgress:
		u.progress(e)
	case events.EngineLaunched:
		if u.Verbose {
			u.Printf("  · %s launched (%s)\n", e.Engine, e.Message)
		}
	case events.EngineReady:
		u.Printf("  · %s server ready, running health check\n", e.Engine)
	case events.HealthFinished:
		if e.Err != nil {
			u.Printf("  ✗ health check failed\n")
		}
	case events.InstallFinished:
		if e.Err != nil {
			u.Printf("  ✗ %s could not be installed:\n%s\n", e.Engine, indent(e.Err.Error(), "      "))
		} else {
			u.Printf("  ✓ %s\n", e.Message)
		}
	case events.Warning:
		u.Printf("  ! %s\n", e.Message)
	case events.BenchPhase:
		u.Printf("  · %s\n", e.Message)
	case events.BenchIteration:
		u.iteration(e)
	case events.BenchEngineDone:
		if e.Err != nil {
			u.Printf("  ✗ %s\n", e.Err)
		} else {
			u.Printf("  ✓ %s\n", e.Message)
		}
	}
}

func (u *UI) iteration(e events.Event) {
	u.mu.Lock()
	defer u.mu.Unlock()
	if !u.TTY {
		if e.Done == e.Total {
			fmt.Fprintf(u.Out, "  · %s: %d iterations\n", e.Message, e.Total)
		}
		return
	}
	if u.progressLine && e.Message != u.lastIter {
		u.clearProgress() // keeps the finished scenario's line
	}
	u.lastIter = e.Message
	fmt.Fprintf(u.Out, "\r\033[K  · %s %d/%d", e.Message, e.Done, e.Total)
	u.progressLine, u.keepProgress = true, true
}

func (u *UI) progress(e events.Event) {
	u.mu.Lock()
	defer u.mu.Unlock()
	done := e.Total > 0 && e.Done >= e.Total
	if !u.TTY {
		if done {
			fmt.Fprintf(u.Out, "  · downloaded %s\n", Bytes(e.Done))
		}
		return
	}
	if !done && time.Since(u.lastProgress) < 100*time.Millisecond {
		return
	}
	u.lastProgress = time.Now()
	label := e.Message
	if label == "" {
		label = "download"
	}
	line := fmt.Sprintf("  · %s %s", label, Bytes(e.Done))
	if e.Total > 0 {
		line += fmt.Sprintf(" / %s (%d%%)", Bytes(e.Total), e.Done*100/e.Total)
	}
	fmt.Fprint(u.Out, "\r\033[K"+line)
	u.progressLine, u.keepProgress = true, false
	if done {
		fmt.Fprintln(u.Out)
		u.progressLine = false
	}
}

func indent(s, prefix string) string {
	return prefix + strings.ReplaceAll(strings.TrimRight(s, "\n"), "\n", "\n"+prefix)
}

// Bytes formats a byte count for humans.
func Bytes(n int64) string {
	switch {
	case n >= 1<<30:
		return fmt.Sprintf("%.1f GB", float64(n)/(1<<30))
	case n >= 1<<20:
		return fmt.Sprintf("%.0f MB", float64(n)/(1<<20))
	case n >= 1<<10:
		return fmt.Sprintf("%.0f KB", float64(n)/(1<<10))
	}
	return fmt.Sprintf("%d B", n)
}

// Truncate shortens s to n runes with an ellipsis.
func Truncate(s string, n int) string {
	r := []rune(s)
	if len(r) <= n {
		return s
	}
	return string(r[:n-1]) + "…"
}
