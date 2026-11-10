package probe

import (
	"context"
	"os"
	"os/exec"
	"regexp"
	"runtime"
	"strings"
	"time"
)

// Runner executes a command and returns its combined stdout. Swappable in tests.
type Runner func(ctx context.Context, name string, args ...string) (string, error)

// ExecRunner runs real commands with a short timeout.
func ExecRunner(ctx context.Context, name string, args ...string) (string, error) {
	ctx, cancel := context.WithTimeout(ctx, 10*time.Second)
	defer cancel()
	out, err := exec.CommandContext(ctx, name, args...).Output()
	return strings.TrimSpace(string(out)), err
}

// Options configures Probe.
type Options struct {
	Run      Runner
	DiskPath string // path whose filesystem free space is reported (KEPLER_HOME)
}

// Detector finds accelerators of one kind. Adding NVIDIA support means adding a detector.
type Detector interface {
	Kind() string
	Detect(ctx context.Context, run Runner, base HostFacts) ([]Accelerator, error)
}

// detectors are the registered accelerator detectors, in priority order. The cpu
// accelerator is appended by Probe and is not a detector.
var detectors []Detector

func registerDetector(d Detector) { detectors = append(detectors, d) }

// Probe detects the host. OS-specific facts come from probeOS (build-tagged per GOOS).
func Probe(ctx context.Context, opt Options) (HostFacts, error) {
	if opt.Run == nil {
		opt.Run = ExecRunner
	}
	h := HostFacts{
		OS:    runtime.GOOS,
		Arch:  runtime.GOARCH,
		Tools: map[string]string{},
		CPU:   CPUInfo{LogicalProcessors: runtime.NumCPU()},
		Power: PowerInfo{Source: "unknown"},
	}
	if err := probeOS(ctx, opt.Run, &h); err != nil {
		return h, err
	}
	for _, d := range detectors {
		accels, err := d.Detect(ctx, opt.Run, h)
		if err != nil {
			continue // a failing detector means "not present", never a fatal probe error
		}
		h.Accelerators = append(h.Accelerators, accels...)
	}
	h.Accelerators = append(h.Accelerators, Accelerator{
		Kind: AccelCPU, Vendor: cpuVendor(h), Name: h.CPU.Brand, MemBytes: h.RAMBytes, Unified: true,
		Cores: h.CPU.PerformanceCores + h.CPU.EfficiencyCores,
	})
	if opt.DiskPath != "" {
		h.FreeDiskBytes = freeDisk(opt.DiskPath)
	}
	probeTools(ctx, opt.Run, &h)
	return h, nil
}

func cpuVendor(h HostFacts) string {
	b := strings.ToLower(h.CPU.Brand)
	switch {
	case strings.Contains(b, "apple"):
		return "apple"
	case strings.Contains(b, "intel"):
		return "intel"
	case strings.Contains(b, "amd"):
		return "amd"
	}
	return ""
}

var versionRe = regexp.MustCompile(`\d+\.\d+(\.\d+)?`)

// ParseVersion extracts the first dotted version number from tool output.
func ParseVersion(s string) string { return versionRe.FindString(s) }

// probeTools records external tools engines may need. Absent tools are not in the map.
func probeTools(ctx context.Context, run Runner, h *HostFacts) {
	type tool struct {
		name string
		args []string
	}
	for _, t := range []tool{
		{"ollama", []string{"--version"}},
		{"cmake", []string{"--version"}},
		{"git", []string{"--version"}},
		{"uv", []string{"--version"}},
	} {
		path, err := exec.LookPath(t.name)
		if err != nil {
			continue
		}
		out, _ := runCombined(ctx, path, t.args...)
		h.Tools[t.name] = ParseVersion(lastVersionLine(t.name, out))
	}
	if h.OS == "darwin" {
		if _, err := run(ctx, "xcode-select", "-p"); err == nil {
			h.Tools["xcode_clt"] = ""
		}
	}
}

// lastVersionLine picks the line carrying the tool's own version. `ollama --version`
// prints connection warnings first and its client version last.
func lastVersionLine(name, out string) string {
	lines := strings.Split(strings.TrimSpace(out), "\n")
	if name == "ollama" {
		for i := len(lines) - 1; i >= 0; i-- {
			if strings.Contains(lines[i], "version") {
				return lines[i]
			}
		}
	}
	if len(lines) > 0 {
		return lines[0]
	}
	return ""
}

func runCombined(ctx context.Context, name string, args ...string) (string, error) {
	ctx, cancel := context.WithTimeout(ctx, 10*time.Second)
	defer cancel()
	cmd := exec.CommandContext(ctx, name, args...)
	cmd.Env = append(os.Environ(), "OLLAMA_HOST=127.0.0.1:1") // keep `ollama --version` from probing a daemon
	out, err := cmd.CombinedOutput()
	return string(out), err
}
