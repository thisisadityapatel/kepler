//go:build darwin

package probe

import (
	"context"
	"regexp"
	"strconv"
	"strings"
)

func init() { registerDetector(metalDetector{}) }

func probeOS(ctx context.Context, run Runner, h *HostFacts) error {
	sysctl := func(key string) string {
		out, err := run(ctx, "sysctl", "-n", key)
		if err != nil {
			return ""
		}
		return strings.TrimSpace(out)
	}
	h.CPU.Brand = sysctl("machdep.cpu.brand_string")
	h.ModelID = sysctl("hw.model")
	h.CPU.PerformanceCores = atoi(sysctl("hw.perflevel0.physicalcpu"))
	h.CPU.EfficiencyCores = atoi(sysctl("hw.perflevel1.physicalcpu"))
	h.RAMBytes = atoi64(sysctl("hw.memsize"))
	if v, err := run(ctx, "sw_vers", "-productVersion"); err == nil {
		h.OSVersion = strings.TrimSpace(v)
	}
	if out, err := run(ctx, "pmset", "-g", "batt"); err == nil {
		h.Power.Source = parsePowerSource(out)
	}
	if out, err := run(ctx, "pmset", "-g"); err == nil {
		h.Power.LowPowerMode = parseLowPowerMode(out)
	}
	return nil
}

// metalDetector reports the Apple GPU on Apple Silicon.
type metalDetector struct{}

func (metalDetector) Kind() string { return AccelMetal }

func (metalDetector) Detect(ctx context.Context, run Runner, h HostFacts) ([]Accelerator, error) {
	if h.Arch != "arm64" {
		return nil, nil // Intel Macs: Metal exists but no v1 engine targets it
	}
	wired := int64(0)
	if out, err := run(ctx, "sysctl", "-n", "iogpu.wired_limit_mb"); err == nil {
		wired = atoi64(strings.TrimSpace(out))
	}
	cores := 0
	if out, err := run(ctx, "ioreg", "-rc", "AGXAccelerator", "-d1"); err == nil {
		cores = parseGPUCores(out)
	}
	return []Accelerator{{
		Kind:     AccelMetal,
		Vendor:   "apple",
		Name:     h.CPU.Brand,
		MemBytes: EffectiveWiredLimit(h.RAMBytes, wired),
		Unified:  true,
		Cores:    cores,
	}}, nil
}

// EffectiveWiredLimit is the GPU-usable share of unified memory. A non-zero
// iogpu.wired_limit_mb is an explicit user override; zero means the OS default,
// approximated as 2/3 of RAM up to 36 GiB and 3/4 above.
func EffectiveWiredLimit(ramBytes, wiredLimitMB int64) int64 {
	if wiredLimitMB > 0 {
		return wiredLimitMB << 20
	}
	if ramBytes <= 36<<30 {
		return ramBytes * 2 / 3
	}
	return ramBytes * 3 / 4
}

var (
	gpuCoresRe  = regexp.MustCompile(`"gpu-core-count"\s*=\s*(\d+)`)
	lowPowerRe  = regexp.MustCompile(`(?m)^\s*lowpowermode\s+(\d)`)
	powerFromRe = regexp.MustCompile(`drawing from '([^']+)'`)
)

func parseGPUCores(ioreg string) int {
	if m := gpuCoresRe.FindStringSubmatch(ioreg); m != nil {
		return atoi(m[1])
	}
	return 0
}

func parsePowerSource(pmsetBatt string) string {
	m := powerFromRe.FindStringSubmatch(pmsetBatt)
	if m == nil {
		return "unknown"
	}
	switch {
	case strings.Contains(m[1], "AC"):
		return "ac"
	case strings.Contains(m[1], "Battery"):
		return "battery"
	}
	return "unknown"
}

func parseLowPowerMode(pmset string) bool {
	m := lowPowerRe.FindStringSubmatch(pmset)
	return m != nil && m[1] == "1"
}

func atoi(s string) int     { n, _ := strconv.Atoi(strings.TrimSpace(s)); return n }
func atoi64(s string) int64 { n, _ := strconv.ParseInt(strings.TrimSpace(s), 10, 64); return n }
