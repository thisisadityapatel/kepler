package probe

import (
	"context"
	"regexp"
	"runtime"
	"strconv"
	"strings"
)

// Thermal states.
const (
	ThermalNominal  = "nominal"
	ThermalElevated = "elevated"
	ThermalUnknown  = "unknown"
)

// Conditions are fast-changing host state sampled around benchmark runs.
type Conditions struct {
	Thermal       string `json:"thermal"`
	ThermalDetail string `json:"thermal_detail,omitempty"`
	MemoryFreePct int    `json:"memory_free_pct,omitempty"` // -1 if unknown
	PowerSource   string `json:"power_source"`
	LowPowerMode  bool   `json:"low_power_mode"`
}

var (
	cpuSpeedLimitRe = regexp.MustCompile(`CPU_Speed_Limit\s*=\s*(\d+)`)
	thermalLevelRe  = regexp.MustCompile(`(?i)thermal warning level\s*(?:set to|=)?\s*(\d+)`)
	memFreeRe       = regexp.MustCompile(`memory free percentage:\s*(\d+)%`)
)

// SampleConditions reads thermal state, memory pressure, and power state.
func SampleConditions(ctx context.Context, run Runner) Conditions {
	if run == nil {
		run = ExecRunner
	}
	c := Conditions{Thermal: ThermalUnknown, MemoryFreePct: -1, PowerSource: "unknown"}
	if runtime.GOOS != "darwin" {
		return c
	}
	if out, err := run(ctx, "pmset", "-g", "therm"); err == nil {
		c.Thermal, c.ThermalDetail = ParseThermal(out)
	}
	if out, err := run(ctx, "memory_pressure", "-Q"); err == nil {
		if m := memFreeRe.FindStringSubmatch(out); m != nil {
			c.MemoryFreePct, _ = strconv.Atoi(m[1])
		}
	}
	if out, err := run(ctx, "pmset", "-g", "batt"); err == nil {
		c.PowerSource = parsePowerSourceAny(out)
	}
	if out, err := run(ctx, "pmset", "-g"); err == nil {
		c.LowPowerMode = strings.Contains(out, "lowpowermode") && regexp.MustCompile(`(?m)^\s*lowpowermode\s+1`).MatchString(out)
	}
	return c
}

// ParseThermal interprets `pmset -g therm`. "No thermal warning level has been
// recorded" and CPU_Speed_Limit = 100 mean nominal; anything else is elevated.
func ParseThermal(out string) (state, detail string) {
	if m := cpuSpeedLimitRe.FindStringSubmatch(out); m != nil {
		if n, _ := strconv.Atoi(m[1]); n < 100 {
			return ThermalElevated, "CPU_Speed_Limit=" + m[1]
		}
	}
	if m := thermalLevelRe.FindStringSubmatch(out); m != nil && m[1] != "0" {
		return ThermalElevated, "thermal warning level " + m[1]
	}
	if strings.Contains(out, "No thermal warning level") || cpuSpeedLimitRe.MatchString(out) {
		return ThermalNominal, ""
	}
	return ThermalUnknown, strings.TrimSpace(out)
}

func parsePowerSourceAny(out string) string {
	switch {
	case strings.Contains(out, "'AC Power'"):
		return "ac"
	case strings.Contains(out, "'Battery Power'"):
		return "battery"
	}
	return "unknown"
}
