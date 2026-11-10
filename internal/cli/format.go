package cli

import (
	"fmt"
	"strings"

	"github.com/thisisadityapatel/kepler/internal/engines/def"
	"github.com/thisisadityapatel/kepler/internal/engines/resolve"
	"github.com/thisisadityapatel/kepler/internal/probe"
	"github.com/thisisadityapatel/kepler/internal/registry"
	"github.com/thisisadityapatel/kepler/internal/ui"
)

// hostLine is the one-line machine summary shown at the top of commands.
func hostLine(h probe.HostFacts) string {
	parts := []string{orDash(h.CPU.Brand), fmt.Sprintf("%s RAM", ui.Bytes(h.RAMBytes))}
	for _, a := range h.Accelerators {
		if a.Kind != probe.AccelCPU {
			parts = append(parts, fmt.Sprintf("%s GPU memory", ui.Bytes(a.MemBytes)))
			break
		}
	}
	osName := h.OS
	if h.OS == "darwin" {
		osName = "macOS"
	}
	parts = append(parts, osName+" "+h.OSVersion, h.PlatformKey())
	return strings.Join(parts, " · ")
}

// hostWarnings lists conditions that would distort benchmark numbers.
func hostWarnings(h probe.HostFacts) []string {
	var w []string
	if h.Power.Source == "battery" {
		w = append(w, "running on battery power; plug in for consistent results")
	}
	if h.Power.LowPowerMode {
		w = append(w, "Low Power Mode is on; it throttles CPU and GPU")
	}
	return w
}

// methodChain renders a target's install methods, e.g. "prebuilt → source".
func methodChain(t *def.Target) string {
	var kinds []string
	for _, m := range t.Install {
		kinds = append(kinds, m.Kind)
	}
	return strings.Join(kinds, " → ")
}

// decisionRow is one engines-table row.
func decisionRow(d resolve.Decision, e *registry.Engine) []string {
	status, version, detail := d.Status, orDash(d.Def.Version), ""
	switch d.Status {
	case resolve.Eligible:
		status = d.Install
		switch d.Install {
		case resolve.InstallReady:
			version = e.Version
			detail = e.Install.Path
		case resolve.InstallUnhealthy:
			detail = "run `kepler engines install " + d.Engine + "` to retry"
			if e != nil && e.Reason != "" {
				detail = firstLine(e.Reason)
			}
		case resolve.InstallOutdated:
			detail = fmt.Sprintf("installed %s; `kepler engines install %s` to upgrade", e.Version, d.Engine)
		case resolve.InstallNotInstalled:
			detail = "install: " + methodChain(d.Target)
			if d.Def.Tier == def.TierLazy {
				detail = "on demand: kepler engines install " + d.Engine
			}
		case resolve.InstallSkipped:
			detail = "skipped; `kepler engines install " + d.Engine + "` to add"
		}
	default:
		detail = d.Reason
	}
	target := "—"
	if d.Target != nil {
		target = d.Target.ID
	}
	return []string{d.Engine, status, version, target, ui.Truncate(detail, 80)}
}

func firstLine(s string) string {
	if i := strings.IndexByte(s, '\n'); i >= 0 {
		return s[:i]
	}
	return s
}

func orDash(s string) string {
	if s == "" {
		return "—"
	}
	return s
}
