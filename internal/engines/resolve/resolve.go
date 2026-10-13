// Package resolve decides which engines (and which target of each) can run on a host.
// It is a pure function of HostFacts, definitions, and install state: no I/O.
package resolve

import (
	"fmt"
	"sort"
	"strings"

	"github.com/thisisadityapatel/kepler/internal/engines/def"
	"github.com/thisisadityapatel/kepler/internal/probe"
)

// Decision statuses.
const (
	Eligible   = "eligible"
	Ineligible = "ineligible"
	Excluded   = "excluded"
)

// Install states, derived from the registry.
const (
	InstallReady        = "ready"
	InstallUnhealthy    = "installed_unhealthy"
	InstallNotInstalled = "not_installed"
	InstallOutdated     = "outdated"
	InstallSkipped      = "user_skipped"
)

// InstalledEngine is the subset of registry state the resolver needs.
type InstalledEngine struct {
	Status   string // registry status: ready | installed_unhealthy | user_skipped | ...
	Version  string
	TargetID string
	Adopted  bool // adopted installs are version-checked against min_version, not pinned version
}

// Query narrows resolution. A nil Query resolves every engine (used by init/doctor).
type Query struct {
	Formats []string // formats the model is available in; nil = any
	Allow   []string // --engines
	Deny    []string // --skip
}

// Decision is the outcome for one engine.
type Decision struct {
	Engine  string
	Def     *def.EngineDef
	Target  *def.Target // nil unless Eligible
	Status  string
	Reason  string // set whenever Status != Eligible
	Install string // install state; only meaningful when Eligible
}

// Resolve returns one decision per definition in stable order.
func Resolve(facts probe.HostFacts, set def.Set, installed map[string]InstalledEngine, q *Query) []Decision {
	var out []Decision
	for _, d := range set.Sorted() {
		out = append(out, decide(facts, d, installed[d.Name], q))
	}
	return out
}

func decide(facts probe.HostFacts, d *def.EngineDef, inst InstalledEngine, q *Query) Decision {
	dec := Decision{Engine: d.Name, Def: d}
	if q != nil {
		if len(q.Allow) > 0 && !contains(q.Allow, d.Name) {
			dec.Status, dec.Reason = Excluded, "not in --engines"
			return dec
		}
		if contains(q.Deny, d.Name) {
			dec.Status, dec.Reason = Excluded, "excluded by --skip"
			return dec
		}
	}

	var closest []string
	for i := range d.Targets {
		t := &d.Targets[i]
		fails := Explain(t.Match, facts)
		if len(fails) == 0 {
			dec.Target = t
			break
		}
		if closest == nil || len(fails) < len(closest) {
			closest = fails
		}
	}
	if dec.Target == nil {
		dec.Status = Ineligible
		for _, u := range d.Unsupported {
			if len(Explain(u.Match, facts)) == 0 {
				dec.Reason = u.Reason
				return dec
			}
		}
		if len(d.Targets) == 0 {
			dec.Reason = fmt.Sprintf("no target for %s", facts.PlatformKey())
		} else {
			dec.Reason = strings.Join(closest, "; ")
		}
		return dec
	}

	if q != nil && q.Formats != nil && !overlaps(d.Formats, q.Formats) {
		dec.Status = Excluded
		dec.Reason = fmt.Sprintf("engine runs %s; model available as %s",
			strings.Join(d.Formats, "/"), strings.Join(q.Formats, "/"))
		return dec
	}

	dec.Status = Eligible
	dec.Install = installState(d, dec.Target, inst)
	return dec
}

func installState(d *def.EngineDef, t *def.Target, inst InstalledEngine) string {
	switch inst.Status {
	case "":
		return InstallNotInstalled
	case InstallSkipped:
		return InstallSkipped
	case InstallUnhealthy:
		return InstallUnhealthy
	case InstallReady:
		if inst.TargetID != t.ID {
			return InstallOutdated // host now matches a different target (e.g. OS upgrade)
		}
		if !inst.Adopted && inst.Version != d.Version {
			return InstallOutdated
		}
		return InstallReady
	}
	return InstallNotInstalled
}

// Explain returns every match clause that fails against facts, as human-readable
// reasons. An empty result means the match holds.
func Explain(m def.Match, h probe.HostFacts) []string {
	var fails []string
	if len(m.OS) > 0 && !m.OS.Has(h.OS) {
		fails = append(fails, fmt.Sprintf("needs os %s (found %s)", strings.Join(m.OS, "|"), h.OS))
	}
	if len(m.Arch) > 0 && !m.Arch.Has(h.Arch) {
		fails = append(fails, fmt.Sprintf("needs arch %s (found %s)", strings.Join(m.Arch, "|"), h.Arch))
	}
	if m.OSVersion != "" {
		c, err := def.ParseConstraint(m.OSVersion)
		if err != nil || !c.Satisfied(h.OSVersion) {
			fails = append(fails, fmt.Sprintf("needs os_version %s (found %s)", m.OSVersion, orNone(h.OSVersion)))
		}
	}
	if m.RAMMin > 0 && h.RAMBytes < int64(m.RAMMin) {
		fails = append(fails, fmt.Sprintf("needs ram >= %s (found %s)", gib(int64(m.RAMMin)), gib(h.RAMBytes)))
	}
	for _, t := range m.Tools {
		if _, ok := h.Tools[t]; !ok {
			fails = append(fails, fmt.Sprintf("needs tool %s on PATH", t))
		}
	}

	// Accelerator clauses must all hold for the same accelerator.
	if hasAccelClause(m) {
		if _, why := matchAccelerator(m, h.Accelerators); why != "" {
			fails = append(fails, why)
		}
	}
	return fails
}

func hasAccelClause(m def.Match) bool {
	return len(m.Accelerator) > 0 || len(m.AcceleratorVendor) > 0 || m.AcceleratorAPI != "" ||
		m.ComputeCapability != "" || m.GPUMemMin > 0
}

// matchAccelerator finds an accelerator satisfying every accelerator clause. If none
// does, it explains the failure for the best candidate (one with the right kind, if any).
func matchAccelerator(m def.Match, accels []probe.Accelerator) (*probe.Accelerator, string) {
	var bestWhy string
	for i := range accels {
		a := &accels[i]
		var why []string
		if len(m.Accelerator) > 0 && !m.Accelerator.Has(a.Kind) {
			why = append(why, "kind")
		}
		if len(m.AcceleratorVendor) > 0 && !m.AcceleratorVendor.Has(a.Vendor) {
			why = append(why, fmt.Sprintf("needs accelerator vendor %s (found %s)", strings.Join(m.AcceleratorVendor, "|"), orNone(a.Vendor)))
		}
		if m.AcceleratorAPI != "" {
			c, err := def.ParseConstraint(m.AcceleratorAPI)
			if err != nil || !c.Satisfied(a.APIVersion) {
				why = append(why, fmt.Sprintf("needs %s api %s (found %s)", a.Kind, m.AcceleratorAPI, orNone(a.APIVersion)))
			}
		}
		if m.ComputeCapability != "" {
			c, err := def.ParseConstraint(m.ComputeCapability)
			if err != nil || !c.Satisfied(a.ComputeCap) {
				why = append(why, fmt.Sprintf("needs compute capability %s (found %s)", m.ComputeCapability, orNone(a.ComputeCap)))
			}
		}
		if m.GPUMemMin > 0 && a.MemBytes < int64(m.GPUMemMin) {
			why = append(why, fmt.Sprintf("needs gpu memory >= %s (found %s)", gib(int64(m.GPUMemMin)), gib(a.MemBytes)))
		}
		if len(why) == 0 {
			return a, ""
		}
		if why[0] != "kind" {
			bestWhy = strings.Join(why, "; ") // right kind, other clause failed: most useful reason
		}
	}
	if bestWhy != "" {
		return nil, bestWhy
	}
	return nil, fmt.Sprintf("needs accelerator %s (found %s)", strings.Join(m.Accelerator, "|"), kinds(accels))
}

func kinds(accels []probe.Accelerator) string {
	var ks []string
	for _, a := range accels {
		ks = append(ks, a.Kind)
	}
	sort.Strings(ks)
	if len(ks) == 0 {
		return "none"
	}
	return strings.Join(ks, ", ")
}

func contains(ss []string, v string) bool {
	for _, s := range ss {
		if s == v {
			return true
		}
	}
	return false
}

func overlaps(a, b []string) bool {
	for _, x := range a {
		if contains(b, x) {
			return true
		}
	}
	return false
}

func orNone(s string) string {
	if s == "" {
		return "none"
	}
	return s
}

func gib(b int64) string { return fmt.Sprintf("%.1f GiB", float64(b)/(1<<30)) }
