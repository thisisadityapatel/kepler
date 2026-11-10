//go:build !darwin

package probe

import "context"

// probeOS is a placeholder until Linux/Windows probers land. It leaves only the
// runtime-derived facts, so the resolver reports engines as ineligible with reasons.
func probeOS(ctx context.Context, run Runner, h *HostFacts) error { return nil }
