package adapter

import (
	"context"
	"sort"
)

// Hooks is a set of step overrides for one engine. A hooks value implements any subset
// of the *Hook interfaces below; steps it doesn't implement use the generic behavior.
type Hooks interface{}

// MaterializeHook overrides pre-launch model preparation.
type MaterializeHook interface {
	Materialize(ctx context.Context, g *GenericAdapter, m Model, run RunEnv) (EngineModel, error)
}

// LaunchHook overrides launch (it may wrap g.LaunchGeneric).
type LaunchHook interface {
	Launch(ctx context.Context, g *GenericAdapter, em EngineModel, run RunEnv) (*Running, error)
}

// PostReadyHook runs once the server answers its readiness probe.
type PostReadyHook interface {
	PostReady(ctx context.Context, g *GenericAdapter, r *Running, em EngineModel) (EngineModel, error)
}

// EndpointHook overrides endpoint/request defaults.
type EndpointHook interface {
	Endpoint(g *GenericAdapter, r *Running, em EngineModel) Endpoint
}

var hooksRegistry = map[string]Hooks{}

// RegisterHooks registers hooks under name. Called explicitly from the composition root.
func RegisterHooks(name string, h Hooks) { hooksRegistry[name] = h }

// Names returns every valid `adapter:` value: "generic" plus registered hook sets.
func Names() map[string]bool {
	out := map[string]bool{Generic: true}
	for n := range hooksRegistry {
		out[n] = true
	}
	return out
}

// SortedNames is Names as a sorted slice.
func SortedNames() []string {
	var out []string
	for n := range Names() {
		out = append(out, n)
	}
	sort.Strings(out)
	return out
}
