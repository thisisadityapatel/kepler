// Package def defines the engine definition schema (one YAML file per engine),
// loading from the embedded set and ~/.kepler/engines.d, validation, and templating.
// It knows nothing about benchmarking.
package def

import (
	"fmt"
	"strconv"
	"strings"
	"time"

	"gopkg.in/yaml.v3"
)

// SchemaVersion is the engine definition schema this build understands.
const SchemaVersion = 1

// Tiers.
const (
	TierDefault  = "default"
	TierOptional = "optional"
	TierLazy     = "lazy"
)

// Sources.
const (
	SourceBuiltin = "builtin"
	SourceUser    = "user"
)

// EngineDef is one engine definition file.
type EngineDef struct {
	Schema      int             `yaml:"schema"`
	Name        string          `yaml:"name"`
	Display     string          `yaml:"display"`
	Version     string          `yaml:"version"`
	Formats     []string        `yaml:"formats"`
	Tier        string          `yaml:"tier"`
	Adapter     string          `yaml:"adapter"`
	Homepage    string          `yaml:"homepage"`
	Launch      Launch          `yaml:"launch"`
	Materialize *Materialize    `yaml:"materialize"`
	Request     Request         `yaml:"request"`
	Caps        map[string]bool `yaml:"caps"`
	Targets     []Target        `yaml:"targets"`
	Unsupported []Unsupported   `yaml:"unsupported"`

	// Set by the loader, not by YAML.
	Source string `yaml:"-"`
	File   string `yaml:"-"`
}

// Target is one way of running the engine on one class of hardware.
type Target struct {
	ID      string          `yaml:"id"`
	Match   Match           `yaml:"match"`
	Install []InstallMethod `yaml:"install"`
	Launch  *Launch         `yaml:"launch"`
	Caps    map[string]bool `yaml:"caps"`
}

// Unsupported attaches a human reason to hosts the engine is known not to support.
type Unsupported struct {
	Match  Match  `yaml:"match"`
	Reason string `yaml:"reason"`
}

// Match clauses are ANDed. Empty clauses always hold. The key set is closed;
// unknown keys fail strict YAML decoding.
type Match struct {
	OS                StringList `yaml:"os"`
	Arch              StringList `yaml:"arch"`
	OSVersion         string     `yaml:"os_version"`
	Accelerator       StringList `yaml:"accelerator"`
	AcceleratorVendor StringList `yaml:"accelerator_vendor"`
	AcceleratorAPI    string     `yaml:"accelerator_api"`
	ComputeCapability string     `yaml:"compute_capability"`
	GPUMemMin         ByteSize   `yaml:"gpu_mem_min"`
	RAMMin            ByteSize   `yaml:"ram_min"`
	Tools             []string   `yaml:"tools"`
}

// Launch describes how to start the engine's server.
type Launch struct {
	Cmd    string              `yaml:"cmd"`
	Args   []string            `yaml:"args"`
	ArgsIf map[string][]string `yaml:"args_if"`
	Env    map[string]string   `yaml:"env"`
	Ready  Ready               `yaml:"ready"`
}

// Ready is the readiness probe.
type Ready struct {
	HTTPGet string   `yaml:"http_get"`
	Expect  int      `yaml:"expect"`
	Timeout Duration `yaml:"timeout"`
}

// Materialize prepares an engine-specific model artifact (e.g. `ollama create`).
type Materialize struct {
	Modelfile string   `yaml:"modelfile"`
	Cmd       []string `yaml:"cmd"`
}

// Request holds per-engine request defaults merged into every chat request.
type Request struct {
	API       string         `yaml:"api"`
	Model     string         `yaml:"model"`
	ExtraBody map[string]any `yaml:"extra_body"`
}

// InstallMethod is one install recipe. Which fields apply depends on Kind.
type InstallMethod struct {
	Kind       string   `yaml:"kind"`
	URL        string   `yaml:"url"`
	SHA256     string   `yaml:"sha256"`
	Bin        string   `yaml:"bin"`
	Repo       string   `yaml:"repo"`
	Ref        string   `yaml:"ref"`
	CMake      []string `yaml:"cmake"`
	Targets    []string `yaml:"build_targets"`
	Needs      []string `yaml:"needs"`
	Python     string   `yaml:"python"`
	Packages   []string `yaml:"packages"`
	BinName    string   `yaml:"bin_name"`
	MinVersion string   `yaml:"min_version"`
}

// Caps known to Kepler. Unknown caps keys are validation errors.
var KnownCaps = map[string]string{
	"ignore_eos":           "engine honors ignore_eos so it always emits max_tokens",
	"disable_prompt_cache": "engine prompt cache can be disabled per request",
	"stream_usage":         "engine reports usage in the stream with stream_options.include_usage",
	"engine_timings":       "engine returns its own timing breakdown",
}

// ResolvedTarget is a definition with one target applied: target launch and caps
// merged over the top-level defaults.
type ResolvedTarget struct {
	Def    *EngineDef
	Target *Target
	Launch Launch
	Caps   map[string]bool
}

// Resolve merges target over the definition defaults.
func (d *EngineDef) Resolve(t *Target) ResolvedTarget {
	l := d.Launch
	if t.Launch != nil {
		l = mergeLaunch(d.Launch, *t.Launch)
	}
	caps := map[string]bool{}
	for k, v := range d.Caps {
		caps[k] = v
	}
	for k, v := range t.Caps {
		caps[k] = v
	}
	return ResolvedTarget{Def: d, Target: t, Launch: l, Caps: caps}
}

func mergeLaunch(base, over Launch) Launch {
	out := base
	if over.Cmd != "" {
		out.Cmd = over.Cmd
	}
	if over.Args != nil {
		out.Args = over.Args
	}
	out.ArgsIf = mergeMap(base.ArgsIf, over.ArgsIf)
	out.Env = mergeMap(base.Env, over.Env)
	if over.Ready.HTTPGet != "" {
		out.Ready.HTTPGet = over.Ready.HTTPGet
	}
	if over.Ready.Expect != 0 {
		out.Ready.Expect = over.Ready.Expect
	}
	if over.Ready.Timeout != 0 {
		out.Ready.Timeout = over.Ready.Timeout
	}
	return out
}

func mergeMap[V any](a, b map[string]V) map[string]V {
	if a == nil && b == nil {
		return nil
	}
	out := make(map[string]V, len(a)+len(b))
	for k, v := range a {
		out[k] = v
	}
	for k, v := range b {
		out[k] = v
	}
	return out
}

// StringList accepts a scalar or a sequence in YAML.
type StringList []string

func (s *StringList) UnmarshalYAML(n *yaml.Node) error {
	switch n.Kind {
	case yaml.ScalarNode:
		*s = StringList{n.Value}
		return nil
	case yaml.SequenceNode:
		var v []string
		if err := n.Decode(&v); err != nil {
			return err
		}
		*s = v
		return nil
	}
	return fmt.Errorf("line %d: expected string or list", n.Line)
}

// Has reports whether v is in the list. An empty list matches nothing.
func (s StringList) Has(v string) bool {
	for _, x := range s {
		if x == v {
			return true
		}
	}
	return false
}

// ByteSize parses "8GiB", "512MiB", "16GB" or a plain byte count.
type ByteSize int64

func (b *ByteSize) UnmarshalYAML(n *yaml.Node) error {
	v, err := ParseByteSize(n.Value)
	if err != nil {
		return fmt.Errorf("line %d: %w", n.Line, err)
	}
	*b = ByteSize(v)
	return nil
}

// ParseByteSize parses a human byte size.
func ParseByteSize(s string) (int64, error) {
	s = strings.TrimSpace(s)
	units := []struct {
		suffix string
		mult   int64
	}{
		{"TiB", 1 << 40}, {"GiB", 1 << 30}, {"MiB", 1 << 20}, {"KiB", 1 << 10},
		{"TB", 1e12}, {"GB", 1e9}, {"MB", 1e6}, {"KB", 1e3}, {"B", 1},
	}
	for _, u := range units {
		if strings.HasSuffix(s, u.suffix) {
			f, err := strconv.ParseFloat(strings.TrimSpace(strings.TrimSuffix(s, u.suffix)), 64)
			if err != nil {
				return 0, fmt.Errorf("invalid size %q", s)
			}
			return int64(f * float64(u.mult)), nil
		}
	}
	n, err := strconv.ParseInt(s, 10, 64)
	if err != nil {
		return 0, fmt.Errorf("invalid size %q", s)
	}
	return n, nil
}

// Duration parses Go duration strings ("300s", "5m").
type Duration time.Duration

func (d *Duration) UnmarshalYAML(n *yaml.Node) error {
	v, err := time.ParseDuration(n.Value)
	if err != nil {
		return fmt.Errorf("line %d: invalid duration %q", n.Line, n.Value)
	}
	*d = Duration(v)
	return nil
}
