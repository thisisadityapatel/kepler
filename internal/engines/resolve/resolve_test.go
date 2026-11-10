package resolve

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"gopkg.in/yaml.v3"

	"github.com/thisisadityapatel/kepler/definitions"
	"github.com/thisisadityapatel/kepler/internal/engines/def"
	"github.com/thisisadityapatel/kepler/internal/probe"
)

func host(t *testing.T, name string) probe.HostFacts {
	t.Helper()
	b, err := os.ReadFile(filepath.Join("..", "..", "..", "testdata", "hosts", name+".yaml"))
	if err != nil {
		t.Fatal(err)
	}
	var h probe.HostFacts
	if err := yaml.Unmarshal(b, &h); err != nil {
		t.Fatal(err)
	}
	return h
}

func builtin(t *testing.T) def.Set {
	t.Helper()
	set, err := def.Load(definitions.Engines(), "", def.ValidateOptions{})
	if err != nil {
		t.Fatal(err)
	}
	return set
}

func byEngine(ds []Decision) map[string]Decision {
	out := map[string]Decision{}
	for _, d := range ds {
		out[d.Engine] = d
	}
	return out
}

func TestResolveFixtureHosts(t *testing.T) {
	set := builtin(t)
	type want struct {
		status string
		reason string // substring; "" = don't check
		target string
	}
	cases := map[string]map[string]want{
		"m4-16gb": {
			"llamacpp": {Eligible, "", "macos-metal"},
			"mlx":      {Eligible, "", "macos-metal"},
			"ollama":   {Eligible, "", "macos-metal"},
			"vllm":     {Ineligible, "vllm-metal is experimental", ""},
			"sglang":   {Ineligible, "SGLang's MLX backend", ""},
		},
		"m1-8gb-macos13": {
			"llamacpp": {Ineligible, "needs os_version >=14.0 (found 13.6)", ""},
			"mlx":      {Ineligible, "needs os_version >=14.0 (found 13.6)", ""},
			"ollama":   {Eligible, "", "macos-metal"},
		},
		"intel-mac": {
			"llamacpp": {Ineligible, "needs arch arm64 (found amd64)", ""},
			"ollama":   {Ineligible, "needs accelerator metal (found cpu)", ""},
		},
		"linux-rtx4090": {
			"llamacpp": {Ineligible, "needs os darwin (found linux)", ""},
			"mlx":      {Ineligible, "MLX runs only on Apple Silicon", ""},
			"vllm":     {Ineligible, "no target for linux/amd64/cuda", ""},
		},
		"windows-rx7900": {
			"mlx":    {Ineligible, "MLX runs only on Apple Silicon", ""},
			"ollama": {Ineligible, "needs accelerator metal (found cpu, vulkan)", ""},
		},
	}
	for hostName, engines := range cases {
		t.Run(hostName, func(t *testing.T) {
			got := byEngine(Resolve(host(t, hostName), set, nil, nil))
			for engine, w := range engines {
				d := got[engine]
				if d.Status != w.status {
					t.Errorf("%s: status = %s (%s), want %s", engine, d.Status, d.Reason, w.status)
					continue
				}
				if w.reason != "" && !strings.Contains(d.Reason, w.reason) {
					t.Errorf("%s: reason = %q, want substring %q", engine, d.Reason, w.reason)
				}
				if w.target != "" && (d.Target == nil || d.Target.ID != w.target) {
					t.Errorf("%s: target = %v, want %s", engine, d.Target, w.target)
				}
			}
		})
	}
}

// Guard for the core extensibility promise: adding a target in YAML makes an engine
// eligible on new hardware with zero code changes.
func TestAddingTargetIsCodeFree(t *testing.T) {
	b, err := os.ReadFile(filepath.Join("..", "..", "..", "definitions", "engines", "llamacpp.yaml"))
	if err != nil {
		t.Fatal(err)
	}
	extra := `
  - id: linux-cuda
    match: { os: linux, arch: amd64, accelerator: cuda, accelerator_api: ">=12.4", gpu_mem_min: 8GiB }
    install:
      - kind: prebuilt
        url: "https://example.invalid/llama-{version}-cuda.tar.gz"
        sha256: "0000000000000000000000000000000000000000000000000000000000000000"
        bin: "llama-server"
`
	src := strings.Replace(string(b), "\n  # Future targets", extra+"\n  # Future targets", 1)
	d, err := def.Parse([]byte(src), "llamacpp+cuda.yaml", def.ValidateOptions{})
	if err != nil {
		t.Fatal(err)
	}
	set := def.Set{Defs: map[string]*def.EngineDef{"llamacpp": d}}

	got := byEngine(Resolve(host(t, "linux-rtx4090"), set, nil, nil))["llamacpp"]
	if got.Status != Eligible || got.Target.ID != "linux-cuda" {
		t.Fatalf("linux host: %+v", got)
	}
	got = byEngine(Resolve(host(t, "m4-16gb"), set, nil, nil))["llamacpp"]
	if got.Status != Eligible || got.Target.ID != "macos-metal" {
		t.Fatalf("mac host must keep its own target: %+v", got)
	}

	small := host(t, "linux-rtx4090")
	small.Accelerators[0].MemBytes = 4 << 30
	got = byEngine(Resolve(small, set, nil, nil))["llamacpp"]
	if got.Status != Ineligible || !strings.Contains(got.Reason, "needs gpu memory >= 8.0 GiB (found 4.0 GiB)") {
		t.Fatalf("small gpu: %+v", got)
	}
}

func TestQueryFiltersAndInstallState(t *testing.T) {
	set := builtin(t)
	h := host(t, "m4-16gb")

	got := byEngine(Resolve(h, set, nil, &Query{Formats: []string{"mlx"}}))
	if got["llamacpp"].Status != Excluded || !strings.Contains(got["llamacpp"].Reason, "model available as mlx") {
		t.Errorf("llamacpp = %+v", got["llamacpp"])
	}
	if got["mlx"].Status != Eligible {
		t.Errorf("mlx = %+v", got["mlx"])
	}

	got = byEngine(Resolve(h, set, nil, &Query{Allow: []string{"mlx"}, Deny: []string{"mlx"}}))
	if got["ollama"].Status != Excluded || got["mlx"].Status != Excluded {
		t.Errorf("allow/deny: ollama=%s mlx=%s", got["ollama"].Status, got["mlx"].Status)
	}

	inst := map[string]InstalledEngine{
		"llamacpp": {Status: InstallReady, Version: "b11574", TargetID: "macos-metal"},
		"mlx":      {Status: InstallReady, Version: "0.1.0", TargetID: "macos-metal"},
		"ollama":   {Status: InstallReady, Version: "0.21.2", TargetID: "macos-metal", Adopted: true},
	}
	got = byEngine(Resolve(h, set, inst, nil))
	for engine, want := range map[string]string{"llamacpp": InstallReady, "mlx": InstallOutdated, "ollama": InstallReady} {
		if got[engine].Install != want {
			t.Errorf("%s install = %s, want %s", engine, got[engine].Install, want)
		}
	}
}

func TestUserDefinitionOverridesBuiltin(t *testing.T) {
	dir := t.TempDir()
	b, err := os.ReadFile(filepath.Join("..", "..", "..", "definitions", "engines", "ollama.yaml"))
	if err != nil {
		t.Fatal(err)
	}
	src := strings.Replace(string(b), "tier: default", "tier: optional", 1)
	if err := os.WriteFile(filepath.Join(dir, "ollama.yaml"), []byte(src), 0o644); err != nil {
		t.Fatal(err)
	}
	set, err := def.Load(definitions.Engines(), dir, def.ValidateOptions{})
	if err != nil {
		t.Fatal(err)
	}
	o := set.Defs["ollama"]
	if o.Source != def.SourceUser || o.Tier != def.TierOptional {
		t.Fatalf("ollama = source %s tier %s", o.Source, o.Tier)
	}
}
