package def

import (
	"strings"
	"testing"

	"github.com/thisisadityapatel/kepler/definitions"
)

const minimal = `
schema: 1
name: demo
display: Demo
version: "1.0"
formats: [gguf]
tier: optional
adapter: generic
launch:
  cmd: "{bin}"
  args: ["--port", "{port}"]
  ready: { http_get: /health, expect: 200, timeout: 30s }
request: { api: openai }
targets:
  - id: any
    match: { os: darwin }
    install: [{ kind: adopt, bin_name: demo }]
`

func TestParseMinimal(t *testing.T) {
	d, err := Parse([]byte(minimal), "demo.yaml", ValidateOptions{})
	if err != nil {
		t.Fatal(err)
	}
	if d.Launch.Ready.Timeout == 0 || d.Targets[0].Match.OS[0] != "darwin" {
		t.Fatalf("decoded = %+v", d)
	}
}

func TestValidationErrorsNameFileAndField(t *testing.T) {
	cases := map[string]struct{ from, to, want string }{
		"unknown match key": {"match: { os: darwin }", "match: { os: darwin, gpu: big }", "field gpu not found"},
		"unknown template":  {`"--port", "{port}"`, `"--port", "{prot}"`, "launch.args: unknown template variable {prot}"},
		"bad tier":          {"tier: optional", "tier: sometimes", "tier: must be default, optional or lazy"},
		"unknown cap":       {"request: { api: openai }", "request: { api: openai }\ncaps: { fast: true }", `caps: unknown capability "fast"`},
		"bad constraint":    {"match: { os: darwin }", `match: { os: darwin, os_version: ">=fourteen" }`, "os_version: invalid version constraint"},
		"placeholder":       {"bin_name: demo", `bin_name: demo, min_version: "1", url: "<pin>"`, "unresolved placeholder"},
		"no install":        {"install: [{ kind: adopt, bin_name: demo }]", "install: []", "at least one install method"},
	}
	for name, c := range cases {
		t.Run(name, func(t *testing.T) {
			src := strings.Replace(minimal, c.from, c.to, 1)
			if src == minimal {
				t.Fatalf("replacement %q did not apply", c.from)
			}
			_, err := Parse([]byte(src), "demo.yaml", ValidateOptions{})
			if err == nil || !strings.Contains(err.Error(), c.want) || !strings.Contains(err.Error(), "demo.yaml") {
				t.Fatalf("err = %v, want substring %q", err, c.want)
			}
		})
	}
}

func TestRegistriesAreEnforced(t *testing.T) {
	opt := ValidateOptions{
		InstallKinds:  map[string]func(InstallMethod) error{"prebuilt": nil},
		Adapters:      map[string]bool{"generic": true},
		CanaryFormats: map[string]bool{"mlx": true},
	}
	_, err := Parse([]byte(minimal), "demo.yaml", opt)
	for _, want := range []string{`unknown install kind "adopt"`, `no health-check canary for format "gguf"`} {
		if err == nil || !strings.Contains(err.Error(), want) {
			t.Errorf("err = %v, want %q", err, want)
		}
	}
}

func TestBuiltinDefinitionsLoad(t *testing.T) {
	set, err := Load(definitions.Engines(), "", ValidateOptions{})
	if err != nil {
		t.Fatal(err)
	}
	for _, name := range []string{"llamacpp", "mlx", "ollama", "vllm", "sglang"} {
		if _, ok := set.Defs[name]; !ok {
			t.Errorf("missing builtin %s", name)
		}
	}
	for _, name := range []string{"llamacpp", "mlx", "ollama"} {
		if set.Defs[name].Tier != TierDefault {
			t.Errorf("%s must be default tier", name)
		}
	}
	c, err := LoadCanaries(definitions.Root(), definitions.CanariesFile)
	if err != nil {
		t.Fatal(err)
	}
	if !c.Formats()["gguf"] || !c.Formats()["mlx"] {
		t.Errorf("canaries = %v", c.Formats())
	}
}

func TestResolveMergesTargetOverrides(t *testing.T) {
	src := strings.Replace(minimal, "install: [{ kind: adopt, bin_name: demo }]",
		"install: [{ kind: adopt, bin_name: demo }]\n    launch: { env: { A: \"1\" }, ready: { timeout: 90s } }\n    caps: { ignore_eos: true }", 1)
	d, err := Parse([]byte(src), "demo.yaml", ValidateOptions{})
	if err != nil {
		t.Fatal(err)
	}
	rt := d.Resolve(&d.Targets[0])
	if rt.Launch.Cmd != "{bin}" || rt.Launch.Env["A"] != "1" || rt.Launch.Ready.HTTPGet != "/health" || !rt.Caps["ignore_eos"] {
		t.Fatalf("resolved = %+v", rt)
	}
}

func TestExpandAndVersions(t *testing.T) {
	got, err := Expand("llama-{version}/x", map[string]string{"version": "b1"})
	if err != nil || got != "llama-b1/x" {
		t.Fatalf("Expand = %q, %v", got, err)
	}
	if _, err := Expand("{port}", nil); err == nil {
		t.Fatal("missing var must error")
	}
	for _, c := range []struct {
		cons, have string
		ok         bool
	}{{">=14.0", "26.6.2", true}, {">=14.0", "13.6", false}, {"0.6.0", "0.21.2", true}, {">=b11000", "b11574", true}, {"==8.9", "8.9", true}} {
		pc, err := ParseConstraint(c.cons)
		if err != nil {
			t.Fatal(err)
		}
		if pc.Satisfied(c.have) != c.ok {
			t.Errorf("%s vs %s: want %v", c.cons, c.have, c.ok)
		}
	}
}
