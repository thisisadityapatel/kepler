package cli

import (
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/thisisadityapatel/kepler/internal/bench/results"
	"github.com/thisisadityapatel/kepler/internal/fakeengine"
	"github.com/thisisadityapatel/kepler/internal/hf/hftest"
	"github.com/thisisadityapatel/kepler/internal/home"
	"github.com/thisisadityapatel/kepler/internal/ledger"
	"github.com/thisisadityapatel/kepler/internal/registry"
)

func TestMain(m *testing.M) {
	fakeengine.MaybeServe()
	os.Exit(m.Run())
}

// A user-supplied engine definition (engines.d) backed by the fake engine: this also
// proves engines can be added without code changes.
const fakeEngineYAML = `
schema: 1
name: fake
display: Fake Engine
version: "1.0"
formats: [gguf]
tier: optional
adapter: generic
launch:
  cmd: "{bin}"
  args: ["-test.run=^$", "--", "-m", "{model_path}", "--port", "{port}"]
  env: { KEPLER_FAKE_ENGINE: "1", FAKE_TOKEN_DELAY: "1ms" }
  ready: { http_get: /health, expect: 200, timeout: 10s }
request: { api: openai, extra_body: { ignore_eos: true } }
caps: { ignore_eos: true, disable_prompt_cache: true, stream_usage: true }
targets:
  - id: any
    match: {}
    install: [{ kind: adopt, bin_name: kepler-fake-engine }]
`

func TestBenchEndToEndOffline(t *testing.T) {
	root := t.TempDir()
	paths := home.New(root)
	if err := paths.Ensure(); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(paths.EnginesD, "fake.yaml"), []byte(fakeEngineYAML), 0o644); err != nil {
		t.Fatal(err)
	}
	reg := &registry.Registry{Schema: registry.SchemaVersion, Engines: map[string]*registry.Engine{
		"fake": {Status: registry.StatusReady, Source: "user", Target: "any", Version: "1.0", Formats: []string{"gguf"},
			Install: &registry.Install{Kind: "adopt", Path: os.Args[0]}},
	}}
	if err := reg.Save(paths.Registry); err != nil {
		t.Fatal(err)
	}

	model := hftest.GGUF("llama", 4, 4, 2, 256, 8192, 2<<20)
	hub := hftest.NewHub(&hftest.Repo{ID: "org/Tiny-GGUF", SHA: strings.Repeat("d", 40), Tags: []string{"gguf"},
		Files: map[string][]byte{"Tiny-Q4_K_M.gguf": model, "Tiny-Q8_0.gguf": append(model, 0), "README.md": []byte("# tiny")}})
	defer hub.Close()
	t.Setenv("HF_ENDPOINT", hub.URL)

	code := run([]string{"--home", root, "bench", "org/Tiny-GGUF", "--engines", "fake", "--preset", "quick",
		"--iterations", "1", "--cooldown", "0s", "--yes"})
	if code != ExitOK {
		t.Fatalf("bench exit code = %d", code)
	}

	files, _ := filepath.Glob(filepath.Join(paths.Results, "*_org-Tiny-GGUF_fake_quick.json"))
	if len(files) != 1 {
		t.Fatalf("result files = %v", files)
	}
	r, err := results.Read(files[0])
	if err != nil {
		t.Fatal(err)
	}
	if r.Status != results.StatusComplete || r.Model.Quant != "Q4_K_M" || r.Model.Arch.Layers != 4 || len(r.Scenarios) != 2 {
		t.Fatalf("result = status %s quant %s arch %+v scenarios %d", r.Status, r.Model.Quant, r.Model.Arch, len(r.Scenarios))
	}
	if r.Model.Files[0].SHA256 == "" || r.Model.Revision != strings.Repeat("d", 40) {
		t.Errorf("model not pinned: %+v", r.Model)
	}
	if hub.Downloads["Tiny-Q8_0.gguf"] != 0 {
		t.Error("unselected quant was downloaded")
	}
	b, _ := os.ReadFile(files[0])
	var raw map[string]any
	if err := json.Unmarshal(b, &raw); err != nil || raw["schema_version"] != float64(2) {
		t.Fatalf("raw json: %v", err)
	}

	// The model is deleted and nothing temporary is left behind.
	if entries, _ := os.ReadDir(paths.Models); len(entries) != 0 {
		t.Errorf("tmp/models not empty: %v", entries)
	}
	if entries, _ := os.ReadDir(paths.Downloads); len(entries) != 0 {
		t.Errorf("tmp/downloads not empty: %v", entries)
	}
	l, err := ledger.Open(paths)
	if err != nil {
		t.Fatal(err)
	}
	defer l.Close()
	if rows, _ := l.List(context.Background(), ledger.StateTemp); len(rows) != 0 {
		t.Errorf("temp ledger rows: %+v", rows)
	}

	if code := run([]string{"--home", root, "results", "list"}); code != ExitOK {
		t.Errorf("results list exit = %d", code)
	}
	if code := run([]string{"--home", root, "results", "show", filepath.Base(files[0])}); code != ExitOK {
		t.Errorf("results show exit = %d", code)
	}

	// --keep retains the model and records it as kept.
	if code := run([]string{"--home", root, "bench", "org/Tiny-GGUF:Q8_0", "--engines", "fake", "--preset", "quick",
		"--scenarios", "chat", "--iterations", "1", "--cooldown", "0s", "--yes", "--keep"}); code != ExitOK {
		t.Fatalf("bench --keep exit = %d", code)
	}
	if rows, _ := l.List(context.Background(), ledger.StateKept); len(rows) != 1 {
		t.Errorf("kept rows = %+v", rows)
	}
	if code := run([]string{"--home", root, "clean", "--all"}); code != ExitOK {
		t.Errorf("clean --all exit = %d", code)
	}
	if entries, _ := os.ReadDir(paths.Models); len(entries) != 0 {
		t.Errorf("kept model not removed by clean --all: %v", entries)
	}
}

func TestBenchExitCodes(t *testing.T) {
	root := t.TempDir()
	hub := hftest.NewHub(&hftest.Repo{ID: "org/Plain", SHA: strings.Repeat("e", 40),
		Files: map[string][]byte{"config.json": []byte("{}"), "model.safetensors": []byte("x")}})
	defer hub.Close()
	t.Setenv("HF_ENDPOINT", hub.URL)

	for name, tc := range map[string]struct {
		args []string
		want int
	}{
		"missing repo":    {[]string{"bench", "org/Missing", "--yes"}, ExitUsage},
		"no gguf/mlx":     {[]string{"bench", "org/Plain", "--yes"}, ExitUsage},
		"bad preset":      {[]string{"bench", "org/Plain", "--preset", "huge", "--yes"}, ExitUsage},
		"unknown engine":  {[]string{"bench", "org/Plain", "--engines", "nope"}, ExitUsage},
		"not a reference": {[]string{"bench", "llama3"}, ExitUsage},
	} {
		if got := run(append([]string{"--home", root}, tc.args...)); got != tc.want {
			t.Errorf("%s: exit = %d, want %d", name, got, tc.want)
		}
	}
}
