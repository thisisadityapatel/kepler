package setup

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/thisisadityapatel/kepler/internal/download"
	"github.com/thisisadityapatel/kepler/internal/engines/adapter"
	"github.com/thisisadityapatel/kepler/internal/engines/def"
	"github.com/thisisadityapatel/kepler/internal/engines/resolve"
	"github.com/thisisadityapatel/kepler/internal/events"
	"github.com/thisisadityapatel/kepler/internal/fakeengine"
	"github.com/thisisadityapatel/kepler/internal/home"
	"github.com/thisisadityapatel/kepler/internal/install"
	"github.com/thisisadityapatel/kepler/internal/ledger"
	"github.com/thisisadityapatel/kepler/internal/probe"
	"github.com/thisisadityapatel/kepler/internal/registry"
)

func TestMain(m *testing.M) {
	fakeengine.MaybeServe()
	os.Exit(m.Run())
}

// The definition tries a prebuilt with a bad checksum first, then adopts the fake
// engine (this test binary, symlinked onto PATH as "fakeengine").
const engineDef = `
schema: 1
name: fake
display: Fake
version: "9.9"
formats: [gguf]
tier: default
adapter: generic
launch:
  cmd: "{bin}"
  args: ["-test.run=^$", "--", "-m", "{model_path}", "--port", "{port}"]
  env: { KEPLER_FAKE_ENGINE: "1", FAKE_EXIT_AT_START: "EXIT" }
  ready: { http_get: /health, expect: 200, timeout: 10s }
request: { api: openai }
targets:
  - id: any
    match: { os: [darwin, linux] }
    install:
      - { kind: prebuilt, url: "BADURL/x.tar.gz", sha256: "0000000000000000000000000000000000000000000000000000000000000000", bin: x }
      - { kind: adopt, bin_name: fakeengine }
`

func newSetup(t *testing.T, exitAtStart bool) (*Setup, resolve.Decision) {
	t.Helper()
	canary := []byte("tiny gguf bytes")
	sum := sha256.Sum256(canary)
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if strings.HasSuffix(r.URL.Path, "/x.tar.gz") {
			w.Write([]byte("not the archive"))
			return
		}
		w.Write(canary)
	}))
	t.Cleanup(srv.Close)

	bindir := t.TempDir()
	if err := os.Symlink(os.Args[0], filepath.Join(bindir, "fakeengine")); err != nil {
		t.Fatal(err)
	}
	t.Setenv("PATH", bindir+":"+os.Getenv("PATH"))

	exit := "0"
	if exitAtStart {
		exit = "1"
	}
	src := strings.NewReplacer("BADURL", srv.URL, `"EXIT"`, `"`+exit+`"`).Replace(engineDef)
	d, err := def.Parse([]byte(src), "fake.yaml", def.ValidateOptions{InstallKinds: install.Kinds(), Adapters: adapter.Names()})
	if err != nil {
		t.Fatal(err)
	}
	d.Source = def.SourceBuiltin

	p := home.New(t.TempDir())
	if err := p.Ensure(); err != nil {
		t.Fatal(err)
	}
	l, err := ledger.Open(p)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { l.Close() })
	dl := download.New(p.Downloads)
	dl.Retries = 0

	s := &Setup{
		Paths: p, Ledger: l, Downloader: dl, Bus: &events.Bus{}, Version: "test", HFEndpoint: srv.URL,
		Defs:     def.Set{Defs: map[string]*def.EngineDef{"fake": d}},
		Canaries: def.Canaries{"gguf": {Repo: "org/tiny", Revision: strings.Repeat("a", 40), Entry: "tiny.gguf", Files: []def.CanaryFile{{Path: "tiny.gguf", Size: int64(len(canary)), SHA256: hex.EncodeToString(sum[:])}}}},
		Facts:    probe.HostFacts{OS: "darwin", Arch: "arm64"},
		Registry: &registry.Registry{Schema: registry.SchemaVersion, Engines: map[string]*registry.Engine{}},
	}
	return s, s.Resolve(nil)[0]
}

func TestInstallFallsBackAndHealthChecks(t *testing.T) {
	s, d := newSetup(t, false)
	if d.Status != resolve.Eligible {
		t.Fatalf("decision = %+v", d)
	}
	if err := s.Install(context.Background(), d); err != nil {
		t.Fatal(err)
	}
	e := s.Registry.Engines["fake"]
	if e.Status != registry.StatusReady || e.Install.Kind != "adopt" || !e.Health.OK || e.Health.DecodeTokPerS <= 0 {
		t.Fatalf("registry entry = %+v health = %+v", e, e.Health)
	}
	if _, err := os.Stat(filepath.Join(s.CanaryDir("gguf"), ".complete")); err != nil {
		t.Error("canary not kept")
	}
	if rows, _ := s.Ledger.List(context.Background(), ledger.StateTemp); len(rows) != 0 {
		t.Errorf("temp ledger rows remain: %+v", rows)
	}
	saved, err := registry.Load(s.Paths.Registry)
	if err != nil || saved.Engines["fake"].Status != registry.StatusReady {
		t.Fatalf("saved registry = %+v, %v", saved, err)
	}
	if got := s.Resolve(nil)[0].Install; got != resolve.InstallReady {
		t.Errorf("resolver install state = %s", got)
	}
}

func TestInstallRecordsUnhealthyWhenEveryMethodFails(t *testing.T) {
	s, d := newSetup(t, true)
	err := s.Install(context.Background(), d)
	if err == nil {
		t.Fatal("expected failure")
	}
	for _, want := range []string{"prebuilt:", "digest mismatch", "adopt: installed but health check failed", "simulated startup crash"} {
		if !strings.Contains(err.Error(), want) {
			t.Errorf("error missing %q:\n%v", want, err)
		}
	}
	if e := s.Registry.Engines["fake"]; e.Status != registry.StatusUnhealthy {
		t.Fatalf("status = %s", e.Status)
	}
}
