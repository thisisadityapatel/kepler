package adapter

import (
	"context"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/thisisadityapatel/kepler/internal/engines/def"
	"github.com/thisisadityapatel/kepler/internal/fakeengine"
	"github.com/thisisadityapatel/kepler/internal/home"
	"github.com/thisisadityapatel/kepler/internal/ledger"
	"github.com/thisisadityapatel/kepler/internal/openai"
)

func TestMain(m *testing.M) {
	fakeengine.MaybeServe()
	os.Exit(m.Run())
}

const fakeDef = `
schema: 1
name: fake
display: Fake
version: "1"
formats: [gguf]
tier: default
adapter: generic
launch:
  cmd: "{bin}"
  args: ["-test.run=^$", "--", "-m", "{model_path}", "--port", "{port}", "-c", "{ctx}"]
  args_if: { mmproj: ["--mmproj", "{mmproj_path}"] }
  env: { KEPLER_FAKE_ENGINE: "1", FAKE_STARTUP_DELAY: "300ms" }
  ready: { http_get: /health, expect: 200, timeout: 10s }
request: { api: openai, model: "fake-{run_id}", extra_body: { ignore_eos: true } }
targets:
  - id: any
    match: {}
    install: [{ kind: adopt, bin_name: fake }]
`

func setup(t *testing.T, src string) (*GenericAdapter, RunEnv, *ledger.Ledger) {
	t.Helper()
	d, err := def.Parse([]byte(src), "fake.yaml", def.ValidateOptions{})
	if err != nil {
		t.Fatal(err)
	}
	p := home.New(t.TempDir())
	if err := p.Ensure(); err != nil {
		t.Fatal(err)
	}
	l, err := ledger.Open(p)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { l.Close() })
	g, err := New(d.Resolve(&d.Targets[0]), Installed{Bin: os.Args[0]}, p)
	if err != nil {
		t.Fatal(err)
	}
	return g, RunEnv{RunID: "r1", Ctx: 4096, LogDir: filepath.Join(p.Logs, "r1"), Ledger: l}, l
}

func TestGenericLifecycle(t *testing.T) {
	ctx := context.Background()
	g, run, l := setup(t, fakeDef)

	em, err := g.Materialize(ctx, Model{Path: "/models/x.gguf"}, run)
	if err != nil {
		t.Fatal(err)
	}
	r, err := g.Launch(ctx, em, run)
	if err != nil {
		t.Fatal(err)
	}
	if err := g.WaitReady(ctx, r); err != nil {
		t.Fatal(err)
	}
	em, err = g.PostReady(ctx, r, em)
	if err != nil {
		t.Fatal(err)
	}
	ep := g.Endpoint(r, em)
	if ep.Model != "fake-r1" || ep.ExtraBody["ignore_eos"] != true || !strings.HasSuffix(ep.BaseURL, "/v1") {
		t.Fatalf("endpoint = %+v", ep)
	}
	res, err := openai.New(ep.BaseURL).Stream(ctx, openai.ChatRequest{Model: ep.Model, MaxTokens: 8, Extra: ep.ExtraBody,
		Messages: []openai.Message{{Role: "user", Content: "hi"}}})
	if err != nil {
		t.Fatal(err)
	}
	if res.Events() != 8 || res.Usage.CompletionTokens != 8 {
		t.Fatalf("stream = %d events, usage %+v", res.Events(), res.Usage)
	}
	if strings.Contains(strings.Join(r.Proc.Cmd.Args, " "), "--mmproj") {
		t.Error("args_if applied without mmproj")
	}

	if err := StopAndRelease(ctx, g, r, run); err != nil {
		t.Fatal(err)
	}
	if exited, _ := r.Proc.Exited(); !exited {
		t.Fatal("engine still running")
	}
	if rows, _ := l.List(ctx, ledger.StateTemp); len(rows) != 0 {
		t.Fatalf("temp ledger rows remain: %+v", rows)
	}
}

func TestWaitReadyReportsCrashWithLog(t *testing.T) {
	ctx := context.Background()
	g, run, _ := setup(t, strings.Replace(fakeDef, `FAKE_STARTUP_DELAY: "300ms"`, `FAKE_EXIT_AT_START: "1"`, 1))
	em, _ := g.Materialize(ctx, Model{Path: "/m.gguf", MMProj: "/p.gguf"}, run)
	r, err := g.Launch(ctx, em, run)
	if err != nil {
		t.Fatal(err)
	}
	defer StopAndRelease(ctx, g, r, run)
	if !strings.Contains(strings.Join(r.Proc.Cmd.Args, " "), "--mmproj /p.gguf") {
		t.Error("args_if not applied with mmproj")
	}
	err = g.WaitReady(ctx, r)
	if err == nil || !strings.Contains(err.Error(), "simulated startup crash") {
		t.Fatalf("err = %v", err)
	}
}

func TestUnknownHooksRejected(t *testing.T) {
	d, err := def.Parse([]byte(strings.Replace(fakeDef, "adapter: generic", "adapter: nope", 1)), "fake.yaml", def.ValidateOptions{})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := New(d.Resolve(&d.Targets[0]), Installed{}, home.New(t.TempDir())); err == nil {
		t.Fatal("expected error for unregistered adapter")
	}
}
