package install

import (
	"archive/tar"
	"bytes"
	"compress/gzip"
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
	"github.com/thisisadityapatel/kepler/internal/engines/def"
	"github.com/thisisadityapatel/kepler/internal/home"
	"github.com/thisisadityapatel/kepler/internal/ledger"
)

type entry struct {
	name, body, link string
	mode             int64
}

func tarGz(t *testing.T, entries []entry) []byte {
	t.Helper()
	var buf bytes.Buffer
	gz := gzip.NewWriter(&buf)
	tw := tar.NewWriter(gz)
	for _, e := range entries {
		h := &tar.Header{Name: e.name, Mode: e.mode, Size: int64(len(e.body)), Typeflag: tar.TypeReg}
		if e.link != "" {
			h.Typeflag, h.Linkname, h.Size = tar.TypeSymlink, e.link, 0
		}
		if err := tw.WriteHeader(h); err != nil {
			t.Fatal(err)
		}
		if e.link == "" {
			tw.Write([]byte(e.body))
		}
	}
	tw.Close()
	gz.Close()
	return buf.Bytes()
}

func testEnv(t *testing.T) Env {
	t.Helper()
	p := home.New(t.TempDir())
	if err := p.Ensure(); err != nil {
		t.Fatal(err)
	}
	l, err := ledger.Open(p)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { l.Close() })
	return Env{Paths: p, Ledger: l, Downloader: download.New(p.Downloads), Engine: "demo", Version: "b1", RunID: "r1"}
}

func TestPrebuiltInstall(t *testing.T) {
	archive := tarGz(t, []entry{
		{name: "llama-b1/llama-server", body: "#!/bin/sh\necho ok\n", mode: 0o755},
		{name: "llama-b1/libggml.0.dylib", body: "lib", mode: 0o644},
		{name: "llama-b1/libggml.dylib", link: "libggml.0.dylib"},
	})
	sum := sha256.Sum256(archive)
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) { w.Write(archive) }))
	defer srv.Close()

	env := testEnv(t)
	m := def.InstallMethod{Kind: "prebuilt", URL: srv.URL + "/llama-{version}.tar.gz", SHA256: hex.EncodeToString(sum[:]), Bin: "llama-{version}/llama-server"}
	if err := (prebuilt{}).Validate(m); err != nil {
		t.Fatal(err)
	}
	res, err := prebuilt{}.Install(context.Background(), env, m)
	if err != nil {
		t.Fatal(err)
	}
	if res.Bin != filepath.Join(env.Paths.Engines, "demo", "b1", "llama-b1", "llama-server") || res.LedgerID == 0 {
		t.Fatalf("res = %+v", res)
	}
	if target, err := os.Readlink(filepath.Join(res.Dir, "llama-b1", "libggml.dylib")); err != nil || target != "libggml.0.dylib" {
		t.Errorf("symlink = %q %v", target, err)
	}
	// Scratch dir disposed, install dir still temp until health check releases it.
	rows, _ := env.Ledger.List(context.Background(), ledger.StateTemp)
	if len(rows) != 1 || rows[0].Kind != ledger.KindEngineInstall {
		t.Fatalf("temp rows = %+v", rows)
	}
	if entries, _ := os.ReadDir(filepath.Join(env.Paths.Tmp, "install")); len(entries) != 0 {
		t.Errorf("scratch left behind: %v", entries)
	}
}

func TestPrebuiltRejectsBadChecksum(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) { w.Write([]byte("evil")) }))
	defer srv.Close()
	env := testEnv(t)
	env.Downloader.Retries = 0
	m := def.InstallMethod{Kind: "prebuilt", URL: srv.URL + "/x.tar.gz", SHA256: strings.Repeat("a", 64), Bin: "x"}
	if _, err := (prebuilt{}).Install(context.Background(), env, m); err == nil {
		t.Fatal("expected checksum failure")
	}
	if _, err := os.Stat(env.Dir()); !os.IsNotExist(err) {
		t.Error("install dir created despite checksum failure")
	}
}

func TestExtractRejectsTraversal(t *testing.T) {
	dir := t.TempDir()
	for name, entries := range map[string][]entry{
		"dotdot":  {{name: "../evil", body: "x", mode: 0o644}},
		"symlink": {{name: "a/link", link: "../../etc/passwd"}},
		"abslink": {{name: "link", link: "/etc/passwd"}},
	} {
		archive := filepath.Join(dir, name+".tar.gz")
		os.WriteFile(archive, tarGz(t, entries), 0o644)
		if err := Extract(archive, filepath.Join(dir, "out-"+name)); err == nil {
			t.Errorf("%s: expected rejection", name)
		}
	}
}

func TestAdoptMinVersion(t *testing.T) {
	bin := filepath.Join(t.TempDir(), "fakeengine")
	os.WriteFile(bin, []byte("#!/bin/sh\necho 'Warning: could not connect'\necho 'client version is 0.21.2'\n"), 0o755)
	t.Setenv("PATH", filepath.Dir(bin)+":"+os.Getenv("PATH"))
	env := testEnv(t)

	res, err := adopt{}.Install(context.Background(), env, def.InstallMethod{Kind: "adopt", BinName: "fakeengine", MinVersion: "0.6.0"})
	if err != nil || res.Version != "0.21.2" || res.Dir != "" {
		t.Fatalf("res = %+v err = %v", res, err)
	}
	if _, err := (adopt{}).Install(context.Background(), env, def.InstallMethod{Kind: "adopt", BinName: "fakeengine", MinVersion: "1.0"}); err == nil {
		t.Fatal("expected too-old error")
	}
}

func TestKindsCoverShippedDefinitions(t *testing.T) {
	for _, k := range []string{"prebuilt", "source", "uv_venv", "adopt"} {
		if _, ok := Kinds()[k]; !ok {
			t.Errorf("kind %s not registered", k)
		}
	}
}
