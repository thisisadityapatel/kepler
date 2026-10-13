package download

import (
	"context"
	"crypto/sha1"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"sync/atomic"
	"testing"
)

func sha(b []byte) string { s := sha256.Sum256(b); return hex.EncodeToString(s[:]) }

func gitOID(b []byte) string {
	h := sha1.New()
	fmt.Fprintf(h, "blob %d\x00", len(b))
	h.Write(b)
	return hex.EncodeToString(h.Sum(nil))
}

// rangeServer serves body honoring Range requests and counts bytes sent.
func rangeServer(body []byte, sent *int64) *httptest.Server {
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		start := 0
		if rg := r.Header.Get("Range"); rg != "" {
			start, _ = strconv.Atoi(strings.TrimSuffix(strings.TrimPrefix(rg, "bytes="), "-"))
			w.Header().Set("Content-Range", fmt.Sprintf("bytes %d-%d/%d", start, len(body)-1, len(body)))
			w.WriteHeader(http.StatusPartialContent)
		}
		n, _ := w.Write(body[start:])
		atomic.AddInt64(sent, int64(n))
	}))
}

func TestFetchResumesPartial(t *testing.T) {
	body := []byte(strings.Repeat("kepler", 10000))
	var sent int64
	srv := rangeServer(body, &sent)
	defer srv.Close()

	dir := t.TempDir()
	c := New(filepath.Join(dir, "parts"))
	r := Request{URL: srv.URL, Dest: filepath.Join(dir, "out", "m.gguf"), Size: int64(len(body)), Digest: Digest{SHA256: sha(body)}}

	_ = os.MkdirAll(c.PartsDir, 0o755)
	half := len(body) / 2
	if err := os.WriteFile(c.PartPath(r), body[:half], 0o644); err != nil {
		t.Fatal(err)
	}
	if err := c.Fetch(context.Background(), r, nil); err != nil {
		t.Fatal(err)
	}
	got, _ := os.ReadFile(r.Dest)
	if string(got) != string(body) {
		t.Fatal("content mismatch")
	}
	if sent != int64(len(body)-half) {
		t.Errorf("sent %d bytes, want %d (resume)", sent, len(body)-half)
	}
	if _, err := os.Stat(c.PartPath(r)); !os.IsNotExist(err) {
		t.Error("part file left behind")
	}
	// Already present and valid → no network.
	sent = 0
	if err := c.Fetch(context.Background(), r, nil); err != nil || sent != 0 {
		t.Fatalf("refetch: err=%v sent=%d", err, sent)
	}
}

func TestFetchGitOIDAndMismatch(t *testing.T) {
	body := []byte(`{"model_type":"llama"}`)
	var sent int64
	srv := rangeServer(body, &sent)
	defer srv.Close()
	dir := t.TempDir()
	c := New(filepath.Join(dir, "parts"))
	c.Retries = 0

	ok := Request{URL: srv.URL, Dest: filepath.Join(dir, "config.json"), Size: int64(len(body)), Digest: Digest{GitOID: gitOID(body)}}
	if err := c.Fetch(context.Background(), ok, nil); err != nil {
		t.Fatal(err)
	}

	bad := Request{URL: srv.URL, Dest: filepath.Join(dir, "bad.json"), Size: int64(len(body)), Digest: Digest{SHA256: strings.Repeat("0", 64)}}
	if err := c.Fetch(context.Background(), bad, nil); !errors.Is(err, ErrDigestMismatch) {
		t.Fatalf("err = %v", err)
	}
	if _, err := os.Stat(bad.Dest); !os.IsNotExist(err) {
		t.Error("unverified file was moved into place")
	}
}

func TestFetchRequiresDigest(t *testing.T) {
	c := New(t.TempDir())
	if err := c.Fetch(context.Background(), Request{URL: "http://x", Dest: "/tmp/x"}, nil); err == nil {
		t.Fatal("expected error")
	}
}
