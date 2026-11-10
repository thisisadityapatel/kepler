package home

import (
	"errors"
	"os"
	"path/filepath"
	"testing"
)

func TestContains(t *testing.T) {
	p := New("/tmp/kh")
	cases := map[string]bool{
		"/tmp/kh":                false,
		"/tmp/kh/tmp/models/x":   true,
		"/tmp/kh/../etc":         false,
		"/tmp/khx/file":          false,
		"/etc/passwd":            false,
		"/tmp/kh/tmp/../results": true,
	}
	for path, want := range cases {
		if got := p.Contains(path); got != want {
			t.Errorf("Contains(%q) = %v, want %v", path, got, want)
		}
	}
}

func TestSafeRemoveAllRefusesOutside(t *testing.T) {
	root := t.TempDir()
	outside := t.TempDir()
	victim := filepath.Join(outside, "keep")
	if err := os.WriteFile(victim, []byte("x"), 0o644); err != nil {
		t.Fatal(err)
	}
	p := New(root)
	if err := p.SafeRemoveAll(victim); !errors.Is(err, ErrOutsideHome) {
		t.Fatalf("expected ErrOutsideHome, got %v", err)
	}
	if _, err := os.Stat(victim); err != nil {
		t.Fatalf("victim was deleted: %v", err)
	}
}

func TestLockIsExclusive(t *testing.T) {
	p := New(t.TempDir())
	l1, err := p.TryLock()
	if err != nil {
		t.Fatal(err)
	}
	if _, err := p.TryLock(); err == nil {
		t.Fatal("second lock should fail")
	} else {
		var le ErrLocked
		if !errors.As(err, &le) || le.PID != os.Getpid() {
			t.Fatalf("unexpected error %v", err)
		}
	}
	if err := l1.Unlock(); err != nil {
		t.Fatal(err)
	}
	l2, err := p.TryLock()
	if err != nil {
		t.Fatalf("relock after unlock: %v", err)
	}
	_ = l2.Unlock()
}
