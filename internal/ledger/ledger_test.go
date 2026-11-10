package ledger

import (
	"context"
	"errors"
	"os"
	"os/exec"
	"path/filepath"
	"syscall"
	"testing"
	"time"

	"github.com/thisisadityapatel/kepler/internal/home"
)

func open(t *testing.T) (*Ledger, home.Paths) {
	t.Helper()
	p := home.New(t.TempDir())
	if err := p.Ensure(); err != nil {
		t.Fatal(err)
	}
	l, err := Open(p)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { l.Close() })
	return l, p
}

// deadPID returns the pid of a process that has already exited.
func deadPID(t *testing.T) int {
	t.Helper()
	cmd := exec.Command("true")
	if err := cmd.Run(); err != nil {
		t.Fatal(err)
	}
	return cmd.Process.Pid
}

func TestReapDeletesArtifactsOfDeadOwners(t *testing.T) {
	ctx := context.Background()
	l, p := open(t)

	orphan := filepath.Join(p.Models, "run1", "model.gguf")
	mine := filepath.Join(p.Models, "run2", "model.gguf")
	for _, f := range []string{orphan, mine} {
		_ = os.MkdirAll(filepath.Dir(f), 0o755)
		_ = os.WriteFile(f, []byte("x"), 0o644)
	}
	if _, err := l.Record(ctx, Artifact{RunID: "run1", Kind: KindDir, Path: filepath.Dir(orphan), OwnerPID: deadPID(t)}); err != nil {
		t.Fatal(err)
	}
	if _, err := l.Record(ctx, Artifact{RunID: "run2", Kind: KindDir, Path: filepath.Dir(mine)}); err != nil {
		t.Fatal(err)
	}

	rep, err := l.Reap(ctx, false)
	if err != nil || len(rep.Errors) > 0 {
		t.Fatal(err, rep.Errors)
	}
	if len(rep.Reaped) != 1 || rep.Reaped[0].RunID != "run1" {
		t.Fatalf("reaped = %+v", rep.Reaped)
	}
	if _, err := os.Stat(orphan); !os.IsNotExist(err) {
		t.Error("orphan survived")
	}
	if _, err := os.Stat(mine); err != nil {
		t.Error("live owner's artifact was deleted")
	}
}

func TestRecordRefusesPathsOutsideHome(t *testing.T) {
	l, _ := open(t)
	_, err := l.Record(context.Background(), Artifact{RunID: "r", Kind: KindFile, Path: "/etc/hosts"})
	if !errors.Is(err, home.ErrOutsideHome) {
		t.Fatalf("err = %v", err)
	}
}

func TestReapKillsOrphanProcessGroup(t *testing.T) {
	ctx := context.Background()
	l, _ := open(t)
	cmd := exec.Command("sleep", "60")
	cmd.SysProcAttr = &syscall.SysProcAttr{Setpgid: true}
	if err := cmd.Start(); err != nil {
		t.Fatal(err)
	}
	done := make(chan struct{})
	go func() { _ = cmd.Wait(); close(done) }()

	id, _ := l.Record(ctx, Artifact{RunID: "r", Kind: KindProcessGroup, OwnerPID: deadPID(t)})
	_ = l.SetPGID(ctx, id, cmd.Process.Pid)
	if _, err := l.Reap(ctx, false); err != nil {
		t.Fatal(err)
	}
	select {
	case <-done:
	case <-time.After(5 * time.Second):
		_ = cmd.Process.Kill()
		t.Fatal("orphan process group still running")
	}
}

func TestKeptAndReleased(t *testing.T) {
	ctx := context.Background()
	l, p := open(t)
	kept := filepath.Join(p.Models, "kept")
	rel := filepath.Join(p.Engines, "llamacpp")
	_ = os.MkdirAll(kept, 0o755)
	_ = os.MkdirAll(rel, 0o755)
	dead := deadPID(t)
	k, _ := l.Record(ctx, Artifact{RunID: "r", Kind: KindDir, Path: kept, OwnerPID: dead})
	r, _ := l.Record(ctx, Artifact{RunID: "r", Kind: KindEngineInstall, Path: rel, OwnerPID: dead})
	_ = l.Keep(ctx, k)
	_ = l.Release(ctx, r)

	if rep, _ := l.Reap(ctx, false); len(rep.Reaped) != 0 {
		t.Fatalf("normal reap touched kept/released: %+v", rep.Reaped)
	}
	if rep, _ := l.Reap(ctx, true); len(rep.Reaped) != 1 || rep.Reaped[0].ID != k {
		t.Fatalf("reap --all: %+v", rep.Reaped)
	}
	if _, err := os.Stat(rel); err != nil {
		t.Error("released engine install was deleted")
	}
}
