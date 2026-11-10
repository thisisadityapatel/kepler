package proc

import (
	"path/filepath"
	"strings"
	"testing"
	"time"
)

func TestStopKillsWholeGroup(t *testing.T) {
	log := filepath.Join(t.TempDir(), "p.log")
	// The leader ignores SIGTERM and spawns a child; Stop must still clear the group.
	p, err := Start(Spec{Path: "/bin/sh", Args: []string{"-c", "trap '' TERM; sleep 60 & echo started; wait"}, LogPath: log})
	if err != nil {
		t.Fatal(err)
	}
	time.Sleep(200 * time.Millisecond)
	start := time.Now()
	if err := p.Stop(500 * time.Millisecond); err != nil {
		t.Fatal(err)
	}
	if groupAlive(p.PGID) {
		t.Fatal("group still alive")
	}
	if time.Since(start) > 10*time.Second {
		t.Fatal("stop took too long")
	}
	if !strings.Contains(p.LogTail(5), "started") {
		t.Errorf("log tail = %q", p.LogTail(5))
	}
}

func TestFreePort(t *testing.T) {
	p, err := FreePort()
	if err != nil || p <= 0 {
		t.Fatalf("port = %d, %v", p, err)
	}
}
