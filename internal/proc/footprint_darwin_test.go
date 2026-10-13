//go:build darwin

package proc

import (
	"os"
	"syscall"
	"testing"
	"time"
)

func TestFootprintOfSelfAndGroup(t *testing.T) {
	self, err := Footprint(os.Getpid())
	if err != nil || self <= 0 {
		t.Fatalf("self footprint = %d, %v", self, err)
	}
	// A shell that allocates ~64 MB in a child keeps the group's footprint well above a bare shell.
	p, err := Start(Spec{Path: "/bin/sh", Args: []string{"-c", "sleep 30 & /usr/bin/perl -e '$x = \"a\" x (64*1024*1024); sleep 30'"}})
	if err != nil {
		t.Fatal(err)
	}
	defer p.Stop(time.Second)
	var pids []int
	var total int64
	deadline := time.Now().Add(5 * time.Second)
	for time.Now().Before(deadline) {
		pids, _ = GroupPIDs(p.PGID)
		total, _ = GroupFootprint(p.PGID)
		if len(pids) >= 3 && total > 60<<20 {
			break
		}
		time.Sleep(100 * time.Millisecond)
	}
	if len(pids) < 3 {
		t.Fatalf("group pids = %v, want sh + sleep + perl", pids)
	}
	if total < 60<<20 {
		t.Fatalf("group footprint = %d, want > 60 MiB", total)
	}
}

func TestMappedResidentCountsMappedFiles(t *testing.T) {
	dir := t.TempDir()
	f, err := os.Create(dir + "/weights.bin")
	if err != nil {
		t.Fatal(err)
	}
	const size = 32 << 20
	if _, err := f.Write(make([]byte, size)); err != nil { // real bytes: holes in sparse files are never resident
		t.Fatal(err)
	}
	data, err := syscall.Mmap(int(f.Fd()), 0, size, syscall.PROT_READ, syscall.MAP_SHARED)
	if err != nil {
		t.Fatal(err)
	}
	defer syscall.Munmap(data)
	var sum byte
	for i := 0; i < size; i += os.Getpagesize() {
		sum += data[i] // touch every page so it is resident
	}
	_ = sum
	got, err := MappedResident(os.Getpid(), dir)
	if err != nil {
		t.Fatal(err)
	}
	if got < size*9/10 || got > size {
		t.Fatalf("mapped resident = %d, want ~%d", got, size)
	}
	if other, _ := MappedResident(os.Getpid(), "/nonexistent-prefix"); other != 0 {
		t.Errorf("unrelated prefix counted %d bytes", other)
	}
}
