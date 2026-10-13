//go:build darwin

package proc

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"unsafe"

	"github.com/ebitengine/purego"
)

// On macOS the memory figure that matches Activity Monitor's "Memory" column is
// phys_footprint from proc_pid_rusage. Unlike RSS it includes GPU (Metal) allocations
// on unified memory, which is where model weights and KV cache live.

const (
	rusageInfoV4      = 4
	physFootprintOffs = 72  // uuid[16] + 7 × uint64 precede ri_phys_footprint
	rusageBufSize     = 512 // rusage_info_v4 is ~296 bytes
)

var (
	libOnce          sync.Once
	libErr           error
	procPidRusage    func(pid int32, flavor int32, buf unsafe.Pointer) int32
	procListPgrpPids func(pgrp int32, buf unsafe.Pointer, size int32) int32
	procPidInfo      func(pid int32, flavor int32, arg uint64, buf unsafe.Pointer, size int32) int32
)

func loadLib() error {
	libOnce.Do(func() {
		h, err := purego.Dlopen("/usr/lib/libSystem.B.dylib", purego.RTLD_NOW|purego.RTLD_GLOBAL)
		if err != nil {
			libErr = err
			return
		}
		purego.RegisterLibFunc(&procPidRusage, h, "proc_pid_rusage")
		purego.RegisterLibFunc(&procListPgrpPids, h, "proc_listpgrppids")
		purego.RegisterLibFunc(&procPidInfo, h, "proc_pidinfo")
	})
	return libErr
}

// Footprint returns the physical footprint of one process in bytes.
func Footprint(pid int) (int64, error) {
	if err := loadLib(); err != nil {
		return 0, err
	}
	var buf [rusageBufSize]byte
	if rc := procPidRusage(int32(pid), rusageInfoV4, unsafe.Pointer(&buf[0])); rc != 0 {
		return 0, fmt.Errorf("proc_pid_rusage(%d) failed", pid)
	}
	return int64(*(*uint64)(unsafe.Pointer(&buf[physFootprintOffs]))), nil
}

// GroupPIDs lists the processes in a process group.
func GroupPIDs(pgid int) ([]int, error) {
	if err := loadLib(); err != nil {
		return nil, err
	}
	var buf [1024]int32
	n := procListPgrpPids(int32(pgid), unsafe.Pointer(&buf[0]), int32(len(buf)*4))
	if n < 0 {
		return nil, fmt.Errorf("proc_listpgrppids(%d) failed", pgid)
	}
	// libproc returns the number of pids on some releases and bytes on others; both
	// are handled by bounding to the buffer and skipping zero entries.
	count := int(n)
	if count > len(buf) {
		count = count / 4
	}
	if count > len(buf) {
		count = len(buf)
	}
	var out []int
	for _, p := range buf[:count] {
		if p > 0 {
			out = append(out, int(p))
		}
	}
	return out, nil
}

// GroupFootprint sums Footprint over every process in the group.
func GroupFootprint(pgid int) (int64, error) {
	pids, err := GroupPIDs(pgid)
	if err != nil {
		return 0, err
	}
	if len(pids) == 0 {
		pids = []int{pgid}
	}
	var total int64
	for _, p := range pids {
		if f, err := Footprint(p); err == nil {
			total += f
		}
	}
	return total, nil
}

// FootprintMethod names the metric for result files.
const FootprintMethod = "phys_footprint+mapped_model_files"

// struct proc_regionwithpathinfo layout (sys/proc_info.h): proc_regioninfo (96 bytes)
// followed by vnode_info (152 bytes) and the path (MAXPATHLEN).
const (
	procPidRegionPathInfo = 8
	regionInfoSize        = 96 + 152 + 1024
	offPagesResident      = 36
	offAddress            = 80
	offSize               = 88
	offPath               = 96 + 152
)

// MappedResident sums resident bytes of file-backed regions whose path starts with
// prefix. Engines that mmap model weights (llama.cpp, Ollama) hold them as clean
// file-backed pages, which phys_footprint does not count.
func MappedResident(pid int, prefix string) (int64, error) {
	if err := loadLib(); err != nil {
		return 0, err
	}
	if resolved, err := filepath.EvalSymlinks(prefix); err == nil {
		prefix = resolved // the kernel reports real paths (/tmp → /private/tmp)
	}
	page := int64(os.Getpagesize())
	var buf [regionInfoSize]byte
	var total int64
	addr := uint64(0)
	for i := 0; i < 1<<20; i++ {
		n := procPidInfo(int32(pid), procPidRegionPathInfo, addr, unsafe.Pointer(&buf[0]), regionInfoSize)
		if n <= 0 {
			break
		}
		start := *(*uint64)(unsafe.Pointer(&buf[offAddress]))
		size := *(*uint64)(unsafe.Pointer(&buf[offSize]))
		if size == 0 {
			break
		}
		path := cString(buf[offPath:])
		if path != "" && strings.HasPrefix(path, prefix) {
			total += int64(*(*uint32)(unsafe.Pointer(&buf[offPagesResident]))) * page
		}
		addr = start + size
	}
	return total, nil
}

// GroupMappedResident sums MappedResident over a process group.
func GroupMappedResident(pgid int, prefix string) (int64, error) {
	pids, err := GroupPIDs(pgid)
	if err != nil {
		return 0, err
	}
	if len(pids) == 0 {
		pids = []int{pgid}
	}
	var total int64
	for _, p := range pids {
		if m, err := MappedResident(p, prefix); err == nil {
			total += m
		}
	}
	return total, nil
}

func cString(b []byte) string {
	for i, c := range b {
		if c == 0 {
			return string(b[:i])
		}
	}
	return string(b)
}
