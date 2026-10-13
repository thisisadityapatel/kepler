//go:build windows

package proc

import (
	"os"
	"syscall"
)

// Windows is not a v1 target. These keep the code compiling; a real implementation
// would use job objects to contain child processes.

func newGroupAttr() *syscall.SysProcAttr { return &syscall.SysProcAttr{} }

func terminateGroup(pgid int) { KillGroup(pgid) }

// KillGroup kills the leader process.
func KillGroup(pgid int) {
	if p, err := os.FindProcess(pgid); err == nil {
		_ = p.Kill()
	}
}

func groupAlive(pgid int) bool { return false }

// Alive reports whether pid exists.
func Alive(pid int) bool {
	p, err := os.FindProcess(pid)
	if err != nil {
		return false
	}
	_ = p
	return true
}
