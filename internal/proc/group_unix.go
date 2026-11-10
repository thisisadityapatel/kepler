//go:build unix

package proc

import (
	"errors"
	"syscall"
)

func newGroupAttr() *syscall.SysProcAttr { return &syscall.SysProcAttr{Setpgid: true} }

func terminateGroup(pgid int) { _ = syscall.Kill(-pgid, syscall.SIGTERM) }

// KillGroup sends SIGKILL to every process in the group.
func KillGroup(pgid int) {
	if pgid > 1 {
		_ = syscall.Kill(-pgid, syscall.SIGKILL)
	}
}

func groupAlive(pgid int) bool {
	err := syscall.Kill(-pgid, 0)
	return err == nil || errors.Is(err, syscall.EPERM)
}

// Alive reports whether pid exists.
func Alive(pid int) bool {
	if pid <= 0 {
		return false
	}
	err := syscall.Kill(pid, 0)
	return err == nil || errors.Is(err, syscall.EPERM)
}
