//go:build unix

package home

import (
	"os"
	"syscall"
)

var errWouldBlock = syscall.EWOULDBLOCK

func tryFlock(f *os.File) error { return syscall.Flock(int(f.Fd()), syscall.LOCK_EX|syscall.LOCK_NB) }

func unflock(f *os.File) error { return syscall.Flock(int(f.Fd()), syscall.LOCK_UN) }
