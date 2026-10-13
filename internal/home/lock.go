package home

import (
	"errors"
	"fmt"
	"os"
	"strconv"
	"strings"
)

// Lock is an exclusive advisory lock on Paths.Lock. Only one Kepler command that
// mutates engines or runs benchmarks may hold it at a time.
type Lock struct {
	f *os.File
}

// ErrLocked means another Kepler process holds the lock.
type ErrLocked struct{ PID int }

func (e ErrLocked) Error() string {
	if e.PID > 0 {
		return fmt.Sprintf("another kepler command is running (pid %d)", e.PID)
	}
	return "another kepler command is running"
}

// TryLock acquires the lock without blocking and records our pid in the file.
func (p Paths) TryLock() (*Lock, error) {
	if err := os.MkdirAll(p.State, 0o755); err != nil {
		return nil, err
	}
	f, err := os.OpenFile(p.Lock, os.O_RDWR|os.O_CREATE, 0o644)
	if err != nil {
		return nil, err
	}
	if err := tryFlock(f); err != nil {
		defer f.Close()
		if errors.Is(err, errWouldBlock) {
			b, _ := os.ReadFile(p.Lock)
			pid, _ := strconv.Atoi(strings.TrimSpace(string(b)))
			return nil, ErrLocked{PID: pid}
		}
		return nil, err
	}
	if err := f.Truncate(0); err == nil {
		_, _ = f.WriteAt([]byte(strconv.Itoa(os.Getpid())+"\n"), 0)
	}
	return &Lock{f: f}, nil
}

// Unlock releases the lock.
func (l *Lock) Unlock() error {
	if l == nil || l.f == nil {
		return nil
	}
	_ = l.f.Truncate(0)
	_ = unflock(l.f)
	err := l.f.Close()
	l.f = nil
	return err
}
