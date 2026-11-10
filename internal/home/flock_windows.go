//go:build windows

package home

import (
	"errors"
	"os"
)

var errWouldBlock = errors.New("lock held")

// Windows is not a v1 target; locking is a no-op so the code compiles.
func tryFlock(f *os.File) error { return nil }

func unflock(f *os.File) error { return nil }
