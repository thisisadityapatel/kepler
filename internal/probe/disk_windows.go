//go:build windows

package probe

// freeDisk is not implemented until the Windows prober lands.
func freeDisk(path string) int64 { return 0 }
