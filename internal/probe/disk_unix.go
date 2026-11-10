//go:build unix

package probe

import (
	"path/filepath"
	"syscall"
)

// freeDisk reports free bytes on the filesystem holding path, walking up to the
// nearest existing ancestor (KEPLER_HOME may not exist yet on first run).
func freeDisk(path string) int64 {
	for p := path; ; p = filepath.Dir(p) {
		var st syscall.Statfs_t
		if err := syscall.Statfs(p, &st); err == nil {
			return int64(st.Bavail) * int64(st.Bsize)
		}
		if p == filepath.Dir(p) {
			return 0
		}
	}
}
