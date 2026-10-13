package install

import (
	"archive/tar"
	"archive/zip"
	"compress/gzip"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
)

// Extract unpacks a .tar.gz/.tgz/.zip archive into dest. Entries (and symlink targets)
// that would land outside dest are rejected.
func Extract(archive, dest string) error {
	switch {
	case strings.HasSuffix(archive, ".tar.gz"), strings.HasSuffix(archive, ".tgz"):
		return extractTarGz(archive, dest)
	case strings.HasSuffix(archive, ".zip"):
		return extractZip(archive, dest)
	}
	return fmt.Errorf("unsupported archive type: %s", filepath.Base(archive))
}

// within resolves name under dest and reports whether it stays inside.
func within(dest, name string) (string, error) {
	p := filepath.Join(dest, name)
	rel, err := filepath.Rel(dest, p)
	if err != nil || rel == ".." || strings.HasPrefix(rel, ".."+string(filepath.Separator)) {
		return "", fmt.Errorf("archive entry escapes destination: %s", name)
	}
	return p, nil
}

func checkLink(dest, linkPath, target string) error {
	if filepath.IsAbs(target) {
		return fmt.Errorf("archive symlink with absolute target: %s -> %s", linkPath, target)
	}
	resolved := filepath.Join(filepath.Dir(linkPath), target)
	rel, err := filepath.Rel(dest, resolved)
	if err != nil || rel == ".." || strings.HasPrefix(rel, ".."+string(filepath.Separator)) {
		return fmt.Errorf("archive symlink escapes destination: %s -> %s", linkPath, target)
	}
	return nil
}

func extractTarGz(archive, dest string) error {
	f, err := os.Open(archive)
	if err != nil {
		return err
	}
	defer f.Close()
	gz, err := gzip.NewReader(f)
	if err != nil {
		return err
	}
	defer gz.Close()
	tr := tar.NewReader(gz)
	for {
		h, err := tr.Next()
		if err == io.EOF {
			return nil
		}
		if err != nil {
			return err
		}
		p, err := within(dest, h.Name)
		if err != nil {
			return err
		}
		switch h.Typeflag {
		case tar.TypeDir:
			if err := os.MkdirAll(p, 0o755); err != nil {
				return err
			}
		case tar.TypeReg:
			if err := writeFile(p, tr, os.FileMode(h.Mode)&0o777); err != nil {
				return err
			}
		case tar.TypeSymlink:
			if err := checkLink(dest, p, h.Linkname); err != nil {
				return err
			}
			if err := os.MkdirAll(filepath.Dir(p), 0o755); err != nil {
				return err
			}
			_ = os.Remove(p)
			if err := os.Symlink(h.Linkname, p); err != nil {
				return err
			}
		default:
			// hard links, devices, etc. are not expected in engine releases; skip
		}
	}
}

func extractZip(archive, dest string) error {
	zr, err := zip.OpenReader(archive)
	if err != nil {
		return err
	}
	defer zr.Close()
	for _, zf := range zr.File {
		p, err := within(dest, zf.Name)
		if err != nil {
			return err
		}
		mode := zf.Mode()
		switch {
		case mode.IsDir():
			if err := os.MkdirAll(p, 0o755); err != nil {
				return err
			}
		case mode&os.ModeSymlink != 0:
			rc, err := zf.Open()
			if err != nil {
				return err
			}
			target, err := io.ReadAll(io.LimitReader(rc, 4096))
			rc.Close()
			if err != nil {
				return err
			}
			if err := checkLink(dest, p, string(target)); err != nil {
				return err
			}
			if err := os.MkdirAll(filepath.Dir(p), 0o755); err != nil {
				return err
			}
			_ = os.Remove(p)
			if err := os.Symlink(string(target), p); err != nil {
				return err
			}
		default:
			rc, err := zf.Open()
			if err != nil {
				return err
			}
			err = writeFile(p, rc, mode.Perm())
			rc.Close()
			if err != nil {
				return err
			}
		}
	}
	return nil
}

func writeFile(p string, r io.Reader, mode os.FileMode) error {
	if err := os.MkdirAll(filepath.Dir(p), 0o755); err != nil {
		return err
	}
	if mode == 0 {
		mode = 0o644
	}
	f, err := os.OpenFile(p, os.O_CREATE|os.O_WRONLY|os.O_TRUNC, mode)
	if err != nil {
		return err
	}
	if _, err := io.Copy(f, r); err != nil {
		f.Close()
		return err
	}
	return f.Close()
}
