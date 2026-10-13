// Package home owns the layout of KEPLER_HOME (default ~/.kepler) and the global lock.
package home

import (
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"
)

// EnvHome overrides the default home directory.
const EnvHome = "KEPLER_HOME"

// Paths is every location Kepler reads or writes. All paths are absolute.
type Paths struct {
	Root      string
	Bin       string
	Engines   string
	Canary    string
	EnginesD  string
	Registry  string
	Config    string
	State     string
	Ledger    string
	Lock      string
	Tmp       string
	Downloads string
	Models    string
	Ollama    string
	Logs      string
	Results   string
}

// Resolve returns the home root from flag, then KEPLER_HOME, then ~/.kepler.
func Resolve(flag string) (string, error) {
	root := flag
	if root == "" {
		root = os.Getenv(EnvHome)
	}
	if root == "" {
		h, err := os.UserHomeDir()
		if err != nil {
			return "", fmt.Errorf("locate user home: %w", err)
		}
		root = filepath.Join(h, ".kepler")
	}
	if strings.HasPrefix(root, "~/") {
		h, err := os.UserHomeDir()
		if err != nil {
			return "", fmt.Errorf("locate user home: %w", err)
		}
		root = filepath.Join(h, root[2:])
	}
	abs, err := filepath.Abs(root)
	if err != nil {
		return "", err
	}
	return abs, nil
}

// New builds the path set for root without touching the filesystem.
func New(root string) Paths {
	j := func(p ...string) string { return filepath.Join(append([]string{root}, p...)...) }
	return Paths{
		Root:      root,
		Bin:       j("bin"),
		Engines:   j("engines"),
		Canary:    j("canary"),
		EnginesD:  j("engines.d"),
		Registry:  j("registry.json"),
		Config:    j("config.yaml"),
		State:     j("state"),
		Ledger:    j("state", "ledger.db"),
		Lock:      j("state", "kepler.lock"),
		Tmp:       j("tmp"),
		Downloads: j("tmp", "downloads"),
		Models:    j("tmp", "models"),
		Ollama:    j("tmp", "ollama"),
		Logs:      j("logs"),
		Results:   j("results"),
	}
}

// Ensure creates every directory in the layout.
func (p Paths) Ensure() error {
	for _, d := range []string{p.Root, p.Bin, p.Engines, p.Canary, p.EnginesD, p.State,
		p.Tmp, p.Downloads, p.Models, p.Ollama, p.Logs, p.Results} {
		if err := os.MkdirAll(d, 0o755); err != nil {
			return fmt.Errorf("create %s: %w", d, err)
		}
	}
	return nil
}

// ErrOutsideHome is returned when a path that would be deleted is not inside Root.
var ErrOutsideHome = errors.New("path is outside KEPLER_HOME")

// Contains reports whether path is strictly inside Root (never Root itself).
func (p Paths) Contains(path string) bool {
	abs, err := filepath.Abs(path)
	if err != nil {
		return false
	}
	rel, err := filepath.Rel(p.Root, abs)
	if err != nil || rel == "." || rel == ".." || strings.HasPrefix(rel, ".."+string(filepath.Separator)) {
		return false
	}
	return !filepath.IsAbs(rel)
}

// SafeRemoveAll deletes path only if it is inside Root. Missing paths are not an error.
func (p Paths) SafeRemoveAll(path string) error {
	if !p.Contains(path) {
		return fmt.Errorf("%w: %s", ErrOutsideHome, path)
	}
	return os.RemoveAll(path)
}
