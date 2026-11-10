// Package registry persists what is installed and healthy on this machine
// (~/.kepler/registry.json). The resolver remains the source of truth for eligibility.
package registry

import (
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"time"

	"github.com/thisisadityapatel/kepler/internal/engines/resolve"
	"github.com/thisisadityapatel/kepler/internal/probe"
)

// SchemaVersion of registry.json.
const SchemaVersion = 1

// Engine statuses.
const (
	StatusReady        = "ready"
	StatusUnhealthy    = "installed_unhealthy"
	StatusNotInstalled = "eligible_not_installed"
	StatusIneligible   = "ineligible"
	StatusSkipped      = "user_skipped"
)

// Registry is registry.json.
type Registry struct {
	Schema            int                `json:"schema"`
	KeplerVersion     string             `json:"kepler_version"`
	DefinitionsSchema int                `json:"definitions_schema"`
	ProbedAt          time.Time          `json:"probed_at"`
	Host              probe.HostFacts    `json:"host"`
	Engines           map[string]*Engine `json:"engines"`
}

// Engine is one engine's install and health record.
type Engine struct {
	Status  string   `json:"status"`
	Reason  string   `json:"reason,omitempty"`
	Source  string   `json:"source,omitempty"`
	Target  string   `json:"target,omitempty"`
	Version string   `json:"version,omitempty"`
	Install *Install `json:"install,omitempty"`
	Formats []string `json:"formats,omitempty"`
	Health  *Health  `json:"health,omitempty"`
}

// Install records how and where the engine was installed.
type Install struct {
	Kind string `json:"kind"`
	Path string `json:"path"`           // absolute path of the server binary
	Dir  string `json:"dir,omitempty"`  // install root owned by Kepler ("" for adopted)
	Venv string `json:"venv,omitempty"` // uv_venv installs
	At   string `json:"at"`
}

// Health is the last health-check outcome.
type Health struct {
	OK              bool    `json:"ok"`
	At              string  `json:"at"`
	LoadS           float64 `json:"load_s,omitempty"`
	DecodeTokPerS   float64 `json:"decode_tok_per_s,omitempty"`
	Error           string  `json:"error,omitempty"`
	LogPath         string  `json:"log_path,omitempty"`
	ReportedVersion string  `json:"reported_version,omitempty"`
}

// Load reads the registry; a missing file yields an empty registry.
func Load(path string) (*Registry, error) {
	r := &Registry{Schema: SchemaVersion, Engines: map[string]*Engine{}}
	b, err := os.ReadFile(path)
	if errors.Is(err, os.ErrNotExist) {
		return r, nil
	}
	if err != nil {
		return nil, err
	}
	if err := json.Unmarshal(b, r); err != nil {
		return nil, fmt.Errorf("%s: %w", path, err)
	}
	if r.Schema != SchemaVersion {
		return nil, fmt.Errorf("%s: unsupported schema %d", path, r.Schema)
	}
	if r.Engines == nil {
		r.Engines = map[string]*Engine{}
	}
	return r, nil
}

// Save writes atomically (temp file + rename).
func (r *Registry) Save(path string) error {
	b, err := json.MarshalIndent(r, "", "  ")
	if err != nil {
		return err
	}
	tmp, err := os.CreateTemp(filepath.Dir(path), ".registry-*.json")
	if err != nil {
		return err
	}
	if _, err := tmp.Write(append(b, '\n')); err != nil {
		tmp.Close()
		os.Remove(tmp.Name())
		return err
	}
	if err := tmp.Close(); err != nil {
		os.Remove(tmp.Name())
		return err
	}
	return os.Rename(tmp.Name(), path)
}

// Installed converts registry state into the resolver's input.
func (r *Registry) Installed() map[string]resolve.InstalledEngine {
	out := map[string]resolve.InstalledEngine{}
	for name, e := range r.Engines {
		ie := resolve.InstalledEngine{Version: e.Version, TargetID: e.Target}
		switch e.Status {
		case StatusReady:
			ie.Status = resolve.InstallReady
		case StatusUnhealthy:
			ie.Status = resolve.InstallUnhealthy
		case StatusSkipped:
			ie.Status = resolve.InstallSkipped
		default:
			continue
		}
		ie.Adopted = e.Install != nil && e.Install.Kind == "adopt"
		out[name] = ie
	}
	return out
}
