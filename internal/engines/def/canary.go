package def

import (
	"bytes"
	"fmt"
	"io/fs"

	"gopkg.in/yaml.v3"
)

// Canary is the tiny model used to health-check engines of one format.
type Canary struct {
	Repo     string       `yaml:"repo"`
	Revision string       `yaml:"revision"`
	Entry    string       `yaml:"entry"` // file passed to the engine; "" = the directory (MLX)
	Files    []CanaryFile `yaml:"files"`
}

// CanaryFile is one pinned file. LFS files carry sha256; small git files carry git_oid
// (the git blob sha1, which is what Hugging Face reports for non-LFS files).
type CanaryFile struct {
	Path   string `yaml:"path"`
	Size   int64  `yaml:"size"`
	SHA256 string `yaml:"sha256"`
	GitOID string `yaml:"git_oid"`
}

// Canaries maps format → canary.
type Canaries map[string]Canary

// SizeBytes is the total download size of the canary.
func (c Canary) SizeBytes() int64 {
	var n int64
	for _, f := range c.Files {
		n += f.Size
	}
	return n
}

// LoadCanaries strictly decodes canaries.yaml from fsys.
func LoadCanaries(fsys fs.FS, name string) (Canaries, error) {
	b, err := fs.ReadFile(fsys, name)
	if err != nil {
		return nil, err
	}
	dec := yaml.NewDecoder(bytes.NewReader(b))
	dec.KnownFields(true)
	var doc struct {
		Schema   int      `yaml:"schema"`
		Canaries Canaries `yaml:"canaries"`
	}
	if err := dec.Decode(&doc); err != nil {
		return nil, fmt.Errorf("%s: %w", name, err)
	}
	if doc.Schema != SchemaVersion {
		return nil, fmt.Errorf("%s: unsupported schema %d", name, doc.Schema)
	}
	for format, c := range doc.Canaries {
		if c.Repo == "" || len(c.Revision) != 40 || len(c.Files) == 0 {
			return nil, fmt.Errorf("%s: canary %q needs repo, a 40-char revision, and files", name, format)
		}
		for _, f := range c.Files {
			if f.Size <= 0 || (len(f.SHA256) != 64 && len(f.GitOID) != 40) {
				return nil, fmt.Errorf("%s: canary %q file %q needs size and sha256 or git_oid", name, format, f.Path)
			}
		}
	}
	return doc.Canaries, nil
}

// Formats returns the set of formats with a canary.
func (c Canaries) Formats() map[string]bool {
	out := map[string]bool{}
	for f := range c {
		out[f] = true
	}
	return out
}
