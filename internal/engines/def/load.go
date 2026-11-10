package def

import (
	"bytes"
	"errors"
	"fmt"
	"io/fs"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strings"

	"gopkg.in/yaml.v3"
)

// ValidateOptions supplies the registries a definition is checked against. They are
// passed in (not imported) so this package stays independent of install/adapter code.
type ValidateOptions struct {
	// InstallKinds maps each registered install kind to its field validator.
	InstallKinds map[string]func(InstallMethod) error
	// Adapters is the set of registered adapter names ("generic" plus hook sets).
	Adapters map[string]bool
	// CanaryFormats is the set of formats that have a health-check canary.
	CanaryFormats map[string]bool
}

// Set is the loaded, validated definitions keyed by name.
type Set struct {
	Defs map[string]*EngineDef
}

// Sorted returns definitions in a stable order: default tier first, then by name.
func (s Set) Sorted() []*EngineDef {
	out := make([]*EngineDef, 0, len(s.Defs))
	for _, d := range s.Defs {
		out = append(out, d)
	}
	rank := map[string]int{TierDefault: 0, TierOptional: 1, TierLazy: 2}
	sort.Slice(out, func(i, j int) bool {
		if rank[out[i].Tier] != rank[out[j].Tier] {
			return rank[out[i].Tier] < rank[out[j].Tier]
		}
		return out[i].Name < out[j].Name
	})
	return out
}

// Load reads builtin definitions from fsys (every *.yaml at its root) and then user
// definitions from userDir (may be empty or missing). A user definition with the same
// name replaces the builtin one.
func Load(fsys fs.FS, userDir string, opt ValidateOptions) (Set, error) {
	set := Set{Defs: map[string]*EngineDef{}}
	var errs []error

	builtin, err := fs.Glob(fsys, "*.yaml")
	if err != nil {
		return set, err
	}
	sort.Strings(builtin)
	for _, name := range builtin {
		b, err := fs.ReadFile(fsys, name)
		if err != nil {
			errs = append(errs, err)
			continue
		}
		d, err := Parse(b, "builtin:"+name, opt)
		if err != nil {
			errs = append(errs, err)
			continue
		}
		d.Source = SourceBuiltin
		if prev, dup := set.Defs[d.Name]; dup {
			errs = append(errs, fmt.Errorf("%s: duplicate engine name %q (also in %s)", d.File, d.Name, prev.File))
			continue
		}
		set.Defs[d.Name] = d
	}

	if userDir != "" {
		user, _ := filepath.Glob(filepath.Join(userDir, "*.yaml"))
		sort.Strings(user)
		for _, path := range user {
			b, err := os.ReadFile(path)
			if err != nil {
				errs = append(errs, err)
				continue
			}
			d, err := Parse(b, path, opt)
			if err != nil {
				errs = append(errs, err)
				continue
			}
			d.Source = SourceUser
			set.Defs[d.Name] = d
		}
	}
	return set, errors.Join(errs...)
}

// Parse strictly decodes and validates one definition. file names the source in errors.
func Parse(b []byte, file string, opt ValidateOptions) (*EngineDef, error) {
	dec := yaml.NewDecoder(bytes.NewReader(b))
	dec.KnownFields(true)
	var d EngineDef
	if err := dec.Decode(&d); err != nil {
		return nil, fmt.Errorf("%s: %w", file, err)
	}
	d.File = file
	if err := Validate(&d, opt); err != nil {
		return nil, err
	}
	return &d, nil
}

var nameRe = regexp.MustCompile(`^[a-z0-9][a-z0-9_-]*$`)

// Validate checks semantic rules strict decoding cannot express. All problems are
// reported together, each prefixed with file and field.
func Validate(d *EngineDef, opt ValidateOptions) error {
	var errs []error
	bad := func(field, format string, a ...any) {
		errs = append(errs, fmt.Errorf("%s: %s: %s", d.File, field, fmt.Sprintf(format, a...)))
	}

	if d.Schema != SchemaVersion {
		bad("schema", "unsupported schema %d (this kepler understands %d)", d.Schema, SchemaVersion)
	}
	if !nameRe.MatchString(d.Name) {
		bad("name", "must match %s, got %q", nameRe, d.Name)
	}
	if d.Display == "" {
		bad("display", "required")
	}
	if len(d.Formats) == 0 {
		bad("formats", "at least one format required")
	}
	switch d.Tier {
	case TierDefault, TierOptional, TierLazy:
	default:
		bad("tier", "must be default, optional or lazy, got %q", d.Tier)
	}
	if d.Adapter == "" {
		bad("adapter", "required (use \"generic\" for YAML-only engines)")
	} else if opt.Adapters != nil && !opt.Adapters[d.Adapter] {
		bad("adapter", "no adapter registered as %q", d.Adapter)
	}
	if d.Request.API != "openai" {
		bad("request.api", "only \"openai\" is supported, got %q", d.Request.API)
	}
	checkCaps(d.Caps, "caps", bad)

	if len(d.Targets) > 0 {
		if opt.CanaryFormats != nil {
			for _, f := range d.Formats {
				if !opt.CanaryFormats[f] {
					bad("formats", "no health-check canary for format %q", f)
				}
			}
		}
		if d.Version == "" {
			bad("version", "required when targets are defined")
		}
	}

	checkLaunch(d.Launch, "launch", true, bad)
	checkTemplates(d.Request.Model, "request.model", bad)
	if d.Materialize != nil {
		checkTemplates(d.Materialize.Modelfile, "materialize.modelfile", bad)
		for _, s := range d.Materialize.Cmd {
			checkTemplates(s, "materialize.cmd", bad)
		}
	}

	ids := map[string]bool{}
	for i, t := range d.Targets {
		f := fmt.Sprintf("targets[%d]", i)
		if t.ID == "" {
			bad(f+".id", "required")
		} else if ids[t.ID] {
			bad(f+".id", "duplicate target id %q", t.ID)
		}
		ids[t.ID] = true
		checkMatch(t.Match, f+".match", bad)
		if len(t.Install) == 0 {
			bad(f+".install", "at least one install method required")
		}
		for j, m := range t.Install {
			mf := fmt.Sprintf("%s.install[%d]", f, j)
			validate, ok := opt.InstallKinds[m.Kind]
			if opt.InstallKinds != nil && !ok {
				bad(mf+".kind", "unknown install kind %q", m.Kind)
				continue
			}
			if validate != nil {
				if err := validate(m); err != nil {
					bad(mf, "%v", err)
				}
			}
			for _, s := range []string{m.URL, m.Bin, m.Ref} {
				checkTemplates(s, mf, bad)
			}
			for _, s := range m.Packages {
				checkTemplates(s, mf+".packages", bad)
			}
		}
		if t.Launch != nil {
			checkLaunch(*t.Launch, f+".launch", false, bad)
		}
		checkCaps(t.Caps, f+".caps", bad)
	}
	for i, u := range d.Unsupported {
		f := fmt.Sprintf("unsupported[%d]", i)
		checkMatch(u.Match, f+".match", bad)
		if u.Reason == "" {
			bad(f+".reason", "required")
		}
	}
	return errors.Join(errs...)
}

func checkCaps(caps map[string]bool, field string, bad func(string, string, ...any)) {
	for k := range caps {
		if _, ok := KnownCaps[k]; !ok {
			bad(field, "unknown capability %q", k)
		}
	}
}

func checkLaunch(l Launch, field string, top bool, bad func(string, string, ...any)) {
	if top {
		if l.Cmd == "" {
			bad(field+".cmd", "required")
		}
		if l.Ready.HTTPGet == "" {
			bad(field+".ready.http_get", "required")
		}
	}
	checkTemplates(l.Cmd, field+".cmd", bad)
	for _, a := range l.Args {
		checkTemplates(a, field+".args", bad)
	}
	for k, args := range l.ArgsIf {
		if _, ok := TemplateVars[k+"_path"]; !ok {
			bad(field+".args_if", "unknown condition %q", k)
		}
		for _, a := range args {
			checkTemplates(a, field+".args_if."+k, bad)
		}
	}
	for k, v := range l.Env {
		checkTemplates(v, field+".env."+k, bad)
	}
}

func checkTemplates(s, field string, bad func(string, string, ...any)) {
	for _, p := range Placeholders(s) {
		if _, ok := TemplateVars[p]; !ok {
			bad(field, "unknown template variable {%s}", p)
		}
	}
	if strings.Contains(s, "<pin>") || strings.Contains(s, "<verify>") {
		bad(field, "unresolved placeholder in %q", s)
	}
}

func checkMatch(m Match, field string, bad func(string, string, ...any)) {
	for _, c := range []struct{ name, v string }{
		{"os_version", m.OSVersion},
		{"accelerator_api", m.AcceleratorAPI},
		{"compute_capability", m.ComputeCapability},
	} {
		if c.v == "" {
			continue
		}
		if _, err := ParseConstraint(c.v); err != nil {
			bad(field+"."+c.name, "%v", err)
		}
	}
}
