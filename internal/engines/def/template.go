package def

import (
	"fmt"
	"regexp"
	"sort"
)

// TemplateVars are the only placeholders allowed in definition strings.
var TemplateVars = map[string]string{
	"bin":            "absolute path of the installed engine binary",
	"port":           "free ephemeral port bound to 127.0.0.1",
	"ctx":            "context length in tokens",
	"model_path":     "absolute path of the model file or directory",
	"mmproj_path":    "absolute path of the vision projector file",
	"run_id":         "id of the current run",
	"kepler_home":    "KEPLER_HOME",
	"version":        "engine definition version",
	"venv":           "engine virtualenv directory (uv_venv installs)",
	"modelfile_path": "path of the generated Ollama Modelfile",
}

var placeholderRe = regexp.MustCompile(`\{([a-z_]+)\}`)

// Placeholders returns the variable names referenced in s.
func Placeholders(s string) []string {
	var out []string
	for _, m := range placeholderRe.FindAllStringSubmatch(s, -1) {
		out = append(out, m[1])
	}
	return out
}

// Expand substitutes {var} placeholders. A referenced variable missing from vars is an error.
func Expand(s string, vars map[string]string) (string, error) {
	var missing []string
	out := placeholderRe.ReplaceAllStringFunc(s, func(m string) string {
		name := m[1 : len(m)-1]
		v, ok := vars[name]
		if !ok {
			missing = append(missing, name)
			return m
		}
		return v
	})
	if len(missing) > 0 {
		sort.Strings(missing)
		return "", fmt.Errorf("template %q: no value for %v", s, missing)
	}
	return out, nil
}

// ExpandAll expands every string in ss.
func ExpandAll(ss []string, vars map[string]string) ([]string, error) {
	out := make([]string, len(ss))
	for i, s := range ss {
		v, err := Expand(s, vars)
		if err != nil {
			return nil, err
		}
		out[i] = v
	}
	return out, nil
}
