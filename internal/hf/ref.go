// Package hf resolves Hugging Face model references into concrete, pinned files:
// parse the user's input, fetch repo metadata, classify the format, group GGUF files
// into quantization variants, and read architecture info without a full download.
package hf

import (
	"fmt"
	"net/url"
	"regexp"
	"strings"
)

// Ref is a parsed model reference.
type Ref struct {
	Repo     string // org/name
	Revision string // branch, tag, or commit; "" = main
	File     string // exact file path inside the repo, if given
	Quant    string // quant selector such as Q4_K_M, if given
}

func (r Ref) String() string {
	s := r.Repo
	if r.Revision != "" {
		s += "@" + r.Revision
	}
	if r.File != "" {
		s += "/" + r.File
	}
	if r.Quant != "" {
		s += ":" + r.Quant
	}
	return s
}

var repoRe = regexp.MustCompile(`^[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*$`)

// ParseRef accepts:
//
//	org/repo
//	org/repo:Q4_K_M
//	org/repo@revision
//	https://huggingface.co/org/repo[/tree/<rev>]
//	https://huggingface.co/org/repo/blob/<rev>/path/file.gguf   (also /resolve/)
//	hf.co/org/repo...
func ParseRef(s string) (Ref, error) {
	in := strings.TrimSpace(s)
	if in == "" {
		return Ref{}, fmt.Errorf("empty model reference")
	}
	var r Ref

	if strings.Contains(in, "://") || strings.HasPrefix(in, "huggingface.co/") || strings.HasPrefix(in, "hf.co/") {
		if !strings.Contains(in, "://") {
			in = "https://" + in
		}
		u, err := url.Parse(in)
		if err != nil {
			return Ref{}, fmt.Errorf("invalid URL %q: %w", s, err)
		}
		host := strings.TrimPrefix(u.Host, "www.")
		if host != "huggingface.co" && host != "hf.co" {
			return Ref{}, fmt.Errorf("unsupported host %q: only huggingface.co links are supported", u.Host)
		}
		parts := strings.Split(strings.Trim(u.Path, "/"), "/")
		if len(parts) < 2 {
			return Ref{}, fmt.Errorf("URL %q does not name a repository (expected huggingface.co/org/repo)", s)
		}
		if parts[0] == "datasets" || parts[0] == "spaces" {
			return Ref{}, fmt.Errorf("%q is a %s link, not a model", s, parts[0])
		}
		r.Repo = parts[0] + "/" + parts[1]
		rest := parts[2:]
		if len(rest) >= 2 {
			switch rest[0] {
			case "tree":
				r.Revision = rest[1]
			case "blob", "resolve":
				r.Revision = rest[1]
				r.File = strings.Join(rest[2:], "/")
			}
		}
		if r.Revision == "main" {
			r.Revision = ""
		}
	} else {
		repo := in
		if i := strings.LastIndex(repo, ":"); i > 0 {
			r.Quant = strings.ToUpper(repo[i+1:])
			repo = repo[:i]
		}
		if i := strings.Index(repo, "@"); i > 0 {
			r.Revision = repo[i+1:]
			repo = repo[:i]
		}
		r.Repo = repo
	}

	if !repoRe.MatchString(r.Repo) {
		return Ref{}, fmt.Errorf("invalid model reference %q (expected org/repo or a huggingface.co link)", s)
	}
	if r.File != "" && !strings.HasSuffix(strings.ToLower(r.File), ".gguf") {
		return Ref{}, fmt.Errorf("file links must point at a .gguf file, got %q", r.File)
	}
	return r, nil
}
