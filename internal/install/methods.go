package install

import (
	"context"
	"fmt"
	"os"
	"os/exec"
	"path"
	"path/filepath"
	"regexp"
	"runtime"
	"strconv"
	"strings"

	"github.com/thisisadityapatel/kepler/internal/download"
	"github.com/thisisadityapatel/kepler/internal/engines/def"
	"github.com/thisisadityapatel/kepler/internal/probe"
	"github.com/thisisadityapatel/kepler/internal/proc"
)

var sha256Re = regexp.MustCompile(`^[0-9a-f]{64}$`)

// ---------------------------------------------------------------- prebuilt

// prebuilt downloads a release archive, verifies its sha256, and extracts it.
type prebuilt struct{}

func (prebuilt) Kind() string { return "prebuilt" }

func (prebuilt) Validate(m def.InstallMethod) error {
	switch {
	case m.URL == "":
		return fmt.Errorf("prebuilt: url required")
	case !sha256Re.MatchString(m.SHA256):
		return fmt.Errorf("prebuilt: sha256 must be 64 lowercase hex chars")
	case m.Bin == "":
		return fmt.Errorf("prebuilt: bin required")
	}
	return nil
}

func (prebuilt) Install(ctx context.Context, env Env, m def.InstallMethod) (Result, error) {
	url, err := def.Expand(m.URL, env.Vars())
	if err != nil {
		return Result{}, err
	}
	binRel, err := def.Expand(m.Bin, env.Vars())
	if err != nil {
		return Result{}, err
	}
	work, workID, err := scratch(ctx, env, "prebuilt")
	if err != nil {
		return Result{}, err
	}
	defer env.Ledger.Dispose(context.WithoutCancel(ctx), workID)

	archive := filepath.Join(work, path.Base(url))
	step(env, "downloading %s", url)
	err = env.Downloader.Fetch(ctx, download.Request{URL: url, Dest: archive, Digest: download.Digest{SHA256: m.SHA256}},
		progressPublisher(env))
	if err != nil {
		return Result{}, err
	}

	dir := env.Dir()
	id, err := claimDir(ctx, env, dir, "prebuilt "+env.Version)
	if err != nil {
		return Result{LedgerID: id}, err
	}
	step(env, "extracting into %s", dir)
	if err := Extract(archive, dir); err != nil {
		return Result{LedgerID: id}, err
	}
	stripQuarantine(ctx, dir)
	bin := filepath.Join(dir, binRel)
	if err := checkExecutable(bin); err != nil {
		return Result{LedgerID: id}, err
	}
	return Result{Kind: "prebuilt", Bin: bin, Dir: dir, Version: env.Version, LedgerID: id}, nil
}

// stripQuarantine removes the macOS quarantine flag so Gatekeeper doesn't block
// downloaded binaries. Downloads made by Go don't normally get it; this is defensive.
func stripQuarantine(ctx context.Context, dir string) {
	if runtime.GOOS == "darwin" {
		_ = exec.CommandContext(ctx, "xattr", "-dr", "com.apple.quarantine", dir).Run()
	}
}

// ---------------------------------------------------------------- source

// source clones a git ref and builds it with cmake. Used as a fallback when a
// prebuilt binary fails its health check.
type source struct{}

func (source) Kind() string { return "source" }

func (source) Validate(m def.InstallMethod) error {
	switch {
	case m.Repo == "":
		return fmt.Errorf("source: repo required")
	case m.Ref == "":
		return fmt.Errorf("source: ref required")
	case m.Bin == "":
		return fmt.Errorf("source: bin required")
	}
	return nil
}

// needHints tells the user how to get a missing build prerequisite.
var needHints = map[string]string{
	"xcode_clt": "install with: xcode-select --install",
	"cmake":     "install with: brew install cmake",
	"git":       "install with: xcode-select --install",
}

func (source) Install(ctx context.Context, env Env, m def.InstallMethod) (Result, error) {
	for _, n := range m.Needs {
		if _, ok := env.Facts.Tools[n]; !ok {
			return Result{}, fmt.Errorf("source build needs %s (%s)", n, needHints[n])
		}
	}
	ref, err := def.Expand(m.Ref, env.Vars())
	if err != nil {
		return Result{}, err
	}
	binRel, err := def.Expand(m.Bin, env.Vars())
	if err != nil {
		return Result{}, err
	}
	work, workID, err := scratch(ctx, env, "source")
	if err != nil {
		return Result{}, err
	}
	defer env.Ledger.Dispose(context.WithoutCancel(ctx), workID)

	src := filepath.Join(work, "src")
	build := filepath.Join(src, "build")
	run := func(name string, args ...string) error {
		return proc.Run(ctx, proc.Spec{Path: name, Args: args, LogPath: env.LogPath})
	}
	step(env, "cloning %s at %s", m.Repo, ref)
	if err := run("git", "clone", "--depth", "1", "--branch", ref, m.Repo, src); err != nil {
		return Result{}, err
	}
	step(env, "configuring with cmake")
	if err := run("cmake", append([]string{"-S", src, "-B", build}, m.CMake...)...); err != nil {
		return Result{}, err
	}
	step(env, "building (this can take several minutes)")
	args := []string{"--build", build, "--config", "Release", "-j", strconv.Itoa(runtime.NumCPU())}
	for _, t := range m.Targets {
		args = append(args, "--target", t)
	}
	if err := run("cmake", args...); err != nil {
		return Result{}, err
	}

	dir := env.Dir()
	id, err := claimDir(ctx, env, dir, "source "+env.Version)
	if err != nil {
		return Result{LedgerID: id}, err
	}
	// Copy the whole build/bin dir: shared-library builds keep dylibs next to binaries.
	if err := run("cp", "-R", filepath.Join(build, "bin"), filepath.Join(dir, "bin")); err != nil {
		return Result{LedgerID: id}, err
	}
	bin := filepath.Join(dir, binRel)
	if err := checkExecutable(bin); err != nil {
		return Result{LedgerID: id}, err
	}
	return Result{Kind: "source", Bin: bin, Dir: dir, Version: env.Version, LedgerID: id}, nil
}

// ---------------------------------------------------------------- uv_venv

// uvVenv creates an isolated virtualenv with a Kepler-managed uv and Python, so the
// user needs no Python installation. Everything lands inside KEPLER_HOME.
type uvVenv struct{}

func (uvVenv) Kind() string { return "uv_venv" }

func (uvVenv) Validate(m def.InstallMethod) error {
	switch {
	case m.Python == "":
		return fmt.Errorf("uv_venv: python required")
	case len(m.Packages) == 0:
		return fmt.Errorf("uv_venv: packages required")
	case m.Bin == "":
		return fmt.Errorf("uv_venv: bin required")
	}
	return nil
}

func (uvVenv) Install(ctx context.Context, env Env, m def.InstallMethod) (Result, error) {
	uv, err := ensureUV(ctx, env)
	if err != nil {
		return Result{}, err
	}
	pkgs, err := def.ExpandAll(m.Packages, env.Vars())
	if err != nil {
		return Result{}, err
	}
	binRel, err := def.Expand(m.Bin, env.Vars())
	if err != nil {
		return Result{}, err
	}
	cacheDir, cacheID, err := scratch(ctx, env, "uv-cache")
	if err != nil {
		return Result{}, err
	}
	defer env.Ledger.Dispose(context.WithoutCancel(ctx), cacheID)

	dir := env.Dir()
	id, err := claimDir(ctx, env, dir, "uv_venv "+env.Version)
	if err != nil {
		return Result{LedgerID: id}, err
	}
	venv := filepath.Join(dir, "venv")
	uvEnv := map[string]string{
		"UV_PYTHON_INSTALL_DIR": filepath.Join(env.Paths.Engines, "_python"),
		"UV_CACHE_DIR":          cacheDir,
		"UV_PYTHON_PREFERENCE":  "only-managed",
		"UV_NO_CONFIG":          "1",
	}
	run := func(args ...string) error {
		return proc.Run(ctx, proc.Spec{Path: uv, Args: args, Env: uvEnv, LogPath: env.LogPath})
	}
	step(env, "creating Python %s virtualenv", m.Python)
	if err := run("venv", "--python", m.Python, venv); err != nil {
		return Result{LedgerID: id}, err
	}
	step(env, "installing %s", strings.Join(pkgs, " "))
	if err := run(append([]string{"pip", "install", "--python", filepath.Join(venv, "bin", "python")}, pkgs...)...); err != nil {
		return Result{LedgerID: id}, err
	}
	bin := filepath.Join(venv, binRel)
	if err := checkExecutable(bin); err != nil {
		return Result{LedgerID: id}, err
	}
	return Result{Kind: "uv_venv", Bin: bin, Dir: dir, Venv: venv, Version: env.Version, LedgerID: id}, nil
}

// Pinned uv used for Python engines. Kepler never relies on a uv the user installed.
var uvReleases = map[string]struct{ URL, SHA256, Inner string }{
	"darwin/arm64": {
		URL:    "https://github.com/astral-sh/uv/releases/download/0.13.0/uv-aarch64-apple-darwin.tar.gz",
		SHA256: "a9c1b29002cf3c83f07fa9cd8a887a3be0107d90e23189721221e7257db8e3d6",
		Inner:  "uv-aarch64-apple-darwin/uv",
	},
}

func ensureUV(ctx context.Context, env Env) (string, error) {
	uv := filepath.Join(env.Paths.Bin, "uv")
	if checkExecutable(uv) == nil {
		return uv, nil
	}
	rel, ok := uvReleases[env.Facts.OS+"/"+env.Facts.Arch]
	if !ok {
		return "", fmt.Errorf("no pinned uv for %s/%s", env.Facts.OS, env.Facts.Arch)
	}
	work, workID, err := scratch(ctx, env, "uv")
	if err != nil {
		return "", err
	}
	defer env.Ledger.Dispose(context.WithoutCancel(ctx), workID)
	step(env, "downloading uv")
	archive := filepath.Join(work, path.Base(rel.URL))
	if err := env.Downloader.Fetch(ctx, download.Request{URL: rel.URL, Dest: archive, Digest: download.Digest{SHA256: rel.SHA256}}, nil); err != nil {
		return "", err
	}
	if err := Extract(archive, work); err != nil {
		return "", err
	}
	b, err := os.ReadFile(filepath.Join(work, rel.Inner))
	if err != nil {
		return "", err
	}
	if err := os.MkdirAll(env.Paths.Bin, 0o755); err != nil {
		return "", err
	}
	tmp := uv + ".tmp"
	if err := os.WriteFile(tmp, b, 0o755); err != nil {
		return "", err
	}
	return uv, os.Rename(tmp, uv)
}

// ---------------------------------------------------------------- adopt

// adopt reuses a binary already on PATH. Kepler never modifies or removes it.
type adopt struct{}

func (adopt) Kind() string { return "adopt" }

func (adopt) Validate(m def.InstallMethod) error {
	if m.BinName == "" {
		return fmt.Errorf("adopt: bin_name required")
	}
	if m.MinVersion != "" {
		if _, err := def.ParseConstraint(m.MinVersion); err != nil {
			return fmt.Errorf("adopt: %w", err)
		}
	}
	return nil
}

func (adopt) Install(ctx context.Context, env Env, m def.InstallMethod) (Result, error) {
	bin, err := exec.LookPath(m.BinName)
	if err != nil {
		return Result{}, fmt.Errorf("%s not found on PATH", m.BinName)
	}
	bin, _ = filepath.Abs(bin)
	version := ToolVersion(ctx, bin)
	if m.MinVersion != "" {
		c, _ := def.ParseConstraint(m.MinVersion)
		if !c.Satisfied(version) {
			return Result{}, fmt.Errorf("%s %s is older than required %s", bin, orUnknown(version), m.MinVersion)
		}
	}
	step(env, "adopting %s (version %s)", bin, orUnknown(version))
	return Result{Kind: "adopt", Bin: bin, Version: version}, nil
}

// ToolVersion runs `bin --version` and extracts a version number. Lines mentioning
// "version" are preferred, last first (Ollama prints connection warnings before it).
func ToolVersion(ctx context.Context, bin string) string {
	cmd := exec.CommandContext(ctx, bin, "--version")
	cmd.Env = append(os.Environ(), "OLLAMA_HOST=127.0.0.1:1")
	out, _ := cmd.CombinedOutput()
	lines := strings.Split(strings.TrimSpace(string(out)), "\n")
	for i := len(lines) - 1; i >= 0; i-- {
		if strings.Contains(strings.ToLower(lines[i]), "version") {
			if v := probe.ParseVersion(lines[i]); v != "" {
				return v
			}
		}
	}
	return probe.ParseVersion(string(out))
}

func orUnknown(s string) string {
	if s == "" {
		return "unknown"
	}
	return s
}
