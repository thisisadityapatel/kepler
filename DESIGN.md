# Kepler 2.0 — Design (Benchmarking, macOS)

Status: draft v1 · Scope: benchmarking CLI on Apple Silicon · Language: Go

---

## 1. Summary

Kepler is a single-binary CLI that benchmarks a Hugging Face model across every
inference engine viable on the user's Mac, then writes one JSON result per engine
and leaves no model files behind.

```
curl -fsSL <install-url> | sh        # install + kepler init
kepler bench unsloth/Qwen3-8B-GGUF   # resolve → download → bench each engine → JSON → cleanup
kepler results                       # view past runs
```

### Goals (v1)

- One-command install on Apple Silicon macOS.
- `kepler init` detects hardware and installs viable engines, chosen by matching per-engine YAML definitions against detected hardware.
- `kepler bench <hf-link>` resolves the model (GGUF first, MLX counterpart when available),
  checks it fits, benchmarks each engine one at a time under a fixed, fair spec.
- Results saved as JSON (evolution of the current `perf/*.json` format).
- Every downloaded or created artifact is deleted after the run — including after crashes.

### Non-goals (v1)

- Inference / serving / gateway (separate plan).
- NVIDIA, AMD, Linux, Windows, Intel Macs. Engine definitions and the probe are designed to
  extend to them; nothing for them is implemented.
- Output quality / accuracy evaluation. Kepler measures speed and memory only.
- Concurrency (multi-stream) benchmarks — deferred to v1.1.
- Energy measurement (`powermetrics` requires sudo) — deferred.
- Result sharing / leaderboard.

### v1 engines

All three are installed by default and benchmarked on every `kepler bench`:

| Engine | Format | Install | Tier |
|---|---|---|---|
| llama.cpp (`llama-server`) | GGUF | prebuilt release (Metal), source build fallback | default |
| MLX (`mlx_lm.server`) | MLX | `uv`-managed venv | default |
| Ollama | GGUF | adopt existing install, else prebuilt | default |

vLLM (vllm-metal) and SGLang (MLX backend) are experimental on Apple Silicon. Their engine
definitions ship with no macOS target, so the engine resolver reports them as `ineligible`
with a reason until targets are added.

### Extensibility principle

Kepler core has **no hard-coded knowledge of any engine or platform**. Engines are YAML
definition files; platforms are detected facts. Adding an engine, or adding NVIDIA/AMD support
to an existing engine, means adding YAML (plus a hardware detector the first time a new
accelerator family appears) — not changing the resolver, bench runner, or results code (§7).

### Module boundaries

Benchmarking and (future) inference are separate modules that share one engine layer:

```
            ┌─────────────────────┐      ┌──────────────────────────────┐
            │  bench  (v1)        │      │  inference  (future, own     │
            │  scenarios, stats,  │      │  design: serve, gateway,     │
            │  results, report    │      │  daemon, menu bar monitor)   │
            └──────────┬──────────┘      └──────────────┬───────────────┘
                       │                                │
   ════════════════════╪════════ shared core ═══════════╪════════════════════
                       ▼                                ▼
   probe · engine definitions · resolver · install · registry · adapters
   (launch / ready / stop) · hf resolver · download · ledger + reaper · proc
```

Rules:
- The shared core **never imports** `bench` (or future `inference`). Nothing in the engine
  layer knows what a scenario, preset, or result file is.
- An engine `ready` in the registry is usable by both modules: `init` installs once, and
  inference later launches the same binaries through the same adapters with the same
  definitions. Benchmark results are an *input* inference may use (e.g. "pick the fastest
  engine"), never a dependency.
- Module-specific behavior lives in the module: bench builds its own requests (§8.5) and owns
  the fairness rules; inference will own gateway, auth, and long-running lifecycle.
- The core reports progress as structured events on an in-process event stream (engine
  launched, ready, memory sample, stopped). `bench` subscribes for its terminal UI today;
  a future daemon can forward the same events to a menu bar app.

---

## 2. Target platform

- Apple Silicon (arm64), macOS ≥ 14.0 (verify MLX's current minimum at implementation).
- Intel Macs: `kepler init` exits with `unsupported: Kepler v1 requires Apple Silicon`.
- No dependency on system Python, Homebrew, Xcode, git, or cmake for the default path.
  The source-build fallback for llama.cpp requires Xcode CLT + cmake; Kepler detects and
  reports this instead of failing opaquely.

---

## 3. CLI surface

```
kepler init      [--yes] [--engines llamacpp,mlx,ollama] [--skip ollama]
kepler doctor                                   # re-probe, re-health-check, report problems
kepler engines   list | install <name> | remove <name>
kepler bench     <model> [flags]
kepler results   list | show <file|id> | compare <comparison_id>
kepler clean     [--all]                        # run reaper; --all also removes canaries/logs
kepler version
```

`kepler bench` flags:

| Flag | Default | Meaning |
|---|---|---|
| `--preset` | `standard` | `quick` \| `standard` \| `full` (§9.4) |
| `--engines` | all ready | allowlist, e.g. `llamacpp,mlx` |
| `--skip` | — | denylist |
| `--formats` | `auto` | `gguf`, `mlx`, or `gguf,mlx`; `auto` = given format + discovered counterparts |
| `--quant` | auto | e.g. `Q4_K_M`, `8bit`; overrides quant selection |
| `--scenarios` | from preset | e.g. `chat,prefill-8k` |
| `--iterations` | from preset | override |
| `--cooldown` | `10s` | pause between engines |
| `--keep` | false | do not delete model files after run |
| `--yes` / `-y` | false | accept resolver choices + confirmation prompts |
| `--json` | false | machine-readable stdout (paths of written results) |

Global flags: `--home` (env `KEPLER_HOME`, default `~/.kepler`), `--verbose`, `--no-color`.

Exit codes: `0` ok · `1` generic error · `2` usage · `3` preflight failed (doesn't fit) ·
`4` no eligible engine · `5` partial (some engines failed; results written for the rest) ·
`130` interrupted.

---

## 4. Home directory layout

```
~/.kepler/
  bin/
    kepler                      # the binary
    uv                          # downloaded by init; used for Python engines
  engines/
    llamacpp/b8971/             # extracted prebuilt (or source build output)
    mlx/venv/                   # uv venv with mlx-lm pinned
    ollama/<version>/           # only if Kepler installed it (adopted installs are not here)
  canary/                       # tiny health-check models, kept permanently
  engines.d/                    # optional user/community engine definitions (*.yaml, §7.1)
  registry.json                 # what is installed + health (written by init/doctor)
  config.yaml                   # optional user overrides (default preset, cooldown)
  state/
    ledger.db                   # artifact ledger (SQLite)
    kepler.lock                 # global flock: one bench at a time
  tmp/
    downloads/                  # *.part files during resumable download
    models/<run_id>/            # downloaded model files for the active run
    ollama/<run_id>/            # private OLLAMA_MODELS dir for the active run
  logs/<run_id>/<engine>.log    # engine stdout/stderr; keep last 20 runs
  results/                      # JSON results (never auto-deleted)
```

Kepler does **not** use `~/.cache/huggingface` or the user's `~/.ollama/models`. Owning every
path is what makes cleanup guaranteed.

---

## 5. Install flow

```
install.sh ─► download kepler-darwin-arm64 + .sha256 from GitHub Releases
           ─► verify checksum ─► ~/.kepler/bin/kepler
           ─► append PATH line to ~/.zshrc (idempotent, marked "# kepler")
           ─► exec kepler init
```

- `install.sh` is POSIX sh, refuses non-arm64 / non-Darwin with a clear message.
- Re-running the installer upgrades the binary in place; it never touches `results/`.

### 5.1 `kepler init`

```
probe (HostFacts) ─► load engine definitions ─► engine resolver ─► show plan (sizes) ─► confirm
      ─► install (prebuilt → fallback) ─► health check (canary) ─► write registry.json
```

1. **Probe** hardware into `HostFacts` (§6). Abort only if no detector recognizes the host.
2. **Resolve**: run the engine resolver (§7.3) over every engine definition → per engine,
   the matched target or `ineligible(reason)`.
3. **Plan**: list eligible engines by tier with download size and disk estimate.
   `default` tier pre-selected (on macOS v1: llama.cpp, MLX, Ollama), `optional` asked,
   ineligible shown with reason ("vLLM: no target for darwin/arm64/metal").
4. **Install** each selected engine using the first install method that succeeds
   (§7.2). Each install step is recorded in the ledger as `state=temp` until it passes
   the health check, so an interrupted init is rolled back on the next run.
5. **Health check**: launch the engine with its canary model, wait for ready, run one
   16-token completion, record version, load time, tok/s. Failure → try next install
   method (e.g. source build) → else mark `installed_unhealthy` with the log path.
6. **Write `registry.json`** atomically (write temp + rename).

`init` is idempotent: re-running it skips engines already `ready` at the definition's version.

---

## 6. Hardware probe

The probe produces a platform-neutral `HostFacts` struct. Engine definitions match against
these facts only — never against OS-specific commands.

```go
type HostFacts struct {
    OS           string        // darwin | linux | windows
    OSVersion    string        // "26.0", "22.04", "10.0.22631"
    Arch         string        // arm64 | amd64
    CPU          CPUInfo       // brand, performance/efficiency cores
    RAMBytes     int64
    Accelerators []Accelerator // zero or more
    FreeDiskBytes int64
    Power        PowerInfo     // source, low-power mode (laptops)
    Tools        map[string]string // name → version: ollama, cmake, xcode_clt, git, docker…
}

type Accelerator struct {
    Kind        string // metal | cuda | rocm | vulkan | cpu
    Vendor      string // apple | nvidia | amd | intel
    Name        string // "Apple M4 Pro", "NVIDIA GeForce RTX 4090"
    MemBytes    int64  // usable GPU memory (wired limit on Apple; VRAM elsewhere)
    Unified     bool   // shares system RAM (Apple Silicon)
    APIVersion  string // Metal family, CUDA driver version, ROCm version
    ComputeCap  string // e.g. "8.9" for CUDA; "" if n/a
    Cores       int
}
```

Implementation is split into:
- **OS probers** (`probe_darwin.go`, later `probe_linux.go`, `probe_windows.go`) selected by
  Go build tags; fill OS, CPU, RAM, disk, power, tools.
- **Accelerator detectors**, a registered list each returning zero or more `Accelerator`s:
  `metal` (v1), later `cuda` (`nvidia-smi`/NVML), `rocm` (`rocm-smi`/`amd-smi`),
  `vulkan` (`vulkaninfo`). A `cpu` accelerator is always appended as the fallback.

Adding NVIDIA support = adding the `cuda` detector. Nothing downstream changes.

### 6.1 macOS prober (v1)

| Field | Source |
|---|---|
| chip | `sysctl -n machdep.cpu.brand_string` (e.g. `Apple M4 Pro`) |
| model id | `sysctl -n hw.model` |
| P / E cores | `hw.perflevel0.physicalcpu`, `hw.perflevel1.physicalcpu` |
| GPU cores | `ioreg -rc AGXAccelerator` → `gpu-core-count` (best effort) |
| RAM | `hw.memsize` |
| GPU memory limit | `iogpu.wired_limit_mb`; `0` = OS default (~⅔–¾ of RAM depending on size — compute and record the effective value) |
| OS version | `sw_vers -productVersion` |
| free disk | `statfs` on `KEPLER_HOME` |
| power | `pmset -g batt` (AC/battery), `pmset -g` → `lowpowermode` |
| existing tools | `ollama` on PATH + version, Xcode CLT (`xcode-select -p`), `cmake` |

Probe output is a Go struct serialized into both `registry.json` and every result file.

---

## 7. Engines: definitions, resolver, registry

Three layers, each with one job:

| Layer | What | Where |
|---|---|---|
| **Engine definitions** | What an engine is, where it can run, how to install/launch/query it | `engines/*.yaml` (embedded) + `~/.kepler/engines.d/*.yaml` |
| **Engine resolver** | Pure function: `HostFacts` + definitions (+ model format) → which engines/targets are eligible and why | `internal/engines/resolve` |
| **Registry** | What is actually installed and healthy on this machine | `~/.kepler/registry.json` |

### 7.1 Engine definitions — one YAML file per engine

- Shipped definitions live in the repo at `engines/<name>.yaml` and are embedded with
  `//go:embed engines/*.yaml`.
- Users and contributors can drop additional or overriding definitions in
  `~/.kepler/engines.d/*.yaml` (same schema). A user file with the same `name` replaces the
  shipped one; the registry records `source: user` for it.
- Every definition is validated at load (unknown keys, unknown match keys, unknown install
  kinds, bad templates → hard error naming the file and field). A CI test validates all
  shipped files.

An engine has **targets**. A target is one way of running the engine on one class of
hardware, with a `match` block evaluated against `HostFacts`. Install recipe, launch args,
and capabilities can differ per target; top-level values are defaults that targets override.

#### Schema

```yaml
schema: 1
name: <id>                    # unique, [a-z0-9_-]
display: <human name>
version: <pinned version>
formats: [gguf | mlx | safetensors | ...]
tier: default | optional | lazy
adapter: generic | <name>     # generic = fully YAML-driven; <name> = registered Go hooks (§8.5)
homepage: <url>

launch:     { cmd, args, env, ready }      # defaults for all targets
request:    { api, model, extra_body }
caps:       { ignore_eos, disable_prompt_cache, stream_usage, engine_timings, ... }

targets:
  - id: <target id>
    match: { <fact>: <constraint>, ... }   # all clauses must hold
    install: [ <method>, <method>, ... ]   # tried in order (§7.2)
    launch:  { ... }                       # merged over top-level launch
    caps:    { ... }                       # merged over top-level caps

unsupported:                               # optional human messages for known non-targets
  - match: { os: darwin }
    reason: "Apple Silicon support is experimental"
```

#### Match keys

Evaluated against `HostFacts` (§6). The set is closed and versioned with `schema`; an
unknown key is a validation error, never silently ignored.

| Key | Example | Meaning |
|---|---|---|
| `os` | `darwin` / `[linux, windows]` | host OS in list |
| `arch` | `arm64` | host arch in list |
| `os_version` | `">=14.0"` | semver-ish constraint |
| `accelerator` | `metal` / `[cuda, rocm]` | host has at least one accelerator of this kind |
| `accelerator_vendor` | `nvidia` | ... from this vendor |
| `accelerator_api` | `">=12.4"` | API version of the matched accelerator (CUDA driver, ROCm, Metal family) |
| `compute_capability` | `">=8.0"` | CUDA compute capability |
| `gpu_mem_min` | `8GiB` | matched accelerator memory |
| `ram_min` | `16GiB` | system RAM |
| `tools` | `[cmake]` | binaries required on PATH for this target |

When an engine has several matching targets, the first in file order wins — so files list
the most specific / fastest targets first (e.g. `cuda` before `vulkan` before `cpu`).

#### Shipped definitions (v1)

`engines/llamacpp.yaml`

```yaml
schema: 1
name: llamacpp
display: "llama.cpp (llama-server)"
version: "b8971"
formats: [gguf]
tier: default
adapter: generic

launch:
  cmd: "{bin}"
  args: ["-m", "{model_path}", "--host", "127.0.0.1", "--port", "{port}",
         "-ngl", "999", "-c", "{ctx}", "-fa", "on", "-np", "1", "--no-webui"]
  args_if: { mmproj: ["--mmproj", "{mmproj_path}"] }
  ready: { http_get: "/health", expect: 200, timeout: 300s }
request:
  api: openai
  extra_body: { ignore_eos: true, cache_prompt: false }
caps: { ignore_eos: true, disable_prompt_cache: true, stream_usage: true, engine_timings: true }

targets:
  - id: macos-metal
    match: { os: darwin, arch: arm64, accelerator: metal, os_version: ">=14.0" }
    install:
      - kind: prebuilt
        url: "https://github.com/ggml-org/llama.cpp/releases/download/{version}/llama-{version}-bin-macos-arm64.zip"   # verify asset name
        sha256: "<pin>"
        bin: "build/bin/llama-server"
      - kind: source
        repo: "https://github.com/ggml-org/llama.cpp"
        ref: "{version}"
        cmake: ["-DGGML_METAL=ON", "-DLLAMA_CURL=OFF"]
        needs: [xcode_clt, cmake]

  # Future targets are added here, e.g.:
  # - id: linux-cuda
  #   match: { os: linux, arch: amd64, accelerator: cuda, accelerator_api: ">=12.4" }
  #   install: [{ kind: prebuilt, url: "...", sha256: "..." }]
  # - id: windows-vulkan
  #   match: { os: windows, arch: amd64, accelerator: vulkan }
  #   install: [...]
```

`engines/mlx.yaml`

```yaml
schema: 1
name: mlx
display: "MLX (mlx_lm.server)"
version: "<pin>"
formats: [mlx]
tier: default
adapter: generic

launch:
  cmd: "{bin}"
  args: ["--model", "{model_path}", "--host", "127.0.0.1", "--port", "{port}"]
  ready: { http_get: "/v1/models", expect: 200, timeout: 300s }
request: { api: openai }
caps: { ignore_eos: false, disable_prompt_cache: false, stream_usage: "<verify>", engine_timings: false }

targets:
  - id: macos-metal
    match: { os: darwin, arch: arm64, accelerator: metal, os_version: ">=14.0" }
    install:
      - kind: uv_venv
        python: "3.12"
        packages: ["mlx-lm=={version}"]
        bin: "bin/mlx_lm.server"

unsupported:
  - match: { os: [linux, windows] }
    reason: "MLX runs only on Apple Silicon"
```

`engines/ollama.yaml`

```yaml
schema: 1
name: ollama
display: "Ollama"
version: "<pin>"              # minimum version when adopting
formats: [gguf]
tier: default
adapter: ollama               # needs a materialize hook (§8.5)

launch:
  cmd: "{bin}"
  args: ["serve"]
  env:
    OLLAMA_HOST: "127.0.0.1:{port}"
    OLLAMA_MODELS: "{kepler_home}/tmp/ollama/{run_id}"
    OLLAMA_NUM_PARALLEL: "1"
    OLLAMA_MAX_LOADED_MODELS: "1"
    OLLAMA_KEEP_ALIVE: "-1"
  ready: { http_get: "/api/version", expect: 200, timeout: 60s }
materialize:
  modelfile: |
    FROM {model_path}
    PARAMETER num_ctx {ctx}
    PARAMETER temperature 0
  cmd: ["create", "kepler-{run_id}", "-f", "{modelfile_path}"]
request: { api: openai, model: "kepler-{run_id}" }
caps: { ignore_eos: false, disable_prompt_cache: false, stream_usage: true, engine_timings: false }

targets:
  - id: macos-metal
    match: { os: darwin, arch: arm64, accelerator: metal }
    install:
      - kind: adopt           # use `ollama` on PATH if version >= pin
        bin_name: ollama
      - kind: prebuilt
        url: "https://github.com/ollama/ollama/releases/download/v{version}/ollama-darwin.tgz"   # verify asset name
        sha256: "<pin>"
        bin: "ollama"
```

`engines/vllm.yaml`, `engines/sglang.yaml` — shipped with metadata, `formats`, `launch`,
`request`, `caps`, **no targets**, and an `unsupported` entry for `os: darwin`. They resolve
to `ineligible` on every v1 host and become eligible the moment a target is added.

Notes:
- Canary (health-check) models live in `canaries.yaml`, keyed by format, not in engine files:
  `gguf: { repo, file, sha256 }` (target < 100 MB), `mlx: { repo, revision }`. An engine whose
  format has no canary fails validation.
- Template variables: `{bin}`, `{port}`, `{ctx}`, `{model_path}`, `{mmproj_path}`,
  `{run_id}`, `{kepler_home}`, `{version}`, `{venv}`, `{modelfile_path}`. Unknown variables
  are validation errors. `{port}` is always a free ephemeral port bound to 127.0.0.1.
- Ollama runs as a **private `ollama serve`** on its own port with `OLLAMA_MODELS` pointed
  into the run's temp dir. It never touches the user's own Ollama models or daemon, and
  cleanup is `rm -rf` of that dir.
- `caps` drives fairness handling (§9.6). Values marked `<verify>` must be confirmed against
  the pinned engine version during implementation.

### 7.2 Install methods

| kind | Behavior |
|---|---|
| `prebuilt` | Download to `tmp/downloads`, verify sha256, extract to `engines/<name>/<version>/`, strip `com.apple.quarantine` xattr, `chmod +x`. |
| `source` | Check `needs`; shallow clone at `ref`; cmake configure + build; copy binary into `engines/<name>/<version>/`; delete source tree. |
| `uv_venv` | Ensure `bin/uv` (download pinned uv release, verify sha256); `uv venv --python {python}`; `uv pip install {packages}`. |
| `adopt` | Find binary on PATH; check version ≥ pin; record path. Kepler never modifies or removes adopted installs. |

### 7.3 Engine resolver

A pure, deterministic function — no I/O — so it is fully unit-testable with fixture hosts:

```go
func Resolve(facts HostFacts, defs []EngineDef, reg Registry, want *ResolveQuery) []Decision

type ResolveQuery struct {
    Formats []string   // from the model profile(s); nil = all (used by init)
    Allow   []string   // --engines
    Deny    []string   // --skip
}

type Decision struct {
    Engine   string
    Target   *Target          // nil if ineligible
    Status   string           // eligible | ineligible | excluded
    Reason   string           // always set when not eligible
    Install  InstallState     // from registry: ready | installed_unhealthy | not_installed | outdated
}
```

Algorithm, per engine definition:
1. Excluded by `--engines`/`--skip` → `excluded`.
2. Walk `targets` in order; first whose every `match` clause holds against `facts` → that
   target. If none:
   - first matching `unsupported` entry → `ineligible` with its `reason`;
   - else → `ineligible`, reason generated from the closest target's failing clauses, e.g.
     `"needs accelerator cuda (found: metal)"`, `"needs os_version >=14.0 (found 13.6)"`.
3. If `want.Formats` is set and the engine supports none of them → `excluded`,
   `"engine runs gguf; model available as mlx only"`.
4. Attach install state from the registry (version compare against definition `version`).

Callers:
- `kepler init` / `kepler doctor`: `Resolve(facts, defs, reg, nil)` → install plan.
- `kepler bench`: re-probes facts (cheap), then `Resolve` with the model's formats → engines
  that are eligible **and** `ready` are benchmarked; eligible-but-not-installed are listed
  with `kepler engines install <name>`; ineligible are shown with reasons.
- `kepler engines list`: prints the full decision table.

Resolver tests run against fixture hosts in `testdata/hosts/` (e.g. `m4-32gb.yaml`,
`m1-8gb-macos13.yaml`, `linux-rtx4090.yaml`, `windows-rx7900.yaml`). The non-Mac fixtures
exist from day one to prove that adding a target to a YAML file makes an engine eligible on
that host with zero code changes.

### 7.4 Adding a new engine or platform

**New engine that speaks the OpenAI streaming API** (most do — e.g. LM Studio's `lms`,
ik_llama.cpp, mistral.rs, TabbyAPI, a future vLLM target):
1. Add `engines/<name>.yaml` with `adapter: generic`, targets, launch, ready probe, caps.
2. Add a canary entry if it introduces a new format.
3. Add a resolver fixture test. Done — no Go changes.

**New engine that needs custom behavior** (model import, non-OpenAI API, odd lifecycle):
also implement the `Hooks` interface (§8.5) and register it under the `adapter:` name.

**New platform for an existing engine** (e.g. llama.cpp on NVIDIA):
1. If the accelerator kind is new to Kepler, add a detector (§6). One-time per kind.
2. Add a target to the engine's YAML.

**New install method** (e.g. `docker`, `pip_wheel`, `winget`): implement the `Installer`
interface and register its `kind`. Existing definitions are unaffected.

### 7.5 Registry (`~/.kepler/registry.json`)

```json
{
  "schema": 1,
  "kepler_version": "2.0.0",
  "definitions_schema": 1,
  "probed_at": "2026-10-12T15:00:00Z",
  "host": { "...": "HostFacts (§6)" },
  "engines": {
    "llamacpp": {
      "status": "ready",
      "source": "builtin",
      "target": "macos-metal",
      "version": "b8971",
      "install": { "kind": "prebuilt", "path": "~/.kepler/engines/llamacpp/b8971/build/bin/llama-server" },
      "formats": ["gguf"],
      "health": { "ok": true, "at": "2026-10-12T15:02:11Z", "load_s": 0.8, "decode_tok_per_s": 410.2 }
    },
    "mlx":    { "status": "ready", "source": "builtin", "target": "macos-metal", "...": "…" },
    "ollama": { "status": "ready", "source": "builtin", "target": "macos-metal",
                "install": { "kind": "adopt", "path": "/usr/local/bin/ollama" }, "...": "…" },
    "vllm":   { "status": "ineligible", "reason": "Apple Silicon support is experimental" }
  }
}
```

Status enum: `ready` · `installed_unhealthy` · `eligible_not_installed` · `ineligible` · `user_skipped`.

The registry is a cache of resolver decisions plus install results; the resolver (§7.3) is
the source of truth for eligibility. If the host changes (e.g. macOS upgrade, new GPU), the
next command's re-probe produces different decisions and `kepler doctor` reconciles.

`bench` only uses `ready` engines. If a definition's version is newer than the installed
version, `bench` warns and suggests `kepler engines install <name>`; it does not auto-upgrade.

---

## 8. Benchmark pipeline

```
parse ─► resolve (HF) ─► plan formats ─► preflight ─► confirm
      ─► for each format:
           download ─► for each engine of that format:
                          launch ─► load ─► warmup ─► scenarios ─► stop ─► write JSON ─► cooldown
                     ─► delete model files
      ─► write comparison JSON ─► print report ─► reaper
```

Formats are processed one at a time (download GGUF → bench llama.cpp + Ollama → delete →
download MLX → bench MLX → delete) so peak disk usage is one model, not all of them.

Before anything else, `bench` acquires `state/kepler.lock` (flock) and runs the reaper (§8.7).
A second concurrent `bench` exits with "another benchmark is running (pid N)" — running two
would also invalidate both sets of numbers.

### 8.1 Input parsing

Accepted forms:
- `org/repo`
- `https://huggingface.co/org/repo` (also `hf.co/...`, trailing `/tree/main`)
- `org/repo:Q4_K_M` — quant selector
- `https://huggingface.co/org/repo/blob/<rev>/path/file.gguf` — exact file
- `org/repo@<revision>` — pin revision

Output: `ModelRef{Repo, Revision?, File?, Quant?}`.

### 8.2 HF resolver

Auth: `HF_TOKEN` env if set (needed for gated repos; error message explains how to set it).

1. **Info**: `GET https://huggingface.co/api/models/{repo}` (at revision if given) →
   resolve `sha` (pinned commit), `tags`, `gated`, `cardData.base_model`.
2. **Files**: `GET /api/models/{repo}/tree/{sha}?recursive=true` → paths, sizes, `lfs.oid`
   (sha256) for every file.
3. **Classify** the repo:
   - any `*.gguf` → `gguf`
   - `config.json` + `*.safetensors` + (`mlx` tag or `quantization` key in config) → `mlx`
   - `config.json` + `*.safetensors` otherwise → `safetensors` (no v1 engine; triggers discovery)
4. **Base model**: from `cardData.base_model` or tags `base_model:quantized:<id>` /
   `base_model:finetune:<id>`. Falls back to the repo itself.
5. **Discover counterparts** (when `--formats auto` or the given repo has no usable format):
   query HF for repos tagged `base_model:quantized:<base>` with library `gguf` / `mlx`
   (confirm exact query params at implementation; fall back to name search
   `?search=<base-name>&filter=gguf`). Rank by publisher:
   `ggml-org`, model's own org > `unsloth`, `bartowski`, `lmstudio-community`, `mlx-community` > others;
   tie-break by downloads.
6. **Pick files per format**:
   - GGUF: parse quant from filename (`Q4_K_M`, `Q5_K_M`, `Q6_K`, `Q8_0`, `IQ4_XS`, `BF16`, …).
     Default = largest quant that passes preflight, preferring `Q4_K_M` ≤ x ≤ `Q8_0`.
     Collect all shards (`*-00001-of-0000N.gguf`) and the `mmproj-*.gguf` if present.
   - MLX: whole repo snapshot (config, tokenizer, `*.safetensors`, index). Match quant to the
     GGUF choice where possible (`Q4_*` ↔ `4bit`, `Q8_0` ↔ `8bit`).
7. **Confirm**: show the chosen repo/file/quant/size per format. Discovered counterparts
   always require confirmation unless `--yes`.

Output: `ModelProfile` per format:

```go
type ModelProfile struct {
    Repo, Revision, Format, Quant string
    Files       []HFFile        // path, size, sha256
    EntryFile   string          // GGUF shard 1, or "" for MLX dir
    MMProj      *HFFile
    SizeBytes   int64
    Arch        ArchInfo        // layers, kv_heads, head_dim, context_max, params
    Vision      bool
    Discovered  bool            // true if Kepler found it, not the user
}
```

`ArchInfo` comes from:
- GGUF: read the GGUF header via HTTP `Range: bytes=0-<N>` (metadata is at the file start;
  fetch more if the KV section is larger) — no full download needed.
- MLX: `config.json` (`num_hidden_layers`, `num_key_value_heads`, `head_dim` or
  `hidden_size/num_attention_heads`, `max_position_embeddings`).

### 8.3 Preflight

Per format × engine, using the largest context the preset needs (`ctx`, §9.4):

```
kv_bytes    = 2 × layers × kv_heads × head_dim × ctx × 2      # f16 K and V
need_mem    = weights_bytes + kv_bytes + 1 GiB overhead
budget_mem  = MemBytes of the accelerator matched by the engine's target (§6)
              (Apple: effective GPU wired limit; discrete GPU: free VRAM)
need_disk   = max over formats(model bytes) × 1.1  (+ model bytes again if Ollama runs: `ollama create` copies)
```

- `need_mem > budget_mem` → for that format try the next smaller quant; if none fits, drop
  scenarios whose ctx is too large (§9.4); if still no fit, the format is skipped with reason.
- `need_disk > free_disk` → exit `3` with the shortfall.
- Warnings (not errors): on battery, Low Power Mode on, memory pressure high
  (`memory_pressure -Q`), another known engine process already running.

### 8.4 Downloader

- Own HTTP client, not `huggingface_hub`. URL:
  `https://huggingface.co/{repo}/resolve/{sha}/{path}`.
- Writes to `tmp/downloads/<sha256>.part`; resumes with `Range` on retry/restart.
- Verifies sha256 against `lfs.oid`; on mismatch deletes and retries once.
- Moves to `tmp/models/<run_id>/<path>` on success.
- Progress bar with bytes/s and ETA. 4 parallel files max; single stream per file.
- Ledger entry is written **before** the `.part` file is created.

### 8.5 Engine adapter interface

The bench runner only talks to this interface, never to a specific engine:

```go
type Adapter interface {
    Name() string
    Formats() []string
    Caps() Caps

    // Prepare engine-specific model artifacts (e.g. `ollama create`). Records ledger entries.
    Materialize(ctx context.Context, m ModelProfile, run RunEnv) (EngineModel, error)

    // Spawn the server in its own process group on run.Port. Does not wait for ready.
    Launch(ctx context.Context, em EngineModel, opts LaunchOpts, run RunEnv) (*Proc, error)

    // Poll the ready probe until success or timeout.
    WaitReady(ctx context.Context, p *Proc) error

    // Endpoint + per-engine request defaults (model name, extra_body, supported caps).
    // Module-agnostic: bench merges these into its scenario requests; inference will
    // merge them into proxied client requests.
    Endpoint(p *Proc, em EngineModel) Endpoint   // {BaseURL, Model, ExtraBody}

    // Kill the process group (SIGTERM → 10s → SIGKILL) and wait for exit + memory release.
    Stop(p *Proc) error

    Version(ctx context.Context) (string, error)
}
```

There is exactly **one** implementation: `GenericAdapter`, constructed from a resolved
`(EngineDef, Target)`. It implements every method from YAML — templated launch args/env,
HTTP ready probe, `extra_body`, process-group stop, `--version` parsing. Materialize is a
no-op that passes the model path through.

Engines that need custom behavior name a hooks set via `adapter: <name>`. Hooks override
individual steps; anything not overridden falls back to the generic behavior:

```go
type Hooks interface{}   // marker; a hooks set implements any subset of:

type MaterializeHook interface { Materialize(ctx context.Context, g *GenericAdapter, m ModelProfile, run RunEnv) (EngineModel, error) }
type LaunchHook      interface { Launch(ctx context.Context, g *GenericAdapter, em EngineModel, opts LaunchOpts, run RunEnv) (*Proc, error) }
type EndpointHook    interface { Endpoint(g *GenericAdapter, p *Proc, em EngineModel) Endpoint }
type VersionHook     interface { Version(ctx context.Context, g *GenericAdapter) (string, error) }

// registered at init: hooks.Register("ollama", ollamaHooks{})
```

v1: llama.cpp and MLX use `generic`; Ollama uses `ollama` hooks (materialize via
`ollama create` from the definition's `materialize` block). A definition naming an
unregistered adapter fails validation.

All engines are benchmarked through their **OpenAI-compatible `/v1/chat/completions`
streaming endpoint**, with one shared HTTP/SSE client. This is the same path a future
`serve` command would use.

### 8.6 Process management

- Spawn with `Setpgid: true`; stdout/stderr → `logs/<run_id>/<engine>.log`.
- `Stop` signals the whole group (Ollama spawns a runner subprocess; this kills it too).
- After stop, wait until the group is gone and system free memory recovers (poll up to 30s)
  before the next engine starts.
- SIGINT/SIGTERM to Kepler → cancel root context → stop active engine → delete temp
  artifacts → write partial results with `status: "interrupted"` → exit 130.

### 8.7 Artifact ledger and reaper

SQLite (`modernc.org/sqlite`, pure Go, no cgo) at `state/ledger.db`:

```sql
CREATE TABLE artifacts (
  id          INTEGER PRIMARY KEY,
  run_id      TEXT NOT NULL,
  kind        TEXT NOT NULL,     -- file | dir | process_group | engine_install
  path        TEXT,              -- for file/dir
  pgid        INTEGER,           -- for process_group
  owner_pid   INTEGER NOT NULL,  -- kepler process that created it
  state       TEXT NOT NULL,     -- temp | kept | deleted
  created_at  TEXT NOT NULL,
  note        TEXT
);
```

Rules:
1. **Write-ahead**: insert `state=temp` *before* creating the artifact.
2. On successful normal cleanup: delete the artifact, set `state=deleted`.
3. `--keep`: set `state=kept`; `kepler clean` lists and offers to delete kept artifacts.
4. **Reaper** runs at the start of every command (and `kepler clean`): for every `temp`
   row whose `owner_pid` is not alive → kill pgid if any, delete path, mark `deleted`.
5. Paths are only ever deleted if they are inside `KEPLER_HOME` (hard safety check).

---

## 9. Benchmark methodology

### 9.1 Principles

1. One engine resident at a time.
2. Identical request parameters across engines: `temperature=0`, `seed=42`, fixed
   `max_tokens`, `stream=true`, `stream_options.include_usage=true`.
3. Every engine should emit exactly `max_tokens`: use `ignore_eos` where supported; elsewhere
   use a prompt designed to elicit long output and flag runs that stop short (§9.6).
4. **Defeat prompt caching**: every iteration's prompt starts with a unique nonce line, and
   `cache_prompt: false` is sent where supported. Otherwise TTFT after iteration 1 measures
   a cache hit, not prefill.
5. Same context size (`ctx`) passed to every engine that accepts it.
6. Warmup discarded (also triggers Metal shader compilation).
7. Report median and p10/p90 — never the mean.

### 9.2 Per-engine sequence

```
launch ──► first 1-token completion succeeds  ═► load_s
       ──► warmup: 1 × chat scenario (discarded)
       ──► scenarios in fixed order, N iterations each
       ──► stop ──► cooldown (--cooldown, then wait for thermal "nominal" up to 60s)
```

`load_s` is defined uniformly as *process spawn → first successful 1-token completion*,
because engines differ in whether they load weights at start (llama-server) or on first
request (Ollama).

### 9.3 Scenarios

Prompts are deterministic synthetic text (seeded filler from an embedded corpus) sized by
target token count. Actual prompt tokens are taken from the engine's `usage.prompt_tokens`
and recorded; the target is approximate because tokenizers differ.

| Name | Prompt tokens | max_tokens | Measures |
|---|---|---|---|
| `chat` | ~64 | 512 | decode tok/s, ITL — "does it feel fast in chat?" |
| `prefill-2k` | ~2,048 | 32 | TTFT, prefill tok/s |
| `prefill-8k` | ~8,192 | 32 | TTFT, prefill tok/s at RAG/agent sizes |
| `depth-4k` | ~4,096 | 128 | decode tok/s with 4k context filled |
| `depth-16k` | ~16,384 | 128 | decode tok/s with 16k context filled |
| `depth-32k` | ~32,768 | 128 | decode tok/s with 32k context filled |

`chat` is depth 0 in the depth curve. Depth scenarios report decode tok/s only (their TTFT
is recorded but is a prefill measurement).

Scenarios whose prompt + max_tokens exceed the model's `context_max`, or that failed
preflight, are skipped and listed in the result under `skipped_scenarios` with a reason.

### 9.4 Presets

| Preset | Scenarios | Iterations | ctx | Approx time / engine (8B Q4, M-series) |
|---|---|---|---|---|
| `quick` | chat (256 out), prefill-2k | 3 | 4096 | ~1 min |
| `standard` | chat, prefill-2k, prefill-8k, depth-16k | 5 | 16896 | ~4 min |
| `full` | all six | 10 | 33280 | ~12 min |

`ctx` = largest prompt + max_tokens in the preset, rounded up to a multiple of 512.
Load is measured in every preset. Times are estimates to sanity-check, not requirements.

### 9.5 Metrics (per iteration)

Timestamps come from a monotonic clock in the SSE client. A "token event" is an SSE delta
containing `content` or `reasoning_content`.

| Metric | Definition |
|---|---|
| `ttft_ms` | request sent → first token event |
| `prefill_tok_per_s` | `prompt_tokens / ttft_s` |
| `decode_tok_per_s` | `(generated_tokens − 1) / (t_last − t_first)` |
| `e2e_tok_per_s` | `generated_tokens / wall_s` |
| `itl_p50_ms`, `itl_p99_ms` | percentiles of gaps between successive token events |
| `wall_s` | request sent → stream closed |
| `generated_tokens` | `usage.completion_tokens`; fallback = token-event count (flagged) |
| `prompt_tokens` | `usage.prompt_tokens`; fallback = Kepler's target (flagged) |
| `peak_mem_bytes` | max over the iteration of Σ `phys_footprint` of the engine's process group |
| `engine_reported` | engine's own timings when available (llama-server `timings`), for cross-checking |

Memory: `phys_footprint` (via `proc_pid_rusage(RUSAGE_INFO_V4)`, called through
`github.com/ebitengine/purego` to keep `CGO_ENABLED=0`) is what Activity Monitor shows as
"Memory" and includes Metal allocations on unified memory; RSS alone undercounts. Sampled
every 100 ms by a goroutine for the life of the engine; `load.peak_mem_bytes` and
per-iteration peaks are both recorded.

If token events ≠ `completion_tokens` (an engine batches tokens per SSE chunk), ITL is
marked `approximate`.

Per run (not per iteration): `load_s`, `mem_after_load_bytes`, thermal before/after
(`pmset -g therm`, raw string recorded), power source.

### 9.6 Comparability

Each result carries `comparability.status`:

- `full` — every principle in §9.1 was honored.
- `degraded` — something wasn't; `notes` lists exactly what, e.g.:
  - `"engine does not support ignore_eos; 2/5 chat iterations stopped at <512 tokens"`
  - `"prompt cache could not be disabled; nonce prefix used"`
  - `"usage not reported in stream; token counts from SSE events"`
  - `"thermal state was 'fair' during run"`
  - `"on battery power"`

Cross-format comparisons (GGUF vs MLX) always add the note
`"different quantization schemes (Q4_K_M vs 4bit); comparison is approximate"` to the
comparison file. The report shows a `~` marker next to degraded/approximate numbers.

---

## 10. Results

### 10.1 Files

- One file per engine: `~/.kepler/results/{date}T{HHMMSS}_{repo}_{engine}_{preset}.json`
  (`repo` sanitized as today: `[^A-Za-z0-9._-]+` → `-`). Time is included because the
  current naming collides when the same model is run twice in one day.
- One comparison per `bench` run with ≥ 2 engines: `comparison_{comparison_id}.json`
  referencing the per-engine files and holding the ranked summary.
- Results are never auto-deleted.

### 10.2 Per-engine schema (v2)

Keeps the v1 shape (`config`, `iterations`, `summary`, `system_info`, `timestamp`,
`engine`, `comparison_id`) and extends it. Keys are snake_case, written with sorted keys and
2-space indent like today.

```json
{
  "schema_version": 2,
  "kepler_version": "2.0.0",
  "timestamp": "2026-10-12T15:08:30.241554Z",
  "run_id": "r_01JA…",
  "comparison_id": "c_01JA…",
  "status": "complete",

  "model": {
    "repo_id": "unsloth/gemma-4-E4B-it-GGUF",
    "revision": "a1b2c3d4…",
    "format": "gguf",
    "quant": "Q8_0",
    "files": [{ "path": "gemma-4-E4B-it-Q8_0.gguf", "size_bytes": 8123456789, "sha256": "…" }],
    "size_bytes": 8123456789,
    "arch": { "layers": 42, "kv_heads": 8, "head_dim": 256, "context_max": 131072 },
    "vision": false,
    "discovered": false,
    "requested_as": "unsloth/gemma-4-E4B-it-GGUF"
  },

  "engine": {
    "name": "llamacpp",
    "version": "b8971",
    "install_kind": "prebuilt",
    "launch_args": ["-ngl", "999", "-c", "16896", "-fa", "on", "-np", "1"]
  },

  "system_info": {
    "os": "darwin", "os_version": "26.0", "arch": "arm64",
    "chip": "Apple M4", "model_id": "Mac16,1",
    "cpu_cores": { "performance": 4, "efficiency": 6 }, "gpu_cores": 10,
    "ram_bytes": 34359738368, "gpu_mem_limit_bytes": 25769803776,
    "power_source": "ac", "low_power_mode": false
  },

  "config": {
    "preset": "standard",
    "temperature": 0,
    "seed": 42,
    "ctx": 16896,
    "warmup_iterations": 1,
    "iterations": 5,
    "cooldown_s": 10
  },

  "load": { "load_s": 3.42, "mem_after_load_bytes": 9100000000, "peak_mem_bytes": 9350000000 },

  "scenarios": [
    {
      "name": "chat",
      "target_prompt_tokens": 64,
      "max_tokens": 512,
      "iterations": [
        {
          "prompt_tokens": 71,
          "generated_tokens": 512,
          "ttft_ms": 173.3,
          "prefill_tok_per_s": 409.7,
          "decode_tok_per_s": 19.12,
          "e2e_tok_per_s": 18.81,
          "itl_p50_ms": 52.1,
          "itl_p99_ms": 61.0,
          "wall_s": 27.22,
          "peak_mem_bytes": 9300000000,
          "engine_reported": { "prompt_per_second": 412.0, "predicted_per_second": 19.2 }
        }
      ],
      "summary": {
        "ttft_ms":          { "p10": 172.9, "median": 173.3, "p90": 176.8 },
        "prefill_tok_per_s":{ "p10": 401.6, "median": 409.7, "p90": 410.5 },
        "decode_tok_per_s": { "p10": 18.87, "median": 19.12, "p90": 19.17 },
        "itl_p99_ms":       { "p10": 59.8,  "median": 61.0,  "p90": 63.2 },
        "peak_mem_bytes":   { "max": 9300000000 }
      },
      "flags": []
    }
  ],
  "skipped_scenarios": [{ "name": "depth-32k", "reason": "not in preset" }],

  "comparability": { "status": "full", "notes": [] },
  "thermal": { "before": "nominal", "after": "nominal" },
  "sample_output": "first 500 characters of the first measured chat iteration",
  "log_path": "~/.kepler/logs/r_01JA…/llamacpp.log"
}
```

`status`: `complete` · `partial` (some scenarios failed) · `failed` (engine never became
ready; file still written with the error) · `interrupted`.

Median = upper-middle for even counts, matching v1 (`legacy/src/benchmark.cpp`). p10/p90 use
nearest-rank.

### 10.3 Comparison schema

```json
{
  "schema_version": 2,
  "comparison_id": "c_01JA…",
  "timestamp": "…",
  "requested_as": "Qwen/Qwen3-8B",
  "preset": "standard",
  "system_info": { "...": "same as per-engine" },
  "entries": [
    { "engine": "mlx",      "format": "mlx",  "quant": "4bit",   "result_file": "…", "comparability": "full" },
    { "engine": "llamacpp", "format": "gguf", "quant": "Q4_K_M", "result_file": "…", "comparability": "full" },
    { "engine": "ollama",   "format": "gguf", "quant": "Q4_K_M", "result_file": "…", "comparability": "degraded" }
  ],
  "ranking": {
    "decode_tok_per_s": ["mlx", "llamacpp", "ollama"],
    "ttft_ms_prefill_2k": ["llamacpp", "mlx", "ollama"],
    "load_s": ["llamacpp", "mlx", "ollama"],
    "peak_mem_bytes": ["mlx", "llamacpp", "ollama"]
  },
  "notes": ["different quantization schemes (Q4_K_M vs 4bit); comparison is approximate"]
}
```

### 10.4 Terminal report

Printed at the end of `bench` and by `kepler results show/compare`:

```
Qwen3-8B · standard · Apple M4 (32 GB) · 2026-10-12 15:08

engine     format  quant   load    decode tok/s   TTFT 2k    TTFT 8k    decode@16k   peak mem
mlx        mlx     4bit    2.1s    41.3           0.61s      2.9s       33.0         5.1 GB
llamacpp   gguf    Q4_K_M  1.4s    38.7           0.55s      2.6s       31.2         5.4 GB
ollama     gguf    Q4_K_M  3.9s    36.9 ~         0.71s      3.1s       29.8 ~       5.9 GB

~ degraded comparability — see `kepler results show <file>`
Results: ~/.kepler/results/comparison_c_01JA….json
Cleaned up 9.8 GB of model files.
```

Best value per column highlighted. Values are medians.

---

## 11. Go implementation

Module: `github.com/thisisadityapatel/kepler`. Build: `CGO_ENABLED=0 GOOS=darwin GOARCH=arm64`
(v1 release target; code must still compile for linux/windows so later ports are additive).

```
cmd/kepler/main.go
engines/                     # engine definitions — one YAML per engine (embedded)
  llamacpp.yaml  mlx.yaml  ollama.yaml  vllm.yaml  sglang.yaml
canaries.yaml                # health-check model per format (embedded)
internal/
  cli/          cobra commands; flag parsing; exit codes
  # ── shared core (must not import bench/) ──
  home/         KEPLER_HOME paths, lockfile
  events/       in-process event stream (typed events, fan-out subscribers)
  probe/        HostFacts; OS probers (probe_darwin.go, build-tagged stubs for others)
    accel/      accelerator detectors: metal (v1); cuda, rocm, vulkan later; cpu fallback
  engines/
    def/        YAML schema types, loading (embedded + engines.d), validation, templating
    resolve/    engine resolver: match evaluation, decisions, reasons (pure)
    adapter/    Adapter interface, GenericAdapter, hooks registry
    hooks/      per-engine hooks: ollama/ (v1)
  registry/     registry.json load/save (atomic)
  install/      Installer interface + kinds: prebuilt | source | uv_venv | adopt
  hf/           ref parsing, API client, classify, discover, quant parse, GGUF header reader
  download/     resumable sha256-verified downloads, progress
  ledger/       SQLite ledger, reaper, safe delete
  proc/         process groups, phys_footprint sampling (purego)
  openai/       streaming chat client (SSE parsing, token-event timestamps)
  # ── bench module ──
  bench/
    preflight/  memory/disk estimates, quant fallback, scenario pruning
    scenario/   scenarios, presets, prompt generator
    runner/     per-engine sequence, mem sampler, fairness/comparability
    stats/      median, percentiles
    results/    schema v2 types, write, filename, comparison, report rendering
  # ── inference module (future, separate design) ──
  ui/           progress bars, tables, prompts (lipgloss / bubbletea, no-TTY fallback)
```

A CI check (`go list -deps` or `depguard`) fails the build if any shared-core package imports
`internal/bench/...`.

Dependencies: `spf13/cobra`, `gopkg.in/yaml.v3`, `modernc.org/sqlite`,
`ebitengine/purego`, `charmbracelet/lipgloss` (+ `bubbles` for progress), `oklog/ulid`
for ids. Everything else from the standard library (`net/http`, `archive/zip`, `crypto/sha256`,
`os/exec`, `syscall`).

UI must degrade to plain line output when stdout is not a TTY or `--json` is set.

### Existing code

The current Python implementation (`src/kepler/`, `pyproject.toml`, `uv.lock`) moves to
`legacy/python/` for reference. Port-worthy logic: result filename sanitization and median
behavior (`src/kepler/benchmark/results.py`), presets (`config/models.yaml`). Existing
`legacy/perf/*.json` files are v1 results; `kepler results` should read and display them
(schema_version absent ⇒ v1) but never rewrite them.

---

## 12. Testing

- **Unit**: ref parsing, repo classification, quant parsing/ranking, preflight math,
  stats (median/percentiles), filename sanitization, templating, ledger
  reaper logic (fake PIDs), GGUF header parsing (fixture files).
- **Engine definitions**: every file in `engines/` loads and validates; invalid fixtures in
  `testdata/defs/` fail with the right file/field message.
- **Engine resolver**: table tests of `Resolve` over fixture hosts (`testdata/hosts/*.yaml`,
  including Linux/NVIDIA and Windows/AMD hosts) × definitions, asserting eligibility and the
  exact reason strings. Includes a test that adds a fixture target to a definition and sees
  the engine become eligible — the guard that extensibility stays code-free.
- **HF resolver**: recorded API responses as fixtures (`testdata/hf/*.json`); no network in
  unit tests.
- **Fake engine**: a Go test binary that serves `/health` and an OpenAI streaming endpoint
  emitting tokens at a configured rate, delay, and chunking. Drives the full bench runner,
  process management, interrupt handling, and result writing deterministically in CI.
- **Crash test**: start a bench against the fake engine, SIGKILL Kepler mid-download and
  mid-run, run `kepler clean`, assert `tmp/` is empty and no fake-engine process survives.
- **Integration (manual / macOS runner)**: `kepler init` + `kepler bench --preset quick`
  against the canary models on a real Apple Silicon machine.

---

## 13. Milestones

Each milestone ends with something runnable.

| # | Deliverable | Done when |
|---|---|---|
| M0 | Skeleton: cobra CLI, `home`, `probe` (HostFacts + metal detector), `ledger` + reaper, `kepler doctor` prints facts | `kepler doctor` shows correct chip/RAM/GPU limit; reaper crash test passes with a dummy artifact |
| M1 | Engine definitions (schema, loader, validation), engine resolver with fixture hosts, `kepler engines list`, `init` for llama.cpp (prebuilt + source fallback), canary health check, registry | Resolver tests pass for Mac and non-Mac fixtures; fresh `~/.kepler` → `kepler init -y` → llama.cpp `ready` |
| M2 | HF resolver (GGUF only), preflight, downloader, GenericAdapter, bench runner, scenarios, results v2, cleanup | `kepler bench <gguf repo> --preset quick` writes a valid v2 JSON and leaves `tmp/` empty |
| M3 | MLX (generic + `uv_venv` installer) and Ollama (`ollama` hooks, adopt/prebuilt, private serve) — all three default | `kepler init -y` installs all three; one command benches all GGUF engines sequentially |
| M4 | Counterpart discovery (GGUF ↔ MLX), comparison file, terminal report, `results` commands | `kepler bench Qwen/Qwen3-8B` finds GGUF + MLX, benches all, prints ranked report |
| M5 | `install.sh`, release pipeline (GoReleaser, checksums), v1 result reading, polish | One-line install works on a clean Mac |

---

## 14. Future (not in v1, designed for)

- **Platforms**: add `probe_linux.go` / `probe_windows.go` and `cuda` / `rocm` / `vulkan`
  detectors; add targets to `engines/*.yaml` (llama.cpp CUDA/ROCm/Vulkan, Ollama, vLLM,
  SGLang). Resolver, bench runner and results need no changes (§7.4).
- **Concurrency scenario** (1/4/8 streams, aggregate tok/s) — most relevant once vLLM/SGLang land.
- **Energy**: optional `--energy` using `powermetrics` (sudo) for tokens/joule.
- **Inference module** (separate design): uses only the shared core — registry `ready`
  engines, the same definitions/adapters/launch path — plus its own gateway and lifecycle.
  Planned shape: a `kepler daemon` exposing a local HTTP + SSE API (current model, engine,
  tok/s, TTFT, memory over time), and a native SwiftUI menu bar app (`MenuBarExtra` + Swift
  Charts) that consumes it. The core stays Go; the UI is a thin native client.
- **Definition updates** without a Kepler release (`kepler engines update`, signed bundle of
  `engines/*.yaml`).
- **Result sharing**: schema already carries model revision/sha256 and full system info.

## 15. Open questions

1. Canary model picks (GGUF < 100 MB, smallest viable MLX) — choose and pin during M1.
2. Exact HF query parameters for `base_model:quantized:` discovery — verify during M4.
3. MLX `mlx_lm.server`: stream usage support and whether `max_tokens` is always honored at
   the pinned version — verify during M3 and set `caps` accordingly.
4. Should `quick` be the default preset for first-time users (faster first impression) with
   `standard` as the documented recommendation?

---

## 16. Implementation notes (as built: M0 + M1)

Deviations from the sections above, decided during implementation:

- **Embedded files** live in `definitions/` (`definitions/engines/*.yaml`, `definitions/canaries.yaml`,
  `definitions/embed.go`) because `go:embed` cannot reach parent directories.
- **Canaries** pinned (closes §15 Q1): GGUF `bartowski/SmolLM2-135M-Instruct-GGUF` Q4_K_M (101 MB),
  MLX `mlx-community/SmolLM2-135M-Instruct-8bit` (141 MB) — same model in both formats. Small
  non-LFS files are verified by git blob sha1 (`git_oid`), LFS files by sha256.
- **Pins**: llama.cpp `b11574` (asset is `.tar.gz`, binary at `llama-{version}/llama-server`),
  mlx-lm `0.32.0`, Ollama `0.40.3` prebuilt; uv `0.13.0` pinned in `internal/install`.
- **Adopt** has its own `min_version` (Ollama: `0.6.0`), separate from the prebuilt `version`.
- **Adapter hooks**: added `PostReadyHook` (runs after the readiness probe; Ollama's
  `ollama create` needs the live server). `Running.Owned` holds extra ledger rows (Ollama's
  private model store) disposed by `adapter.StopAndRelease`. Materialize is generic-only today.
- **Ledger states**: added `released` — ownership handed to the registry (engine installs,
  canaries) after a passing health check; the reaper never touches released rows.
- **`internal/setup`** (shared core) holds install-with-fallback, canary download, and health
  checks used by `init`, `engines install`, and `doctor`.
- **`internal/fakeengine`**: tests re-exec their own binary as a fake OpenAI server.
- **`HF_ENDPOINT`** env (or `Setup.HFEndpoint`) overrides `https://huggingface.co`.
- **Portability**: Unix-only primitives (flock, statfs, process groups) sit behind build tags;
  `make cross` keeps linux/windows compiling.
- **Doctor** separates warnings (battery, Low Power Mode → exit 0) from problems (exit 1).
- **Not yet built**: everything under §8.1–8.4 and §9–10 (`kepler bench`, HF resolver, preflight,
  results), `install.sh`, phys_footprint memory sampling, and the bench import-boundary CI check.

## 17. Implementation notes (as built: M2)

- **Default quant is `Q4_K_M`, not "largest that fits"** (changes §8.2 step 6). Reason: results stay
  comparable across machines and downloads stay reasonable. `hf.DefaultQuantPreference` falls back
  through other Q4-class quants, then Q5/Q6/Q8, then smaller; preflight takes the first that fits.
  `--quant` / `repo:QUANT` override.
- **Vision projectors are not downloaded or loaded** (text-only scenarios); `model.vision` is still
  recorded and an info note is added. Loading mmproj would add memory and load time to one engine only.
- **Memory metric** is `phys_footprint + resident pages of memory-mapped files under KEPLER_HOME/tmp`
  (`config.memory_metric = "phys_footprint+mapped_model_files"`). phys_footprint alone omits mmapped
  weights (llama.cpp and Ollama mmap GGUFs): a 469 MB model showed 228 MB. Mapped pages are scanned via
  `proc_pidinfo(PROC_PIDREGIONPATHINFO)` once per second; footprint every 100 ms. `load.mapped_model_bytes`
  records the mmapped part.
- **Prompt calibration**: before warmup each engine gets a ~512-token and a tiny prompt (1 output token
  each); chars/token = Δchars / Δprompt_tokens, which cancels chat-template overhead. Recorded as
  `config.chars_per_token`. Falls back to 4.0 with an info note if the engine reports no usage.
- **load_s** = spawn → first token of a 1-token request, minus engine-specific import time
  (`load.materialize_s`, Ollama's `ollama create`).
- **Comparability** splits `notes` (degrading: early stop, missing usage, battery, Low Power Mode,
  thermal) from `info` (non-degrading: approximate ITL, prompt cache not disableable, vision skipped).
- **Partial downloads are ledgered** and deleted on interrupt/crash (disk hygiene over cross-run resume;
  resume still works across retries within a run).
- **Host conditions** (`probe.SampleConditions`: thermal via `pmset -g therm`, memory free via
  `memory_pressure -Q`, power) are recorded before/after each engine in `conditions`.
- **Observed engine behavior (M4, Qwen2.5-0.5B)**: llama.cpp, Ollama 0.21.2, and mlx-lm 0.32.0 all report
  usage in the stream and emitted the full max_tokens with the long-essay prompt; MLX caps updated.
- **`kepler results list|show`** implemented early (was M4). `show` accepts a result file, a comparison
  file, or a run/comparison id.
- **Exit codes**: invalid input (bad reference, preset, scenario, quant) → 2; preflight → 3; no engine → 4;
  partial → 5; interrupted → 130.
- **Import boundary** enforced by `internal/arch_test.go` (shared core must not import `internal/bench`).
- **Not yet built**: counterpart discovery GGUF↔MLX (M4), multi-format runs in one command, `install.sh`
  and release pipeline (M5).
- **llama-server `--cache-ram 0`**: b11574 keeps a host-RAM prompt cache (default 8 GiB) that grows with
  every unique prompt; with Kepler's per-iteration nonce it inflated peak memory from 1.5 GB to 5.0 GB on a
  1 GB model. Disabled in `llamacpp.yaml`; TTFT unchanged.
- **Results never contain `null` arrays** (scenarios, iterations, flags, notes are always `[]`).
