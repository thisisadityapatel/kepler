# Kepler: Everything You Need to Know

A guide for a software engineer who is new to LLM inference, benchmarking, and Go.
Read it top to bottom once. Each part builds on the previous one.

1. How an LLM generates text
2. Model files: parameters, precision, quantization
3. File formats: safetensors, GGUF, MLX
4. Hugging Face
5. Inference engines
6. Apple Silicon specifics
7. Benchmarking: what we measure and why
8. Kepler's architecture, end to end
9. Go, explained through this codebase
10. Working on Kepler day to day
11. Glossary

---

## 1. How an LLM generates text

**Tokens.** Models don't see characters; they see *tokens*, which are integer IDs for chunks of
text ("Hello" → 9906, " world" → 1917). A *tokenizer* (shipped with the model) converts text
to tokens and back. In English, roughly 1 token ≈ 4 characters ≈ ¾ of a word. Different models
use different tokenizers, so the same text has a different token count per model. That's why
Kepler **calibrates** prompt sizes per model (§7.5).

**The model is a function.** Given a sequence of tokens, it outputs a probability for every
possible next token. Generation is a loop:

```
tokens = tokenize(prompt)
repeat:
    probs = model(tokens)          # one "forward pass"
    next  = sample(probs)          # pick a token (temperature controls randomness)
    tokens.append(next)
    stream next to the user
until next == EOS (end-of-sequence) or max_tokens reached
```

**Two phases, two very different speeds:**

| Phase | What happens | Bottleneck | Metric |
|---|---|---|---|
| **Prefill** | All prompt tokens are processed at once, in parallel | **Compute** (matrix multiplies) | prefill tok/s, TTFT |
| **Decode** | Output tokens are generated one at a time | **Memory bandwidth** (all weights are read for every token) | decode tok/s |

Decode is memory-bound: to produce *each* token, the GPU reads the entire model from memory.
A 4 GB model on a machine with 120 GB/s of bandwidth tops out around 120/4 = 30 tok/s. This
rule of thumb explains most of what you'll see in results.

**KV cache.** During attention, every token produces a "key" and a "value" vector per layer.
Engines cache these so earlier tokens aren't recomputed on every step. The cache grows with
context length:

```
KV bytes = 2 (K and V) × layers × kv_heads × head_dim × context_tokens × 2 bytes (fp16)
```

Llama‑3‑8B (32 layers, 8 KV heads, head_dim 128) needs 128 KiB per token, so a 32k context
costs 4 GiB on top of the weights. Kepler's preflight uses exactly this formula
(`internal/bench/preflight`).

**Context length (ctx).** This is the maximum number of tokens (prompt + output) the engine
holds. The model has a trained maximum (`context_max`, e.g. 32,768). Kepler launches engines
with the smallest ctx its scenarios need, because a bigger ctx means a bigger KV allocation and
more memory.

**Why decode slows down with long context.** Each new token attends to *all* previous tokens,
so the work per token grows with context. That's the "depth" scenarios (§7.3).

**Temperature / seed.** Temperature 0 means always picking the most likely token, which makes
output deterministic. A seed fixes any remaining randomness. Kepler uses `temperature=0, seed=42`
so every engine does identical work.

**Chat template.** Chat models expect special formatting around messages
(`<|im_start|>user\n...<|im_end|>`). The engine applies it, which adds tokens you didn't type.
That's why the "chat" prompt of ~64 tokens shows up as ~93–123 prompt tokens in results.

**Reasoning ("thinking") models** emit hidden reasoning tokens before the answer, often
streamed as `reasoning_content`. Kepler counts them as output tokens.

---

## 2. Model files: parameters, precision, quantization

**Parameters** are the learned numbers (weights). "8B" means 8 billion of them. File size ≈
parameters × bytes per parameter.

| Precision | Bits per weight | 8B model size | Notes |
|---|---|---|---|
| FP32 | 32 | 32 GB | training only |
| FP16 / BF16 | 16 | 16 GB | "full precision" for inference; BF16 has FP32's range |
| Q8_0 | ~8.5 | 8.5 GB | near-lossless |
| Q4_K_M | ~4.8 | 4.9 GB | the standard sweet spot |
| Q2_K | ~2.6 | 3 GB | noticeable quality loss |

**Quantization** stores weights in fewer bits: groups ("blocks") of weights share a scale
factor, and each weight becomes a small integer. This trades some accuracy for less memory
and **faster decode**, since there are fewer bytes to read per token (see the bandwidth rule
above).

**Reading GGUF quant names** (llama.cpp's naming):
- `Q4` = ~4 bits per weight, `Q8` = 8 bits, and so on.
- `_0` / `_1` = older simple block formats.
- `_K` = "k-quants": smarter block structure with super-blocks and better accuracy per bit.
- `_S` / `_M` / `_L` = small, medium, or large mix. Some tensors (e.g. attention, output) are
  kept at higher precision. `Q4_K_M` mixes Q4_K with some Q6_K.
- `IQ4_XS`, `IQ3_M` = "i-quants": better quality at very low bits, usually built with an
  importance matrix (imatrix) computed on calibration data.
- `UD-Q4_K_XL` = Unsloth "dynamic" quants, an Unsloth-specific per-layer mix.
- `F16`, `BF16` = not quantized.

**MLX quant names** are simpler: `4bit`, `8bit` (with a group size, typically 64), or `bf16`.

**Why Kepler defaults to Q4_K_M:** it's the community standard, so results compare across
machines. Q4_K_M and MLX 4‑bit are *different* algorithms, so cross-format comparisons are
approximate, and Kepler says so in the result.

**Sharding.** Big models are split into parts: `model-00001-of-00003.gguf`. All parts must be
downloaded; llama.cpp is given only the first and finds the rest. Kepler groups shards into
one "variant" (`hf.GGUFVariants`).

**Vision models** ship a separate *projector* (`mmproj-F16.gguf`) that turns images into
tokens. Kepler benchmarks text only, so it skips the projector (it would add memory and load
time to only some engines).

---

## 3. File formats: safetensors, GGUF, MLX

| Format | What it is | Who runs it | How Kepler detects it |
|---|---|---|---|
| **safetensors** | PyTorch/Transformers weights: `config.json` + `*.safetensors` + tokenizer files. The "original" release of most models. | vLLM, SGLang (NVIDIA mostly) | `config.json` + `.safetensors`, no `mlx` tag |
| **GGUF** | llama.cpp's **single-file** format: a header with metadata (architecture, layer count, tokenizer, chat template) followed by tensors. Usually quantized. | llama.cpp, Ollama, LM Studio | any `*.gguf` file |
| **MLX** | Apple's MLX framework format: safetensors files in MLX's layout, with `quantization` in `config.json`. | mlx-lm (Apple Silicon only) | `mlx` tag or `library_name: mlx` |

**The key consequence:** an engine can only load its own format. MLX can't read GGUF, and
llama.cpp can't read MLX. That's why benchmarking a GGUF repo runs only llama.cpp and Ollama.
The planned M4 feature finds the MLX build of the same base model automatically.

**GGUF header trick.** Preflight needs layer count, KV heads, and so on *before* downloading
gigabytes. GGUF puts metadata at the start of the file, so Kepler sends an HTTP
`Range: bytes=0-` request and parses only the header, stopping before the large tokenizer
arrays (`internal/hf/gguf.go`). For MLX it reads `config.json`.

Header layout (little-endian): magic `"GGUF"`, version u32, tensor_count u64, kv_count u64,
then kv pairs (`key: string`, `type: u32`, `value`). Keys look like `general.architecture` =
"qwen2", `qwen2.block_count` = 24, `qwen2.attention.head_count_kv` = 2.

---

## 4. Hugging Face

A git-hosting site for models. Each model is a **repo** (`org/name`) whose large files are
stored with **Git LFS**.

- **Revision** = branch, tag, or commit SHA. Kepler always resolves to a 40-char **commit SHA**
  and records it, so a result says exactly which bytes were tested.
- **Integrity.** LFS files have a **sha256** (`lfs.oid`); small files have a **git blob SHA‑1**
  (sha1 of `"blob <size>\0" + content`). Kepler verifies every download with one or the other.
- **Gated repos** (Llama, Gemma) need you to accept a license and set `HF_TOKEN`.
- **APIs Kepler uses** (`internal/hf/client.go`):
  - `GET /api/models/{repo}`: commit SHA, tags, `base_model` (the upstream model a quant was
    made from)
  - `GET /api/models/{repo}/tree/{sha}?recursive=true`: every file with size and hashes
  - `GET /{repo}/resolve/{sha}/{path}`: download (supports `Range` for resume and header reads)
- **Typical publishers:** `ggml-org` (llama.cpp team), `bartowski`, `unsloth`,
  `lmstudio-community` (GGUF); `mlx-community` (MLX).
- `HF_ENDPOINT` overrides the base URL. The tests use it to point at a fake hub.

---

## 5. Inference engines

An **engine** loads weights and runs the generation loop efficiently on the GPU. All Kepler
engines run as an **HTTP server** exposing an **OpenAI-compatible API**:

```
POST /v1/chat/completions
{"model": "...", "messages": [{"role":"user","content":"..."}],
 "max_tokens": 512, "temperature": 0, "seed": 42, "stream": true,
 "stream_options": {"include_usage": true}, "ignore_eos": true}
```

With `stream: true` the response is **SSE** (Server-Sent Events): lines like
`data: {"choices":[{"delta":{"content":"Hel"}}]}`, one per token (roughly), ending with a usage
chunk (`prompt_tokens`, `completion_tokens`) and `data: [DONE]`. Kepler timestamps each line as
it arrives (`internal/openai/client.go`); that's how it measures latency precisely. Because
every engine speaks this API, one client measures all of them identically.

| Engine | Language | Notes in Kepler |
|---|---|---|
| **llama.cpp** (`llama-server`) | C++ on top of **ggml** (its tensor library); Metal backend on Mac | Prebuilt binary from GitHub releases. Supports `ignore_eos`, disabling the prompt cache, and its own timing stats. Kepler passes `--cache-ram 0` (its RAM prompt cache otherwise grows with every request). |
| **Ollama** | Go app wrapping a llama.cpp/ggml runner | Model management ("pull", Modelfile). Kepler runs a **private** `ollama serve` on its own port with its own model folder, then imports the GGUF via `ollama create`. No `ignore_eos`. |
| **MLX** (`mlx_lm.server`) | Python on Apple's MLX framework | Installed in an isolated Python env using **uv** (a fast Python package manager). Apple Silicon only. |
| vLLM, SGLang | Python, CUDA-focused, high throughput, batching | Defined in YAML but not runnable on Mac yet. |

**Important engine flags/behaviors:**
- `-ngl 999`: offload all layers to GPU.
- `-c N`: context size.
- `-fa on`: flash attention, a faster attention kernel.
- `-np 1`: one parallel slot.
- **`ignore_eos`** forces exactly `max_tokens` output. Without it, a model may stop early and
  look "faster", so Kepler flags runs that stopped short.
- **Prompt cache.** Engines reuse KV for prompts that share a *prefix* with earlier requests,
  which makes repeated benchmarks look unrealistically fast. Kepler puts a unique
  `[session ...]` line at the **start** of every prompt, so no prefix is ever shared.

---

## 6. Apple Silicon specifics

- **Unified memory.** CPU and GPU share one RAM pool. No PCIe copy, so the GPU can use most of
  RAM. The limit is the **GPU wired limit**: by default about ⅔ of RAM (≤36 GB machines) or ¾
  (larger). It's overridable via `sysctl iogpu.wired_limit_mb`. On your 16 GB M4 that's
  ~10.7 GB. Preflight's memory budget is this number.
- **Metal** is Apple's GPU API. The engines' GPU kernels are written in it.
- **Memory bandwidth** drives decode speed: M4 ≈ 120 GB/s, M4 Pro ≈ 273, M4 Max ≈ 546 GB/s.
- **Thermals and power.** Laptops throttle when hot or on battery / Low Power Mode. Kepler
  records `pmset -g therm` and power state before and after each engine, and marks results
  `degraded` if conditions were bad.
- **Measuring memory correctly (a real bug we hit):**
  - **RSS** = resident pages, including file-backed pages, but missing GPU allocations.
  - **phys_footprint** (what Activity Monitor shows) includes GPU/Metal memory but **excludes
    memory-mapped files**.
  - llama.cpp and Ollama **mmap** the GGUF: the file's pages are mapped into memory instead of
    read. So phys_footprint showed 228 MB for a 469 MB model.
  - Kepler's metric is **phys_footprint + resident pages of mapped model files**, read via
    macOS `libproc` (`proc_pid_rusage`, `proc_pidinfo(PROC_PIDREGIONPATHINFO)`), summed over
    the engine's whole **process group**. Code: `internal/proc/footprint_darwin.go`.

---

## 7. Benchmarking: what we measure and why

### 7.1 Metrics (per request; `internal/bench/runner/metrics.go`)

| Metric | Formula | Meaning |
|---|---|---|
| `ttft_ms` | first token arrival − request sent | Time To First Token: how long until anything appears (mostly prefill) |
| `prefill_tok_per_s` | prompt_tokens / TTFT | prompt processing speed |
| `decode_tok_per_s` | (generated − 1) / (last token − first token) | generation speed; the headline number |
| `e2e_tok_per_s` | generated / total wall time | what a user experiences overall |
| `itl_p50_ms`, `itl_p99_ms` | percentiles of gaps between tokens | **smoothness**: p99 catches stutters a median hides |
| `peak_mem_bytes` | max sampled memory during the request | how much RAM the engine really needs |
| `load_s` (per engine) | process start → first token of a 1-token request | cold start cost |

### 7.2 Statistics

Each scenario runs N iterations and reports **p10 / median / p90**, never the mean (one outlier
skews a mean). The median uses the "upper middle" for even counts, matching Kepler v1.
Percentiles are nearest-rank.

### 7.3 Scenarios: each one stresses a different regime

| Scenario | Prompt → output | Answers |
|---|---|---|
| `chat` | ~64 → 512 | "Does it feel fast to talk to?" (decode, ITL) |
| `prefill-2k` / `prefill-8k` | 2k / 8k → 32 | "How fast with big inputs?" (RAG, coding agents; TTFT) |
| `depth-4k/16k/32k` | 4k/16k/32k → 128 | "How much does decode slow as context fills?" |

Presets: `quick` (chat-256 + prefill-2k, 3 iterations), `standard` (chat, prefill-2k/8k,
depth-16k, 5), `full` (all six, 10).

### 7.4 Fairness rules (what makes the numbers trustworthy)

1. One engine at a time; each is fully stopped before the next starts; cooldown in between.
2. Identical requests: temp 0, seed 42, same max_tokens, `ignore_eos` where supported.
3. Unique prompt prefix per request, which defeats prompt caching.
4. Same context size for every engine that accepts one.
5. Warmup request discarded (the first run compiles GPU shaders and warms caches).
6. Medians and percentiles.
7. Everything that couldn't be honored is written into the result.

### 7.5 Prompt calibration

Prompts are synthetic, deterministic prose (seeded sentence sampling), with an instruction to
"write a long essay" so engines without `ignore_eos` rarely stop early. To hit "2048 tokens"
on *this* model's tokenizer, Kepler first sends a ~512-token prompt and a tiny prompt (1 output
token each):

```
chars_per_token = (len(big) − len(small)) / (tokens(big) − tokens(small))
```

The subtraction cancels the chat template's extra tokens. All later prompts are sized with
that ratio (we measured ≈4.99 for Qwen).

### 7.6 Comparability

Each result has `comparability.status` = `full` or `degraded`:
- **`notes`** (these degrade the result): stopped before max_tokens, missing token usage,
  battery, Low Power Mode, thermal throttling.
- **`info`** (these don't): approximate ITL, prompt cache not disableable, vision skipped.

---

## 8. Kepler's architecture, end to end

### 8.1 Commands

`init`, `doctor`, `engines list|install|remove`, `bench`, `results list|show`, `clean`,
`version`. Exit codes: 0 OK, 1 error, 2 bad input, 3 doesn't fit, 4 no engine, 5 partial,
130 Ctrl‑C.

### 8.2 `~/.kepler` (or `$KEPLER_HOME`)

```
bin/uv                 Kepler-managed uv (for MLX's Python env)
engines/<name>/<ver>/  installed engines (llama.cpp binary, MLX venv)
canary/                tiny health-check models (SmolLM2-135M, GGUF + MLX), kept
engines.d/*.yaml       your own engine definitions (override/extend builtins)
registry.json          what's installed and healthy
state/ledger.db        SQLite: every temp artifact ever created
state/kepler.lock      only one kepler command mutates at a time (flock)
tmp/models/<run>/      downloaded model for the current run (deleted after)
tmp/ollama/<run>/      Ollama's private model store (deleted after)
tmp/downloads/         *.part files during download
logs/<run>/<engine>.log  engine stdout/stderr (look here when something fails)
results/*.json         results, never auto-deleted
```

### 8.3 The big ideas

**1. Engines are data, not code.** Each engine is a YAML file (`definitions/engines/*.yaml`):
version, formats, how to launch (`cmd`, `args` with `{port}` / `{model_path}` placeholders),
the readiness URL, request extras, capabilities, and **targets**. A target is
`match` (hardware conditions) + `install` (ordered methods: `prebuilt` → `source` / `uv_venv`
/ `adopt`). Adding an engine or a platform = editing YAML.

**2. Hardware is facts.** `probe` produces `HostFacts` (OS, arch, RAM, accelerators like
`{kind: metal, mem_bytes: ...}`, tools). The **resolver** (`internal/engines/resolve`) is a
**pure function** `Resolve(facts, defs, installed, query) → decisions`: for each engine,
eligible (with the matching target), ineligible (with a human reason like
`needs accelerator cuda (found: metal)`), or excluded. It's tested against fake Linux/Windows
machines to prove new platforms need only YAML.

**3. One generic adapter.** `GenericAdapter` runs *any* YAML engine: Materialize → Launch
(free port, process group, log file) → WaitReady (poll the HTTP URL) → PostReady → Endpoint →
Stop. Engines with quirks register **hooks**; only Ollama has them (its `ollama create` needs
the server running, hence `PostReady`).

**4. The ledger guarantees cleanup.** Before creating *anything* (a download, a temp dir, an
engine process), Kepler writes a row `{path or pgid, owner_pid, state: temp}` to SQLite (a
**write-ahead** record). Normal completion deletes the thing and marks the row deleted. If
Kepler crashes or is killed, the **reaper** runs at the start of the next command: any temp row
whose owner process is dead gets its files deleted and its process group killed. States:
`temp`, `kept` (`--keep`), `released` (handed to the registry, like installed engines),
`deleted`. Deletions are refused outside KEPLER_HOME.

**5. Process groups.** Engines start with `Setpgid`, so the engine and all its children share a
group. Stopping sends SIGTERM to `-pgid` (the whole group), waits, then SIGKILL. Ollama spawns
a runner child; without this it would be orphaned holding GPU memory.

**6. Module boundary.** Shared core (probe, engines, install, registry, ledger, proc, hf,
download, openai, events) **never imports** `internal/bench`. A future inference module reuses
the core. `internal/arch_test.go` enforces this.

### 8.4 `kepler init` flow

probe → load YAML (validated strictly: unknown keys are errors) → resolve → print plan →
for each eligible engine, try install methods in order. Each method must pass a **health
check** (launch on the canary model, generate 32 tokens) or its files are rolled back and the
next method is tried → write `registry.json`.

### 8.5 `kepler bench` flow (`internal/bench/bench.go`)

1. `hf.ParseRef`: accepts `org/repo`, `org/repo:Q8_0`, `@revision`, full URLs.
2. `hf.Resolve`: pin the commit, list files, classify format, group GGUF variants, rank quants.
3. Choose engines: resolver query filtered to the model's format, keeping only `ready` engines.
4. Profile: read architecture from the GGUF header / `config.json`.
5. **Preflight**: for each quant in preference order, does weights + KV(ctx) + 1 GiB fit the GPU
   budget? If none fits, drop the largest scenarios. Then check disk (×2 if Ollama will copy
   the model).
6. Show the plan, confirm, download (resumable, verified, ledgered).
7. For each engine, `runner.Run`: launch → load probe → calibrate → warmup → scenarios
   (a memory sampler goroutine polls every 100 ms) → stop → result JSON. Cooldown between
   engines.
8. Comparison file + ranked report. Delete the model unless `--keep`.

### 8.6 Code map

```
cmd/kepler/main.go            entry point → cli.Execute()
definitions/                  embedded YAML: engines/*.yaml, canaries.yaml
internal/cli/                 cobra commands; the "composition root" wiring everything
internal/home/                paths, safe delete, lock
internal/probe/               HostFacts, macOS sysctl/pmset/ioreg parsing, thermal
internal/engines/def/         YAML schema, strict loader, validation, {templates}, versions
internal/engines/resolve/     the resolver (pure)
internal/engines/adapter/     Adapter interface, GenericAdapter, hooks registry
internal/engines/hooks/ollama Ollama-specific steps
internal/install/             prebuilt (download+sha256+extract), source (git+cmake), uv_venv, adopt
internal/setup/               install-with-fallback, canaries, health checks (init/doctor)
internal/registry/            registry.json
internal/ledger/              SQLite ledger + reaper
internal/proc/                process groups, memory (purego → libproc)
internal/download/            resumable HTTP download with integrity check
internal/openai/              SSE streaming client with per-token timestamps
internal/hf/                  Hugging Face refs, API, formats, quants, GGUF header parser
internal/events/              tiny pub/sub bus (core → UI; later → menu-bar app)
internal/ui/                  terminal tables, prompts, progress
internal/bench/               orchestrator + preflight/, scenario/, runner/, stats/, results/
internal/fakeengine/          fake OpenAI server used by tests
testdata/                     fake hosts, recorded HF API responses
```

---

## 9. Go, explained through this codebase

**Module and packages.** `go.mod` declares the module `github.com/thisisadityapatel/kepler`.
Each directory is a **package**; the import path is module + directory. Anything under
`internal/` can only be imported from inside this module (enforced by the compiler).
Capitalized names are **exported** (public); lowercase names are private to the package.

**Structs, methods, interfaces.** There are no classes. A struct holds data, and methods attach
to it (`func (l *Ledger) Record(...)`). **Interfaces are implicit**: any type with the right
methods satisfies one, with no `implements` keyword. Example: `install.Installer` is satisfied
by `prebuilt{}`, `source{}`, and so on. `adapter.Hooks` uses this trick: it's an empty
interface, and the adapter checks at runtime `if h, ok := g.hooks.(PostReadyHook); ok { ... }`
(a **type assertion**) to see which optional steps a hooks value implements.

**Errors are values.** Functions return `(result, error)`; you check `if err != nil`.
- Wrap with context: `fmt.Errorf("download %s: %w", path, err)`. The `%w` keeps the original
  error.
- Test with `errors.Is(err, hf.ErrNotFound)` (matches anywhere in the chain) or
  `errors.As(err, &fitErr)` (extracts a typed error).
- The CLI maps these to exit codes (`internal/cli/bench.go`). `errors.Join` combines several
  errors (used to report every validation problem at once).

**`defer`** runs a call when the function returns. It's used for guaranteed cleanup:
`defer adapter.StopAndRelease(...)` means the engine stops even if a scenario errors out.

**`context.Context`** carries cancellation and deadlines through call chains. `cli` creates one
with `signal.NotifyContext`, so **Ctrl‑C cancels it**, which aborts HTTP requests, downloads,
and waits everywhere. Cleanup paths use `context.WithoutCancel(ctx)` so they still run after
cancellation. Per-request timeouts use `context.WithTimeout`.

**Goroutines and synchronization.** `go f()` starts a lightweight thread.
- The memory `Sampler` runs in one, polling on a `time.Ticker`.
- It stops via a `stop` channel and signals completion on `done`.
- Shared fields are protected by `sync.Mutex`.
- `select` waits on several channels at once (stop vs tick, or done vs timeout).
- `go test -race` detects unsynchronized access. It caught two real bugs here.

**Generics** appear once: `mergeMap[V any](a, b map[string]V)`.

**Build tags.** `//go:build darwin` at the top of a file compiles it only on macOS. That's how
`probe_darwin.go` / `probe_other.go` and `group_unix.go` / `group_windows.go` coexist.
`make cross` proves Linux and Windows still compile.

**`//go:embed`** bakes files into the binary at compile time (the engine YAMLs), so there's
nothing to install alongside it.

**cgo vs purego.** Calling C normally needs **cgo**, which needs a C toolchain and complicates
cross-compiling. Kepler uses **purego** to `dlopen` macOS `libSystem` and call `proc_pid_rusage`
directly, with the struct offsets hand-computed. This keeps `CGO_ENABLED=0` and a single static
binary.

**`os/exec` and `syscall`.** Engines run via `exec.Command`. `SysProcAttr{Setpgid: true}` puts
each one in a new process group. `syscall.Kill(-pgid, SIGTERM)` signals the whole group.
`syscall.Flock` implements the global lock.

**Testing conventions.**
- Files `*_test.go`, functions `TestXxx(t *testing.T)`.
- **Table-driven tests**: a map or slice of cases, looped.
- `t.TempDir()` gives auto-cleaned dirs; `httptest.NewServer` gives real local HTTP servers.
- **TestMain re-exec trick**: a test binary sets `KEPLER_FAKE_ENGINE=1` and launches *itself*
  as an engine; `fakeengine.MaybeServe()` in `TestMain` turns it into a fake OpenAI server. This
  tests real process management without real engines.
- `go vet` catches suspicious code; `gofmt` is the one true formatting.

**Libraries used:** cobra (CLI), yaml.v3 (strict decoding via `KnownFields(true)`),
modernc.org/sqlite (pure-Go SQLite, no cgo), purego. Everything else is the standard library.

**Good Go habits visible here:** small packages with one job; accepting interfaces and returning
structs; passing dependencies in explicitly (`Setup`, `Env` structs) instead of using globals;
pure functions where possible (the resolver and preflight are trivially testable).

---

## 10. Working on Kepler day to day

```shell
make build          # bin/kepler  (CGO off, version from git)
make test           # all unit tests, offline
make race           # tests with race detector
make vet            # gofmt check + go vet
make cross          # compile for linux/windows too
go test ./internal/hf/ -run TestParseRef -v    # one test, verbose
```

**Debugging a failed engine:** the error (and the result JSON `error` field) includes the
engine log's last lines; full log at `~/.kepler/logs/<run_id>/<engine>.log`. `kepler doctor`
re-runs health checks.

**Adding an engine** (e.g. one with an OpenAI-compatible server): copy
`definitions/engines/llamacpp.yaml`, change name, version, launch args, readiness URL, caps,
and the target's install method. Drop it in `~/.kepler/engines.d/` to try it without
rebuilding. `kepler engines list` shows whether it resolved. `kepler engines install <name>`
installs it and runs the health check.

**Reading a result file:** `model` (repo, commit, quant, sha256s, arch), `engine` (version,
exact launch args), `system_info`, `config` (preset, ctx, chars_per_token, memory metric),
`load`, `scenarios[]` (every iteration and its summary), `comparability`, `conditions`
(before/after), `log_path`.

**What's next:**
- **M4**: GGUF↔MLX counterpart discovery, so one command runs all three engines.
- **M5**: one-line `install.sh` and a release pipeline.
- **Later**: Linux/NVIDIA targets + vLLM/SGLang, a concurrency scenario, and the inference
  module + SwiftUI menu-bar monitor.

---

## 11. Glossary

| Term | Meaning |
|---|---|
| Token | integer ID for a chunk of text |
| Prefill / decode | process the prompt (parallel, compute-bound) / generate tokens (serial, bandwidth-bound) |
| KV cache | stored attention keys/values; grows with context |
| ctx | context length the engine allocates |
| TTFT | time to first token |
| ITL | inter-token latency |
| Quantization | storing weights in fewer bits |
| Q4_K_M | 4-bit k-quant, medium mix; Kepler's default |
| GGUF | llama.cpp single-file model format |
| safetensors | standard Transformers weight format |
| MLX | Apple's ML framework and its model format |
| mmproj | vision projector file for multimodal GGUFs |
| Shard | one part of a split model |
| LFS | Git Large File Storage; LFS files carry a sha256 |
| Revision / SHA | exact commit of a HF repo |
| Engine | program that runs a model (llama.cpp, Ollama, MLX…) |
| SSE | Server-Sent Events, the streaming HTTP format engines use |
| ignore_eos | forces generating exactly max_tokens |
| Prompt cache | engine reuses KV for repeated prompt prefixes |
| Unified memory | CPU+GPU share RAM (Apple Silicon) |
| Wired limit | max RAM the GPU may use |
| phys_footprint | macOS per-process memory incl. GPU, excl. mapped files |
| mmap | mapping a file into memory instead of reading it |
| Process group / pgid | a process and its children, signaled together |
| Ledger / reaper | write-ahead record of temp artifacts / startup cleanup of dead runs' leftovers |
| Canary | tiny model used for health checks |
| Target | one way to install/run an engine on one class of hardware |
| Resolver | pure function deciding which engines can run on a host |
| Preflight | "will it fit?" check before downloading |
| Comparability | whether a run honored the fairness rules |
| uv | fast Python package/env manager used for MLX |
| purego | call C libraries from Go without cgo |
