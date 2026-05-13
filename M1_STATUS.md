# Kepler Python rewrite — M1 status

Branch: `version3.0`. Plan: `~/.claude/plans/here-is-the-original-gleaming-beacon.md`.

## What works end-to-end

`kepler benchmark <model> --format gguf` downloads a GGUF artifact from HF Hub, runs every available compatible engine sequentially in isolated subprocesses, writes one per-engine JSON and one comparison JSON, and renders a Rich comparison table.

Verified on Apple M4 with `Qwen/Qwen2.5-0.5B-Instruct --format gguf --mode quick`:

```
Engine     tok/s   ttft (ms)   gen tok/s   wall (s)   status
llamacpp    83.1   12          84.7        0.60       ok
ik_llama    —      —           —           —          unavailable: not yet implemented (M4)
Winner: llamacpp (83.1 tok/s)
```

Sub-commands all working: `kepler list-engines`, `list-models`, `results`, `benchmark --help`.

## Files created (M1)

- `pyproject.toml`, `config/models.yaml` (copied from `legacy/config/models.yaml` verbatim)
- `src/kepler/__init__.py`, `__main__.py`
- `src/kepler/cli/app.py` — Typer CLI: `benchmark`, `list-engines`, `list-models`, `results`
- `src/kepler/config/loader.py` — YAML → `BenchmarkPreset` / `HardQuestion` / `AppConfig`
- `src/kepler/engines/base.py` — `InferenceEngine` ABC, `EngineResult`, `EngineCapabilities`, `ModelFormat`
- `src/kepler/engines/registry.py` — availability probes, canonical engine ordering, format support map
- `src/kepler/engines/llamacpp.py` — in-process GGUF, streaming for real TTFT
- `src/kepler/engines/ollama.py` — daemon discovery / spawn, `/api/pull`, `/api/generate`, unload via `keep_alive: 0`
- `src/kepler/models/registry.py` — `ModelSpec`
- `src/kepler/models/resolver.py` — single-format probe → `list[ResolvedEngine]`, local-path detection, HF fallback to `<owner>/<name>-GGUF`
- `src/kepler/models/downloader.py` — `hf_hub_download`/`snapshot_download` wrappers; case-insensitive GGUF quant picker (Q5_K_M > Q4_K_M > Q8_0 > …)
- `src/kepler/benchmark/system_info.py` — `platform.system()` + `sysctl machdep.cpu.brand_string`
- `src/kepler/benchmark/results.py` — `BenchmarkResult`, `IterationRecord`, `BenchmarkSummary`, `vec_median` (matches `legacy/src/benchmark.cpp:28`), `save_result`
- `src/kepler/benchmark/lifecycle.py` — `multiprocessing.spawn` subprocess isolation, `verify_released` (PID reaped + port closed + no descendants), `cooldown`, `find_free_port`
- `src/kepler/benchmark/runner.py` — warmup + N iterations
- `src/kepler/benchmark/orchestrator.py` — sequential driver, child entrypoint dispatch, builds `BenchmarkResult` per engine
- `src/kepler/benchmark/comparison.py` — `ComparisonReport`, winner selection, narrative line, save
- `src/kepler/ui/display.py` — banner, info/success/warn/error helpers
- `src/kepler/ui/progress.py` — Rich progress factories
- `src/kepler/ui/report.py` — final comparison table with best-per-column highlighting

JSON outputs preserve the legacy `legacy/perf/2026-05-01_*.json` schema exactly. New optional field: `comparison_id` (ULID). New companion file: `perf/comparison_<date>_<repo>_<mode>.json` (schema_version=1).

## Bugs found and fixed during smoke test

1. **Resolver crashed unavailable engines** — engines without their deps installed (M2/M3/M4 stubs) were sent to `_run_one_engine` and showed up as `failed: not wired in child entrypoint yet`. Fixed by filtering them to `unavailable` ResolvedEngine entries before resolution.
2. **llama-cpp-python had no `timings` field** — non-streaming response shape doesn't include it. Switched `LlamaCppEngine.infer` to streaming and time TTFT manually.
3. **GGUF picker case-mismatch** — repos use lowercase `q5_k_m` filenames; my `Q5_K_M` priority list missed them and fell through to FP16. Now case-insensitive.
4. **Long error stack traces wrecked the report table** — `_short_reason` truncates to first line, ~80 chars.

## Dev workflow caveat

macOS Sequoia auto-tags files written by Claude Code in `~/Documents/...` venvs with `com.apple.provenance` and the `hidden` flag. Python's `site` module silently skips hidden `.pth` files, so `uv pip install -e .` (editable) doesn't expose the package. **Workaround:** use a non-editable install — `uv pip install . --no-deps --reinstall` after code changes. Wheel install copies the package into `site-packages/kepler/` directly, no `.pth` involved.

## Setup

```
uv venv --python 3.12 .venv
uv pip install . --no-deps
uv pip install typer rich pyyaml psutil python-ulid huggingface_hub httpx openai llama-cpp-python
```

`llama-cpp-python` 0.3.22 ships a wheel with Metal pre-enabled on Apple Silicon — no `CMAKE_ARGS` needed.

## What's next

- **M2** — MLX engine (`src/kepler/engines/mlx.py`), wire MLX branch in resolver, child dispatch in orchestrator. Unlocks `--format mlx`.
- **M3** — vllm-mlx (`src/kepler/engines/vllm.py`), spawned OpenAI-compat server, ephemeral port allocation, process-tree teardown.
- **M4** — `ik_llama` (PATH-only spawn), `sglang` stub, `kepler doctor` command.
- **M5** — narrative tuning, `--strict`, `--offline` end-to-end test, `--allow-mixed-quants`, README.

## Open items

- Comparison runs with only one ready engine still write a `comparison_*.json` file (intentional — keeps schema deterministic). Consider suppressing when only one engine ran successfully.
- `verify_released` is currently called from inside the orchestrator implicitly via `proc.join` — there's no explicit post-engine verification step yet for the spawned-server engines (M3 will need it).
- Two-region live progress UI is not yet wired; current UX uses simple `info`/`success` lines per engine. Plan calls for `rich.live.Live` two-region tracker — punted to M2 once a second engine exists to compare against.
- Thermal cool-down currently sleeps without a visible countdown UI for `--cooldown > 0`. The `cooldown_tick` callback is wired but not exercised in M1's smoke test (we ran with `--cooldown 0`).
