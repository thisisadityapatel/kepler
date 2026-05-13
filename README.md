# Kepler

Benchmark LLMs on macOS Metal GPUs across multiple inference engines.

Supports [llama.cpp](https://github.com/ggml-org/llama.cpp), [mlx-lm](https://github.com/ml-explore/mlx-lm), and [Ollama](https://github.com/ollama/ollama) out of the box, with a unified CLI for running the same model across engines and comparing throughput, TTFT, and memory.

Work in progress to add [ik-llama](https://github.com/ikawrakow/ik_llama.cpp), [SGLang](https://github.com/sgl-project/sglang) and [vLLM](https://github.com/vllm-project/vllm) inference engines.

## Requirements

- Python 3.12+
- [uv](https://github.com/astral-sh/uv)
- macOS on Apple Silicon (M Series)

## Setup

```shell
git clone https://github.com/thisisadityapatel/kepler.git
cd kepler
uv venv
uv sync
```

## Get a model

GGUF (for `llama.cpp` / `Ollama`) — drop the .gguf file in `models/`:

```shell
hf download Qwen/Qwen2.5-0.5B-Instruct-GGUF qwen2.5-0.5b-instruct-q5_k_m.gguf --local-dir models/
```

MLX engines fetch their own artifacts from Hugging Face on first run.

## Run

MLX format:

```shell
uv run kepler benchmark mlx-community/Qwen2.5-0.5B-Instruct-4bit --format mlx
```

GGUF format:

```shell
uv run kepler benchmark qwen2.5:0.5b --format gguf
```

Results land in `perf/` as JSON, plus a `comparison_<id>.json` when more than one engine runs.

### Options

| Flag | Values | Description |
|------|--------|-------------|
| `--format` | `gguf`, `mlx` | Model format. Required unless `--engine` implies one. |
| `--mode` | `quick`, `standard`, `performance` | Preset for iterations, max tokens, and prompt. Default `standard`. |
| `--engine` | `llamacpp`, `mlx`, `ollama`, `vllm`, `sglang` | Run a single engine (skips comparison output). |
| `--only-engine` | comma-separated list | Allowlist of engines to run. |
| `--skip-engine` | comma-separated list | Denylist of engines to skip. |
| `--cooldown` | integer (seconds) | Pause between engines for thermal recovery. Default `8`. |
| `--iterations` | integer | Override the preset iteration count. |
| `--max-tokens` | integer | Override the preset max tokens. |
| `--temperature` | float | Override the preset temperature. |
| `--ollama-tag` | string | Override the Ollama tag derived from the model id. |
| `--offline` | flag | Skip Hugging Face Hub probes; use the local cache only. |

## Other commands

```shell
uv run kepler list-engines     # show engines and availability
uv run kepler list-models      # show locally downloaded models
uv run kepler results          # summarize past runs
```

## License

MIT — see [LICENSE](LICENSE).
