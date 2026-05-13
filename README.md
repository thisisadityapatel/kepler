# Kepler

Benchmark LLMs on macOS Metal GPUs across multiple inference engines.

Supports `llama.cpp`, `mlx-lm`, and `Ollama` out of the box, with a unified CLI for running the same model across engines and comparing throughput, TTFT, and memory.

Work in progress to add `ik-llama`, `SGLang` and `vLLM` inference engines.

## Requirements

- Python 3.12+
- [uv](https://github.com/astral-sh/uv)
- [Ollama](https://ollama.com) running locally (optional, for the Ollama engine)
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

```shell
uv run kepler benchmark mlx-community/Qwen2.5-0.5B-Instruct-4bit --format mlx
```

Modes: `quick`, `standard`, `performance`. Results land in `perf/` as JSON.

## Other commands

```shell
uv run kepler list-engines     # show engines and availability
uv run kepler list-models      # show locally downloaded models
uv run kepler results          # summarize past runs
```

## License

MIT — see [LICENSE](LICENSE).
