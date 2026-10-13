```shell
██╗  ██╗███████╗██████╗ ██╗     ███████╗██████╗
██║ ██╔╝██╔════╝██╔══██╗██║     ██╔════╝██╔══██╗
█████╔╝ █████╗  ██████╔╝██║     █████╗  ██████╔╝
██╔═██╗ ██╔══╝  ██╔═══╝ ██║     ██╔══╝  ██╔══██╗
██║  ██╗███████╗██║     ███████╗███████╗██║  ██║
╚═╝  ╚═╝╚══════╝╚═╝     ╚══════╝╚══════╝╚═╝  ╚═╝
```

Benchmark any LLM across every inference engine your machine can run.

> **Status:** Kepler 2.0 is a rewrite in Go, under active development. Engine setup and
> benchmarking work today on Apple Silicon.

## Supported

| Engine | Format | Apple Silicon |
|---|---|---|
| [llama.cpp](https://github.com/ggml-org/llama.cpp) (`llama-server`) | GGUF | ✓ |
| [MLX](https://github.com/ml-explore/mlx-lm) (`mlx_lm.server`) | MLX | ✓ |
| [Ollama](https://github.com/ollama/ollama) | GGUF | ✓ |
| [vLLM](https://github.com/vllm-project/vllm), [SGLang](https://github.com/sgl-project/sglang) | safetensors | planned |

NVIDIA and AMD support is planned. Engines are YAML definitions matched against detected
hardware, so new engines and platforms are added without changing Kepler's core.

## Build

Requires Go 1.25+.

```shell
make build        # → bin/kepler
make test         # unit tests (no network)
```

## Usage

```shell
kepler init                                        # detect hardware, install + health-check engines
kepler bench Qwen/Qwen2.5-1.5B-Instruct-GGUF       # benchmark every ready engine for this model
kepler bench unsloth/gemma-3-4b-it-GGUF:Q8_0 --preset quick
kepler bench https://huggingface.co/mlx-community/Qwen2.5-0.5B-Instruct-4bit
kepler results list                                # saved results, newest first
kepler results show <file>                         # re-render a result or comparison
kepler engines list                                # every engine, whether it can run here, and why
kepler doctor                                      # hardware report + re-run health checks
kepler clean                                       # remove leftovers from interrupted runs
```

`kepler bench` resolves the Hugging Face repo to a pinned commit, picks a quant
(`Q4_K_M` by default, or the best that fits), checks memory and disk, downloads the
model, benchmarks each engine one at a time, writes one JSON file per engine to
`~/.kepler/results`, and deletes the model afterwards (`--keep` to retain it).

| Preset | Scenarios | Iterations |
|---|---|---|
| `quick` | chat (256 tokens out), prefill-2k | 3 |
| `standard` (default) | chat, prefill-2k, prefill-8k, depth-16k | 5 |
| `full` | chat, prefill-2k/8k, depth-4k/16k/32k | 10 |

Each result records load time, TTFT, prefill and decode tokens/s, inter-token latency
(p50/p99), and peak memory (phys_footprint plus resident memory-mapped weights), with
p10/median/p90 per scenario. Every engine gets the same prompts (calibrated to the
model's tokenizer), `temperature=0`, a fixed seed, and a unique per-request prefix so
prompt caches can't help. Runs where an engine couldn't honor the spec, or the machine
was on battery or throttled, are marked `degraded` with the reasons.

Set `HF_TOKEN` for gated repositories.

Everything Kepler installs or downloads lives in `~/.kepler` (override with
`KEPLER_HOME` or `--home`). Kepler never uses your Hugging Face cache or your Ollama
model store, and an existing `ollama` on your `PATH` is reused but never modified.

## Adding an engine

Drop a YAML definition in `definitions/engines/` (shipped) or `~/.kepler/engines.d/`
(local). Engines that expose an OpenAI-compatible streaming endpoint need no Go code.
See `definitions/engines/llamacpp.yaml` for a complete example and
`internal/engines/def/types.go` for the schema.

## License

MIT — see [LICENSE](LICENSE).
