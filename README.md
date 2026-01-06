# Kepler

LLM benchmarking workbench for macOS. Select, serve, and benchmark GGUF models via llama.cpp in Docker.

## Setup

```bash
make install
```

## Run

Interactive mode: select model and benchmark.

```bash
python utilities/workbench.py
```

Direct mode with specific model and benchmark preset (quick, standard, performance, hard).

```bash
python utilities/workbench.py --model path/to/model.gguf --benchmark quick
```