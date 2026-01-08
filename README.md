# Kepler

Personal LLM workbench for macOS. Select, serve, and benchmark models (GGUF) via llama.cpp in Docker.

## Setup

```bash
make install
```

## Run

Download the desired .gguf model into the `models` directory. ([Instructions](https://github.com/thisisadityapatel/kepler/blob/main/models/readme.md))

Then run:

```bash
make run
```

Direct mode with specific model and benchmark preset (quick, standard, performance, hard) using:

```bash
uv run python utilities/workbench.py --model path/to/model.gguf --benchmark quick
```
