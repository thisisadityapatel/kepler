# Kepler

Personal LLM workbench for macOS. Select, serve, and benchmark models (GGUF) via llama.cpp in Docker.

<div align="center">
  <video src="https://github.com/user-attachments/assets/e382215c-9c1f-4462-bdbe-778982e87d3e"></video>
</div>

## Setup

```bash
make install
```

## Run

Download .gguf model into the `models` directory. ([Instructions](https://github.com/thisisadityapatel/kepler/blob/main/models/readme.md))

Then run:

```bash
make run
```

Direct mode with specific model and benchmark preset (quick, standard, performance, hard) using:

```bash
uv run python utilities/workbench.py --model path/to/model.gguf --benchmark quick
```
