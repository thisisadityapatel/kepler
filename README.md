# Kepler

Personal LLM workbench for macOS. Select, serve, and benchmark models (GGUF) via llama.cpp in Docker.

<div align="center">
  <video src="https://github.com/user-attachments/assets/e682d0c0-749a-464d-848a-d2ee92155fc4">
    Your browser does not support the video tag.
  </video>
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
