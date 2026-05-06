## Kepler

Inference, benchmark, and compare LLMs (GGUF) via llama.cpp engine on macOS with native Metal GPU acceleration.

<div align="center">
  <video src="https://github.com/user-attachments/assets/b967c2fc-588f-4bcb-9971-87030a692327"></video>
</div>

---

### Requirements

- macOS on Apple Silicon (M1 or later)
- Xcode Command Line Tools: `xcode-select --install`

---

### Setup

**1. Install and build**

```bash
make setup
```

This installs [Nix](https://nixos.org/) if you don't have it, then builds the binary. If Nix was just installed, open a new terminal and run `make build` to finish.

**2. Add models**

Drop `.gguf` model files into the `models/` directory, Example:

```
models/
  gemma-4-E4B-it-Q8_0.gguf
```

Models can be downloaded from [Hugging Face](https://huggingface.co/models?library=gguf).

---

### Run

```bash
make run
```

Kepler will prompt you to select a model and benchmark type, then run the benchmark and save results to `perf/`.

**Pass CLI flags:**

```bash
make run ARGS="-m models/gemma-4-E4B-it-Q8_0.gguf -b standard"
```

**List available models:**

```bash
make run ARGS="--list-models"
```

---

### Benchmark types

| Type          | Iterations | Tokens |
|---------------|-----------|--------|
| `quick`       | 2         | 50     |
| `standard`    | 3         | 100    |
| `performance` | 5         | 200    |
| `hard`        | 3 × 3     | varies |

Results are saved as JSON in `perf/`.
