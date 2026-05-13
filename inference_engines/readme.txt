# LLM Inference engine informationss

## Engine profile

llama.cpp
  Speed: Moderate
  Hardware Needs: CPU/Any GPU
  Best Use Case: Local, personal use, universal compatibility
  macOS Metal Available: Yes (Mature)

vLLM
  Speed: Very Fast
  Hardware Needs: High-end GPU
  Best Use Case: Multi-user serving, enterprise throughput
  macOS Metal Available: No (NVIDIA/AMD focused)

vLLM-MLX
  Speed: Very Fast
  Hardware Needs: High-end Apple Silicon (M2 Ultra / M4 Max / M5)
  Best Use Case: Multi-user serving on Mac Studio/Pro hardware
  macOS Metal Available: Yes (Native MLX)

TensorRT-LLM
  Speed: Fastest
  Hardware Needs: NVIDIA GPU (required)
  Best Use Case: Production, maximum performance on NVIDIA hardware
  macOS Metal Available: No

MLX-LM
  Speed: Fast (Native on Apple Silicon)
  Hardware Needs: Apple Silicon (M-series)
  Best Use Case: Apple-native development, local research, and fine-tuning (LoRA)
  macOS Metal Available: Yes (Native)

Ollama
  Speed: Fast (MLX-backed for Mac)
  Hardware Needs: NVIDIA GPU or Apple Silicon (8GB+ RAM)
  Best Use Case: General consumer use; "it just works" simplicity
  macOS Metal Available: Yes (MLX-backed)

ik_llama.cpp
  Speed: Moderate-Fast
  Hardware Needs: CPU/Any GPU
  Best Use Case: Optimized local use, hybrid workflows
  macOS Metal Available: Yes (Metal)

SGLang
  Speed: Very Fast
  Hardware Needs: High-end GPU
  Best Use Case: Structured outputs (JSON), complex agents
  macOS Metal Available: Partial (In-development)

---

## Engine Support Matrix

| Engine           | Model Format       | Interface              | macOS Metal Available | Release Phase |
|------------------|--------------------|------------------------|-----------------------|---------------|
| llama-cpp-python | GGUF               | In-process Python      | Yes (Mature)          | Phase 1       |
| mlx-lm           | MLX / SafeTensors  | In-process Python      | Yes (Native)          | Phase 1       |
| Ollama           | GGUF + MLX         | REST API               | Yes (MLX-backed)      | Phase 1       |
| vllm-mlx         | MLX / SafeTensors  | REST API (OpenAI-comp) | Yes (Native MLX)      | Phase 1       |
| ik_llama         | GGUF               | Subprocess + REST      | Yes (Metal)           | Phase 1 (Opt) |
| SGLang           | SafeTensors        | REST API               | In-development        | Stub only     |

---

## Tradeoffs and Scenarios

- Maximum speed on NVIDIA? -> TensorRT-LLM
- Running on Apple Silicon? -> MLX-LM or Ollama (MLX-backed)
- Serving many concurrent users? -> vLLM or SGLang
- Running locally on a standard laptop? -> llama.cpp or ik_llama.cpp
- Local Fine-tuning (LoRA) on Mac? -> MLX-LM
- Need Structured JSON/Regex Outputs? -> SGLang or llama.cpp (GBNF)