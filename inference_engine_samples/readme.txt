Inference Engines

llama.cpp
  Speed: Moderate
  Hardware Needs: CPU/Any GPU
  Best Use Case: Local, personal use

vLLM
  Speed: Very Fast
  Hardware Needs: High-end GPU
  Best Use Case: Multi-user serving

TensorRT-LLM
  Speed: Fastest
  Hardware Needs: NVIDIA GPU (required)
  Best Use Case: Production, max performance

ik_llama.cpp
  Speed: Moderate-Fast
  Hardware Needs: CPU/Any GPU
  Best Use Case: Optimized local use

SGLang
  Speed: Very Fast
  Hardware Needs: High-end GPU
  Best Use Case: Structured outputs, agents

Different tradeoffs for different scenarios:
- Need max speed + have NVIDIA GPU? → TensorRT-LLM
- Serving many users? → vLLM or SGLang
- Running locally on laptop? → llama.cpp or ik_llama.cpp