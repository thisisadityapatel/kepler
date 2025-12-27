"""
Kepler Benchmark Runner
(llama.cpp backend for benchmaring the model on macOS metal silicon GPUs)

Use --model to define the model to benchmark.

Example:
uv run python scripts/run_bench.py --model models/zai-org/GLM-4.6V-FP8
"""

import time

import requests


def bench_once_llama(
    prompt: str, max_tokens: int, temperature: float, host: str, port: int
) -> dict:
    """Run single inference and extract timing metrics."""

    # 1. Build request payload
    url = f"http://{host}:{port}/completion"
    payload = {
        "prompt": prompt,
        "n_predict": max_tokens,
        "temperature": temperature,
    }

    # 2. Send request and measure wall time
    t0 = time.perf_counter()
    response = requests.post(url, json=payload, timeout=300)
    response.raise_for_status()
    t1 = time.perf_counter()

    wall_time = t1 - t0

    # 3. Extract metrics from llama.cpp response
    data = response.json()
    text = data.get("content", "")
    timings = data.get("timings", {})

    # llama.cpp gives you detailed timings!
    prompt_tokens = data.get("tokens_evaluated")  # Input tokens
    prompt_ms = timings.get("prompt_ms")  # Time to process prompt
    gen_tokens = timings.get("predicted_n")  # Output tokens
    gen_ms = timings.get("predicted_ms")  # Time to generate
    gen_tok_per_s = timings.get("predicted_per_second")  # Tokens/sec

    # 4. Compute overall throughput
    tok_per_s = gen_tokens / wall_time if wall_time > 0 else None

    # 5. Return standardized metrics
    return {
        "wall_s": wall_time,
        "prompt_tokens": prompt_tokens,
        "generated_tokens": gen_tokens,
        "ttft_ms": prompt_ms,  # Time to first token
        "tok_per_s": tok_per_s,  # Overall throughput
        "generation_tok_per_s": gen_tok_per_s,  # Decode-only throughput
        "generation_ms": gen_ms,
        "output_text": text,
    }
