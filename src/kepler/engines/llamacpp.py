from __future__ import annotations

import time
from typing import Any

from kepler.engines.base import (
    EngineCapabilities,
    EngineResult,
    InferenceEngine,
    ModelFormat,
)
from kepler.models.registry import ModelSpec


class LlamaCppEngine(InferenceEngine):
    """In-process GGUF via llama-cpp-python with Metal offload."""

    name = "llamacpp-python"

    def __init__(self, n_ctx: int = 4096, n_gpu_layers: int = -1):
        self.n_ctx = n_ctx
        self.n_gpu_layers = n_gpu_layers
        self._llm: Any = None
        self._model: ModelSpec | None = None

    @property
    def capabilities(self) -> EngineCapabilities:
        avail, reason = _probe()
        return EngineCapabilities(
            name=self.name,
            supported_formats={ModelFormat.GGUF},
            available=avail,
            unavailable_reason=reason,
            requires_server=False,
        )

    def load(self, model: ModelSpec) -> None:
        if self._llm is not None:
            return
        if model.local_path is None:
            raise RuntimeError("LlamaCppEngine.load: ModelSpec.local_path is required")
        import llama_cpp

        self._llm = llama_cpp.Llama(
            model_path=str(model.local_path),
            n_gpu_layers=self.n_gpu_layers,
            n_ctx=self.n_ctx,
            flash_attn=True,
            verbose=False,
        )
        self._model = model

    def infer(self, prompt: str, max_tokens: int, temperature: float) -> EngineResult:
        if self._llm is None:
            raise RuntimeError("LlamaCppEngine.infer called before load()")

        # Stream so we can measure TTFT directly. llama-cpp-python's non-streaming
        # response does not include a `timings` field, so we time it ourselves.
        chunks: list[str] = []
        t0 = time.perf_counter()
        ttft = None
        gen_started = None
        first_chunk_received = False

        stream = self._llm(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            stream=True,
        )
        for chunk in stream:
            if not first_chunk_received:
                ttft = time.perf_counter() - t0
                gen_started = time.perf_counter()
                first_chunk_received = True
            choice = chunk.get("choices", [{}])[0] or {}
            piece = choice.get("text", "")
            if piece:
                chunks.append(piece)
        wall_s = time.perf_counter() - t0

        text = "".join(chunks)
        prompt_tokens = len(self._llm.tokenize(prompt.encode("utf-8")))
        generated_tokens = len(self._llm.tokenize(text.encode("utf-8"))) if text else 0

        prefill_ms = (ttft or 0.0) * 1000.0
        gen_end = time.perf_counter()
        generation_ms = ((gen_end - gen_started) * 1000.0) if gen_started else 0.0
        gen_tps = generated_tokens / (generation_ms / 1000.0) if generation_ms > 0 else 0.0
        tok_per_s = generated_tokens / wall_s if wall_s > 0 else 0.0

        return EngineResult(
            output_text=text,
            prompt_tokens=prompt_tokens,
            generated_tokens=generated_tokens,
            wall_s=wall_s,
            ttft_ms=prefill_ms,
            prefill_ms=prefill_ms,
            generation_ms=generation_ms,
            generation_tok_per_s=gen_tps,
            tok_per_s=tok_per_s,
        )

    def unload(self) -> None:
        self._llm = None
        self._model = None


def _probe() -> tuple[bool, str | None]:
    try:
        import llama_cpp
    except ImportError as exc:
        return False, f"llama-cpp-python not installed: {exc}"
    try:
        if hasattr(llama_cpp, "llama_supports_gpu_offload"):
            if not llama_cpp.llama_supports_gpu_offload():
                return True, "Metal GPU offload unavailable — will run on CPU"
    except Exception:
        pass
    return True, None
