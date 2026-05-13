from __future__ import annotations

import platform
import time
from typing import Any

from kepler.engines.base import (
    EngineCapabilities,
    EngineResult,
    InferenceEngine,
    ModelFormat,
)
from kepler.models.registry import ModelSpec


class MlxEngine(InferenceEngine):
    """In-process MLX inference via mlx-lm. Apple Silicon (Darwin/arm64) only."""

    name = "mlx"

    def __init__(self) -> None:
        self._model: Any = None
        self._tokenizer: Any = None
        self._spec: ModelSpec | None = None

    @property
    def capabilities(self) -> EngineCapabilities:
        avail, reason = _probe()
        return EngineCapabilities(
            name=self.name,
            supported_formats={ModelFormat.MLX},
            available=avail,
            unavailable_reason=reason,
            requires_server=False,
        )

    def load(self, model: ModelSpec) -> None:
        if self._model is not None:
            return
        if model.local_path is not None:
            target = str(model.local_path)
        elif model.hf_repo is not None:
            target = model.hf_repo
        else:
            raise RuntimeError("MlxEngine.load: ModelSpec needs local_path or hf_repo")

        from mlx_lm import load as mlx_load

        self._model, self._tokenizer = mlx_load(target)
        self._spec = model

    def infer(self, prompt: str, max_tokens: int, temperature: float) -> EngineResult:
        if self._model is None or self._tokenizer is None:
            raise RuntimeError("MlxEngine.infer called before load()")

        from mlx_lm import stream_generate
        from mlx_lm.sample_utils import make_sampler

        sampler = make_sampler(temp=temperature)

        chunks: list[str] = []
        t0 = time.perf_counter()
        ttft: float | None = None
        gen_started: float | None = None
        last: Any = None

        for resp in stream_generate(
            self._model,
            self._tokenizer,
            prompt,
            max_tokens=max_tokens,
            sampler=sampler,
        ):
            if ttft is None:
                ttft = time.perf_counter() - t0
                gen_started = time.perf_counter()
            if resp.text:
                chunks.append(resp.text)
            last = resp
        wall_s = time.perf_counter() - t0

        text = "".join(chunks)
        prompt_tokens = int(getattr(last, "prompt_tokens", 0) or 0) if last is not None else 0
        generated_tokens = (
            int(getattr(last, "generation_tokens", 0) or 0) if last is not None else 0
        )

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
        spec = self._spec
        self._model = None
        self._tokenizer = None
        self._spec = None
        try:
            import mlx.core as mx

            if hasattr(mx, "clear_cache"):
                mx.clear_cache()
            elif hasattr(mx, "metal") and hasattr(mx.metal, "clear_cache"):
                mx.metal.clear_cache()
        except Exception:
            pass
        # Remove downloaded HF weights for this repo only. Never touch any other
        # repo in the cache; never touch user-supplied local_path directories.
        if spec is not None and spec.hf_repo and spec.local_path is None:
            _delete_hf_repo_cache(spec.hf_repo)


def _delete_hf_repo_cache(repo_id: str) -> None:
    """Remove all cached revisions of `repo_id` from the HF hub cache. Best-effort:
    deletion failures are swallowed so they can't fail a successful benchmark."""
    try:
        from huggingface_hub import scan_cache_dir

        info = scan_cache_dir()
        hashes: list[str] = []
        for repo in info.repos:
            if repo.repo_type == "model" and repo.repo_id == repo_id:
                hashes.extend(rev.commit_hash for rev in repo.revisions)
        if hashes:
            info.delete_revisions(*hashes).execute()
    except Exception:
        pass


def _probe() -> tuple[bool, str | None]:
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        return False, "MLX requires macOS on Apple Silicon (arm64)"
    try:
        import mlx.core  # noqa: F401
    except ImportError as exc:
        return False, f"mlx not installed: {exc}"
    try:
        import mlx_lm  # noqa: F401
    except ImportError as exc:
        return False, f"mlx-lm not installed: {exc}"
    return True, None
