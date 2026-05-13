from __future__ import annotations

from kepler.engines.base import EngineCapabilities, ModelFormat


def list_capabilities() -> list[EngineCapabilities]:
    """Probe every known engine for availability. Imports are lazy and isolated
    so an engine missing its deps doesn't crash the rest of the registry."""
    out: list[EngineCapabilities] = []
    for name in ENGINE_ORDER:
        out.append(_probe(name))
    return out


def get_capability(name: str) -> EngineCapabilities:
    return _probe(name)


def engines_for_format(fmt: ModelFormat) -> list[str]:
    """Engine names known to support `fmt`, in deterministic order."""
    return [n for n in ENGINE_ORDER if fmt in FORMAT_SUPPORT.get(n, set())]


# Order is the canonical comparison-report ordering.
# ik_llama is intentionally excluded — its only backends are CPU and CUDA
# (per its README), so it can't run on macOS Metal GPU like the rest.
ENGINE_ORDER: list[str] = [
    "llamacpp",
    "mlx",
    "ollama",
    "vllm",
    "sglang",
]

FORMAT_SUPPORT: dict[str, set[ModelFormat]] = {
    "llamacpp": {ModelFormat.GGUF},
    "mlx": {ModelFormat.MLX},
    "ollama": {ModelFormat.GGUF, ModelFormat.MLX},
    "vllm": {ModelFormat.MLX},
    "sglang": {ModelFormat.MLX},
}


def _probe(name: str) -> EngineCapabilities:
    if name == "llamacpp":
        from kepler.engines.llamacpp import LlamaCppEngine

        return LlamaCppEngine().capabilities
    if name == "ollama":
        from kepler.engines.ollama import OllamaEngine

        return OllamaEngine().capabilities
    if name == "mlx":
        from kepler.engines.mlx import MlxEngine

        return MlxEngine().capabilities
    if name == "vllm":
        return EngineCapabilities(
            name="vllm",
            supported_formats={ModelFormat.MLX},
            available=False,
            unavailable_reason="not yet implemented (M3)",
            requires_server=True,
        )
    if name == "sglang":
        return EngineCapabilities(
            name="sglang",
            supported_formats={ModelFormat.MLX},
            available=False,
            unavailable_reason="stub — Metal support in development",
        )
    return EngineCapabilities(
        name=name, supported_formats=set(), available=False, unavailable_reason="unknown engine"
    )
