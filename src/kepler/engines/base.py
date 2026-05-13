from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum


class ModelFormat(str, Enum):
    GGUF = "gguf"
    MLX = "mlx"


@dataclass
class EngineResult:
    """One inference result with timing — the unit a runner collects per iteration."""

    output_text: str
    prompt_tokens: int
    generated_tokens: int
    wall_s: float
    ttft_ms: float
    prefill_ms: float
    generation_ms: float
    generation_tok_per_s: float
    tok_per_s: float


@dataclass
class EngineCapabilities:
    name: str
    supported_formats: set[ModelFormat] = field(default_factory=set)
    available: bool = False
    unavailable_reason: str | None = None
    requires_server: bool = False


class InferenceEngine(ABC):
    """All engines implement this. Each engine is run inside a fresh subprocess so
    the GPU is fully released by process exit before the next engine starts."""

    @property
    @abstractmethod
    def capabilities(self) -> EngineCapabilities: ...

    @abstractmethod
    def load(self, model) -> None:
        """Download (if needed), load into memory or start a server. Idempotent."""

    @abstractmethod
    def infer(self, prompt: str, max_tokens: int, temperature: float) -> EngineResult: ...

    @abstractmethod
    def unload(self) -> None:
        """Release memory / stop server process."""

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.unload()
