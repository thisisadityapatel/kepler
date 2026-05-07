from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from kepler.engines.base import ModelFormat


@dataclass
class ModelSpec:
    """What an engine needs to load a model. Built by the resolver."""

    repo_id: str
    display_name: str
    format: ModelFormat
    hf_repo: str | None = None
    local_path: Path | None = None
    gguf_filename: str | None = None
    ollama_tag: str | None = None
    quantization: str | None = None
