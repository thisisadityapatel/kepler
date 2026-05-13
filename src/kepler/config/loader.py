from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class BenchmarkPreset:
    prompt_set: str
    prompt: str
    max_tokens: int
    temperature: float
    iterations: int


@dataclass
class HardQuestion:
    prompt: str
    max_tokens: int
    description: str


@dataclass
class AppConfig:
    presets: dict[str, BenchmarkPreset] = field(default_factory=dict)
    hard_questions: list[HardQuestion] = field(default_factory=list)
    backends: dict = field(default_factory=dict)


PRESET_KEYS = {
    "quick": "quick_benchmark",
    "standard": "standard_benchmark",
    "performance": "performance_benchmark",
}


def load_config(path: Path) -> AppConfig:
    with path.open() as f:
        raw = yaml.safe_load(f) or {}
    presets: dict[str, BenchmarkPreset] = {}
    for mode, key in PRESET_KEYS.items():
        block = raw.get(key)
        if block is None:
            continue
        presets[mode] = BenchmarkPreset(
            prompt_set=block.get("prompt_set", mode),
            prompt=block["prompt"],
            max_tokens=int(block["max_tokens"]),
            temperature=float(block["temperature"]),
            iterations=int(block["iterations"]),
        )
    hard: list[HardQuestion] = []
    for q in (raw.get("benchmark_questions") or {}).get("hard_questions", []) or []:
        hard.append(
            HardQuestion(
                prompt=q["prompt"],
                max_tokens=int(q["max_tokens"]),
                description=q.get("description", ""),
            )
        )
    backends = (raw.get("defaults") or {}).get("backends", {})
    return AppConfig(presets=presets, hard_questions=hard, backends=backends)


def find_config(start: Path | None = None) -> Path:
    """Walk up from start (or CWD) looking for config/models.yaml."""
    here = (start or Path.cwd()).resolve()
    for parent in [here, *here.parents]:
        candidate = parent / "config" / "models.yaml"
        if candidate.is_file():
            return candidate
    raise FileNotFoundError("config/models.yaml not found in cwd or any parent directory")
