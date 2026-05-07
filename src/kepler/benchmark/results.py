from __future__ import annotations

import json
import re
import statistics
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

from kepler.engines.base import EngineResult


@dataclass
class IterationRecord:
    wall_s: float
    output_text: str
    prompt_tokens: int
    generated_tokens: int
    tok_per_s: float
    generation_tok_per_s: float
    ttft_ms: float
    prefill_ms: float
    generation_ms: float


@dataclass
class BenchmarkSummary:
    median_wall_s: float
    median_tok_per_s: float
    median_ttft_ms: float
    median_generation_tok_per_s: float


@dataclass
class BenchmarkConfig:
    prompt_set: str
    prompt: str
    max_tokens: int
    temperature: float
    iterations: int
    host: str | None = None
    port: int | None = None


@dataclass
class BenchmarkResult:
    timestamp: str
    repo_id: str
    model_ref: str
    engine: str
    system_info: dict[str, str]
    config: BenchmarkConfig
    iterations: list[IterationRecord]
    summary: BenchmarkSummary
    mode: str = "text-only"
    environment: str = "stable"
    comparison_id: str | None = None


def vec_median(xs: list[float]) -> float:
    """Matches legacy C++ vec_median (legacy/src/benchmark.cpp:28):
    sort ascending, return v[v.size() / 2] — the upper middle for even-length."""
    s = sorted(xs)
    return s[len(s) // 2]


def now_iso8601_micro() -> str:
    return datetime.now().isoformat(timespec="microseconds")


def build_iteration(r: EngineResult) -> IterationRecord:
    return IterationRecord(
        wall_s=r.wall_s,
        output_text=r.output_text,
        prompt_tokens=r.prompt_tokens,
        generated_tokens=r.generated_tokens,
        tok_per_s=r.tok_per_s,
        generation_tok_per_s=r.generation_tok_per_s,
        ttft_ms=r.ttft_ms,
        prefill_ms=r.prefill_ms,
        generation_ms=r.generation_ms,
    )


def build_summary(iters: list[IterationRecord]) -> BenchmarkSummary:
    return BenchmarkSummary(
        median_wall_s=vec_median([i.wall_s for i in iters]),
        median_tok_per_s=vec_median([i.tok_per_s for i in iters]),
        median_ttft_ms=vec_median([i.ttft_ms for i in iters]),
        median_generation_tok_per_s=vec_median([i.generation_tok_per_s for i in iters]),
    )


def sanitize_repo(repo_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", repo_id).strip("-")


def result_filename(result: BenchmarkResult) -> str:
    date = result.timestamp.split("T", 1)[0]
    repo = sanitize_repo(result.repo_id)
    return f"{date}_{repo}_{result.engine}_{result.mode}.json"


def save_result(result: BenchmarkResult, perf_dir: Path) -> Path:
    perf_dir.mkdir(parents=True, exist_ok=True)
    path = perf_dir / result_filename(result)
    with path.open("w") as f:
        json.dump(_to_dict(result), f, indent=2, sort_keys=True)
    return path


def _to_dict(result: BenchmarkResult) -> dict:
    d = asdict(result)
    if d.get("comparison_id") is None:
        d.pop("comparison_id", None)
    return d
