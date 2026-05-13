from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from kepler.benchmark.results import now_iso8601_micro, sanitize_repo
from kepler.config.loader import BenchmarkPreset

if TYPE_CHECKING:
    from kepler.benchmark.orchestrator import EngineRunRecord
    from kepler.models.resolver import ResolvedEngine


@dataclass
class EngineEntry:
    engine: str
    status: str
    reason: str | None = None
    summary: dict | None = None
    result_file: str | None = None
    error_class: str | None = None
    artifact: str | None = None


@dataclass
class ComparisonReport:
    schema_version: int
    comparison_id: str
    timestamp: str
    repo_id: str
    mode: str
    format: str
    system_info: dict[str, str]
    engines: list[EngineEntry] = field(default_factory=list)
    winner: dict | None = None
    narrative: str = ""


def build_report(
    comparison_id: str,
    resolved: list["ResolvedEngine"],
    records: list["EngineRunRecord"],
    preset: BenchmarkPreset,
    system_info: dict[str, str],
) -> ComparisonReport:
    repo_id = ""
    fmt = ""
    artifact_by_engine: dict[str, str] = {}
    for r in resolved:
        if r.spec is not None and not repo_id:
            repo_id = r.spec.repo_id
            fmt = r.spec.format.value
        if r.spec is not None:
            artifact_by_engine[r.engine_name] = _artifact_label(r.spec)

    entries: list[EngineEntry] = []
    for rec in records:
        summary = None
        result_file = None
        if rec.result is not None:
            s = rec.result.summary
            summary = {
                "median_wall_s": s.median_wall_s,
                "median_tok_per_s": s.median_tok_per_s,
                "median_ttft_ms": s.median_ttft_ms,
                "median_generation_tok_per_s": s.median_generation_tok_per_s,
            }
        if rec.result_file is not None:
            result_file = rec.result_file.name
        entries.append(
            EngineEntry(
                engine=rec.engine_name,
                status=rec.status,
                reason=rec.reason,
                summary=summary,
                result_file=result_file,
                error_class=rec.error_class,
                artifact=artifact_by_engine.get(rec.engine_name),
            )
        )

    winner = _pick_winner(entries)
    narrative = _narrative(entries, winner, system_info)

    return ComparisonReport(
        schema_version=1,
        comparison_id=comparison_id,
        timestamp=now_iso8601_micro(),
        repo_id=repo_id,
        mode=preset.prompt_set,
        format=fmt,
        system_info=system_info,
        engines=entries,
        winner=winner,
        narrative=narrative,
    )


def save_report(report: ComparisonReport, perf_dir: Path) -> Path:
    perf_dir.mkdir(parents=True, exist_ok=True)
    date = report.timestamp.split("T", 1)[0]
    repo = sanitize_repo(report.repo_id) if report.repo_id else "unknown"
    path = perf_dir / f"comparison_{date}_{repo}_{report.mode}.json"
    with path.open("w") as f:
        json.dump(_to_dict(report), f, indent=2, sort_keys=True)
    return path


def _to_dict(report: ComparisonReport) -> dict:
    d = asdict(report)
    d["engines"] = [{k: v for k, v in e.items() if v is not None} for e in d["engines"]]
    if d.get("winner") is None:
        d.pop("winner", None)
    return d


def _pick_winner(entries: list[EngineEntry]) -> dict | None:
    candidates = [e for e in entries if e.status == "ok" and e.summary]
    if not candidates:
        return None
    best = max(candidates, key=lambda e: e.summary["median_tok_per_s"])
    return {
        "engine": best.engine,
        "metric": "median_tok_per_s",
        "value": best.summary["median_tok_per_s"],
    }


def _narrative(
    entries: list[EngineEntry], winner: dict | None, system_info: dict[str, str]
) -> str:
    if winner is None:
        ok = [e for e in entries if e.status == "ok"]
        if not ok:
            return "No engine completed successfully."
        return f"{ok[0].engine} ran successfully on {system_info.get('processor', 'this machine')}."
    proc = system_info.get("processor", "this machine")
    ok = [e for e in entries if e.status == "ok" and e.summary]
    ok_sorted = sorted(ok, key=lambda e: e.summary["median_tok_per_s"], reverse=True)
    if len(ok_sorted) < 2:
        return f"{winner['engine']} reached {winner['value']:.1f} tok/s on {proc}."
    leader = ok_sorted[0]
    laggard = ok_sorted[-1]
    if laggard.summary["median_tok_per_s"] <= 0:
        return f"{leader.engine} leads at {leader.summary['median_tok_per_s']:.1f} tok/s on {proc}."
    ratio = leader.summary["median_tok_per_s"] / laggard.summary["median_tok_per_s"]
    return (
        f"{leader.engine} leads at {leader.summary['median_tok_per_s']:.1f} tok/s "
        f"({ratio:.2f}× faster than {laggard.engine}) on {proc}."
    )


def _artifact_label(spec) -> str:
    if spec.gguf_filename:
        return spec.gguf_filename
    if spec.ollama_tag:
        return f"ollama:{spec.ollama_tag}"
    if spec.local_path:
        return str(spec.local_path)
    return spec.hf_repo or spec.repo_id
