from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from ulid import ULID

from kepler.benchmark import comparison, lifecycle
from kepler.benchmark.results import (
    BenchmarkConfig,
    BenchmarkResult,
    build_iteration,
    build_summary,
    now_iso8601_micro,
    save_result,
)
from kepler.benchmark.system_info import get_system_info
from kepler.config.loader import BenchmarkPreset
from kepler.engines.base import EngineResult
from kepler.models.resolver import ResolvedEngine

EngineRunStatus = Literal["ok", "skipped", "unavailable", "failed"]


@dataclass
class EngineRunRecord:
    engine_name: str
    status: EngineRunStatus
    reason: str | None = None
    result: BenchmarkResult | None = None
    result_file: Path | None = None
    error_class: str | None = None


def run_comparison(
    resolved: list[ResolvedEngine],
    preset: BenchmarkPreset,
    perf_dir: Path,
    cooldown_s: int = 8,
    write_comparison: bool = True,
    cooldown_tick=None,
    engine_progress_cb=None,
) -> tuple[comparison.ComparisonReport, list[EngineRunRecord]]:
    comparison_id = str(ULID())
    sys_info = get_system_info()
    completed: list[EngineRunRecord] = []
    runnable_indices = [i for i, r in enumerate(resolved) if r.status == "ready"]

    for idx, r in enumerate(resolved):
        if r.status != "ready":
            completed.append(
                EngineRunRecord(
                    engine_name=r.engine_name,
                    status=("skipped" if r.status == "skipped" else "unavailable"),
                    reason=r.reason,
                )
            )
            continue
        if engine_progress_cb:
            engine_progress_cb(r.engine_name, "starting")
        record = _run_one_engine(
            resolved=r,
            preset=preset,
            comparison_id=comparison_id,
            sys_info=sys_info,
            perf_dir=perf_dir,
            engine_progress_cb=engine_progress_cb,
        )
        completed.append(record)

        is_last_runnable = runnable_indices and idx == runnable_indices[-1]
        if record.status == "ok" and not is_last_runnable and cooldown_s > 0:
            lifecycle.cooldown(cooldown_s, on_tick=cooldown_tick)

    report = comparison.build_report(
        comparison_id=comparison_id,
        resolved=resolved,
        records=completed,
        preset=preset,
        system_info=sys_info,
    )
    if write_comparison:
        comparison.save_report(report, perf_dir)
    return report, completed


def _run_one_engine(
    resolved: ResolvedEngine,
    preset: BenchmarkPreset,
    comparison_id: str,
    sys_info: dict[str, str],
    perf_dir: Path,
    engine_progress_cb,
) -> EngineRunRecord:
    spec = resolved.spec
    if spec is None:
        return EngineRunRecord(
            engine_name=resolved.engine_name,
            status="failed",
            reason="resolver produced no spec",
            error_class="MissingSpec",
        )

    outcome = lifecycle.run_in_subprocess(
        _child_run,
        args=(resolved.engine_name, spec, preset),
    )
    if not outcome.ok:
        return EngineRunRecord(
            engine_name=resolved.engine_name,
            status="failed",
            reason=outcome.error,
            error_class="ChildError",
        )
    payload = outcome.payload or {}
    iterations: list[EngineResult] = payload.get("iterations") or []
    host = payload.get("host")
    port = payload.get("port")
    model_ref = payload.get("model_ref") or _model_ref(spec)
    repo_id = payload.get("repo_id") or spec.repo_id

    iter_records = [build_iteration(it) for it in iterations]
    summary = build_summary(iter_records)

    result = BenchmarkResult(
        timestamp=now_iso8601_micro(),
        repo_id=repo_id,
        model_ref=model_ref,
        engine=resolved.engine_name,
        system_info=sys_info,
        config=BenchmarkConfig(
            prompt_set=preset.prompt_set,
            prompt=preset.prompt,
            max_tokens=preset.max_tokens,
            temperature=preset.temperature,
            iterations=preset.iterations,
            host=host,
            port=port,
        ),
        iterations=iter_records,
        summary=summary,
        mode="text-only",
        environment="stable",
        comparison_id=comparison_id,
    )
    path = save_result(result, perf_dir)
    if engine_progress_cb:
        engine_progress_cb(resolved.engine_name, "done")
    return EngineRunRecord(
        engine_name=resolved.engine_name,
        status="ok",
        result=result,
        result_file=path,
    )


def _child_run(engine_name: str, spec, preset: BenchmarkPreset) -> dict:
    """Runs in the spawned subprocess. Imports its engine lazily so siblings'
    deps aren't dragged in. Returns a plain dict (picklable)."""
    from kepler.benchmark.runner import run_iterations
    from kepler.engines.base import InferenceEngine

    engine: InferenceEngine
    host: str | None = None
    port: int | None = None

    if engine_name == "llamacpp":
        from kepler.engines.llamacpp import LlamaCppEngine

        engine = LlamaCppEngine()
    elif engine_name == "ollama":
        from kepler.engines.ollama import OLLAMA_BASE_URL, OllamaEngine

        engine = OllamaEngine()
        host = "localhost"
        port = int(OLLAMA_BASE_URL.rsplit(":", 1)[-1])
    else:
        raise RuntimeError(f"engine '{engine_name}' is not wired in the child entrypoint yet")

    try:
        out = run_iterations(engine, spec, preset)
    finally:
        try:
            engine.unload()
        except Exception:
            pass

    return {
        "iterations": list(out.iterations),
        "host": host,
        "port": port,
        "model_ref": _model_ref(spec),
        "repo_id": spec.repo_id,
    }


def _model_ref(spec) -> str:
    if spec.local_path is not None:
        return str(spec.local_path)
    if spec.ollama_tag:
        return f"ollama://{spec.ollama_tag}"
    return spec.hf_repo or spec.repo_id
