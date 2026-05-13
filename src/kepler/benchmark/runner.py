from __future__ import annotations

from dataclasses import dataclass

from kepler.config.loader import BenchmarkPreset
from kepler.engines.base import EngineResult, InferenceEngine
from kepler.models.registry import ModelSpec


@dataclass
class RunnerOutput:
    iterations: list[EngineResult]


def run_iterations(
    engine: InferenceEngine,
    model: ModelSpec,
    preset: BenchmarkPreset,
    progress_cb=None,
) -> RunnerOutput:
    """Mirrors the legacy C++ flow: warmup once, then N timed iterations."""
    engine.load(model)
    if progress_cb:
        progress_cb("warmup", 0, preset.iterations)
    engine.infer(prompt=preset.prompt, max_tokens=16, temperature=preset.temperature)

    results: list[EngineResult] = []
    for i in range(preset.iterations):
        if progress_cb:
            progress_cb("iter", i, preset.iterations)
        r = engine.infer(
            prompt=preset.prompt,
            max_tokens=preset.max_tokens,
            temperature=preset.temperature,
        )
        results.append(r)
    if progress_cb:
        progress_cb("done", preset.iterations, preset.iterations)
    return RunnerOutput(iterations=results)
