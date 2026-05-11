from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from kepler.engines.base import ModelFormat
from kepler.engines.registry import FORMAT_SUPPORT, get_capability
from kepler.models import downloader
from kepler.models.registry import ModelSpec

Status = Literal["ready", "skipped", "unavailable"]

GGUF_QUANT_PREFERENCE = ("q5_k_m", "q4_k_m", "q8_0", "q6_k", "q4_0", "f16", "bf16", "f32")
_QUANT_RX = re.compile(r"(Q[0-9]+_[A-Z0-9_]+|F16|BF16|F32)", re.IGNORECASE)


@dataclass
class ResolvedEngine:
    engine_name: str
    spec: ModelSpec | None
    status: Status
    reason: str | None = None


def resolve(
    model_id: str,
    fmt: ModelFormat,
    engines: list[str],
    models_dir: Path,
    offline: bool = False,
    ollama_tag_override: str | None = None,
) -> list[ResolvedEngine]:
    """Map (`model_id`, `fmt`) to a per-engine `ModelSpec` or a `skipped` reason.

    Inputs accepted for `model_id`:
      - An Ollama-style tag like `qwen2.5:0.5b`. For `--format gguf`, kepler looks
        in `models_dir` for a .gguf whose filename contains every piece of the tag
        (split on `:`). For `--format mlx`, the tag is handed to Ollama as-is.
      - An explicit path to a local .gguf file or MLX directory.

    Kepler does not download GGUF artifacts — the user is responsible for placing
    .gguf files in `models_dir`."""

    available_engines: list[str] = []
    unavailable: list[ResolvedEngine] = []
    for e in engines:
        cap = get_capability(e)
        if not cap.available:
            unavailable.append(ResolvedEngine(e, None, "unavailable", cap.unavailable_reason))
        else:
            available_engines.append(e)

    candidate = Path(model_id)
    local_fmt = downloader.detect_local_format(candidate)
    if local_fmt is not None:
        if local_fmt != fmt:
            reason = f"local path is {local_fmt.value} but --format={fmt.value}"
            mismatched = [ResolvedEngine(e, None, "skipped", reason) for e in available_engines]
            return _merge(engines, unavailable + mismatched)
        return _merge(
            engines,
            unavailable
            + _build_local_specs(model_id, candidate, fmt, available_engines, display=candidate.stem),
        )

    if fmt is ModelFormat.GGUF:
        matched = _find_gguf_in_models_dir(model_id, models_dir)
        if matched is None:
            reason = (
                f"no .gguf in {models_dir} matches tag '{model_id}' — "
                f"place a matching .gguf in {models_dir}/ (kepler does not download GGUFs)"
            )
            skipped = [ResolvedEngine(e, None, "skipped", reason) for e in available_engines]
            return _merge(engines, unavailable + skipped)
        return _merge(
            engines,
            unavailable
            + _build_local_specs(model_id, matched, fmt, available_engines, display=model_id),
        )

    return _merge(
        engines,
        unavailable
        + _resolve_mlx_hub(
            model_id,
            available_engines,
            models_dir,
            offline=offline,
            ollama_tag_override=ollama_tag_override,
        ),
    )


def _merge(engines: list[str], resolved: list[ResolvedEngine]) -> list[ResolvedEngine]:
    """Preserve the original engine ordering so the comparison report is deterministic."""
    by_name = {r.engine_name: r for r in resolved}
    return [by_name[e] for e in engines if e in by_name]


def _build_local_specs(
    model_id: str,
    path: Path,
    fmt: ModelFormat,
    engines: list[str],
    display: str,
) -> list[ResolvedEngine]:
    """Build one ModelSpec per engine pointing at a local file. For Ollama with a
    .gguf, derive a kepler-namespaced import tag so we don't collide with anything
    the user pulled from Ollama's library."""
    repo_id = model_id if model_id == display else f"local/{display}"
    quantization = _quant_from_filename(path.name) if path.is_file() else None
    out: list[ResolvedEngine] = []
    for e in engines:
        if fmt not in FORMAT_SUPPORT.get(e, set()):
            out.append(
                ResolvedEngine(e, None, "skipped", f"engine does not support {fmt.value}")
            )
            continue
        if e == "ollama":
            if fmt is not ModelFormat.GGUF or not path.is_file():
                out.append(
                    ResolvedEngine(
                        e,
                        None,
                        "skipped",
                        "ollama local import only supports a single .gguf file",
                    )
                )
                continue
            from kepler.engines.ollama import local_tag_for_path
            spec = ModelSpec(
                repo_id=repo_id,
                display_name=display,
                format=fmt,
                local_path=path,
                gguf_filename=path.name,
                ollama_tag=local_tag_for_path(path),
                quantization=quantization,
            )
            out.append(ResolvedEngine(e, spec, "ready"))
            continue
        spec = ModelSpec(
            repo_id=repo_id,
            display_name=display,
            format=fmt,
            local_path=path,
            gguf_filename=path.name if fmt is ModelFormat.GGUF and path.is_file() else None,
            quantization=quantization,
        )
        out.append(ResolvedEngine(e, spec, "ready"))
    return out


def _resolve_mlx_hub(
    model_id: str,
    engines: list[str],
    models_dir: Path,
    offline: bool,
    ollama_tag_override: str | None,
) -> list[ResolvedEngine]:
    """MLX path: each engine uses its own native source. Ollama pulls via its
    library (`ollama pull <tag>`); MLX-native engines are stubbed until M2."""
    out: list[ResolvedEngine] = []
    for e in engines:
        if ModelFormat.MLX not in FORMAT_SUPPORT.get(e, set()):
            out.append(ResolvedEngine(e, None, "skipped", "engine does not support mlx"))
            continue
        if e == "ollama":
            tag = ollama_tag_override or model_id
            spec = ModelSpec(
                repo_id=model_id,
                display_name=model_id,
                format=ModelFormat.MLX,
                ollama_tag=tag,
            )
            out.append(ResolvedEngine(e, spec, "ready"))
            continue
        out.append(
            ResolvedEngine(e, None, "unavailable", "MLX path not yet wired (planned for M2)")
        )
    return out


def _find_gguf_in_models_dir(tag: str, models_dir: Path) -> Path | None:
    """Find a .gguf in models_dir matching `tag`. Tag is split on `:` and every
    piece must appear as a case-insensitive substring of the filename. If multiple
    files match, prefer the highest-quality quantization."""
    if not models_dir.is_dir():
        return None
    pieces = [p.lower() for p in tag.split(":") if p]
    if not pieces:
        return None
    matches: list[Path] = []
    for p in sorted(models_dir.iterdir()):
        if not p.is_file() or p.suffix.lower() != ".gguf":
            continue
        name = p.name.lower()
        if all(piece in name for piece in pieces):
            matches.append(p)
    if not matches:
        return None
    if len(matches) == 1:
        return matches[0]
    for quant in GGUF_QUANT_PREFERENCE:
        for p in matches:
            if quant in p.name.lower():
                return p
    return matches[0]


def _quant_from_filename(filename: str) -> str | None:
    m = _QUANT_RX.search(filename)
    return m.group(0).upper() if m else None
