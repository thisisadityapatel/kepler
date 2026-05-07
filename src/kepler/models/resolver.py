from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from kepler.engines.base import ModelFormat
from kepler.engines.registry import FORMAT_SUPPORT, engines_for_format, get_capability
from kepler.models import downloader
from kepler.models.registry import ModelSpec

Status = Literal["ready", "skipped", "unavailable"]


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
    """For each requested engine: produce a ResolvedEngine with a concrete ModelSpec
    if the engine supports `fmt` and the artifact is reachable, otherwise a skipped
    entry explaining why. The user picks `fmt` up front — no auto-detection magic."""

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
        return _merge(engines, unavailable + _resolve_local(model_id, candidate, fmt, available_engines))

    if fmt is ModelFormat.GGUF:
        return _merge(
            engines,
            unavailable
            + _resolve_gguf_hub(
                model_id,
                available_engines,
                models_dir,
                offline=offline,
                ollama_tag_override=ollama_tag_override,
            ),
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


def _resolve_local(
    model_id: str, path: Path, fmt: ModelFormat, engines: list[str]
) -> list[ResolvedEngine]:
    display = path.stem if path.is_file() else path.name
    repo_id = f"local/{display}"
    out: list[ResolvedEngine] = []
    for e in engines:
        if fmt not in FORMAT_SUPPORT.get(e, set()):
            out.append(
                ResolvedEngine(e, None, "skipped", f"engine does not support {fmt.value}")
            )
            continue
        if e == "ollama":
            out.append(
                ResolvedEngine(
                    e,
                    None,
                    "skipped",
                    "ollama needs a tag, not a local file — pass an HF repo id with --format gguf",
                )
            )
            continue
        spec = ModelSpec(
            repo_id=repo_id,
            display_name=display,
            format=fmt,
            local_path=path,
            gguf_filename=path.name if fmt is ModelFormat.GGUF and path.is_file() else None,
        )
        out.append(ResolvedEngine(e, spec, "ready"))
    return out


def _resolve_gguf_hub(
    model_id: str,
    engines: list[str],
    models_dir: Path,
    offline: bool,
    ollama_tag_override: str | None,
) -> list[ResolvedEngine]:
    out: list[ResolvedEngine] = []
    gguf_repo: str | None = None
    gguf_file: str | None = None
    download_path: Path | None = None
    quantization: str | None = None
    download_error: str | None = None

    needs_local = any(e in {"llamacpp", "ik_llama"} for e in engines)
    if needs_local:
        gguf_repo, gguf_file = _find_gguf_repo(model_id, offline=offline)
        if gguf_repo is None or gguf_file is None:
            download_error = (
                "no GGUF found in HF Hub (searched the original repo and `<owner>/<name>-GGUF`)"
            )
        else:
            quantization = _quant_from_filename(gguf_file)
            if not offline:
                try:
                    download_path = downloader.download_gguf(gguf_repo, gguf_file, models_dir)
                except Exception as exc:
                    download_error = f"download failed: {exc}"

    for e in engines:
        if ModelFormat.GGUF not in FORMAT_SUPPORT.get(e, set()):
            out.append(ResolvedEngine(e, None, "skipped", "engine does not support gguf"))
            continue
        if e == "ollama":
            tag = ollama_tag_override or _derive_ollama_tag(model_id)
            spec = ModelSpec(
                repo_id=model_id,
                display_name=_display_name(model_id),
                format=ModelFormat.GGUF,
                ollama_tag=tag,
            )
            out.append(ResolvedEngine(e, spec, "ready"))
            continue
        if download_error is not None:
            out.append(ResolvedEngine(e, None, "skipped", download_error))
            continue
        spec = ModelSpec(
            repo_id=model_id,
            display_name=_display_name(model_id),
            format=ModelFormat.GGUF,
            hf_repo=gguf_repo,
            local_path=download_path,
            gguf_filename=gguf_file,
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
    """Stub for M2 — returns 'unavailable' for non-Ollama MLX engines so the CLI
    surfaces a clear message until M2 lands."""
    out: list[ResolvedEngine] = []
    for e in engines:
        if ModelFormat.MLX not in FORMAT_SUPPORT.get(e, set()):
            out.append(ResolvedEngine(e, None, "skipped", "engine does not support mlx"))
            continue
        if e == "ollama":
            tag = ollama_tag_override or _derive_ollama_tag(model_id)
            spec = ModelSpec(
                repo_id=model_id,
                display_name=_display_name(model_id),
                format=ModelFormat.MLX,
                ollama_tag=tag,
            )
            out.append(ResolvedEngine(e, spec, "ready"))
            continue
        out.append(
            ResolvedEngine(e, None, "unavailable", "MLX path not yet wired (planned for M2)")
        )
    return out


def _find_gguf_repo(model_id: str, offline: bool) -> tuple[str | None, str | None]:
    """Probe HF for a GGUF artifact. Try the original repo first, then `<owner>/<name>-GGUF`."""
    if offline:
        return None, None
    candidates = [model_id]
    if "/" in model_id:
        owner, name = model_id.split("/", 1)
        candidates.append(f"{owner}/{name}-GGUF")
    for repo in candidates:
        if not downloader.repo_exists(repo):
            continue
        files = downloader.list_repo_files(repo)
        picked = downloader.pick_gguf_file(files)
        if picked:
            return repo, picked
    return None, None


_QUANT_RX = re.compile(r"(Q[0-9]+_[A-Z0-9_]+|F16|BF16|F32)", re.IGNORECASE)


def _quant_from_filename(filename: str) -> str | None:
    m = _QUANT_RX.search(filename)
    return m.group(0).upper() if m else None


def _display_name(model_id: str) -> str:
    return model_id.split("/", 1)[-1]


def _derive_ollama_tag(model_id: str) -> str:
    from kepler.engines.ollama import derive_tag_from_repo

    return derive_tag_from_repo(model_id)
