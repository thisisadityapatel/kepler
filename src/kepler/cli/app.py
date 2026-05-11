from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
from rich.table import Table

from kepler import __version__
from kepler.benchmark.orchestrator import run_comparison
from kepler.config.loader import find_config, load_config
from kepler.engines.base import ModelFormat
from kepler.engines.registry import (
    ENGINE_ORDER,
    FORMAT_SUPPORT,
    engines_for_format,
    list_capabilities,
)
from kepler.models.resolver import resolve
from kepler.ui import display
from kepler.ui.progress import make_cooldown_progress, make_engine_progress
from kepler.ui.report import render_comparison

app = typer.Typer(
    name="kepler",
    help="Benchmark any LLM on macOS Metal GPU across inference engines.",
    no_args_is_help=True,
    add_completion=False,
)


def _split_csv(value: Optional[str]) -> list[str]:
    if not value:
        return []
    return [v.strip() for v in value.split(",") if v.strip()]


def _project_paths() -> tuple[Path, Path, Path]:
    cfg_path = find_config()
    project_root = cfg_path.parent.parent
    return cfg_path, project_root / "perf", project_root / "models"


@app.command("benchmark")
def cmd_benchmark(
    model: str = typer.Argument(
        ...,
        help=(
            "Model tag (e.g. qwen2.5:0.5b). For --format gguf, kepler looks in models/ "
            "for a .gguf whose name contains every piece of the tag. For --format mlx, "
            "the tag is passed to Ollama; MLX-native engines fetch their own artifacts."
        ),
    ),
    fmt: Optional[str] = typer.Option(
        None, "--format", "-f", help="gguf | mlx (required unless --engine implies one)"
    ),
    mode: str = typer.Option("standard", "--mode", "-m", help="quick | standard | performance"),
    engine: Optional[str] = typer.Option(None, "--engine", help="Run only this engine; skips comparison file"),
    only_engine: Optional[str] = typer.Option(None, "--only-engine", help="Comma-separated allowlist"),
    skip_engine: Optional[str] = typer.Option(None, "--skip-engine", help="Comma-separated denylist"),
    cooldown: int = typer.Option(8, "--cooldown", help="Seconds between engines for thermal recovery"),
    iterations: Optional[int] = typer.Option(None, "--iterations", help="Override preset iterations"),
    max_tokens: Optional[int] = typer.Option(None, "--max-tokens", help="Override preset max_tokens"),
    temperature: Optional[float] = typer.Option(None, "--temperature", help="Override preset temperature"),
    ollama_tag: Optional[str] = typer.Option(None, "--ollama-tag", help="Override Ollama tag derivation"),
    offline: bool = typer.Option(False, "--offline", help="Skip HF Hub probes; use cache only"),
):
    """Run all available engines compatible with --format on this model, sequentially."""
    display.banner(__version__)

    if engine and (only_engine or skip_engine):
        display.error("--engine is mutually exclusive with --only-engine / --skip-engine")
        raise typer.Exit(code=2)

    # Resolve format
    format_enum = _resolve_format(fmt, engine)
    if format_enum is None:
        display.error("--format gguf|mlx is required (unless --engine X implies one)")
        raise typer.Exit(code=2)

    cfg_path, perf_dir, models_dir = _project_paths()
    cfg = load_config(cfg_path)
    preset = cfg.presets.get(mode)
    if preset is None:
        display.error(f"unknown --mode '{mode}' (valid: {', '.join(cfg.presets)})")
        raise typer.Exit(code=2)

    if iterations is not None:
        preset.iterations = iterations
    if max_tokens is not None:
        preset.max_tokens = max_tokens
    if temperature is not None:
        preset.temperature = temperature

    engines = _select_engines(format_enum, engine, only_engine, skip_engine)
    if not engines:
        display.error("no engines selected for this format after filtering")
        raise typer.Exit(code=2)

    display.info(
        f"Model: [bold]{model}[/bold]   format=[cyan]{format_enum.value}[/cyan]   "
        f"mode=[cyan]{mode}[/cyan]   engines=[cyan]{', '.join(engines)}[/cyan]"
    )

    display.info("Resolving artifacts…")
    resolved = resolve(
        model_id=model,
        fmt=format_enum,
        engines=engines,
        models_dir=models_dir,
        offline=offline,
        ollama_tag_override=ollama_tag,
    )
    for r in resolved:
        if r.status == "ready":
            display.success(f"  {r.engine_name}: ready")
        else:
            display.warn(f"  {r.engine_name}: {r.status} — {r.reason}")

    if not any(r.status == "ready" for r in resolved):
        display.error("nothing to run — every engine was skipped or unavailable")
        raise typer.Exit(code=2)

    write_comparison = engine is None
    progress = make_engine_progress()
    cooldown_progress = make_cooldown_progress()
    cool_task = None

    def cooldown_tick(remaining: int):
        nonlocal cool_task
        if cool_task is None:
            cool_task = cooldown_progress.add_task("cooldown", total=cooldown, remaining=remaining)
            cooldown_progress.start()
        cooldown_progress.update(
            cool_task, completed=cooldown - remaining, remaining=remaining
        )
        if remaining <= 0:
            cooldown_progress.stop()
            cool_task = None

    def engine_progress_cb(name: str, phase: str):
        if phase == "starting":
            display.info(f"Running [bold]{name}[/bold]…")
        elif phase == "done":
            display.success(f"  {name}: done")

    try:
        report, _records = run_comparison(
            resolved=resolved,
            preset=preset,
            perf_dir=perf_dir,
            cooldown_s=cooldown,
            write_comparison=write_comparison,
            cooldown_tick=cooldown_tick,
            engine_progress_cb=engine_progress_cb,
        )
    finally:
        if cool_task is not None:
            try:
                cooldown_progress.stop()
            except Exception:
                pass

    render_comparison(report)


@app.command("list-engines")
def cmd_list_engines():
    """Show all engines with their availability status and supported formats."""
    display.banner(__version__)
    table = Table(border_style="cyan")
    table.add_column("Engine", style="bold")
    table.add_column("Formats")
    table.add_column("Available")
    table.add_column("Notes")
    for cap in list_capabilities():
        fmts = ", ".join(sorted(f.value for f in cap.supported_formats))
        avail = "[green]yes[/green]" if cap.available else "[red]no[/red]"
        notes = cap.unavailable_reason or ("server" if cap.requires_server else "in-process")
        table.add_row(cap.name, fmts, avail, notes)
    display.console.print(table)


@app.command("list-models")
def cmd_list_models():
    """Show models cached locally under models/."""
    display.banner(__version__)
    _, _, models_dir = _project_paths()
    if not models_dir.exists():
        display.warn(f"{models_dir} does not exist")
        raise typer.Exit(code=0)
    table = Table(border_style="cyan")
    table.add_column("Path", style="bold")
    table.add_column("Size (MB)", justify="right")
    found = 0
    for path in sorted(models_dir.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix.lower() not in {".gguf", ".safetensors", ".bin"}:
            continue
        rel = path.relative_to(models_dir)
        size_mb = path.stat().st_size / 1024 / 1024
        table.add_row(str(rel), f"{size_mb:.1f}")
        found += 1
    if found == 0:
        display.warn("no models cached yet")
        return
    display.console.print(table)


@app.command("results")
def cmd_results():
    """Summarize all perf/*.json files (legacy + new)."""
    display.banner(__version__)
    _, perf_dir, _ = _project_paths()
    if not perf_dir.exists():
        display.warn(f"{perf_dir} does not exist")
        raise typer.Exit(code=0)
    import json as _json

    table = Table(border_style="cyan")
    table.add_column("File", style="bold")
    table.add_column("Engine")
    table.add_column("Repo")
    table.add_column("tok/s", justify="right")
    table.add_column("ttft (ms)", justify="right")
    rows = 0
    for path in sorted(perf_dir.glob("*.json")):
        if path.name.startswith("comparison_"):
            continue
        try:
            data = _json.loads(path.read_text())
            summary = data.get("summary") or {}
            table.add_row(
                path.name,
                str(data.get("engine", "?")),
                str(data.get("repo_id", "?")),
                f"{summary.get('median_tok_per_s', 0):.1f}",
                f"{summary.get('median_ttft_ms', 0):.0f}",
            )
            rows += 1
        except Exception as exc:
            display.warn(f"could not parse {path.name}: {exc}")
    if rows == 0:
        display.warn("no results yet")
        return
    display.console.print(table)


def _resolve_format(fmt: Optional[str], engine: Optional[str]) -> Optional[ModelFormat]:
    if fmt is not None:
        return ModelFormat(fmt.lower())
    if engine is not None:
        formats = FORMAT_SUPPORT.get(engine, set())
        if len(formats) == 1:
            return next(iter(formats))
    return None


def _select_engines(
    fmt: ModelFormat,
    engine: Optional[str],
    only: Optional[str],
    skip: Optional[str],
) -> list[str]:
    if engine is not None:
        if engine not in ENGINE_ORDER:
            return []
        if fmt not in FORMAT_SUPPORT.get(engine, set()):
            return []
        return [engine]
    base = engines_for_format(fmt)
    only_list = _split_csv(only)
    skip_list = _split_csv(skip)
    if only_list:
        base = [e for e in base if e in only_list]
    if skip_list:
        base = [e for e in base if e not in skip_list]
    return base


def main():
    app()


if __name__ == "__main__":
    main()
