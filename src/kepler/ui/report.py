from __future__ import annotations

from rich.table import Table

from kepler.benchmark.comparison import ComparisonReport
from kepler.ui.display import console


def render_comparison(report: ComparisonReport) -> None:
    if not report.engines:
        console.print("[red]No engines were attempted.[/red]")
        return

    ok_summaries: dict[str, dict] = {
        e.engine: e.summary for e in report.engines if e.status == "ok" and e.summary
    }
    bests = _column_bests(ok_summaries)

    table = Table(title=f"Kepler comparison — {report.repo_id} [{report.format}]", border_style="cyan")
    table.add_column("Engine", style="bold")
    table.add_column("tok/s", justify="right")
    table.add_column("ttft (ms)", justify="right")
    table.add_column("gen tok/s", justify="right")
    table.add_column("wall (s)", justify="right")
    table.add_column("status")

    for e in report.engines:
        if e.status == "ok" and e.summary:
            s = e.summary
            row = [
                e.engine,
                _cell(s["median_tok_per_s"], bests.get("tok"), e.engine, fmt="{:.1f}"),
                _cell(s["median_ttft_ms"], bests.get("ttft"), e.engine, fmt="{:.0f}", lower_better=True),
                _cell(
                    s["median_generation_tok_per_s"],
                    bests.get("gen"),
                    e.engine,
                    fmt="{:.1f}",
                ),
                _cell(s["median_wall_s"], bests.get("wall"), e.engine, fmt="{:.2f}", lower_better=True),
                "[green]ok[/green]",
            ]
        else:
            label = e.status
            if e.reason:
                short = _short_reason(e.reason)
                label = f"{e.status}: {short}"
            color = "yellow" if e.status in {"skipped", "unavailable"} else "red"
            row = [e.engine, "—", "—", "—", "—", f"[{color}]{label}[/{color}]"]
        table.add_row(*row)

    console.print(table)
    if report.winner is not None:
        console.print(
            f"[bold green]Winner:[/bold green] {report.winner['engine']} "
            f"({report.winner['value']:.1f} tok/s)"
        )
    if report.narrative:
        console.print(f"[dim]{report.narrative}[/dim]")


def _column_bests(ok: dict[str, dict]) -> dict[str, str]:
    if not ok:
        return {}
    by_metric_higher = {
        "tok": "median_tok_per_s",
        "gen": "median_generation_tok_per_s",
    }
    by_metric_lower = {
        "ttft": "median_ttft_ms",
        "wall": "median_wall_s",
    }
    out: dict[str, str] = {}
    for k, m in by_metric_higher.items():
        out[k] = max(ok.items(), key=lambda kv: kv[1][m])[0]
    for k, m in by_metric_lower.items():
        out[k] = min(ok.items(), key=lambda kv: kv[1][m])[0]
    return out


def _cell(value: float, best_engine: str | None, engine: str, fmt: str, lower_better: bool = False) -> str:
    text = fmt.format(value)
    if best_engine == engine:
        return f"[bold green]{text}[/bold green]"
    return text


def _short_reason(reason: str, limit: int = 80) -> str:
    """Trim a long error/skip reason to one line so the table stays readable."""
    first_line = reason.strip().splitlines()[0] if reason else ""
    if len(first_line) > limit:
        return first_line[: limit - 1] + "…"
    return first_line
