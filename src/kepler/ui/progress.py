from __future__ import annotations

from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)

from kepler.ui.display import console as _shared_console


def make_engine_progress(console: Console | None = None) -> Progress:
    return Progress(
        SpinnerColumn(),
        TextColumn("[bold]{task.description}[/bold]"),
        BarColumn(bar_width=30),
        TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(),
        console=console or _shared_console,
        transient=True,
    )


def make_cooldown_progress(console: Console | None = None) -> Progress:
    return Progress(
        SpinnerColumn(),
        TextColumn("[dim]cooling down…[/dim]"),
        BarColumn(bar_width=20),
        TextColumn("{task.fields[remaining]}s left"),
        console=console or _shared_console,
        transient=True,
    )


def make_download_progress(console: Console | None = None) -> Progress:
    return Progress(
        SpinnerColumn(),
        TextColumn("[bold]{task.description}[/bold]"),
        BarColumn(),
        TextColumn("{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        console=console or _shared_console,
        transient=True,
    )
