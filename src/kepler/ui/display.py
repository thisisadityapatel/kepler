from __future__ import annotations

from rich.console import Console
from rich.panel import Panel
from rich.text import Text

console = Console()


def banner(version: str) -> None:
    title = Text("KEPLER", style="bold cyan")
    sub = Text(f"  benchmark any LLM on macOS Metal — v{version}", style="dim")
    console.print(Panel.fit(title + Text("\n") + sub, border_style="cyan"))


def info(msg: str) -> None:
    console.print(f"[cyan]●[/cyan] {msg}")


def success(msg: str) -> None:
    console.print(f"[green]●[/green] {msg}")


def warn(msg: str) -> None:
    console.print(f"[yellow]●[/yellow] {msg}")


def error(msg: str) -> None:
    console.print(f"[red]●[/red] {msg}")


def dim(msg: str) -> None:
    console.print(f"[dim]{msg}[/dim]")
