"""Progress tracking system using Rich for Kepler LLM Workbench."""

import time
from contextlib import contextmanager
from typing import Dict, List

from rich.console import Console
from rich.status import Status
from rich.table import Table


class WorkflowTracker:
    """Clean, futuristic workflow progress tracker using Rich."""

    def __init__(self):
        self.console = Console()
        self.steps: List[Dict] = []
        self.current_step_idx = -1

    def add_step(self, name: str, description: str = ""):
        """Add a step to the workflow."""
        self.steps.append(
            {
                "name": name,
                "description": description,
                "status": "pending",
                "start_time": None,
                "end_time": None,
                "error": None,
            }
        )

    def show_workflow(self):
        """Display the complete workflow status."""
        table = Table(show_header=False, box=None, padding=(0, 1))
        table.add_column("Status", style="bold")
        table.add_column("Step")
        table.add_column("Time", justify="right")

        for i, step in enumerate(self.steps):
            if step["status"] == "completed":
                status = "[green]●[/green]"
                duration = ""
                if step["start_time"] and step["end_time"]:
                    duration = (
                        f"[dim]{step['end_time'] - step['start_time']:.1f}s[/dim]"
                    )
            elif step["status"] == "in_progress":
                status = "[yellow]●[/yellow]"
                duration = "[yellow]...[/yellow]"
            elif step["status"] == "failed":
                status = "[red]●[/red]"
                duration = "[red]FAILED[/red]"
            else:
                status = "[dim]○[/dim]"
                duration = ""

            table.add_row(status, step["name"], duration)

        self.console.print(table)

    @contextmanager
    def step(self, step_name: str):
        """Context manager for executing a step with status tracking."""
        # Find step index
        step_idx = None
        for i, step in enumerate(self.steps):
            if step["name"] == step_name:
                step_idx = i
                break

        if step_idx is None:
            raise ValueError(f"Step '{step_name}' not found")

        # Start step
        self.current_step_idx = step_idx
        self.steps[step_idx]["status"] = "in_progress"
        self.steps[step_idx]["start_time"] = time.time()

        self.console.clear()
        self.console.print("[bold]Kepler LLM Workbench[/bold]")
        self.console.print()
        self.show_workflow()

        try:
            # Don't use spinner for interactive steps like Model Selection
            if step_name == "Model Selection":
                yield
            else:
                with Status(f"[yellow]{step_name}...[/yellow]", console=self.console):
                    yield
            # Success
            self.steps[step_idx]["status"] = "completed"
            self.steps[step_idx]["end_time"] = time.time()
        except Exception as e:
            # Failure
            self.steps[step_idx]["status"] = "failed"
            self.steps[step_idx]["error"] = str(e)
            raise
        finally:
            # Refresh display
            self.console.clear()
            self.console.print("\n[bold]Kepler LLM Workbench[/bold]")
            self.console.print()
            self.show_workflow()

    def print_summary(self):
        """Print final workflow summary."""
        completed = sum(1 for s in self.steps if s["status"] == "completed")
        failed = sum(1 for s in self.steps if s["status"] == "failed")
        total = len(self.steps)

        self.console.print()
        if failed == 0 and completed == total:
            total_time = sum(
                s["end_time"] - s["start_time"]
                for s in self.steps
                if s["end_time"] and s["start_time"]
            )
            self.console.print(
                f"[green]All steps completed successfully in {total_time:.1f}s[/green]"
            )
        elif failed > 0:
            self.console.print(
                f"[red]Workflow failed: {failed}/{total} steps failed[/red]"
            )
        else:
            self.console.print(
                f"[yellow]Workflow incomplete: {completed}/{total} steps completed[/yellow]"
            )


def create_workbench_tracker() -> WorkflowTracker:
    """Create a progress tracker for the standard workbench workflow."""
    tracker = WorkflowTracker()
    tracker.add_step("Model Selection", "Select GGUF model file")
    tracker.add_step("Docker Setup", "Build Docker image")
    tracker.add_step("Container Start", "Start model server")
    tracker.add_step("Health Check", "Wait for server ready")
    tracker.add_step("Benchmarking", "Run performance tests")
    tracker.add_step(
        "Save Results", "Save benchmark results to disk"
    )
    return tracker


# Simple status functions for inline use
console = Console()


def status_info(msg: str):
    """Print an info status message."""
    console.print(f"[blue]●[/blue] {msg}")


def status_success(msg: str):
    """Print a success status message."""
    console.print(f"[green]●[/green] {msg}")


def status_warning(msg: str):
    """Print a warning status message."""
    console.print(f"[yellow]●[/yellow] {msg}")


def status_error(msg: str):
    """Print an error status message."""
    console.print(f"[red]●[/red] {msg}")


def status_pending(msg: str):
    """Print a pending status message."""
    console.print(f"[dim]○[/dim] {msg}")
