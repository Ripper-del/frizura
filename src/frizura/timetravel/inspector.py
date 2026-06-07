"""Rich-powered pipeline inspector for time-travel debugging.

Provides beautiful console output for visualizing pipeline execution
history. Uses Rich Tables, Panels, and Trees to display step timelines,
cost breakdowns, event sequences, and detailed step information.

This is a static display (not an interactive TUI) suitable for CLI use
and debugging sessions.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

from rich.console import Console
from rich.markup import escape
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.tree import Tree

from frizura.core.events import Event, EventType
from frizura.models.execution import StepResult
from frizura.timetravel.replay import ReplayContext, ReplayEngine
from frizura.timetravel.store import EventStore

logger = logging.getLogger(__name__)

# Status styling
_STATUS_ICONS: dict[str, str] = {
    "completed": "✅",
    "failed": "❌",
    "skipped": "⏭️",
    "healed": "🩹",
    "running": "⏳",
}

_STATUS_COLORS: dict[str, str] = {
    "completed": "green",
    "failed": "red",
    "skipped": "dim",
    "healed": "yellow",
    "running": "cyan",
}

_EVENT_ICONS: dict[str, str] = {
    "pipeline.started": "🚀",
    "pipeline.completed": "🏁",
    "pipeline.failed": "💥",
    "step.started": "▶️",
    "step.completed": "✅",
    "step.failed": "❌",
    "step.skipped": "⏭️",
    "llm.request": "📤",
    "llm.response": "📥",
    "schema.validation.ok": "✅",
    "schema.validation.failed": "⚠️",
    "schema.heal.attempt": "🔧",
    "schema.heal.success": "🩹",
    "schema.heal.failed": "💔",
    "routing.decision": "🧭",
    "state.snapshot": "📸",
    "budget.exhausted": "💸",
}


def _format_duration(ms: float) -> str:
    """Format milliseconds into a human-readable duration."""
    if ms < 1:
        return f"{ms * 1000:.0f}µs"
    if ms < 1000:
        return f"{ms:.0f}ms"
    if ms < 60_000:
        return f"{ms / 1000:.1f}s"
    minutes = ms / 60_000
    return f"{minutes:.1f}min"


def _format_cost(usd: float) -> str:
    """Format a USD cost into a readable string."""
    if usd == 0:
        return "$0"
    if usd < 0.001:
        return f"${usd:.6f}"
    if usd < 0.01:
        return f"${usd:.4f}"
    if usd < 1:
        return f"${usd:.3f}"
    return f"${usd:.2f}"


def _truncate(text: str, max_len: int = 80) -> str:
    """Truncate text with ellipsis if too long."""
    if len(text) <= max_len:
        return text
    return text[: max_len - 1] + "…"


class Inspector:
    """Rich console-based pipeline execution inspector.

    Connects to the ``EventStore`` and ``ReplayEngine`` to display
    pipeline history with beautiful formatting.

    Usage::

        inspector = Inspector(store, replay_engine)
        await inspector.show("pipeline-123")
    """

    def __init__(
        self,
        store: EventStore,
        replay_engine: ReplayEngine | None = None,
        console: Console | None = None,
    ) -> None:
        self._store = store
        self._replay = replay_engine
        self._console = console or Console()

    # --- Main display --------------------------------------------------------

    async def show(self, pipeline_id: str) -> None:
        """Display a comprehensive pipeline execution view.

        Shows a summary panel, step timeline table, event tree, and
        cost breakdown.
        """
        summary = await self._store.get_pipeline_summary(pipeline_id)
        events = await self._store.get_events(pipeline_id)

        if summary.get("status") == "not_found":
            self._console.print(
                f"[red bold]Pipeline '{pipeline_id}' not found[/red bold]"
            )
            return

        # 1. Summary panel
        self._print_summary_panel(summary)

        # 2. Step timeline
        if self._replay:
            try:
                step_results = await self._replay.replay_full(pipeline_id)
                self._print_step_timeline(step_results)
            except Exception as exc:
                self._console.print(
                    f"[yellow]Could not replay steps: {exc}[/yellow]"
                )
                self._print_event_timeline(events)
        else:
            self._print_event_timeline(events)

        # 3. Event tree
        self._print_event_tree(events)

        # 4. Cost breakdown
        self._print_cost_breakdown(events)

    # --- Summary panel -------------------------------------------------------

    def _print_summary_panel(self, summary: dict[str, Any]) -> None:
        """Print a styled panel with pipeline summary info."""
        status = summary.get("status", "unknown")
        icon = _STATUS_ICONS.get(status, "❓")
        color = _STATUS_COLORS.get(status, "white")

        info_lines = [
            f"[bold]Pipeline ID:[/bold] {summary['pipeline_id']}",
            f"[bold]Status:[/bold] [{color}]{icon} {status.upper()}[/{color}]",
            f"[bold]Steps:[/bold] {summary.get('step_count', '?')}",
            f"[bold]Events:[/bold] {summary.get('event_count', '?')}",
            f"[bold]Duration:[/bold] {_format_duration(summary.get('duration_ms', 0))}",
            f"[bold]Total Cost:[/bold] {_format_cost(summary.get('total_cost_usd', 0))}",
        ]
        models = summary.get("models_used", [])
        if models:
            info_lines.append(f"[bold]Models:[/bold] {', '.join(models)}")

        first_at = summary.get("first_event_at", "")
        if first_at:
            info_lines.append(f"[bold]Started:[/bold] {first_at}")

        panel = Panel(
            "\n".join(info_lines),
            title="[bold cyan]Pipeline Execution Summary[/bold cyan]",
            border_style="cyan",
            expand=False,
        )
        self._console.print(panel)
        self._console.print()

    # --- Step timeline -------------------------------------------------------

    def _print_step_timeline(self, steps: list[StepResult]) -> None:
        """Print a table showing each step with status, duration, cost, model."""
        if not steps:
            return

        table = Table(
            title="Step Timeline",
            show_lines=True,
            title_style="bold magenta",
        )
        table.add_column("#", style="dim", width=4, justify="right")
        table.add_column("Status", width=6, justify="center")
        table.add_column("Step Name", style="bold", min_width=20)
        table.add_column("Duration", justify="right")
        table.add_column("Cost", justify="right")
        table.add_column("Model", style="dim")
        table.add_column("Healed", justify="center")
        table.add_column("Output Preview", max_width=40)

        for i, step in enumerate(steps, 1):
            icon = _STATUS_ICONS.get(step.status, "❓")
            color = _STATUS_COLORS.get(step.status, "white")

            output_preview = ""
            if step.output is not None:
                output_preview = _truncate(str(step.output), 40)

            healed_str = ""
            if step.healing_attempts > 0:
                healed_str = f"[yellow]🔧 ×{step.healing_attempts}[/yellow]"

            table.add_row(
                str(i),
                icon,
                f"[{color}]{escape(step.step_name)}[/{color}]",
                _format_duration(step.duration_ms),
                _format_cost(step.cost_usd),
                escape(step.model_used) if step.model_used else "[dim]—[/dim]",
                healed_str,
                escape(output_preview) if output_preview else "[dim]—[/dim]",
            )

        self._console.print(table)
        self._console.print()

    # --- Event timeline (fallback when replay unavailable) -------------------

    def _print_event_timeline(self, events: list[Event]) -> None:
        """Print a simple event timeline table."""
        if not events:
            return

        table = Table(
            title="Event Timeline",
            show_lines=False,
            title_style="bold magenta",
        )
        table.add_column("Seq", style="dim", width=4, justify="right")
        table.add_column("", width=3)  # icon
        table.add_column("Event Type", min_width=25)
        table.add_column("Step", style="cyan")
        table.add_column("Time", style="dim")
        table.add_column("Details", max_width=50)

        for event in events:
            icon = _EVENT_ICONS.get(event.event_type.value, "·")
            details = ""
            if event.event_type == EventType.LLM_RESPONSE:
                model = event.data.get("model", "")
                cost = event.data.get("cost_usd", 0)
                details = f"model={model} cost={_format_cost(cost)}"
            elif event.event_type == EventType.STEP_FAILED:
                details = event.data.get("error", "")[:50]
            elif event.event_type == EventType.ROUTING_DECISION:
                details = f"→ {event.data.get('selected_model', '?')}"
            elif event.data:
                # Show first key=value pair
                for k, v in list(event.data.items())[:1]:
                    details = f"{k}={_truncate(str(v), 40)}"

            ts = event.timestamp.strftime("%H:%M:%S.%f")[:-3]
            table.add_row(
                str(event.sequence_number),
                icon,
                escape(event.event_type.value),
                escape(event.step_name or event.step_id or "—"),
                ts,
                escape(details),
            )

        self._console.print(table)
        self._console.print()

    # --- Event tree ----------------------------------------------------------

    def _print_event_tree(self, events: list[Event]) -> None:
        """Print a tree view of events grouped by step."""
        if not events:
            return

        tree = Tree(
            "[bold cyan]📋 Event Log[/bold cyan]",
            guide_style="dim",
        )

        current_step_branch: Tree | None = None
        current_step_id: str | None = None

        for event in events:
            icon = _EVENT_ICONS.get(event.event_type.value, "·")

            # Pipeline-level events
            if event.event_type in (
                EventType.PIPELINE_STARTED,
                EventType.PIPELINE_COMPLETED,
                EventType.PIPELINE_FAILED,
            ):
                tree.add(
                    f"{icon} [bold]{escape(event.event_type.value)}[/bold]"
                    f" [dim]{event.timestamp.strftime('%H:%M:%S')}[/dim]"
                )
                continue

            # Group events under step branches
            if event.step_id and event.step_id != current_step_id:
                current_step_id = event.step_id
                step_label = event.step_name or event.step_id
                current_step_branch = tree.add(
                    f"[bold]{escape(step_label)}[/bold]"
                    f" [dim]({event.step_id})[/dim]"
                )

            branch = current_step_branch or tree
            detail = ""
            if event.event_type == EventType.LLM_RESPONSE:
                model = event.data.get("model", "")
                tokens = event.data.get("total_tokens", "?")
                detail = f" [dim]model={model} tokens={tokens}[/dim]"
            elif event.event_type == EventType.SCHEMA_HEAL_ATTEMPT:
                attempt = event.data.get("attempt", "?")
                detail = f" [dim]attempt #{attempt}[/dim]"

            branch.add(
                f"{icon} {escape(event.event_type.value)}{detail}"
                f" [dim]{event.timestamp.strftime('%H:%M:%S.%f')[:-3]}[/dim]"
            )

        self._console.print(tree)
        self._console.print()

    # --- Cost breakdown ------------------------------------------------------

    def _print_cost_breakdown(self, events: list[Event]) -> None:
        """Print cost breakdown by model and step."""
        cost_by_model: dict[str, float] = {}
        cost_by_step: dict[str, float] = {}
        total_input = 0
        total_output = 0

        for ev in events:
            if ev.event_type == EventType.LLM_RESPONSE:
                model = ev.data.get("model", "unknown")
                cost = ev.data.get("cost_usd", 0.0)
                step_name = ev.step_name or ev.step_id or "unknown"
                cost_by_model[model] = cost_by_model.get(model, 0) + cost
                cost_by_step[step_name] = cost_by_step.get(step_name, 0) + cost
                total_input += ev.data.get("input_tokens", 0)
                total_output += ev.data.get("output_tokens", 0)

        if not cost_by_model:
            return

        table = Table(
            title="Cost Breakdown",
            show_lines=True,
            title_style="bold yellow",
        )
        table.add_column("Category", style="bold")
        table.add_column("Item")
        table.add_column("Cost", justify="right", style="green")

        for model, cost in sorted(
            cost_by_model.items(), key=lambda x: x[1], reverse=True
        ):
            table.add_row("Model", escape(model), _format_cost(cost))

        table.add_section()
        for step_name, cost in sorted(
            cost_by_step.items(), key=lambda x: x[1], reverse=True
        ):
            table.add_row("Step", escape(step_name), _format_cost(cost))

        table.add_section()
        total = sum(cost_by_model.values())
        table.add_row(
            "[bold]Total[/bold]",
            f"[dim]{total_input:,} in + {total_output:,} out tokens[/dim]",
            f"[bold green]{_format_cost(total)}[/bold green]",
        )

        self._console.print(table)
        self._console.print()

    # --- Standalone formatters -----------------------------------------------

    def format_pipeline_summary(self, summary: dict[str, Any]) -> str:
        """Format a one-line pipeline summary for CLI output.

        Args:
            summary: Dict from ``EventStore.get_pipeline_summary()``.

        Returns:
            A single-line string like:
            ``✅ pipeline-123 | 5 steps | 1.2s | $0.003 | gpt-4o-mini``
        """
        status = summary.get("status", "unknown")
        icon = _STATUS_ICONS.get(status, "❓")
        pid = summary.get("pipeline_id", "?")
        steps = summary.get("step_count", "?")
        duration = _format_duration(summary.get("duration_ms", 0))
        cost = _format_cost(summary.get("total_cost_usd", 0))
        models = ", ".join(summary.get("models_used", [])) or "—"

        return f"{icon} {pid} | {steps} steps | {duration} | {cost} | {models}"

    def print_step_detail(self, step: StepResult) -> None:
        """Print a detailed view of a single step result."""
        status = step.status
        icon = _STATUS_ICONS.get(status, "❓")
        color = _STATUS_COLORS.get(status, "white")

        # Header
        header = Text()
        header.append(f"{icon} ", style="bold")
        header.append(step.step_name, style=f"bold {color}")
        header.append(f"  ({step.step_id})", style="dim")

        lines = [
            f"[bold]Status:[/bold]    [{color}]{status.upper()}[/{color}]",
            f"[bold]Duration:[/bold]  {_format_duration(step.duration_ms)}",
            f"[bold]Cost:[/bold]      {_format_cost(step.cost_usd)}",
        ]

        if step.model_used:
            lines.append(f"[bold]Model:[/bold]     {escape(step.model_used)}")

        if step.routing_reason:
            lines.append(
                f"[bold]Routing:[/bold]   {escape(step.routing_reason)}"
            )

        if step.healing_attempts > 0:
            lines.append(
                f"[bold]Healing:[/bold]   [yellow]"
                f"{step.healing_attempts} attempt(s)[/yellow]"
            )

        if step.error:
            lines.append(f"[bold]Error:[/bold]     [red]{escape(step.error)}[/red]")

        # Output
        if step.output is not None:
            output_str = str(step.output)
            if len(output_str) > 500:
                output_str = output_str[:500] + "…"
            lines.append("")
            lines.append("[bold]Output:[/bold]")
            lines.append(escape(output_str))

        # LLM Response details
        if step.llm_response:
            resp = step.llm_response
            lines.append("")
            lines.append("[bold]LLM Response:[/bold]")
            lines.append(f"  Provider: {escape(resp.provider)}")
            lines.append(f"  Model:    {escape(resp.model)}")
            lines.append(
                f"  Tokens:   {resp.usage.input_tokens} in"
                f" + {resp.usage.output_tokens} out"
                f" = {resp.usage.total_tokens} total"
            )
            lines.append(
                f"  Latency:  {_format_duration(resp.latency_ms)}"
            )
            if resp.finish_reason != "stop":
                lines.append(f"  Finish:   {resp.finish_reason}")

        # Timestamps
        lines.append("")
        lines.append(f"[dim]Started:   {step.started_at.isoformat()}[/dim]")
        if step.completed_at:
            lines.append(
                f"[dim]Completed: {step.completed_at.isoformat()}[/dim]"
            )

        panel = Panel(
            "\n".join(lines),
            title=header,
            border_style=color,
            expand=False,
        )
        self._console.print(panel)

    # --- Pipeline list -------------------------------------------------------

    async def show_pipeline_list(self, limit: int = 20) -> None:
        """Display a list of recent pipelines."""
        pipelines = await self._store.list_pipelines(limit=limit)
        if not pipelines:
            self._console.print("[dim]No pipelines recorded yet.[/dim]")
            return

        table = Table(
            title="Recent Pipelines",
            show_lines=False,
            title_style="bold cyan",
        )
        table.add_column("", width=3)  # status icon
        table.add_column("Pipeline ID", style="bold")
        table.add_column("Events", justify="right")
        table.add_column("Started", style="dim")
        table.add_column("Last Event", style="dim")

        for p in pipelines:
            icon = _STATUS_ICONS.get(p["status"], "❓")
            color = _STATUS_COLORS.get(p["status"], "white")
            started = p["first_event_at"][:19] if p["first_event_at"] else "—"
            last = p["last_event_at"][:19] if p["last_event_at"] else "—"
            table.add_row(
                icon,
                f"[{color}]{p['pipeline_id']}[/{color}]",
                str(p["event_count"]),
                started,
                last,
            )

        self._console.print(table)
        self._console.print()
