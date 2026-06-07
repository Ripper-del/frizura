"""CLI application for Frizura.

Provides commands to run pipelines, inspect history (TUI debugger),
replay execution, and manage the local hybrid model swarm.
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console

# Lazy initialization of Typer app
app = typer.Typer(
    name="frizura",
    help="Frizura: Next-gen LLM orchestrator with time-travel and auto-healing.",
    no_args_is_help=True,
)
console = Console()


def main() -> None:
    """CLI entrypoint."""
    try:
        app()
    except Exception as exc:
        console.print(f"[red]Error:[/red] {exc}")
        sys.exit(1)


# ---------------------------------------------------------------------------
# run Command
# ---------------------------------------------------------------------------

@app.command(name="run", help="Execute a pipeline script.")
def run_pipeline(
    script_path: Path = typer.Argument(
        ...,
        help="Path to the Python file containing the Frizura pipeline/task.",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
    ),
    budget_cost: Optional[float] = typer.Option(
        None,
        "--budget-cost",
        "-c",
        help="Max budget cost in USD.",
    ),
    budget_time: Optional[float] = typer.Option(
        None,
        "--budget-time",
        "-t",
        help="Max runtime in seconds.",
    ),
) -> None:
    """Run a pipeline from a python script file."""
    import importlib.util
    from frizura.models.budget import Budget
    from frizura.core.engine import FrizuraEngine
    from frizura.core.graph import Pipeline

    console.print(f"Running pipeline script: [green]{script_path}[/green]...")

    # Load the script dynamically
    try:
        spec = importlib.util.spec_from_file_location("user_script", script_path)
        if not spec or not spec.loader:
            raise ImportError("Could not load script spec.")
        
        module = importlib.util.module_from_spec(spec)
        sys.modules["user_script"] = module
        spec.loader.exec_module(module)
    except Exception as exc:
        console.print(f"[red]Error loading script:[/red] {exc}")
        raise typer.Exit(code=1)

    # Look for a Pipeline object or decorated tasks in the module
    pipeline_obj = None
    for name, obj in inspect_members(module):
        if isinstance(obj, Pipeline):
            pipeline_obj = obj
            break

    # If we have a pipeline object, run it
    if pipeline_obj:
        console.print(f"Found pipeline: [cyan]{pipeline_obj.name}[/cyan]. Running...")
        
        # Build budget constraints
        budget = None
        if budget_cost is not None or budget_time is not None:
            budget = Budget(max_cost=budget_cost, max_time=budget_time)

        engine = FrizuraEngine()
        
        # Run in event loop
        try:
            result = asyncio.run(engine.run(pipeline_obj, input_data={}, budget=budget))
            
            # Print results
            console.print("\n[bold green]Pipeline Completed![/bold green]")
            console.print(f"Pipeline ID: [magenta]{result.pipeline_id}[/magenta]")
            console.print(f"Status: {result.status}")
            console.print(f"Total Cost: [green]${result.total_cost_usd:.5f}[/green]")
            console.print(f"Total Duration: {result.total_duration_ms / 1000:.2f}s")
            console.print(f"Steps executed: {len(result.steps)}")
        except Exception as exc:
            console.print(f"[red]Pipeline execution failed:[/red] {exc}")
            raise typer.Exit(code=1)
    else:
        console.print("[yellow]No Pipeline object found in the script.[/yellow]")
        console.print("Make sure your script instantiates and exposes a `Pipeline` object.")


def inspect_members(module: Any) -> list[tuple[str, Any]]:
    """Helper to get members of a module safely."""
    try:
        import inspect
        return inspect.getmembers(module)
    except Exception:
        return []


# ---------------------------------------------------------------------------
# inspect Command (Time-travel debugger view)
# ---------------------------------------------------------------------------

@app.command(name="inspect", help="Inspect and debug a pipeline's event log.")
def inspect_pipeline(
    pipeline_id: str = typer.Argument(
        ...,
        help="ID of the pipeline to inspect.",
    ),
) -> None:
    """Print a beautiful summary of pipeline execution using Rich."""
    from frizura.timetravel.store import EventStore
    from frizura.timetravel.replay import ReplayEngine
    from frizura.timetravel.inspector import Inspector

    try:
        store = EventStore()
        async def run_inspector():
            await store.init()
            replay_engine = ReplayEngine(store=store)
            inspector = Inspector(store=store, replay_engine=replay_engine)
            await inspector.show(pipeline_id)
            await store.close()
            
        asyncio.run(run_inspector())
    except ImportError:
        # Fallback if inspector is not implemented or fails to import
        console.print("[red]Inspector module not available. Fallback to CLI summary.[/red]")
        _fallback_inspect(pipeline_id)
    except Exception as exc:
        console.print(f"[red]Error inspecting pipeline:[/red] {exc}")


def _fallback_inspect(pipeline_id: str) -> None:
    """Fallback inspect using direct SQLite query if Inspector TUI is missing."""
    import sqlite3
    import json
    db_path = ".frizura/events.db"
    if not os.path.exists(db_path):
        console.print(f"[red]Database not found at {db_path}[/red]")
        return
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Retrieve pipeline summary
    cursor.execute("SELECT event_type, timestamp, data FROM events WHERE pipeline_id = ? ORDER BY sequence_number", (pipeline_id,))
    rows = cursor.fetchall()
    
    if not rows:
        console.print(f"[yellow]No events found for pipeline: {pipeline_id}[/yellow]")
        conn.close()
        return

    console.print(f"\n[bold]Pipeline {pipeline_id} Event Log:[/bold]")
    for row in rows:
        ev_type, ts, data_str = row
        try:
            data = json.loads(data_str)
        except Exception:
            data = data_str
        console.print(f"[{ts}] {ev_type}: {data}")
    conn.close()


# ---------------------------------------------------------------------------
# replay Command
# ---------------------------------------------------------------------------

@app.command(name="replay", help="Replay a pipeline up to a specific step.")
def replay_pipeline(
    pipeline_id: str = typer.Argument(
        ...,
        help="Pipeline ID to replay.",
    ),
    to_step: Optional[str] = typer.Option(
        None,
        "--to",
        help="Step ID or name to stop replay at.",
    ),
) -> None:
    """Replay execution of a pipeline."""
    from frizura.timetravel.replay import ReplayEngine
    
    console.print(f"Replaying pipeline [magenta]{pipeline_id}[/magenta]...")
    engine = ReplayEngine()
    
    try:
        ctx = asyncio.run(engine.replay_to(pipeline_id, to_step))
        console.print("[green]Replay completed successfully![/green]")
        console.print(f"Replayed State: {ctx.state}")
    except Exception as exc:
        console.print(f"[red]Replay failed:[/red] {exc}")
        raise typer.Exit(code=1)


# ---------------------------------------------------------------------------
# swarm Command
# ---------------------------------------------------------------------------

swarm_app = typer.Typer(name="swarm", help="Manage local hybrid models pool.")
app.add_typer(swarm_app)


@swarm_app.command(name="status", help="Show health status of local Ollama pool.")
def swarm_status() -> None:
    """Check health and list local models in Ollama."""
    from frizura.swarm.pool import LocalPool
    pool = LocalPool()
    
    console.print("Checking local pool health...")
    try:
        status = asyncio.run(pool.healthcheck())
        console.print(f"\n[bold]Swarm Pool Status:[/bold]")
        console.print(f"Healthy hosts: {status.healthy_hosts}/{status.total_hosts}")
        console.print(f"Available models: {status.models_available}")
        for host, info in status.details.items():
            color = "green" if info.get("status") == "ok" else "red"
            console.print(f"- {host}: [{color}]{info.get('status')}[/{color}] ({info.get('models', 0)} models)")
    except Exception as exc:
        console.print(f"[red]Failed to check swarm status:[/red] {exc}")
        raise typer.Exit(code=1)


@swarm_app.command(name="discover", help="Discover Ollama models in LAN.")
def swarm_discover() -> None:
    """Scan local network for Ollama hosts."""
    from frizura.swarm.pool import LocalPool
    pool = LocalPool()
    
    console.print("Scanning local pool models...")
    try:
        models = asyncio.run(pool.discover())
        console.print(f"\n[bold]Discovered local models ({len(models)}):[/bold]")
        for m in models:
            console.print(f"- [cyan]{m.name}[/cyan] on {m.host} (Size: {m.size/1e6:.1f}MB, Quantization: {m.quantization})")
    except Exception as exc:
        console.print(f"[red]Failed to discover models:[/red] {exc}")
        raise typer.Exit(code=1)
