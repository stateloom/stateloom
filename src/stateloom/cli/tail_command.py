"""stateloom tail — live event stream from the dashboard WebSocket."""

from __future__ import annotations

import asyncio
import json
from typing import Any

import click
import websockets


def _format_event(data: dict[str, Any]) -> str | None:
    """Format an event dict into a console-friendly line using Rich."""
    from rich.console import Console
    from rich.text import Text

    event_type = data.get("event_type", "")
    line = Text()

    if event_type == "llm_call":
        line.append("[StateLoom] ", style="bold cyan")
        line.append("  ", style="green")
        line.append(f"{data.get('model', '?')} ", style="bold")
        tokens = data.get("total_tokens", 0)
        line.append(f"| {tokens} tok ", style="dim")
        cost = data.get("cost", 0)
        line.append(f"| ${cost:.4f} ", style="yellow")
        latency = data.get("latency_ms", 0)
        line.append(f"| {latency:.0f}ms ", style="dim")
        sid = data.get("session_id", "?")
        line.append(f"| session:{sid[:20]}", style="blue")

    elif event_type == "cache_hit":
        line.append("[StateLoom] ", style="bold cyan")
        line.append("  CACHE HIT ", style="bold yellow")
        match_type = data.get("match_type", "exact")
        saved = data.get("saved_cost", 0)
        if match_type == "semantic":
            score = data.get("similarity_score", 0)
            line.append(f"| semantic match ({score:.3f}) | saved ${saved:.4f}", style="dim")
        else:
            line.append(f"| exact match | saved ${saved:.4f}", style="dim")

    elif event_type == "pii_detection":
        line.append("[StateLoom] ", style="bold cyan")
        action = data.get("action_taken", "")
        if action == "blocked":
            line.append("  PII BLOCKED ", style="bold red")
        else:
            line.append("  PII DETECTED ", style="bold yellow")
        pii_type = data.get("pii_type", "")
        line.append(f"| type:{pii_type}", style="")

    elif event_type == "local_routing":
        line.append("[StateLoom] ", style="bold cyan")
        if data.get("routing_success"):
            line.append("  ROUTED LOCAL ", style="bold green")
        else:
            line.append("  ROUTE FALLBACK ", style="bold yellow")
        line.append(f"| {data.get('local_model', '?')} ", style="bold")
        score = data.get("complexity_score", 0)
        line.append(f"| complexity:{score:.2f}", style="dim")

    elif event_type == "kill_switch":
        line.append("[StateLoom] ", style="bold cyan")
        line.append("  KILL SWITCH ", style="bold red")
        line.append(f"| {data.get('reason', '')}", style="")

    elif event_type == "blast_radius":
        line.append("[StateLoom] ", style="bold cyan")
        line.append("  BLAST RADIUS ", style="bold red")
        line.append(f"| {data.get('trigger', '')} ", style="")
        line.append(f"| {data.get('count', 0)}/{data.get('threshold', 0)} ", style="yellow")
        line.append(f"| action:{data.get('action', '')}", style="dim")

    elif event_type == "budget_enforcement":
        line.append("[StateLoom] ", style="bold cyan")
        line.append("  BUDGET ENFORCED ", style="bold red")
        limit = data.get("limit", 0)
        spent = data.get("spent", 0)
        line.append(f"| limit:${limit:.2f} | spent:${spent:.2f}", style="yellow")

    elif event_type == "tool_call":
        line.append("[StateLoom] ", style="bold cyan")
        line.append("  TOOL ", style="bold magenta")
        line.append(f"| {data.get('tool_name', '?')} ", style="bold")
        latency = data.get("latency_ms", 0)
        line.append(f"| {latency:.0f}ms", style="dim")

    elif event_type == "checkpoint":
        label = data.get("label", data.get("metadata", {}).get("label", ""))
        line.append("[StateLoom] ", style="bold cyan")
        line.append(f"  CHECKPOINT: {label}", style="bold white")

    else:
        # Unknown event type — show raw type
        line.append("[StateLoom] ", style="bold cyan")
        line.append(f"  {event_type.upper()} ", style="bold")
        sid = data.get("session_id", "")
        if sid:
            line.append(f"| session:{sid[:20]}", style="blue")

    if not line.plain.strip():
        return None

    # Render to string (no ANSI when not a terminal)
    console = Console(highlight=False)
    with console.capture() as capture:
        console.print(line, end="")
    return capture.get()


async def _tail(host: str, port: int, session_filter: str | None, as_json: bool) -> None:
    from websockets.exceptions import ConnectionClosed

    uri = f"ws://{host}:{port}/ws"
    try:
        async with websockets.connect(uri) as ws:
            click.echo(f"Connected to {uri} — streaming events (Ctrl+C to stop)")
            try:
                while True:
                    try:
                        raw = await asyncio.wait_for(ws.recv(), timeout=35)
                    except asyncio.TimeoutError:
                        # Send ping to keep alive
                        await ws.send("ping")
                        continue

                    try:
                        msg = json.loads(raw)
                    except (json.JSONDecodeError, TypeError):
                        continue

                    if msg.get("type") == "heartbeat":
                        continue

                    if msg.get("type") != "new_event":
                        continue

                    data = msg.get("data", {})

                    # Session filter
                    if session_filter:
                        sid = data.get("session_id", "")
                        if session_filter not in sid:
                            continue

                    if as_json:
                        click.echo(json.dumps(data))
                    else:
                        formatted = _format_event(data)
                        if formatted:
                            click.echo(formatted)
            except ConnectionClosed:
                click.echo("\nConnection closed by server.")

    except (OSError, Exception) as exc:
        click.echo(f"Error: could not connect to {uri} — {exc}")
        raise SystemExit(1)


@click.command()
@click.option("--port", default=4782, type=int, show_default=True, help="Dashboard port")
@click.option("--host", default="127.0.0.1", show_default=True, help="Dashboard host")
@click.option("--session", "session_filter", default=None, help="Filter by session ID (substring)")
@click.option("--json", "as_json", is_flag=True, help="Output raw JSON")
def tail(port: int, host: str, session_filter: str | None, as_json: bool) -> None:
    """Stream live events from the running StateLoom server."""
    try:
        asyncio.run(_tail(host, port, session_filter, as_json))
    except KeyboardInterrupt:
        click.echo("\nDisconnected.")
