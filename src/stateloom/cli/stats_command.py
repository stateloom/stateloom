"""stateloom stats — quick terminal dashboard from the running server."""

from __future__ import annotations

import json

import click


def _fetch(host: str, port: int, path: str) -> dict:
    import httpx

    resp = httpx.get(f"http://{host}:{port}{path}", timeout=10)
    resp.raise_for_status()
    return resp.json()


def _print_stats(stats: dict, breakdown: dict, latency: dict) -> None:
    from rich.console import Console
    from rich.table import Table
    from rich.text import Text

    console = Console()

    # Overview
    console.print()
    console.print(Text("StateLoom Stats", style="bold"))
    console.print(Text("=" * 50, style="dim"))

    console.print()
    console.print(Text("Overview", style="bold underline"))
    total_cost = stats.get("total_cost", 0)
    total_tokens = stats.get("total_tokens", 0)
    total_calls = stats.get("total_calls", 0)
    active = stats.get("active_sessions", 0)
    cache_hits = stats.get("total_cache_hits", 0)
    cache_savings = stats.get("total_cache_savings", 0)

    console.print(f"  Total cost:       ${total_cost:.4f}")
    console.print(f"  Total tokens:     {total_tokens:,}")
    console.print(f"  Total calls:      {total_calls:,}")
    console.print(f"  Active sessions:  {active}")
    console.print(f"  Cache hits:       {cache_hits}")
    console.print(f"  Cache savings:    ${cache_savings:.4f}")

    # By Provider
    by_provider = breakdown.get("by_provider", {})
    if by_provider:
        console.print()
        console.print(Text("By Provider", style="bold underline"))
        table = Table(show_header=True, header_style="bold", box=None, pad_edge=False)
        table.add_column("Provider", style="cyan")
        table.add_column("Requests", justify="right")
        table.add_column("Cost", justify="right", style="yellow")
        for provider, data in sorted(by_provider.items()):
            if isinstance(data, dict):
                table.add_row(
                    provider,
                    str(data.get("count", 0)),
                    f"${data.get('cost', 0):.4f}",
                )
            else:
                table.add_row(provider, str(data), "—")
        console.print(table)

    # By Model
    by_model = breakdown.get("by_model", {})
    if by_model:
        console.print()
        console.print(Text("By Model", style="bold underline"))
        table = Table(show_header=True, header_style="bold", box=None, pad_edge=False)
        table.add_column("Model", style="cyan")
        table.add_column("Requests", justify="right")
        table.add_column("Cost", justify="right", style="yellow")
        table.add_column("Tokens", justify="right", style="dim")
        for model, data in sorted(by_model.items()):
            if isinstance(data, dict):
                table.add_row(
                    model,
                    str(data.get("count", 0)),
                    f"${data.get('cost', 0):.4f}",
                    f"{data.get('tokens', 0):,}",
                )
            else:
                table.add_row(model, str(data), "—", "—")
        console.print(table)

    # Latency
    percentiles = latency.get("percentiles", {})
    if percentiles:
        console.print()
        console.print(Text("Latency", style="bold underline"))
        for label in ("p50", "p90", "p95", "p99"):
            val = percentiles.get(label)
            if val is not None:
                console.print(f"  {label}: {val:.0f}ms")

    console.print()


@click.command()
@click.option("--port", default=4782, type=int, show_default=True, help="Dashboard port")
@click.option("--host", default="127.0.0.1", show_default=True, help="Dashboard host")
@click.option(
    "--window",
    default="24h",
    show_default=True,
    type=click.Choice(["1h", "6h", "24h", "7d"]),
    help="Time window for breakdown",
)
@click.option("--json", "as_json", is_flag=True, help="Output raw JSON")
def stats(port: int, host: str, window: str, as_json: bool) -> None:
    """Show aggregate stats from the running StateLoom server."""
    try:
        stats_data = _fetch(host, port, "/api/v1/stats")
        breakdown = _fetch(host, port, f"/api/v1/observability/breakdown?window={window}")
        latency = _fetch(host, port, f"/api/v1/observability/latency?window={window}")
    except Exception as exc:
        click.echo(f"Error: could not connect to StateLoom on {host}:{port} — {exc}")
        raise SystemExit(1)

    if as_json:
        combined = {
            "stats": stats_data,
            "breakdown": breakdown,
            "latency": latency,
        }
        click.echo(json.dumps(combined, indent=2))
    else:
        _print_stats(stats_data, breakdown, latency)
