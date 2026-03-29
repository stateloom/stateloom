"""CLI subcommands for managing the bundled Ollama installation."""

from __future__ import annotations

import click

from stateloom.local.manager import DEFAULT_PORT, OllamaManager


@click.group()
def ollama() -> None:
    """Manage the bundled Ollama installation for local models."""


@ollama.command()
@click.option("--version", "ver", default="latest", help="Ollama release tag (e.g. v0.9.0)")
def install(ver: str) -> None:
    """Download the Ollama binary for your platform."""
    mgr = OllamaManager()

    if mgr.is_installed():
        current = mgr.installed_version()
        if ver == "latest" or ver == current:
            click.echo(f"Ollama is already installed (version {current}).")
            click.echo("To reinstall, delete ~/.stateloom/ollama/ first.")
            return

    click.echo(f"Installing Ollama ({ver})...")

    with click.progressbar(length=100, label="Downloading") as bar:
        last_pct = [0]

        def _progress(downloaded: int, total: int) -> None:
            if total > 0:
                pct = min(int(downloaded * 100 / total), 100)
                delta = pct - last_pct[0]
                if delta > 0:
                    bar.update(delta)
                    last_pct[0] = pct

        path, tag = mgr.install(version=ver, progress=_progress)
        # Fill remaining progress
        remaining = 100 - last_pct[0]
        if remaining > 0:
            bar.update(remaining)

    click.echo(f"Installed Ollama {tag} at {path}")


@ollama.command("start")
@click.option("--port", default=DEFAULT_PORT, type=int, show_default=True, help="Port number")
def start_cmd(port: int) -> None:
    """Start the managed Ollama server."""
    mgr = OllamaManager()

    if not mgr.is_installed():
        click.echo("Ollama is not installed. Run: stateloom ollama install")
        raise SystemExit(1)

    if mgr.is_running(port=port):
        click.echo(f"Managed Ollama is already running on port {port}.")
        return

    click.echo(f"Starting managed Ollama on port {port}...")
    pid = mgr.start(port=port)
    click.echo(f"Ollama started (PID {pid}, port {port}).")


@ollama.command("stop")
def stop_cmd() -> None:
    """Stop the managed Ollama server."""
    mgr = OllamaManager()
    pid_path = mgr._pid_path

    if not pid_path.exists():
        click.echo("Managed Ollama is not running.")
        return

    click.echo("Stopping managed Ollama...")
    mgr.stop()
    click.echo("Ollama stopped.")


@ollama.command("status")
@click.option("--port", default=DEFAULT_PORT, type=int, show_default=True, help="Port number")
def status_cmd(port: int) -> None:
    """Show the status of the managed Ollama installation."""
    mgr = OllamaManager()

    if not mgr.is_installed():
        click.echo("Ollama: not installed")
        click.echo("Run: stateloom ollama install")
        return

    version = mgr.installed_version() or "unknown"
    click.echo(f"Ollama version: {version}")
    click.echo(f"Binary: {mgr._bin}")

    if mgr.is_running(port=port):
        click.echo(f"Status: running (port {port})")

        # List models
        try:
            import httpx

            with httpx.Client(
                base_url=f"http://127.0.0.1:{port}",
                timeout=httpx.Timeout(5.0),
            ) as client:
                resp = client.get("/api/tags")
                resp.raise_for_status()
                models = resp.json().get("models", [])
                if models:
                    click.echo(f"Models ({len(models)}):")
                    for m in models:
                        name = m.get("name", "?")
                        size_gb = m.get("size", 0) / (1024**3)
                        click.echo(f"  - {name} ({size_gb:.1f} GB)")
                else:
                    click.echo("Models: none")
        except Exception:
            click.echo("Models: (could not query)")
    else:
        click.echo("Status: not running")


@ollama.command()
@click.argument("model")
@click.option("--port", default=DEFAULT_PORT, type=int, show_default=True, help="Port number")
def pull(model: str, port: int) -> None:
    """Pull (download) a model for local inference."""
    mgr = OllamaManager()

    if not mgr.is_running(port=port):
        click.echo(f"Managed Ollama is not running on port {port}.")
        click.echo("Run: stateloom ollama start")
        raise SystemExit(1)

    click.echo(f"Pulling model '{model}'...")

    last_status = [""]

    def _progress(data: dict) -> None:
        status = data.get("status", "")
        if status != last_status[0]:
            click.echo(f"  {status}")
            last_status[0] = status

    mgr.ensure_model(model, port=port, progress=_progress)
    click.echo(f"Model '{model}' is ready.")


@ollama.command("list")
@click.option("--port", default=DEFAULT_PORT, type=int, show_default=True, help="Port number")
def list_cmd(port: int) -> None:
    """List downloaded models."""
    mgr = OllamaManager()

    if not mgr.is_running(port=port):
        click.echo(f"Managed Ollama is not running on port {port}.")
        click.echo("Run: stateloom ollama start")
        raise SystemExit(1)

    try:
        import httpx

        with httpx.Client(
            base_url=f"http://127.0.0.1:{port}",
            timeout=httpx.Timeout(5.0),
        ) as client:
            resp = client.get("/api/tags")
            resp.raise_for_status()
            models = resp.json().get("models", [])
    except Exception as e:
        click.echo(f"Failed to list models: {e}")
        raise SystemExit(1)

    if not models:
        click.echo("No models downloaded.")
        click.echo("Run: stateloom ollama pull <model>")
        return

    click.echo(f"Downloaded models ({len(models)}):")
    for m in models:
        name = m.get("name", "?")
        size_gb = m.get("size", 0) / (1024**3)
        click.echo(f"  {name:<30} {size_gb:.1f} GB")


@ollama.command()
def recommend() -> None:
    """Show hardware-aware model recommendations."""
    from stateloom.local.hardware import detect_hardware, recommend_models

    hw = detect_hardware()
    click.echo(f"Hardware: {hw.cpu_brand or hw.arch}")
    click.echo(f"  RAM: {hw.total_ram_gb:.1f} GB")
    if hw.gpu_name:
        click.echo(f"  GPU: {hw.gpu_name} ({hw.gpu_vram_gb:.1f} GB VRAM)")
    click.echo()

    recs = recommend_models(hw)
    if not recs:
        click.echo("No model recommendations for your hardware.")
        return

    click.echo("Recommended models:")
    for r in recs:
        name = r.get("model", "?")
        size = r.get("size_gb", 0)
        desc = r.get("description", "")
        click.echo(f"  {name:<25} {size:.1f} GB  {desc}")
