"""StateLoom CLI — start the proxy server and manage configuration."""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any

import click

_PID_DIR = Path(".stateloom")
_PID_FILE = _PID_DIR / "server.pid"
_LOG_FILE = _PID_DIR / "server.log"


def _read_pid() -> int | None:
    """Read the PID from the pidfile, returning None if absent or stale."""
    if not _PID_FILE.exists():
        return None
    try:
        pid = int(_PID_FILE.read_text().strip())
    except (ValueError, OSError):
        return None
    # Check if process is still alive
    try:
        os.kill(pid, 0)
    except OSError:
        # Process is gone — clean up stale pidfile
        _PID_FILE.unlink(missing_ok=True)
        return None
    return pid


def _write_pid(pid: int) -> None:
    _PID_DIR.mkdir(parents=True, exist_ok=True)
    _PID_FILE.write_text(str(pid))


def _remove_pid() -> None:
    _PID_FILE.unlink(missing_ok=True)


@click.group(invoke_without_command=True)
@click.pass_context
def main(ctx: click.Context) -> None:
    """StateLoom — AI gateway with observability and guardrails."""
    if ctx.invoked_subcommand is None:
        click.echo(ctx.get_help())


# Register ollama subcommand group
from stateloom.cli.ollama_commands import ollama  # noqa: E402

main.add_command(ollama)

# Register diagnostic commands
from stateloom.cli.doctor_command import doctor  # noqa: E402
from stateloom.cli.stats_command import stats  # noqa: E402
from stateloom.cli.tail_command import tail  # noqa: E402

main.add_command(doctor)
main.add_command(stats)
main.add_command(tail)

# Register auth commands
from stateloom.cli.login_command import login, logout  # noqa: E402

main.add_command(login)
main.add_command(logout)


@main.command()
@click.option("--host", default="127.0.0.1", show_default=True, help="Bind address")
@click.option("--port", default=4782, type=int, show_default=True, help="Port number")
@click.option("--no-auth", is_flag=True, help="Don't require virtual keys (solo dev mode)")
@click.option("--verbose", is_flag=True, help="Enable console output for each LLM call")
@click.option("--with-ollama", is_flag=True, help="Start managed Ollama for local models")
@click.option(
    "--debug", is_flag=True, help="Enable debug mode (verbose logging, server logs in dashboard)"
)
def serve(
    host: str, port: int, no_auth: bool, verbose: bool, with_ollama: bool, debug: bool
) -> None:
    """Start the StateLoom proxy server (foreground).

    Speaks OpenAI, Anthropic, and Gemini native API formats.

    \b
    Usage with Claude CLI:
        stateloom serve --no-auth
        export ANTHROPIC_BASE_URL=http://localhost:4782
        claude "explain this code"

    \b
    Usage with Gemini CLI:
        stateloom serve --no-auth
        export CODE_ASSIST_ENDPOINT=http://localhost:4782/code-assist
        gemini "explain this code"
    """
    _run_server(host, port, no_auth, verbose, foreground=True, with_ollama=with_ollama, debug=debug)


@main.command()
@click.option("--host", default="127.0.0.1", show_default=True, help="Bind address")
@click.option("--port", default=4782, type=int, show_default=True, help="Port number")
@click.option("--no-auth", is_flag=True, help="Don't require virtual keys (solo dev mode)")
@click.option("--verbose", is_flag=True, help="Enable console output for each LLM call")
@click.option("--with-ollama", is_flag=True, help="Start managed Ollama for local models")
@click.option(
    "--debug", is_flag=True, help="Enable debug mode (verbose logging, server logs in dashboard)"
)
def start(
    host: str, port: int, no_auth: bool, verbose: bool, with_ollama: bool, debug: bool
) -> None:
    """Start the StateLoom proxy server in the background.

    \b
    Usage:
        stateloom start --no-auth
        stateloom stop
    """
    existing = _read_pid()
    if existing is not None:
        click.echo(f"StateLoom is already running (PID {existing}).")
        click.echo("Run 'stateloom stop' first, or 'stateloom status' to check.")
        raise SystemExit(1)

    # Build the command that the child process will run
    cmd = [
        sys.executable,
        "-m",
        "stateloom.cli",
        "serve",
        "--host",
        host,
        "--port",
        str(port),
    ]
    if no_auth:
        cmd.append("--no-auth")
    if verbose:
        cmd.append("--verbose")
    if with_ollama:
        cmd.append("--with-ollama")
    if debug:
        cmd.append("--debug")

    _PID_DIR.mkdir(parents=True, exist_ok=True)
    log_fh = open(_LOG_FILE, "a")

    proc = subprocess.Popen(
        cmd,
        stdout=log_fh,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    _write_pid(proc.pid)

    click.echo(f"StateLoom started (PID {proc.pid}).")
    click.echo("")
    click.echo("Endpoints:")
    click.echo(f"  OpenAI:    POST http://{host}:{port}/v1/chat/completions")
    click.echo(f"  Anthropic: POST http://{host}:{port}/v1/messages")
    click.echo(f"  Gemini:    POST http://{host}:{port}/v1beta/models/MODEL:generateContent")
    click.echo(f"  Dashboard: http://{host}:{port}")
    click.echo("")
    click.echo("Set these environment variables to route CLI traffic through StateLoom:")
    click.echo(f"  export ANTHROPIC_BASE_URL=http://{host}:{port}")
    click.echo(f"  export CODE_ASSIST_ENDPOINT=http://{host}:{port}/code-assist")
    click.echo(f"  export OPENAI_BASE_URL=http://{host}:{port}/v1")
    click.echo("")
    click.echo(f"Logs: {_LOG_FILE.resolve()}")
    click.echo("Run 'stateloom stop' to shut down.")


@main.command()
def stop() -> None:
    """Stop the background StateLoom server."""
    pid = _read_pid()
    if pid is None:
        click.echo("StateLoom is not running (no PID file found).")
        raise SystemExit(1)

    click.echo(f"Stopping StateLoom (PID {pid})...")
    try:
        os.kill(pid, signal.SIGTERM)
    except OSError as exc:
        click.echo(f"Failed to send signal: {exc}")
        _remove_pid()
        raise SystemExit(1)

    # Wait briefly for the process to exit
    import time

    for _ in range(20):  # up to 2 seconds
        time.sleep(0.1)
        try:
            os.kill(pid, 0)
        except OSError:
            break  # process exited

    _remove_pid()
    click.echo("StateLoom stopped.")


@main.command()
def restart() -> None:
    """Restart the background StateLoom server.

    Sends SIGUSR1 to the running process, which triggers a graceful
    re-exec (reload code + config without losing the PID file).

    \b
    Usage:
        stateloom restart
    """
    pid = _read_pid()
    if pid is None:
        click.echo("StateLoom is not running. Use 'stateloom start' first.")
        raise SystemExit(1)

    click.echo(f"Restarting StateLoom (PID {pid})...")
    if not hasattr(signal, "SIGUSR1"):
        # Windows: no SIGUSR1 — fall back to stop + start
        click.echo("SIGUSR1 not available on this platform — using stop+start.")
        try:
            os.kill(pid, signal.SIGTERM)
        except OSError as exc:
            click.echo(f"Failed to stop: {exc}")
            _remove_pid()
            raise SystemExit(1)
        import time

        for _ in range(20):
            time.sleep(0.1)
            try:
                os.kill(pid, 0)
            except OSError:
                break
        _remove_pid()
        click.echo("Stopped. Re-run 'stateloom start' manually.")
        return

    try:
        os.kill(pid, signal.SIGUSR1)
    except OSError as exc:
        click.echo(f"Failed to send restart signal: {exc}")
        _remove_pid()
        raise SystemExit(1)

    click.echo("Restart signal sent. StateLoom is reloading.")


@main.command()
def status() -> None:
    """Check if the StateLoom server is running."""
    pid = _read_pid()
    if pid is None:
        click.echo("StateLoom is not running.")
        raise SystemExit(1)
    click.echo(f"StateLoom is running (PID {pid}).")


def _run_server(
    host: str,
    port: int,
    no_auth: bool,
    verbose: bool,
    *,
    foreground: bool,
    with_ollama: bool = False,
    debug: bool = False,
) -> None:
    """Shared server startup logic for both serve and start commands."""
    import stateloom

    stateloom.init(
        dashboard=True,
        dashboard_host=host,
        dashboard_port=port,
        proxy_require_virtual_key=not no_auth,
        console_output=verbose,
        shadow=False,
        with_ollama=with_ollama,
        debug=debug,
    )

    if foreground:
        click.echo(f"StateLoom running on http://{host}:{port}")
        click.echo("")
        click.echo("Endpoints:")
        click.echo(f"  OpenAI:    POST http://{host}:{port}/v1/chat/completions")
        click.echo(f"  Anthropic: POST http://{host}:{port}/v1/messages")
        click.echo(f"  Gemini:    POST http://{host}:{port}/v1beta/models/MODEL:generateContent")
        click.echo(f"  Dashboard: http://{host}:{port}")
        click.echo("")
        click.echo("Set these environment variables to route CLI traffic through StateLoom:")
        click.echo(f"  export ANTHROPIC_BASE_URL=http://{host}:{port}")
        click.echo(f"  export CODE_ASSIST_ENDPOINT=http://{host}:{port}/code-assist")
        click.echo(f"  export OPENAI_BASE_URL=http://{host}:{port}/v1")
        click.echo("")
        click.echo("Press Ctrl+C to stop.")

    # Handle SIGUSR1 for graceful restart (re-exec the process)
    _restart_requested = threading.Event()

    def _handle_restart(signum: int, frame: Any) -> None:
        click.echo("\nRestart requested — reloading...")
        _restart_requested.set()

    if hasattr(signal, "SIGUSR1"):
        signal.signal(signal.SIGUSR1, _handle_restart)

    try:
        _restart_requested.wait()
    except KeyboardInterrupt:
        if foreground:
            click.echo("\nShutting down...")
        stateloom.shutdown()
        return

    # Restart: shut down cleanly then re-exec the same process
    stateloom.shutdown()
    os.execv(sys.executable, [sys.executable] + sys.argv)
