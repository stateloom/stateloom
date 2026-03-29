"""stateloom doctor — health and diagnostics check."""

from __future__ import annotations

import importlib
import json
import os
from typing import Any

import click
import httpx


def _check(category: str, name: str, status: str, detail: str) -> dict[str, str]:
    return {"category": category, "name": name, "status": status, "detail": detail}


def _check_sdk(module: str, label: str) -> dict[str, str]:
    try:
        mod = importlib.import_module(module)
        version = getattr(mod, "__version__", getattr(mod, "VERSION", "unknown"))
        return _check("SDKs", f"{label} installed", "pass", f"v{version}")
    except ImportError:
        return _check("SDKs", f"{label} installed", "fail", "not installed")


def _run_checks(port: int) -> list[dict[str, str]]:
    results: list[dict[str, str]] = []

    # SDK checks
    results.append(_check_sdk("openai", "OpenAI SDK"))
    results.append(_check_sdk("anthropic", "Anthropic SDK"))
    results.append(_check_sdk("google.generativeai", "Gemini SDK"))
    results.append(_check_sdk("mistralai", "Mistral SDK"))
    results.append(_check_sdk("cohere", "Cohere SDK"))
    results.append(_check_sdk("litellm", "LiteLLM SDK"))

    # Provider API key checks
    for env_var, label in [
        ("OPENAI_API_KEY", "OPENAI_API_KEY"),
        ("ANTHROPIC_API_KEY", "ANTHROPIC_API_KEY"),
    ]:
        val = os.environ.get(env_var)
        if val:
            results.append(_check("Providers", f"{label} is set", "pass", "configured"))
        else:
            results.append(_check("Providers", f"{label} not set", "warn", "not set"))

    # Google key — check both env vars
    google_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
    if google_key:
        results.append(_check("Providers", "Google API key is set", "pass", "configured"))
    else:
        results.append(
            _check("Providers", "Google API key not set", "warn", "GOOGLE_API_KEY / GEMINI_API_KEY")
        )

    # Local — Ollama reachable (system)
    try:
        from stateloom.local.client import OllamaClient

        client = OllamaClient()
        if client.is_available():
            models = client.list_models()
            count = len(models) if models else 0
            results.append(
                _check("Local", "Ollama reachable", "pass", f"{count} model(s) available")
            )
        else:
            results.append(_check("Local", "Ollama reachable", "fail", "not responding"))
    except Exception:
        results.append(_check("Local", "Ollama reachable", "fail", "not responding"))

    # Local — managed Ollama installed
    try:
        from stateloom.local.manager import OllamaManager

        mgr = OllamaManager()
        if mgr.is_installed():
            ver = mgr.installed_version() or "unknown"
            results.append(_check("Local", "Managed Ollama installed", "pass", f"version {ver}"))
        else:
            results.append(_check("Local", "Managed Ollama installed", "warn", "not installed"))
    except Exception:
        results.append(_check("Local", "Managed Ollama installed", "warn", "not installed"))

    # Hardware
    try:
        from stateloom.local.hardware import detect_hardware

        hw = detect_hardware()
        results.append(_check("Hardware", "RAM", "pass", f"{hw.ram_gb:.1f} GB"))
        if hw.gpu_name:
            detail = hw.gpu_name
            if hw.gpu_vram_gb > 0:
                detail += f" — {hw.gpu_vram_gb:.1f} GB"
            results.append(_check("Hardware", "GPU/Accelerator", "pass", detail))
        else:
            results.append(_check("Hardware", "GPU/Accelerator", "warn", "none detected"))
        results.append(_check("Hardware", "Disk free", "pass", f"{hw.disk_free_gb:.1f} GB"))
    except Exception:
        results.append(_check("Hardware", "Hardware detection", "fail", "error"))

    # Store — SQLite writable
    try:
        from pathlib import Path

        store_path = Path(".stateloom") / "data.db"
        store_path.parent.mkdir(parents=True, exist_ok=True)
        # Try opening for append to check writability
        with open(store_path, "a"):
            pass
        results.append(_check("Store", "SQLite store writable", "pass", str(store_path)))
    except Exception as exc:
        results.append(_check("Store", "SQLite store writable", "fail", str(exc)))

    # Server — dashboard reachable
    try:
        resp = httpx.get(f"http://127.0.0.1:{port}/api/v1/health", timeout=3)
        if resp.status_code == 200:
            results.append(_check("Server", "Dashboard reachable", "pass", f"port {port}"))
        else:
            results.append(
                _check("Server", "Dashboard reachable", "fail", f"HTTP {resp.status_code}")
            )
    except Exception:
        results.append(
            _check("Server", "Dashboard reachable", "fail", f"not reachable on port {port}")
        )

    # Optional dependencies
    for module, label in [
        ("sentence_transformers", "sentence-transformers"),
        ("faiss", "faiss-cpu"),
        ("prometheus_client", "prometheus_client"),
    ]:
        try:
            importlib.import_module(module)
            results.append(_check("Dependencies", label, "pass", "installed"))
        except ImportError:
            results.append(_check("Dependencies", label, "warn", "not installed"))

    return results


_STATUS_ICONS = {"pass": "\u2714", "warn": "\u26a0", "fail": "\u2718"}
_STATUS_COLORS = {"pass": "green", "warn": "yellow", "fail": "red"}


def _print_results(results: list[dict[str, str]]) -> None:
    from rich.console import Console
    from rich.text import Text

    console = Console()
    console.print()
    console.print(Text("StateLoom Doctor", style="bold"))
    console.print(Text("=" * 40, style="dim"))

    for r in results:
        line = Text()
        icon = _STATUS_ICONS.get(r["status"], "?")
        color = _STATUS_COLORS.get(r["status"], "white")
        line.append(f"  [{icon}] ", style=f"bold {color}")
        line.append(f"{r['category']:<14}", style="dim")
        line.append(f"{r['name']}", style="")
        if r["detail"]:
            line.append(f" ({r['detail']})", style="dim")
        console.print(line)

    passed = sum(1 for r in results if r["status"] == "pass")
    warned = sum(1 for r in results if r["status"] == "warn")
    failed = sum(1 for r in results if r["status"] == "fail")
    console.print()
    summary = Text()
    summary.append("Summary: ", style="bold")
    summary.append(f"{passed} passed", style="green")
    if warned:
        summary.append(f", {warned} warning{'s' if warned != 1 else ''}", style="yellow")
    if failed:
        summary.append(f", {failed} failed", style="red")
    console.print(summary)
    console.print()


@click.command()
@click.option("--port", default=4782, type=int, show_default=True, help="Dashboard port to check")
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
def doctor(port: int, as_json: bool) -> None:
    """Run diagnostic checks on the StateLoom environment."""
    results = _run_checks(port)

    if as_json:
        click.echo(json.dumps(results, indent=2))
    else:
        _print_results(results)

    # Exit with error code if any check failed
    if any(r["status"] == "fail" for r in results):
        raise SystemExit(1)
