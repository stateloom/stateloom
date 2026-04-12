"""CLI subcommands for launching OpenClaw with StateLoom integration."""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import click

from stateloom.cli import _LOG_FILE, _PID_DIR, _read_pid, _write_pid

logger = logging.getLogger(__name__)

_OPENCLAW_CONFIG_DIR = Path.home() / ".openclaw"
_OPENCLAW_CONFIG_FILE = _OPENCLAW_CONFIG_DIR / "openclaw.json"
_OPENCLAW_BACKUP_FILE = _OPENCLAW_CONFIG_DIR / "openclaw.json.stateloom-backup"
_MANAGED_NODE_DIR = Path.home() / ".stateloom" / "openclaw-env"

# Context windows and max output tokens for well-known models.
# Models not in this table get conservative defaults (128_000 / 8192).
_MODEL_LIMITS: dict[str, tuple[int, int]] = {
    # OpenAI
    "gpt-5.4": (128_000, 16384),
    "gpt-5.4-mini": (128_000, 16384),
    "gpt-5.4-nano": (128_000, 16384),
    "gpt-4o": (128_000, 16384),
    "gpt-4o-mini": (128_000, 16384),
    "gpt-4.1": (1_047_576, 32768),
    "gpt-4.1-mini": (1_047_576, 32768),
    "gpt-4.1-nano": (1_047_576, 32768),
    "gpt-4-turbo": (128_000, 4096),
    "gpt-4": (8192, 4096),
    "gpt-3.5-turbo": (16385, 4096),
    "o1": (200_000, 100_000),
    "o1-mini": (128_000, 65536),
    "o3": (200_000, 100_000),
    "o3-mini": (200_000, 100_000),
    # Anthropic
    "claude-opus-4-20250514": (200_000, 8192),
    "claude-sonnet-4-20250514": (200_000, 8192),
    "claude-3-5-sonnet-20241022": (200_000, 8192),
    "claude-3-5-haiku-20241022": (200_000, 8192),
    "claude-3-haiku-20240307": (200_000, 4096),
    "claude-sonnet-4-6": (200_000, 8192),
    "claude-opus-4-6": (200_000, 8192),
    "claude-haiku-4-5": (200_000, 8192),
    "claude-haiku-4-5-20251001": (200_000, 8192),
    # Google
    "gemini-3.1-pro": (1_048_576, 65536),
    "gemini-3-flash": (1_048_576, 65536),
    "gemini-2.5-pro": (1_048_576, 65536),
    "gemini-2.5-flash": (1_048_576, 65536),
    "gemini-2.5-flash-lite": (1_048_576, 8192),
    "gemini-2.0-flash": (1_048_576, 8192),
    "gemini-2.0-flash-lite": (1_048_576, 8192),
    "gemini-1.5-pro": (2_097_152, 8192),
    "gemini-1.5-flash": (1_048_576, 8192),
    "gemini-1.5-flash-8b": (1_048_576, 8192),
    # Mistral
    "mistral-small-latest": (128_000, 8192),
    "mistral-medium-latest": (128_000, 8192),
    "mistral-large-latest": (128_000, 8192),
    "codestral-latest": (256_000, 8192),
    "pixtral-large-latest": (128_000, 8192),
    "open-mistral-nemo": (128_000, 8192),
    # Cohere
    "command-a-03-2025": (256_000, 8192),
    "command-r-plus": (128_000, 4096),
    "command-r": (128_000, 4096),
    "command-r7b-12-2024": (128_000, 4096),
}

_DEFAULT_CONTEXT = 128_000
_DEFAULT_MAX_OUTPUT = 8192


def _humanize_model_name(model_id: str) -> str:
    """Turn a model ID into a human-readable name for OpenClaw's model picker."""
    parts = model_id.replace("-", " ").split()
    name = " ".join(p.capitalize() if p.isalpha() else p for p in parts)
    return f"{name} (via StateLoom)"


def _build_openclaw_models() -> list[dict[str, Any]]:
    """Build the OpenClaw model catalog from prices.json."""
    prices_path = Path(__file__).resolve().parent.parent / "pricing" / "data" / "prices.json"
    with open(prices_path) as f:
        data = json.load(f)

    models: list[dict[str, Any]] = []
    for model_id in data.get("models", {}):
        ctx_window, max_output = _MODEL_LIMITS.get(
            model_id, (_DEFAULT_CONTEXT, _DEFAULT_MAX_OUTPUT)
        )
        models.append(
            {
                "id": model_id,
                "name": _humanize_model_name(model_id),
                "maxTokens": max_output,
                "contextWindow": ctx_window,
            }
        )
    return models


def _read_openclaw_config() -> dict[str, Any]:
    """Read the OpenClaw config file, handling JSON5 gracefully."""
    if not _OPENCLAW_CONFIG_FILE.exists():
        return {}

    raw = _OPENCLAW_CONFIG_FILE.read_text(encoding="utf-8")
    if not raw.strip():
        return {}

    try:
        return json.loads(raw)  # type: ignore[no-any-return]
    except json.JSONDecodeError:
        # Config probably uses JSON5 features (comments, trailing commas)
        try:
            import json5  # type: ignore[import-not-found]

            return json5.loads(raw)  # type: ignore[no-any-return]
        except ImportError:
            click.echo(
                "Error: OpenClaw config uses JSON5 features (comments or trailing commas).\n"
                "Install json5 to handle it: pip install json5"
            )
            raise SystemExit(1)


def _write_openclaw_config(config: dict[str, Any]) -> None:
    """Write the OpenClaw config as valid JSON."""
    _OPENCLAW_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    _OPENCLAW_CONFIG_FILE.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")


def _backup_openclaw_config() -> None:
    """Backup the existing OpenClaw config if it exists."""
    if _OPENCLAW_CONFIG_FILE.exists():
        shutil.copy2(_OPENCLAW_CONFIG_FILE, _OPENCLAW_BACKUP_FILE)


def _get_known_model_ids() -> list[str]:
    """Return list of valid model IDs from prices.json."""
    prices_path = Path(__file__).resolve().parent.parent / "pricing" / "data" / "prices.json"
    with open(prices_path) as f:
        data = json.load(f)
    return list(data.get("models", {}).keys())


def _apply_stateloom_config(
    config: dict[str, Any], host: str, port: int, model_ids: list[str]
) -> dict[str, Any]:
    """Deep-merge StateLoom provider config into OpenClaw config."""
    openclaw_models = _build_openclaw_models()
    base_url = f"http://{host}:{port}/v1"

    # models.providers.stateloom
    models = config.setdefault("models", {})
    models["mode"] = "merge"
    providers = models.setdefault("providers", {})
    providers["stateloom"] = {
        "baseUrl": base_url,
        "apiKey": "sk-stateloom",
        "api": "openai-completions",
        "models": openclaw_models,
    }

    # agents.defaults.model.primary
    agents = config.setdefault("agents", {})
    defaults = agents.setdefault("defaults", {})
    model_cfg = defaults.setdefault("model", {})
    model_cfg["primary"] = f"stateloom/{model_ids[0]}"

    # agents.defaults.models — register all models
    defaults_models = defaults.setdefault("models", {})
    for m in openclaw_models:
        defaults_models[f"stateloom/{m['id']}"] = {}

    return config


_PROVIDER_ENV_KEYS: dict[str, tuple[str, str]] = {
    "gpt-": ("OPENAI_API_KEY", "OpenAI"),
    "o1": ("OPENAI_API_KEY", "OpenAI"),
    "o3": ("OPENAI_API_KEY", "OpenAI"),
    "claude-": ("ANTHROPIC_API_KEY", "Anthropic"),
    "gemini-": ("GOOGLE_API_KEY", "Google"),
    "mistral-": ("MISTRAL_API_KEY", "Mistral"),
    "codestral-": ("MISTRAL_API_KEY", "Mistral"),
    "pixtral-": ("MISTRAL_API_KEY", "Mistral"),
    "open-mistral-": ("MISTRAL_API_KEY", "Mistral"),
    "command-": ("COHERE_API_KEY", "Cohere"),
}


def _ensure_api_key(model: str) -> None:
    """Prompt for the provider API key if not already in the environment."""
    import os

    env_var: str | None = None
    provider_name: str | None = None
    for prefix, (var, name) in _PROVIDER_ENV_KEYS.items():
        if model.startswith(prefix):
            env_var, provider_name = var, name
            break

    if env_var is None:
        return  # Unknown provider — skip prompting

    if os.environ.get(env_var):
        click.echo(f"Using {provider_name} key from ${env_var}")
        return

    key = click.prompt(f"Enter your {provider_name} API key", hide_input=True)
    os.environ[env_var] = key
    click.echo(f"Tip: export {env_var}=... to skip this prompt next time")


def _get_openclaw_binary() -> str | None:
    """Return path to openclaw binary, checking system PATH then managed install."""
    system = shutil.which("openclaw")
    if system is not None:
        return system
    managed = _MANAGED_NODE_DIR / "bin" / "openclaw"
    if managed.is_file():
        return str(managed)
    return None


def _install_openclaw() -> str:
    """Install OpenClaw into a managed Node.js environment via nodeenv + npm."""
    try:
        import nodeenv  # type: ignore[import-untyped]  # noqa: F401
    except ImportError:
        click.echo("Error: OpenClaw is not installed and the 'openclaw' extra is missing.")
        click.echo("")
        click.echo("Either install the extra for automatic setup:")
        click.echo("  pip install 'stateloom[openclaw]'")
        click.echo("")
        click.echo("Or install OpenClaw manually:")
        click.echo("  npm install -g @anthropic-ai/openclaw")
        raise SystemExit(1)

    click.echo("Installing OpenClaw (one-time setup)...")

    # Create isolated Node.js environment
    click.echo("  Setting up Node.js environment...")
    result = subprocess.run(
        [sys.executable, "-m", "nodeenv", "--prebuilt", str(_MANAGED_NODE_DIR)],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        click.echo(f"Error: Failed to create Node.js environment.\n{result.stderr}")
        raise SystemExit(1)

    # Install OpenClaw globally within the managed environment
    npm_bin = str(_MANAGED_NODE_DIR / "bin" / "npm")
    click.echo("  Installing @anthropic-ai/openclaw...")
    result = subprocess.run(
        [npm_bin, "install", "-g", "@anthropic-ai/openclaw"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        click.echo(f"Error: npm install failed.\n{result.stderr}")
        raise SystemExit(1)

    openclaw_bin = str(_MANAGED_NODE_DIR / "bin" / "openclaw")
    if not Path(openclaw_bin).is_file():
        click.echo("Error: OpenClaw binary not found after installation.")
        raise SystemExit(1)

    click.echo("  OpenClaw installed successfully.")
    return openclaw_bin


def _check_stateloom_health(host: str, port: int, timeout: float = 1.0) -> bool:
    """Check if StateLoom is responding on the given host:port."""
    import urllib.request

    try:
        url = f"http://{host}:{port}/api/health"
        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return bool(resp.status == 200)
    except Exception:
        return False


def _wait_for_health(host: str, port: int, max_wait: float = 10.0) -> bool:
    """Poll health endpoint until it responds or timeout."""
    deadline = time.monotonic() + max_wait
    while time.monotonic() < deadline:
        if _check_stateloom_health(host, port):
            return True
        time.sleep(0.3)
    return False


@click.group()
def openclaw() -> None:
    """Manage OpenClaw integration with StateLoom."""


@openclaw.command()
@click.option("--host", default="127.0.0.1", show_default=True, help="StateLoom host")
@click.option("--port", default=4782, type=int, show_default=True, help="StateLoom port")
@click.option("--model", required=True, help="Model(s) for OpenClaw (comma-separated)")
@click.option("--no-auth", is_flag=True, help="Don't require virtual keys (solo dev mode)")
@click.option("--verbose", is_flag=True, help="Enable console output for each LLM call")
def launch(host: str, port: int, model: str, no_auth: bool, verbose: bool) -> None:
    """Start StateLoom + launch OpenClaw.

    \b
    Automatically configures OpenClaw, prompts for provider API keys
    if not already in the environment, starts StateLoom, and launches
    OpenClaw.

    \b
    Examples:
        stateloom openclaw launch --model gemini-2.5-flash
        stateloom openclaw launch --model gemini-2.5-flash,gpt-4o,claude-haiku-4-5
        stateloom openclaw launch --model claude-haiku-4-5-20251001 --no-auth
    """
    # Parse comma-separated model list
    model_ids = [m.strip() for m in model.split(",") if m.strip()]
    if not model_ids:
        click.echo("Error: --model must specify at least one model.")
        raise SystemExit(1)

    # Validate all models
    known = _get_known_model_ids()
    for m in model_ids:
        if m not in known:
            click.echo(f"Error: Unknown model '{m}'.")
            click.echo(f"Available models: {', '.join(sorted(known))}")
            raise SystemExit(1)

    # Ensure API keys are available (prompt if missing), deduplicate by provider
    seen_env_vars: set[str] = set()
    for m in model_ids:
        for prefix, (var, _name) in _PROVIDER_ENV_KEYS.items():
            if m.startswith(prefix):
                if var not in seen_env_vars:
                    seen_env_vars.add(var)
                    _ensure_api_key(m)
                break

    # Check if StateLoom is already running (pidfile or health check)
    existing_pid = _read_pid()
    if existing_pid is not None and _check_stateloom_health(host, port):
        click.echo(f"StateLoom already running on http://{host}:{port} (PID {existing_pid})")
    else:
        click.echo(f"Starting StateLoom on http://{host}:{port} ...")

        # Spawn detached `stateloom serve` process (same pattern as `stateloom start`)
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

        _PID_DIR.mkdir(parents=True, exist_ok=True)
        log_fh = open(_LOG_FILE, "a")  # noqa: SIM115

        proc = subprocess.Popen(
            cmd,
            stdout=log_fh,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        _write_pid(proc.pid)

        if not _wait_for_health(host, port, max_wait=10.0):
            click.echo("Error: StateLoom failed to start within 10 seconds.")
            click.echo(f"Check logs: {_LOG_FILE.resolve()}")
            raise SystemExit(1)
        click.echo(f"StateLoom started (PID {proc.pid}).")

    # Configure OpenClaw
    config = _read_openclaw_config()
    _backup_openclaw_config()
    config = _apply_stateloom_config(config, host, port, model_ids)
    _write_openclaw_config(config)

    click.echo("")
    click.echo(f"  Dashboard:     http://{host}:{port}")
    click.echo(f"  Primary model: stateloom/{model_ids[0]}")
    if len(model_ids) > 1:
        others = ", ".join(f"stateloom/{m}" for m in model_ids[1:])
        click.echo(f"  Also available: {others}")
    click.echo("")
    click.echo("Now run OpenClaw in this or another terminal:")
    click.echo("")
    click.echo("  openclaw gateway start")
    click.echo("")
    click.echo("Run 'stateloom stop' to shut down StateLoom.")
