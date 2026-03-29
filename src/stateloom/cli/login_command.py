"""CLI login command — authenticate to StateLoom and save credentials."""

from __future__ import annotations

import json
import time
from pathlib import Path

import click

_CREDS_DIR = Path.home() / ".stateloom"
_CREDS_FILE = _CREDS_DIR / "credentials.json"


def _save_credentials(
    server: str,
    email: str,
    access_token: str,
    refresh_token: str,
    expires_in: int,
) -> None:
    """Write credentials to ~/.stateloom/credentials.json."""
    _CREDS_DIR.mkdir(parents=True, exist_ok=True)
    data = {
        "server": server,
        "email": email,
        "access_token": access_token,
        "refresh_token": refresh_token,
        "expires_at": int(time.time()) + expires_in,
    }
    _CREDS_FILE.write_text(json.dumps(data, indent=2))
    _CREDS_FILE.chmod(0o600)


def load_credentials() -> dict | None:
    """Load saved credentials, returning None if missing or unreadable."""
    if not _CREDS_FILE.exists():
        return None
    try:
        return json.loads(_CREDS_FILE.read_text())
    except (json.JSONDecodeError, OSError):
        return None


@click.command()
@click.option("--host", default="127.0.0.1", show_default=True, help="StateLoom host")
@click.option("--port", default=4782, type=int, show_default=True, help="StateLoom port")
def login(host: str, port: int) -> None:
    """Log in to StateLoom dashboard and save credentials."""
    email = click.prompt("Email")
    password = click.prompt("Password", hide_input=True)

    server = f"http://{host}:{port}"
    url = f"{server}/api/v1/auth/login"

    try:
        import urllib.request

        req = urllib.request.Request(
            url,
            data=json.dumps({"email": email, "password": password}).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read())
    except Exception as exc:
        click.echo(f"Login failed: {exc}", err=True)
        raise SystemExit(1)

    access_token = data.get("access_token", "")
    refresh_token = data.get("refresh_token", "")
    if not access_token:
        click.echo("Login failed: no access token in response", err=True)
        raise SystemExit(1)

    _save_credentials(
        server=server,
        email=email,
        access_token=access_token,
        refresh_token=refresh_token,
        expires_in=data.get("expires_in", 900),
    )

    click.echo(f"Logged in as {email}")
    click.echo(f"Credentials saved to {_CREDS_FILE}")


@click.command()
def logout() -> None:
    """Remove saved StateLoom credentials."""
    if _CREDS_FILE.exists():
        _CREDS_FILE.unlink()
        click.echo("Logged out — credentials removed.")
    else:
        click.echo("No credentials found.")
