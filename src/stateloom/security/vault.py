"""In-memory secret vault — move API keys out of os.environ.

Thread-safe storage for sensitive values. Optionally scrubs keys from
the environment to prevent exfiltration by supply-chain attacks.
"""

from __future__ import annotations

import logging
import os
import threading
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger("stateloom.security")

# Default environment variable names to vault
_DEFAULT_KEYS = [
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "GOOGLE_API_KEY",
    "STATELOOM_LICENSE_KEY",
    "STATELOOM_JWT_SECRET",
]


class SecretVault:
    """In-memory secret vault with optional environ scrubbing."""

    def __init__(self) -> None:
        self._secrets: dict[str, str] = {}
        self._scrubbed: list[str] = []
        self._access_log: list[dict[str, Any]] = []
        self._lock = threading.Lock()
        self._enabled = False

    def configure(
        self,
        enabled: bool,
        scrub_environ: bool = False,
        keys: list[str] | None = None,
    ) -> None:
        """Configure the vault: read env vars, store them, optionally scrub.

        Args:
            enabled: Whether the vault is active.
            scrub_environ: If True, delete vaulted keys from os.environ.
            keys: List of env var names to vault. None = use defaults.
        """
        if not enabled:
            self._enabled = False
            return

        self._enabled = True
        target_keys = keys if keys else list(_DEFAULT_KEYS)

        with self._lock:
            for key in target_keys:
                value = os.environ.get(key)
                if value:
                    self._secrets[key] = value
                    if scrub_environ:
                        del os.environ[key]
                        self._scrubbed.append(key)
                        logger.debug("Vaulted and scrubbed: %s", key)
                    else:
                        logger.debug("Vaulted (env preserved): %s", key)

    def store(self, name: str, value: str) -> None:
        """Store a secret in the vault."""
        with self._lock:
            self._secrets[name] = value
            self._access_log.append(
                {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "action": "store",
                    "key": name,
                }
            )

    def retrieve(self, name: str) -> str | None:
        """Retrieve a secret from the vault. Returns None if not found."""
        with self._lock:
            value = self._secrets.get(name)
            if value is not None:
                self._access_log.append(
                    {
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "action": "retrieve",
                        "key": name,
                    }
                )
            return value

    def has(self, name: str) -> bool:
        """Check if a secret exists in the vault."""
        with self._lock:
            return name in self._secrets

    def list_keys(self) -> list[str]:
        """Return sorted list of vaulted key names (NOT values)."""
        with self._lock:
            return sorted(self._secrets.keys())

    def remove(self, name: str) -> bool:
        """Remove a secret from the vault. Returns True if it existed."""
        with self._lock:
            if name in self._secrets:
                del self._secrets[name]
                self._access_log.append(
                    {
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "action": "remove",
                        "key": name,
                    }
                )
                return True
            return False

    def restore_environ(self) -> list[str]:
        """Restore scrubbed keys back to os.environ. Returns restored key names."""
        restored: list[str] = []
        with self._lock:
            for key in self._scrubbed:
                if key in self._secrets:
                    os.environ[key] = self._secrets[key]
                    restored.append(key)
            self._scrubbed.clear()
        return restored

    def get_status(self) -> dict[str, Any]:
        """Return vault status for dashboard API (no secret values)."""
        with self._lock:
            return {
                "enabled": self._enabled,
                "key_count": len(self._secrets),
                "keys": sorted(self._secrets.keys()),
                "scrubbed": list(self._scrubbed),
                "scrubbed_count": len(self._scrubbed),
                "access_log_size": len(self._access_log),
            }
