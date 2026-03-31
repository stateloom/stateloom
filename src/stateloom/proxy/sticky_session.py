"""Sticky session manager for the StateLoom proxy.

Maps client fingerprints (IP + user-agent hash) to session IDs so that
multiple HTTP requests from the same client are grouped into a single
StateLoom session.  Stale mappings auto-expire after a configurable idle
timeout (default 30 min).
"""

from __future__ import annotations

import hashlib
import threading
import time
import uuid
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from starlette.requests import Request

_DEFAULT_IDLE_TIMEOUT = 1800.0


class _StickyEntry:
    __slots__ = ("session_id", "last_access")

    def __init__(self, session_id: str) -> None:
        self.session_id = session_id
        self.last_access = time.monotonic()


class StickySessionManager:
    """Maps client fingerprints to session IDs with idle-based expiry."""

    def __init__(self, idle_timeout: float = _DEFAULT_IDLE_TIMEOUT) -> None:
        self._idle_timeout = idle_timeout
        self._mappings: dict[str, _StickyEntry] = {}
        self._lock = threading.Lock()
        self._last_prune = time.monotonic()
        self._prune_interval = 60.0  # seconds between prune sweeps

    def get_session_id(self, request: Request) -> str:
        """Return a stable session ID for the given request.

        Same IP + user-agent → same session ID, as long as the mapping
        hasn't expired due to idle timeout.
        """
        fingerprint = self._compute_fingerprint(request)
        now = time.monotonic()

        with self._lock:
            self._prune_stale(now)

            entry = self._mappings.get(fingerprint)
            if entry is not None and (now - entry.last_access) < self._idle_timeout:
                entry.last_access = now
                return entry.session_id

            # New or expired → create fresh mapping
            session_id = f"sticky-{uuid.uuid4().hex[:12]}"
            self._mappings[fingerprint] = _StickyEntry(session_id)
            return session_id

    def _compute_fingerprint(self, request: Request) -> str:
        """Hash IP + user-agent into a short fingerprint."""
        # Prefer X-Forwarded-For (first entry) for clients behind a reverse proxy
        forwarded = request.headers.get("x-forwarded-for", "")
        if forwarded:
            ip = forwarded.split(",")[0].strip()
        else:
            ip = request.client.host if request.client else "unknown"

        user_agent = request.headers.get("user-agent", "")
        raw = f"{ip}|{user_agent}"
        return hashlib.sha256(raw.encode()).hexdigest()[:16]

    def _prune_stale(self, now: float) -> None:
        """Remove expired entries.  Runs at most every ``_prune_interval`` seconds.

        Must be called while holding ``self._lock``.
        """
        if (now - self._last_prune) < self._prune_interval:
            return

        self._last_prune = now
        stale = [
            fp
            for fp, entry in self._mappings.items()
            if (now - entry.last_access) >= self._idle_timeout
        ]
        for fp in stale:
            del self._mappings[fp]


def resolve_session_id(
    explicit_id: str,
    request: Request,
    sticky: StickySessionManager | None,
) -> str:
    """Choose a session ID with priority: explicit header > sticky > random UUID."""
    if explicit_id:
        return explicit_id
    if sticky is not None:
        return sticky.get_session_id(request)
    return f"proxy-{uuid.uuid4().hex[:12]}"


# Well-known User-Agent patterns → friendly client names.
# Specific SDK/CLI tools that identify the actual caller:
_UA_PATTERNS: list[tuple[str, str]] = [
    ("claude-code/", "Claude Code"),
    ("claude-cli/", "Claude CLI"),
    ("AnthropicSDK/", "Anthropic SDK"),
    ("claudedesktop/", "Claude Desktop"),
    ("codex/", "Codex CLI"),
    ("OpenAI/", "OpenAI SDK"),
    ("GeminiCLI/", "Gemini CLI"),
    ("Gemini-API/", "Gemini SDK"),
    ("google-api-python-client/", "Google API Client"),
]

# Generic HTTP libraries — matched but NOT used as the session name.
# When a generic client is detected, we prefer the provider name
# (e.g. "Gemini" instead of "httpx").
_GENERIC_UA_PATTERNS: list[str] = [
    "python-requests/",
    "node-fetch/",
    "axios/",
    "curl/",
    "httpx/",
]


def derive_session_name(request: Request, model: str, provider: str) -> str:
    """Derive a short session name from the client User-Agent.

    Returns a name like "Claude Code" or "Gemini".
    Specific SDK/CLI clients get their own name; generic HTTP libraries
    fall through to the provider name.
    """
    ua = request.headers.get("user-agent", "")
    ua_lower = ua.lower()

    # Check specific SDK/CLI clients first
    for pattern, name in _UA_PATTERNS:
        if pattern.lower() in ua_lower:
            return name

    # Provider name is more useful than a generic HTTP library name
    if provider:
        return provider.capitalize()

    # Fallback: first product token from UA
    if ua:
        return ua.split("/")[0].split(" ")[0][:30]
    return "Proxy"
