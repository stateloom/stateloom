"""CPython audit hook manager (PEP 578).

Intercepts dangerous interpreter operations at the C level:
file open, socket connect, subprocess, etc.

sys.addaudithook() is global and irreversible. We install once and
configure via mutable state so the hook can be toggled at runtime.
"""

from __future__ import annotations

import logging
import sys
import threading
from collections import deque
from datetime import datetime, timezone
from fnmatch import fnmatch
from typing import Any

from stateloom.core.types import ActionTaken

logger = logging.getLogger("stateloom.security")

_AUDIT_BUFFER_MAXLEN = 100

# Events we monitor (subset of PEP 578 + custom CPython hooks)
_MONITORED_EVENTS = frozenset(
    {
        "open",
        "socket.connect",
        "socket.bind",
        "subprocess.Popen",
        "os.system",
        "shutil.rmtree",
        "os.remove",
        "import",
        "compile",
        "exec",
        "ctypes.dlopen",
    }
)


class AuditHookManager:
    """Manages a CPython audit hook for security monitoring.

    The hook is installed once via sys.addaudithook() and cannot be removed.
    Configuration changes (enable/disable, mode, deny list) take effect
    immediately via shared mutable state.
    """

    def __init__(self) -> None:
        self._enabled = False
        self._mode = "audit"  # "audit" or "enforce"
        self._deny_events: set[str] = set()
        self._allow_paths: list[str] = []
        self._installed = False
        self._lock = threading.Lock()
        self._recent_events: deque[dict[str, Any]] = deque(maxlen=_AUDIT_BUFFER_MAXLEN)
        self._event_count = 0
        self._blocked_count = 0

        # Set externally by Gate._setup_security()
        self._store: Any = None
        self._session_fn: Any = None

    def configure(
        self,
        enabled: bool,
        mode: str = "audit",
        deny_events: list[str] | None = None,
        allow_paths: list[str] | None = None,
    ) -> None:
        """Update hook configuration at runtime."""
        with self._lock:
            self._enabled = enabled
            self._mode = mode
            if deny_events is not None:
                self._deny_events = set(deny_events)
            if allow_paths is not None:
                self._allow_paths = list(allow_paths)

    def install(self) -> bool:
        """Install the audit hook. Returns False if already installed."""
        if self._installed:
            return False
        sys.addaudithook(self._hook)
        self._installed = True
        return True

    def _hook(self, event: str, args: tuple) -> None:
        """The actual audit hook callback invoked by CPython."""
        if not self._enabled:
            return

        if event not in _MONITORED_EVENTS:
            return

        detail = _extract_detail(event, args)

        # Check allow list for file operations
        if event == "open" and detail:
            for pattern in self._allow_paths:
                if fnmatch(detail, pattern):
                    return

        is_denied = event in self._deny_events

        if not is_denied:
            return

        severity = (
            "high" if event in ("subprocess.Popen", "os.system", "ctypes.dlopen") else "medium"
        )
        blocked = self._mode == "enforce"
        action = ActionTaken.BLOCKED if blocked else ActionTaken.LOGGED

        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "audit_event": event,
            "detail": detail,
            "action": action,
            "severity": severity,
            "blocked": blocked,
        }

        with self._lock:
            self._event_count += 1
            if blocked:
                self._blocked_count += 1
            self._recent_events.append(record)

        # Persist event (fail-open)
        try:
            if self._store is not None:
                from stateloom.core.event import SecurityAuditEvent

                session_id = ""
                if self._session_fn is not None:
                    session = self._session_fn()
                    if session is not None:
                        session_id = session.id

                evt = SecurityAuditEvent(
                    session_id=session_id,
                    audit_event=event,
                    action_taken=action,
                    detail=detail[:500],
                    source="audit_hook",
                    severity=severity,
                    blocked=blocked,
                )
                self._store.save_event(evt)
        except Exception:
            pass

        if blocked:
            raise RuntimeError(f"StateLoom security policy blocked: {event} ({detail})")

    def get_status(self) -> dict[str, Any]:
        """Return current status for dashboard API."""
        with self._lock:
            return {
                "installed": self._installed,
                "enabled": self._enabled,
                "mode": self._mode,
                "deny_events": sorted(self._deny_events),
                "allow_paths": list(self._allow_paths),
                "event_count": self._event_count,
                "blocked_count": self._blocked_count,
                "recent_events": list(self._recent_events),
            }


def _extract_detail(event: str, args: tuple) -> str:
    """Extract a human-readable detail string from audit hook args."""
    try:
        if event == "open" and args:
            return str(args[0]) if args[0] else ""
        if event in ("socket.connect", "socket.bind") and args:
            addr = args[1] if len(args) > 1 else args[0]
            if isinstance(addr, tuple) and len(addr) >= 2:
                return f"{addr[0]}:{addr[1]}"
            return str(addr)
        if event == "subprocess.Popen" and args:
            return str(args[0]) if args[0] else ""
        if event == "os.system" and args:
            return str(args[0]) if args[0] else ""
        if event in ("shutil.rmtree", "os.remove") and args:
            return str(args[0]) if args[0] else ""
        if event == "import" and args:
            return str(args[0]) if args[0] else ""
        if event in ("compile", "exec") and args:
            return str(args[0])[:200] if args[0] else ""
        if event == "ctypes.dlopen" and args:
            return str(args[0]) if args[0] else ""
    except Exception:
        pass
    return ""
