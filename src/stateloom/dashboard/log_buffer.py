"""In-memory circular log buffer for debug mode dashboard streaming."""

from __future__ import annotations

import collections
import logging
import threading
from typing import Any

_MAX_BUFFER = 2000

# Loggers whose messages are excluded from the buffer (internal plumbing noise)
_EXCLUDED_LOGGERS = frozenset(
    {
        "stateloom.dashboard.ws",
    }
)


class LogBuffer(logging.Handler):
    """In-memory circular buffer that captures log records."""

    def __init__(self, maxlen: int = _MAX_BUFFER) -> None:
        super().__init__()
        self._buffer: collections.deque[dict[str, Any]] = collections.deque(maxlen=maxlen)
        self._lock = threading.Lock()
        self._listeners: list[Any] = []  # asyncio.Queue refs for WS streaming

    def emit(self, record: logging.LogRecord) -> None:
        # Skip internal plumbing loggers to avoid noise
        if record.name in _EXCLUDED_LOGGERS:
            return
        entry = {
            "timestamp": record.created,
            "level": record.levelname,
            "logger": record.name,
            "message": self.format(record),
            "module": record.module,
            "lineno": record.lineno,
        }
        with self._lock:
            self._buffer.append(entry)
        # Notify WS listeners (fire-and-forget)
        for q in list(self._listeners):
            try:
                q.put_nowait(entry)
            except Exception:
                pass

    def get_logs(self, limit: int = 200, level: str | None = None) -> list[dict]:
        with self._lock:
            entries = list(self._buffer)
        if level:
            entries = [e for e in entries if e["level"] == level.upper()]
        return entries[-limit:]

    def clear(self) -> None:
        with self._lock:
            self._buffer.clear()

    def subscribe(self, queue: Any) -> None:
        self._listeners.append(queue)

    def unsubscribe(self, queue: Any) -> None:
        try:
            self._listeners.remove(queue)
        except ValueError:
            pass


# Module-level singleton (set by Gate when debug=True)
_log_buffer: LogBuffer | None = None


def get_log_buffer() -> LogBuffer | None:
    return _log_buffer


def install_log_buffer() -> LogBuffer:
    """Install the log buffer handler on the root 'stateloom' logger."""
    global _log_buffer
    buf = LogBuffer()
    buf.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s — %(message)s"))
    logging.getLogger("stateloom").addHandler(buf)
    _log_buffer = buf
    return buf
