"""Tests for the in-memory log buffer (debug mode)."""

from __future__ import annotations

import logging

import pytest

from stateloom.dashboard.log_buffer import LogBuffer


def _make_buffer(maxlen: int = 100) -> LogBuffer:
    buf = LogBuffer(maxlen=maxlen)
    buf.setFormatter(logging.Formatter("%(levelname)s %(name)s — %(message)s"))
    return buf


def _emit(buf: LogBuffer, message: str, level: int = logging.INFO, name: str = "test") -> None:
    record = logging.LogRecord(
        name=name,
        level=level,
        pathname="test.py",
        lineno=1,
        msg=message,
        args=(),
        exc_info=None,
    )
    buf.emit(record)


class TestLogBuffer:
    def test_captures_records(self) -> None:
        buf = _make_buffer()
        _emit(buf, "hello world")
        logs = buf.get_logs()
        assert len(logs) == 1
        assert "hello world" in logs[0]["message"]
        assert logs[0]["level"] == "INFO"
        assert logs[0]["logger"] == "test"

    def test_circular_eviction(self) -> None:
        buf = _make_buffer(maxlen=5)
        for i in range(10):
            _emit(buf, f"msg-{i}")
        logs = buf.get_logs(limit=100)
        assert len(logs) == 5
        # Oldest entries (0-4) should be evicted
        assert "msg-5" in logs[0]["message"]
        assert "msg-9" in logs[4]["message"]

    def test_get_logs_with_limit(self) -> None:
        buf = _make_buffer()
        for i in range(20):
            _emit(buf, f"msg-{i}")
        logs = buf.get_logs(limit=3)
        assert len(logs) == 3
        # Should return the 3 most recent
        assert "msg-17" in logs[0]["message"]
        assert "msg-19" in logs[2]["message"]

    def test_get_logs_with_level_filter(self) -> None:
        buf = _make_buffer()
        _emit(buf, "info-msg", level=logging.INFO)
        _emit(buf, "warn-msg", level=logging.WARNING)
        _emit(buf, "error-msg", level=logging.ERROR)
        _emit(buf, "debug-msg", level=logging.DEBUG)

        info_logs = buf.get_logs(level="INFO")
        assert len(info_logs) == 1
        assert "info-msg" in info_logs[0]["message"]

        warn_logs = buf.get_logs(level="warning")
        assert len(warn_logs) == 1
        assert "warn-msg" in warn_logs[0]["message"]

        error_logs = buf.get_logs(level="ERROR")
        assert len(error_logs) == 1

    def test_clear(self) -> None:
        buf = _make_buffer()
        for i in range(5):
            _emit(buf, f"msg-{i}")
        assert len(buf.get_logs()) == 5
        buf.clear()
        assert len(buf.get_logs()) == 0

    def test_subscribe_unsubscribe(self) -> None:
        import asyncio

        buf = _make_buffer()
        queue: asyncio.Queue = asyncio.Queue(maxsize=10)
        buf.subscribe(queue)
        assert queue in buf._listeners

        _emit(buf, "live-msg")
        # The entry should be pushed to the queue
        assert not queue.empty()
        entry = queue.get_nowait()
        assert "live-msg" in entry["message"]

        buf.unsubscribe(queue)
        assert queue not in buf._listeners

        # After unsubscribe, new messages should not appear
        _emit(buf, "after-unsub")
        assert queue.empty()

    def test_subscribe_full_queue_no_crash(self) -> None:
        import asyncio

        buf = _make_buffer()
        queue: asyncio.Queue = asyncio.Queue(maxsize=1)
        buf.subscribe(queue)

        # Fill the queue
        _emit(buf, "msg-1")
        assert not queue.empty()

        # Emit again — queue is full, should not crash
        _emit(buf, "msg-2")
        # Buffer still has both
        assert len(buf.get_logs()) == 2

        buf.unsubscribe(queue)

    def test_log_entry_fields(self) -> None:
        buf = _make_buffer()
        _emit(buf, "field-check", level=logging.WARNING, name="stateloom.proxy")
        logs = buf.get_logs()
        assert len(logs) == 1
        entry = logs[0]
        assert "timestamp" in entry
        assert entry["level"] == "WARNING"
        assert entry["logger"] == "stateloom.proxy"
        assert "message" in entry
        assert "module" in entry
        assert "lineno" in entry

    def test_excluded_loggers_filtered(self) -> None:
        buf = _make_buffer()
        # WebSocket logger is excluded to reduce noise
        _emit(buf, "ws-noise", name="stateloom.dashboard.ws")
        _emit(buf, "useful-msg", name="stateloom.proxy")
        logs = buf.get_logs()
        assert len(logs) == 1
        assert logs[0]["logger"] == "stateloom.proxy"
