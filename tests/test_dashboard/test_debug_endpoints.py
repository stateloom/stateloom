"""Tests for debug mode REST and WebSocket endpoints."""

from __future__ import annotations

import logging
from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from stateloom.dashboard.api import create_api_router


def _make_app(*, debug: bool = False):
    """Build a minimal FastAPI app with the API router for testing."""
    gate = MagicMock()
    gate.store = MagicMock()
    gate.store.list_sessions.return_value = []
    gate.store.get_session.return_value = None
    gate.config = MagicMock()
    gate.config.async_jobs_enabled = False
    gate.config.debug = debug
    gate._feature_registry = MagicMock()
    gate._feature_registry.status.return_value = {"features": {}}

    app = FastAPI()
    router = create_api_router(gate)
    app.include_router(router, prefix="/api/v1")
    return app, gate


class TestDebugStatusEndpoint:
    def test_debug_status_false(self) -> None:
        app, _ = _make_app(debug=False)
        client = TestClient(app)
        resp = client.get("/api/v1/debug")
        assert resp.status_code == 200
        assert resp.json() == {"debug": False}

    def test_debug_status_true(self) -> None:
        app, _ = _make_app(debug=True)
        client = TestClient(app)
        resp = client.get("/api/v1/debug")
        assert resp.status_code == 200
        assert resp.json() == {"debug": True}


class TestLogsEndpoint:
    def test_logs_returns_404_when_not_debug(self) -> None:
        app, _ = _make_app(debug=False)
        client = TestClient(app)
        resp = client.get("/api/v1/logs")
        assert resp.status_code == 404
        assert "Debug mode is not enabled" in resp.json()["detail"]

    def test_logs_returns_entries_in_debug_mode(self) -> None:
        app, _ = _make_app(debug=True)

        # Install a log buffer and emit some records
        from stateloom.dashboard import log_buffer

        old_buf = log_buffer._log_buffer
        try:
            buf = log_buffer.LogBuffer()
            buf.setFormatter(logging.Formatter("%(levelname)s %(name)s — %(message)s"))
            log_buffer._log_buffer = buf

            # Emit a test log record
            record = logging.LogRecord(
                name="stateloom.test",
                level=logging.INFO,
                pathname="test.py",
                lineno=42,
                msg="test log entry",
                args=(),
                exc_info=None,
            )
            buf.emit(record)

            client = TestClient(app)
            resp = client.get("/api/v1/logs")
            assert resp.status_code == 200
            data = resp.json()
            assert data["debug"] is True
            assert len(data["logs"]) == 1
            assert "test log entry" in data["logs"][0]["message"]
        finally:
            log_buffer._log_buffer = old_buf

    def test_logs_limit_param(self) -> None:
        app, _ = _make_app(debug=True)

        from stateloom.dashboard import log_buffer

        old_buf = log_buffer._log_buffer
        try:
            buf = log_buffer.LogBuffer()
            buf.setFormatter(logging.Formatter("%(message)s"))
            log_buffer._log_buffer = buf

            for i in range(10):
                record = logging.LogRecord(
                    name="test",
                    level=logging.INFO,
                    pathname="t.py",
                    lineno=1,
                    msg=f"msg-{i}",
                    args=(),
                    exc_info=None,
                )
                buf.emit(record)

            client = TestClient(app)
            resp = client.get("/api/v1/logs?limit=3")
            assert resp.status_code == 200
            assert len(resp.json()["logs"]) == 3
        finally:
            log_buffer._log_buffer = old_buf

    def test_logs_level_filter(self) -> None:
        app, _ = _make_app(debug=True)

        from stateloom.dashboard import log_buffer

        old_buf = log_buffer._log_buffer
        try:
            buf = log_buffer.LogBuffer()
            buf.setFormatter(logging.Formatter("%(message)s"))
            log_buffer._log_buffer = buf

            for level in (logging.DEBUG, logging.INFO, logging.WARNING):
                record = logging.LogRecord(
                    name="test",
                    level=level,
                    pathname="t.py",
                    lineno=1,
                    msg=f"{logging.getLevelName(level)}-msg",
                    args=(),
                    exc_info=None,
                )
                buf.emit(record)

            client = TestClient(app)
            resp = client.get("/api/v1/logs?level=WARNING")
            assert resp.status_code == 200
            logs = resp.json()["logs"]
            assert len(logs) == 1
            assert logs[0]["level"] == "WARNING"
        finally:
            log_buffer._log_buffer = old_buf


class TestClearLogsEndpoint:
    def test_clear_returns_404_when_not_debug(self) -> None:
        app, _ = _make_app(debug=False)
        client = TestClient(app)
        resp = client.delete("/api/v1/logs")
        assert resp.status_code == 404

    def test_clear_logs(self) -> None:
        app, _ = _make_app(debug=True)

        from stateloom.dashboard import log_buffer

        old_buf = log_buffer._log_buffer
        try:
            buf = log_buffer.LogBuffer()
            buf.setFormatter(logging.Formatter("%(message)s"))
            log_buffer._log_buffer = buf

            record = logging.LogRecord(
                name="test",
                level=logging.INFO,
                pathname="t.py",
                lineno=1,
                msg="to-be-cleared",
                args=(),
                exc_info=None,
            )
            buf.emit(record)
            assert len(buf.get_logs()) == 1

            client = TestClient(app)
            resp = client.delete("/api/v1/logs")
            assert resp.status_code == 200
            assert resp.json()["cleared"] is True
            assert len(buf.get_logs()) == 0
        finally:
            log_buffer._log_buffer = old_buf


class TestWsLogsEndpoint:
    def test_ws_logs_rejects_when_not_debug(self) -> None:
        from stateloom.dashboard.ws import create_log_websocket_route

        gate = MagicMock()
        gate.config = MagicMock()
        gate.config.debug = False

        app = FastAPI()
        handler = create_log_websocket_route(gate)
        app.add_api_websocket_route("/ws/logs", handler)

        client = TestClient(app)
        with pytest.raises(Exception):
            # WebSocket should close with 1008 when debug is disabled
            with client.websocket_connect("/ws/logs"):
                pass
