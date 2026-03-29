"""Tests for the stateloom tail command."""

from __future__ import annotations

import asyncio
import json
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from click.testing import CliRunner

from stateloom.cli import main
from stateloom.cli.tail_command import _format_event, tail


class TestFormatEvent:
    def test_llm_call_event(self):
        data = {
            "event_type": "llm_call",
            "model": "gpt-4o",
            "total_tokens": 1500,
            "cost": 0.0123,
            "latency_ms": 450.0,
            "session_id": "abc123def456",
        }
        result = _format_event(data)
        assert result is not None
        assert "gpt-4o" in result
        assert "1500 tok" in result
        assert "$0.0123" in result
        assert "450ms" in result
        assert "session:" in result

    def test_cache_hit_exact(self):
        data = {
            "event_type": "cache_hit",
            "match_type": "exact",
            "saved_cost": 0.05,
        }
        result = _format_event(data)
        assert result is not None
        assert "CACHE HIT" in result
        assert "exact match" in result

    def test_cache_hit_semantic(self):
        data = {
            "event_type": "cache_hit",
            "match_type": "semantic",
            "similarity_score": 0.95,
            "saved_cost": 0.03,
        }
        result = _format_event(data)
        assert result is not None
        assert "semantic match" in result
        assert "0.950" in result

    def test_pii_blocked(self):
        data = {
            "event_type": "pii_detection",
            "action_taken": "blocked",
            "pii_type": "email",
        }
        result = _format_event(data)
        assert result is not None
        assert "PII BLOCKED" in result
        assert "email" in result

    def test_pii_detected(self):
        data = {
            "event_type": "pii_detection",
            "action_taken": "redacted",
            "pii_type": "ssn",
        }
        result = _format_event(data)
        assert result is not None
        assert "PII DETECTED" in result

    def test_local_routing_success(self):
        data = {
            "event_type": "local_routing",
            "routing_success": True,
            "local_model": "llama3:8b",
            "complexity_score": 0.35,
        }
        result = _format_event(data)
        assert result is not None
        assert "ROUTED LOCAL" in result
        assert "llama3:8b" in result

    def test_local_routing_fallback(self):
        data = {
            "event_type": "local_routing",
            "routing_success": False,
            "local_model": "llama3:8b",
            "complexity_score": 0.8,
        }
        result = _format_event(data)
        assert result is not None
        assert "ROUTE FALLBACK" in result

    def test_kill_switch(self):
        data = {
            "event_type": "kill_switch",
            "reason": "Emergency shutdown",
        }
        result = _format_event(data)
        assert result is not None
        assert "KILL SWITCH" in result

    def test_blast_radius(self):
        data = {
            "event_type": "blast_radius",
            "trigger": "consecutive_failures",
            "count": 5,
            "threshold": 5,
            "action": "pause_session",
        }
        result = _format_event(data)
        assert result is not None
        assert "BLAST RADIUS" in result
        assert "5/5" in result

    def test_budget_enforcement(self):
        data = {
            "event_type": "budget_enforcement",
            "limit": 10.0,
            "spent": 10.5,
        }
        result = _format_event(data)
        assert result is not None
        assert "BUDGET ENFORCED" in result

    def test_tool_call(self):
        data = {
            "event_type": "tool_call",
            "tool_name": "search_web",
            "latency_ms": 120.0,
        }
        result = _format_event(data)
        assert result is not None
        assert "TOOL" in result
        assert "search_web" in result

    def test_checkpoint(self):
        data = {
            "event_type": "checkpoint",
            "label": "phase_1_done",
        }
        result = _format_event(data)
        assert result is not None
        assert "CHECKPOINT" in result
        assert "phase_1_done" in result

    def test_unknown_event_type(self):
        data = {
            "event_type": "some_new_type",
            "session_id": "sess-123",
        }
        result = _format_event(data)
        assert result is not None
        assert "SOME_NEW_TYPE" in result


class TestTailHelp:
    def test_help(self):
        runner = CliRunner()
        result = runner.invoke(main, ["tail", "--help"])
        assert result.exit_code == 0
        assert "--port" in result.output
        assert "--host" in result.output
        assert "--session" in result.output
        assert "--json" in result.output


class _FakeWS:
    """A fake WebSocket that yields preset messages then raises ConnectionClosed."""

    def __init__(self, messages: list[str]):
        self._messages = list(messages)
        self._idx = 0

    async def recv(self):
        if self._idx < len(self._messages):
            msg = self._messages[self._idx]
            self._idx += 1
            return msg
        # Simulate connection closed after all messages
        from websockets.exceptions import ConnectionClosed

        raise ConnectionClosed(None, None)

    async def send(self, data):
        pass


@asynccontextmanager
async def _fake_connect(messages):
    """Return a factory that creates a fake WS async context manager."""

    async def _connect(uri):
        return _FakeWS(messages)

    # We need an async context manager that yields the fake WS
    yield _FakeWS(messages)


class TestTailAsync:
    @pytest.mark.asyncio
    async def test_tail_processes_events(self):
        """_tail processes new_event messages."""
        from stateloom.cli.tail_command import _tail

        event_msg = json.dumps(
            {
                "type": "new_event",
                "data": {
                    "event_type": "llm_call",
                    "model": "gpt-4o",
                    "total_tokens": 100,
                    "cost": 0.001,
                    "latency_ms": 200,
                    "session_id": "test-sess",
                },
            }
        )
        messages = [event_msg]

        ws = _FakeWS(messages)

        @asynccontextmanager
        async def mock_connect(uri):
            yield ws

        with (
            patch("stateloom.cli.tail_command.websockets.connect", side_effect=mock_connect),
            patch("stateloom.cli.tail_command.click.echo") as mock_echo,
        ):
            # ConnectionClosed will be caught by _tail and it will exit cleanly
            await _tail("127.0.0.1", 4782, None, False)

        # "Connected" message + at least one event
        call_args = [str(c) for c in mock_echo.call_args_list]
        assert any("Connected" in s for s in call_args)
        assert any("gpt-4o" in s for s in call_args)

    @pytest.mark.asyncio
    async def test_tail_session_filter(self):
        """_tail filters events by session ID substring."""
        from stateloom.cli.tail_command import _tail

        messages = [
            json.dumps(
                {
                    "type": "new_event",
                    "data": {
                        "event_type": "llm_call",
                        "session_id": "target-sess-123",
                        "model": "gpt-4o",
                        "total_tokens": 0,
                        "cost": 0,
                        "latency_ms": 0,
                    },
                }
            ),
            json.dumps(
                {
                    "type": "new_event",
                    "data": {
                        "event_type": "llm_call",
                        "session_id": "other-sess-456",
                        "model": "claude-3",
                        "total_tokens": 0,
                        "cost": 0,
                        "latency_ms": 0,
                    },
                }
            ),
        ]

        ws = _FakeWS(messages)

        @asynccontextmanager
        async def mock_connect(uri):
            yield ws

        with (
            patch("stateloom.cli.tail_command.websockets.connect", side_effect=mock_connect),
            patch("stateloom.cli.tail_command.click.echo") as mock_echo,
        ):
            await _tail("127.0.0.1", 4782, "target-sess", False)

        call_args = [str(c) for c in mock_echo.call_args_list]
        # Should have gpt-4o (matched) but NOT claude-3 (filtered out)
        assert any("gpt-4o" in s for s in call_args)
        assert not any("claude-3" in s for s in call_args)

    @pytest.mark.asyncio
    async def test_tail_json_mode(self):
        """_tail --json outputs raw JSON."""
        from stateloom.cli.tail_command import _tail

        event_data = {
            "event_type": "llm_call",
            "model": "gpt-4o",
            "session_id": "s1",
            "total_tokens": 100,
            "cost": 0.01,
            "latency_ms": 50,
        }
        messages = [json.dumps({"type": "new_event", "data": event_data})]

        ws = _FakeWS(messages)

        @asynccontextmanager
        async def mock_connect(uri):
            yield ws

        with (
            patch("stateloom.cli.tail_command.websockets.connect", side_effect=mock_connect),
            patch("stateloom.cli.tail_command.click.echo") as mock_echo,
        ):
            await _tail("127.0.0.1", 4782, None, True)

        # Find the JSON output line (not the "Connected" line)
        json_calls = []
        for call in mock_echo.call_args_list:
            arg = call[0][0] if call[0] else ""
            try:
                parsed = json.loads(arg)
                json_calls.append(parsed)
            except (json.JSONDecodeError, TypeError):
                pass

        assert len(json_calls) == 1
        assert json_calls[0]["model"] == "gpt-4o"

    @pytest.mark.asyncio
    async def test_tail_skips_heartbeat(self):
        """_tail ignores heartbeat messages."""
        from stateloom.cli.tail_command import _tail

        messages = [
            json.dumps({"type": "heartbeat"}),
            json.dumps(
                {
                    "type": "new_event",
                    "data": {
                        "event_type": "llm_call",
                        "model": "gpt-4o",
                        "session_id": "s1",
                        "total_tokens": 0,
                        "cost": 0,
                        "latency_ms": 0,
                    },
                }
            ),
        ]

        ws = _FakeWS(messages)

        @asynccontextmanager
        async def mock_connect(uri):
            yield ws

        with (
            patch("stateloom.cli.tail_command.websockets.connect", side_effect=mock_connect),
            patch("stateloom.cli.tail_command.click.echo") as mock_echo,
        ):
            await _tail("127.0.0.1", 4782, None, True)

        # Only "Connected" + one event JSON, no heartbeat output
        json_outputs = []
        for call in mock_echo.call_args_list:
            arg = call[0][0] if call[0] else ""
            try:
                json.loads(arg)
                json_outputs.append(arg)
            except (json.JSONDecodeError, TypeError):
                pass
        assert len(json_outputs) == 1


class TestTailCommand:
    def test_connection_error(self):
        """Connection error exits with error."""
        runner = CliRunner()
        with patch(
            "stateloom.cli.tail_command.asyncio.run",
            side_effect=SystemExit(1),
        ):
            result = runner.invoke(tail, [])
        assert result.exit_code == 1
