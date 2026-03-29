"""Tests for the stateloom stats command."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

from click.testing import CliRunner

from stateloom.cli import main
from stateloom.cli.stats_command import stats

_MOCK_STATS = {
    "total_cost": 1.2345,
    "total_tokens": 50000,
    "total_calls": 120,
    "active_sessions": 3,
    "total_cache_hits": 15,
    "total_cache_savings": 0.456,
}

_MOCK_BREAKDOWN = {
    "by_provider": {
        "openai": {"count": 80, "cost": 0.8},
        "anthropic": {"count": 40, "cost": 0.4},
    },
    "by_model": {
        "gpt-4o": {"count": 60, "cost": 0.6, "tokens": 30000},
        "claude-3-5-sonnet": {"count": 40, "cost": 0.4, "tokens": 20000},
    },
}

_MOCK_LATENCY = {
    "percentiles": {
        "p50": 250.0,
        "p90": 800.0,
        "p95": 1200.0,
        "p99": 2500.0,
    }
}


def _mock_fetch(host, port, path):
    if "/api/v1/stats" in path and "observability" not in path:
        return _MOCK_STATS
    if "/api/v1/observability/breakdown" in path:
        return _MOCK_BREAKDOWN
    if "/api/v1/observability/latency" in path:
        return _MOCK_LATENCY
    return {}


class TestStatsHelp:
    def test_help(self):
        runner = CliRunner()
        result = runner.invoke(main, ["stats", "--help"])
        assert result.exit_code == 0
        assert "--port" in result.output
        assert "--host" in result.output
        assert "--window" in result.output
        assert "--json" in result.output

    def test_window_choices(self):
        runner = CliRunner()
        result = runner.invoke(main, ["stats", "--help"])
        assert "1h" in result.output
        assert "24h" in result.output


class TestStatsCommand:
    def test_json_output(self):
        """--json outputs combined JSON."""
        runner = CliRunner()
        with patch("stateloom.cli.stats_command._fetch", side_effect=_mock_fetch):
            result = runner.invoke(stats, ["--json"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert "stats" in data
        assert "breakdown" in data
        assert "latency" in data
        assert data["stats"]["total_calls"] == 120

    def test_formatted_output(self):
        """Default output shows formatted tables."""
        runner = CliRunner()
        with patch("stateloom.cli.stats_command._fetch", side_effect=_mock_fetch):
            result = runner.invoke(stats, [])
        assert result.exit_code == 0
        assert "StateLoom Stats" in result.output
        assert "Overview" in result.output
        assert "$1.2345" in result.output
        assert "50,000" in result.output

    def test_provider_breakdown_shown(self):
        """By Provider section shows provider data."""
        runner = CliRunner()
        with patch("stateloom.cli.stats_command._fetch", side_effect=_mock_fetch):
            result = runner.invoke(stats, [])
        assert "By Provider" in result.output
        assert "openai" in result.output
        assert "anthropic" in result.output

    def test_model_breakdown_shown(self):
        """By Model section shows model data."""
        runner = CliRunner()
        with patch("stateloom.cli.stats_command._fetch", side_effect=_mock_fetch):
            result = runner.invoke(stats, [])
        assert "By Model" in result.output
        assert "gpt-4o" in result.output

    def test_latency_shown(self):
        """Latency percentiles are shown."""
        runner = CliRunner()
        with patch("stateloom.cli.stats_command._fetch", side_effect=_mock_fetch):
            result = runner.invoke(stats, [])
        assert "Latency" in result.output
        assert "p50" in result.output
        assert "p99" in result.output

    def test_connection_error(self):
        """Connection error prints message and exits 1."""
        runner = CliRunner()
        with patch(
            "stateloom.cli.stats_command._fetch",
            side_effect=Exception("Connection refused"),
        ):
            result = runner.invoke(stats, [])
        assert result.exit_code == 1
        assert "Error" in result.output

    def test_custom_host_port(self):
        """--host and --port are used for fetching."""
        runner = CliRunner()
        calls = []

        def capture_fetch(host, port, path):
            calls.append((host, port))
            return _mock_fetch(host, port, path)

        with patch("stateloom.cli.stats_command._fetch", side_effect=capture_fetch):
            result = runner.invoke(stats, ["--host", "10.0.0.1", "--port", "9999", "--json"])
        assert result.exit_code == 0
        assert all(h == "10.0.0.1" and p == 9999 for h, p in calls)

    def test_window_param(self):
        """--window is included in breakdown/latency requests."""
        runner = CliRunner()
        calls = []

        def capture_fetch(host, port, path):
            calls.append(path)
            return _mock_fetch(host, port, path)

        with patch("stateloom.cli.stats_command._fetch", side_effect=capture_fetch):
            runner.invoke(stats, ["--window", "7d", "--json"])
        breakdown_calls = [c for c in calls if "breakdown" in c]
        assert any("window=7d" in c for c in breakdown_calls)

    def test_empty_breakdown(self):
        """Handles empty breakdown gracefully."""
        runner = CliRunner()

        def empty_fetch(host, port, path):
            if "stats" in path and "observability" not in path:
                return _MOCK_STATS
            return {}

        with patch("stateloom.cli.stats_command._fetch", side_effect=empty_fetch):
            result = runner.invoke(stats, [])
        assert result.exit_code == 0
        assert "Overview" in result.output
