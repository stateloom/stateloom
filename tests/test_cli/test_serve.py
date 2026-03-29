"""Tests for the StateLoom CLI."""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch

from click.testing import CliRunner

from stateloom.cli import _PID_FILE, _read_pid, _remove_pid, _write_pid, main


class TestCLIHelp:
    def test_cli_help(self):
        """stateloom --help shows commands."""
        runner = CliRunner()
        result = runner.invoke(main, ["--help"])
        assert result.exit_code == 0
        assert "StateLoom" in result.output
        assert "serve" in result.output

    def test_serve_help(self):
        """stateloom serve --help shows options."""
        runner = CliRunner()
        result = runner.invoke(main, ["serve", "--help"])
        assert result.exit_code == 0
        assert "--host" in result.output
        assert "--port" in result.output
        assert "--no-auth" in result.output
        assert "--verbose" in result.output
        assert "4782" in result.output

    def test_start_help(self):
        """stateloom start --help shows options."""
        runner = CliRunner()
        result = runner.invoke(main, ["start", "--help"])
        assert result.exit_code == 0
        assert "--host" in result.output
        assert "--port" in result.output
        assert "--no-auth" in result.output
        assert "background" in result.output.lower()

    def test_stop_help(self):
        """stateloom stop --help shows description."""
        runner = CliRunner()
        result = runner.invoke(main, ["stop", "--help"])
        assert result.exit_code == 0
        assert "Stop" in result.output

    def test_status_help(self):
        """stateloom status --help shows description."""
        runner = CliRunner()
        result = runner.invoke(main, ["status", "--help"])
        assert result.exit_code == 0
        assert "Check" in result.output or "running" in result.output.lower()

    def test_cli_group_no_args(self):
        """stateloom without subcommand shows help."""
        runner = CliRunner()
        result = runner.invoke(main, [])
        assert result.exit_code == 0
        assert "serve" in result.output
        assert "start" in result.output
        assert "stop" in result.output
        assert "status" in result.output


class TestStopNoServer:
    def test_stop_when_not_running(self, tmp_path, monkeypatch):
        """stateloom stop when not running gives clear message."""
        monkeypatch.setattr("stateloom.cli._PID_FILE", tmp_path / "nonexistent.pid")
        runner = CliRunner()
        result = runner.invoke(main, ["stop"])
        assert result.exit_code == 1
        assert "not running" in result.output.lower()


class TestStatusNoServer:
    def test_status_when_not_running(self, tmp_path, monkeypatch):
        """stateloom status when not running gives clear message."""
        monkeypatch.setattr("stateloom.cli._PID_FILE", tmp_path / "nonexistent.pid")
        runner = CliRunner()
        result = runner.invoke(main, ["status"])
        assert result.exit_code == 1
        assert "not running" in result.output.lower()


class TestPidHelpers:
    def test_read_pid_no_file(self, tmp_path, monkeypatch):
        """Returns None when no PID file exists."""
        monkeypatch.setattr("stateloom.cli._PID_FILE", tmp_path / "nope.pid")
        assert _read_pid() is None

    def test_read_pid_stale(self, tmp_path, monkeypatch):
        """Returns None and cleans up when PID belongs to a dead process."""
        pid_file = tmp_path / "server.pid"
        pid_file.write_text("99999999")  # almost certainly not a real PID
        monkeypatch.setattr("stateloom.cli._PID_FILE", pid_file)

        assert _read_pid() is None
        assert not pid_file.exists()

    def test_read_pid_current_process(self, tmp_path, monkeypatch):
        """Returns PID when process is alive (use current process as test)."""
        pid_file = tmp_path / "server.pid"
        pid_file.write_text(str(os.getpid()))
        monkeypatch.setattr("stateloom.cli._PID_FILE", pid_file)

        assert _read_pid() == os.getpid()

    def test_write_and_remove_pid(self, tmp_path, monkeypatch):
        """Write and remove PID file."""
        pid_file = tmp_path / "server.pid"
        monkeypatch.setattr("stateloom.cli._PID_FILE", pid_file)
        monkeypatch.setattr("stateloom.cli._PID_DIR", tmp_path)

        _write_pid(12345)
        assert pid_file.read_text() == "12345"

        _remove_pid()
        assert not pid_file.exists()


class TestStartAlreadyRunning:
    def test_start_already_running(self, tmp_path, monkeypatch):
        """stateloom start refuses when server already running."""
        pid_file = tmp_path / "server.pid"
        pid_file.write_text(str(os.getpid()))
        monkeypatch.setattr("stateloom.cli._PID_FILE", pid_file)

        runner = CliRunner()
        result = runner.invoke(main, ["start"])
        assert result.exit_code == 1
        assert "already running" in result.output.lower()
