"""Tests for OllamaManager — download, start, stop, health-check."""

from __future__ import annotations

import os
import signal
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, call, mock_open, patch

import httpx
import pytest

from stateloom.local.manager import (
    DEFAULT_PORT,
    OllamaManager,
    _get_download_url,
)

# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------


class TestPaths:
    """Verify default path constants."""

    def test_default_port(self):
        assert DEFAULT_PORT == 11435

    def test_default_dir(self):
        mgr = OllamaManager()
        assert str(mgr._dir).endswith(".stateloom/ollama")

    def test_bin_path(self):
        mgr = OllamaManager()
        assert str(mgr._bin).endswith(".stateloom/ollama/bin/ollama")

    def test_pid_path(self):
        mgr = OllamaManager()
        assert str(mgr._pid_path).endswith("ollama.pid")

    def test_log_path(self):
        mgr = OllamaManager()
        assert str(mgr._log_path).endswith("ollama.log")

    def test_version_path(self):
        mgr = OllamaManager()
        assert str(mgr._version_path).endswith("version")

    def test_custom_dir(self, tmp_path):
        mgr = OllamaManager(ollama_dir=tmp_path / "custom")
        assert mgr._dir == tmp_path / "custom"
        assert mgr._bin == tmp_path / "custom" / "bin" / "ollama"


# ---------------------------------------------------------------------------
# Download URL
# ---------------------------------------------------------------------------


class TestDownloadUrl:
    """Test platform-specific download URL generation."""

    def test_macos(self):
        with patch("stateloom.local.manager.platform") as mock_platform:
            mock_platform.system.return_value = "Darwin"
            mock_platform.machine.return_value = "arm64"
            url = _get_download_url("v0.9.0")
        assert url == "https://github.com/ollama/ollama/releases/download/v0.9.0/ollama-darwin.tgz"

    def test_linux_amd64(self):
        with patch("stateloom.local.manager.platform") as mock_platform:
            mock_platform.system.return_value = "Linux"
            mock_platform.machine.return_value = "x86_64"
            url = _get_download_url("v0.9.0")
        assert url == (
            "https://github.com/ollama/ollama/releases/download/v0.9.0/ollama-linux-amd64.tar.zst"
        )

    def test_linux_arm64(self):
        with patch("stateloom.local.manager.platform") as mock_platform:
            mock_platform.system.return_value = "Linux"
            mock_platform.machine.return_value = "aarch64"
            url = _get_download_url("v0.9.0")
        assert url == (
            "https://github.com/ollama/ollama/releases/download/v0.9.0/ollama-linux-arm64.tar.zst"
        )

    def test_linux_amd64_alias(self):
        with patch("stateloom.local.manager.platform") as mock_platform:
            mock_platform.system.return_value = "Linux"
            mock_platform.machine.return_value = "amd64"
            url = _get_download_url("v0.9.0")
        assert "linux-amd64" in url

    def test_linux_arm64_alias(self):
        with patch("stateloom.local.manager.platform") as mock_platform:
            mock_platform.system.return_value = "Linux"
            mock_platform.machine.return_value = "arm64"
            url = _get_download_url("v0.9.0")
        assert "linux-arm64" in url

    def test_unsupported_platform(self):
        with patch("stateloom.local.manager.platform") as mock_platform:
            mock_platform.system.return_value = "Windows"
            mock_platform.machine.return_value = "AMD64"
            with pytest.raises(RuntimeError, match="Unsupported platform"):
                _get_download_url("v0.9.0")

    def test_unsupported_linux_arch(self):
        with patch("stateloom.local.manager.platform") as mock_platform:
            mock_platform.system.return_value = "Linux"
            mock_platform.machine.return_value = "riscv64"
            with pytest.raises(RuntimeError, match="Unsupported Linux architecture"):
                _get_download_url("v0.9.0")


# ---------------------------------------------------------------------------
# Install
# ---------------------------------------------------------------------------


class TestInstall:
    """Test binary download and installation."""

    def test_install_creates_binary(self, tmp_path):
        mgr = OllamaManager(ollama_dir=tmp_path / "ollama")

        with (
            patch.object(mgr, "_resolve_version", return_value="v0.9.0"),
            patch(
                "stateloom.local.manager._get_download_url",
                return_value="https://example.com/ollama-darwin",
            ),
            patch.object(mgr, "_download_and_extract") as mock_dl,
        ):
            # Create the file so chmod works
            (tmp_path / "ollama" / "bin").mkdir(parents=True, exist_ok=True)
            (tmp_path / "ollama" / "bin" / "ollama").write_bytes(b"fake-binary")

            path, version = mgr.install(version="v0.9.0")

        assert path == tmp_path / "ollama" / "bin" / "ollama"
        assert version == "v0.9.0"
        mock_dl.assert_called_once()
        # Version file written
        assert (tmp_path / "ollama" / "version").read_text() == "v0.9.0"

    def test_install_sets_executable(self, tmp_path):
        mgr = OllamaManager(ollama_dir=tmp_path / "ollama")

        with (
            patch.object(mgr, "_resolve_version", return_value="v0.9.0"),
            patch(
                "stateloom.local.manager._get_download_url",
                return_value="https://example.com/ollama-darwin",
            ),
            patch.object(mgr, "_download_and_extract"),
        ):
            (tmp_path / "ollama" / "bin").mkdir(parents=True, exist_ok=True)
            bin_path = tmp_path / "ollama" / "bin" / "ollama"
            bin_path.write_bytes(b"fake-binary")
            # Remove executable bit
            bin_path.chmod(0o644)

            mgr.install(version="v0.9.0")

        assert os.access(bin_path, os.X_OK)

    def test_install_with_progress(self, tmp_path):
        mgr = OllamaManager(ollama_dir=tmp_path / "ollama")
        progress_calls = []

        with (
            patch.object(mgr, "_resolve_version", return_value="v0.9.0"),
            patch(
                "stateloom.local.manager._get_download_url",
                return_value="https://example.com/ollama-darwin",
            ),
            patch.object(mgr, "_download_and_extract") as mock_dl,
        ):
            (tmp_path / "ollama" / "bin").mkdir(parents=True, exist_ok=True)
            (tmp_path / "ollama" / "bin" / "ollama").write_bytes(b"fake-binary")

            mgr.install(version="v0.9.0", progress=lambda d, t: progress_calls.append((d, t)))

        # Progress callback was passed through
        assert mock_dl.call_args is not None
        assert mock_dl.call_args.kwargs.get("progress") is not None

    def test_is_installed_false(self, tmp_path):
        mgr = OllamaManager(ollama_dir=tmp_path / "ollama")
        assert mgr.is_installed() is False

    def test_is_installed_true(self, tmp_path):
        mgr = OllamaManager(ollama_dir=tmp_path / "ollama")
        (tmp_path / "ollama" / "bin").mkdir(parents=True)
        (tmp_path / "ollama" / "bin" / "ollama").write_bytes(b"binary")
        assert mgr.is_installed() is True

    def test_installed_version_none(self, tmp_path):
        mgr = OllamaManager(ollama_dir=tmp_path / "ollama")
        assert mgr.installed_version() is None

    def test_installed_version_returns_tag(self, tmp_path):
        mgr = OllamaManager(ollama_dir=tmp_path / "ollama")
        (tmp_path / "ollama").mkdir(parents=True)
        (tmp_path / "ollama" / "version").write_text("v0.9.0")
        assert mgr.installed_version() == "v0.9.0"


# ---------------------------------------------------------------------------
# Resolve version
# ---------------------------------------------------------------------------


class TestResolveVersion:
    """Test version resolution from GitHub releases."""

    def test_explicit_version(self, tmp_path):
        mgr = OllamaManager(ollama_dir=tmp_path / "ollama")
        assert mgr._resolve_version("v0.9.0") == "v0.9.0"

    def test_latest_via_redirect(self, tmp_path):
        mgr = OllamaManager(ollama_dir=tmp_path / "ollama")

        mock_resp = MagicMock()
        mock_resp.status_code = 302
        mock_resp.headers = {
            "location": "https://github.com/ollama/ollama/releases/tag/v0.9.0",
        }

        with patch("stateloom.local.manager.httpx.Client") as MockClient:
            mock_client = MagicMock()
            MockClient.return_value.__enter__ = MagicMock(return_value=mock_client)
            MockClient.return_value.__exit__ = MagicMock(return_value=False)
            mock_client.head.return_value = mock_resp

            result = mgr._resolve_version("latest")

        assert result == "v0.9.0"

    def test_latest_network_error(self, tmp_path):
        mgr = OllamaManager(ollama_dir=tmp_path / "ollama")

        with patch("stateloom.local.manager.httpx.Client") as MockClient:
            mock_client = MagicMock()
            MockClient.return_value.__enter__ = MagicMock(return_value=mock_client)
            MockClient.return_value.__exit__ = MagicMock(return_value=False)
            mock_client.head.side_effect = httpx.ConnectError("fail")

            with pytest.raises(RuntimeError, match="Failed to resolve"):
                mgr._resolve_version("latest")


# ---------------------------------------------------------------------------
# Start
# ---------------------------------------------------------------------------


class TestStart:
    """Test managed Ollama process start."""

    def test_start_launches_process(self, tmp_path):
        mgr = OllamaManager(ollama_dir=tmp_path / "ollama")
        # Create fake binary
        (tmp_path / "ollama" / "bin").mkdir(parents=True)
        (tmp_path / "ollama" / "bin" / "ollama").write_bytes(b"binary")

        mock_proc = MagicMock()
        mock_proc.pid = 12345

        with (
            patch.object(mgr, "is_running", return_value=False),
            patch("stateloom.local.manager.subprocess.Popen", return_value=mock_proc),
            patch.object(mgr, "_wait_for_health", return_value=True),
            patch("builtins.open", mock_open()),
            patch("stateloom.local.manager.atexit"),
        ):
            pid = mgr.start(port=11435)

        assert pid == 12345
        # PID file written
        assert (tmp_path / "ollama" / "ollama.pid").read_text() == "12345"

    def test_start_already_running_returns_pid(self, tmp_path):
        mgr = OllamaManager(ollama_dir=tmp_path / "ollama")
        (tmp_path / "ollama").mkdir(parents=True)
        (tmp_path / "ollama" / "ollama.pid").write_text("99999")

        with (
            patch.object(mgr, "is_running", return_value=True),
            patch.object(mgr, "_read_pid", return_value=99999),
        ):
            pid = mgr.start(port=11435)

        assert pid == 99999

    def test_start_not_installed_raises(self, tmp_path):
        mgr = OllamaManager(ollama_dir=tmp_path / "ollama")

        with patch.object(mgr, "is_running", return_value=False):
            with pytest.raises(RuntimeError, match="not installed"):
                mgr.start()

    def test_start_health_check_fails_stops(self, tmp_path):
        mgr = OllamaManager(ollama_dir=tmp_path / "ollama")
        (tmp_path / "ollama" / "bin").mkdir(parents=True)
        (tmp_path / "ollama" / "bin" / "ollama").write_bytes(b"binary")

        mock_proc = MagicMock()
        mock_proc.pid = 12345

        with (
            patch.object(mgr, "is_running", return_value=False),
            patch("stateloom.local.manager.subprocess.Popen", return_value=mock_proc),
            patch.object(mgr, "_wait_for_health", return_value=False),
            patch.object(mgr, "stop") as mock_stop,
            patch("builtins.open", mock_open()),
        ):
            with pytest.raises(RuntimeError, match="failed to become healthy"):
                mgr.start(port=11435)

        mock_stop.assert_called_once()

    def test_start_sets_ollama_host_env(self, tmp_path):
        mgr = OllamaManager(ollama_dir=tmp_path / "ollama")
        (tmp_path / "ollama" / "bin").mkdir(parents=True)
        (tmp_path / "ollama" / "bin" / "ollama").write_bytes(b"binary")

        captured_env = {}
        original_popen = subprocess.Popen

        def mock_popen(*args, **kwargs):
            captured_env.update(kwargs.get("env", {}))
            mock = MagicMock()
            mock.pid = 12345
            return mock

        with (
            patch.object(mgr, "is_running", return_value=False),
            patch("stateloom.local.manager.subprocess.Popen", side_effect=mock_popen),
            patch.object(mgr, "_wait_for_health", return_value=True),
            patch("builtins.open", mock_open()),
            patch("stateloom.local.manager.atexit"),
        ):
            mgr.start(port=11435)

        assert captured_env.get("OLLAMA_HOST") == "127.0.0.1:11435"

    def test_start_custom_port(self, tmp_path):
        mgr = OllamaManager(ollama_dir=tmp_path / "ollama")
        (tmp_path / "ollama" / "bin").mkdir(parents=True)
        (tmp_path / "ollama" / "bin" / "ollama").write_bytes(b"binary")

        captured_env = {}

        def mock_popen(*args, **kwargs):
            captured_env.update(kwargs.get("env", {}))
            mock = MagicMock()
            mock.pid = 12345
            return mock

        with (
            patch.object(mgr, "is_running", return_value=False),
            patch("stateloom.local.manager.subprocess.Popen", side_effect=mock_popen),
            patch.object(mgr, "_wait_for_health", return_value=True),
            patch("builtins.open", mock_open()),
            patch("stateloom.local.manager.atexit"),
        ):
            mgr.start(port=22222)

        assert captured_env.get("OLLAMA_HOST") == "127.0.0.1:22222"


# ---------------------------------------------------------------------------
# Stop
# ---------------------------------------------------------------------------


class TestStop:
    """Test managed Ollama process stop."""

    def test_stop_sends_sigterm(self, tmp_path):
        mgr = OllamaManager(ollama_dir=tmp_path / "ollama")
        (tmp_path / "ollama").mkdir(parents=True)
        (tmp_path / "ollama" / "ollama.pid").write_text("12345")

        kill_calls = []

        def mock_kill(pid, sig):
            kill_calls.append((pid, sig))
            if sig == 0 and len([c for c in kill_calls if c[1] == 0]) > 1:
                raise OSError("process gone")

        with patch("stateloom.local.manager.os.kill", side_effect=mock_kill):
            mgr.stop()

        assert any(sig == signal.SIGTERM for _, sig in kill_calls)
        # PID file removed
        assert not (tmp_path / "ollama" / "ollama.pid").exists()

    def test_stop_no_pid_file(self, tmp_path):
        mgr = OllamaManager(ollama_dir=tmp_path / "ollama")
        # Should not raise
        mgr.stop()

    def test_stop_stale_pid(self, tmp_path):
        mgr = OllamaManager(ollama_dir=tmp_path / "ollama")
        (tmp_path / "ollama").mkdir(parents=True)
        (tmp_path / "ollama" / "ollama.pid").write_text("12345")

        def mock_kill(pid, sig):
            raise OSError("No such process")

        with patch("stateloom.local.manager.os.kill", side_effect=mock_kill):
            mgr.stop()

        assert not (tmp_path / "ollama" / "ollama.pid").exists()


# ---------------------------------------------------------------------------
# is_running
# ---------------------------------------------------------------------------


class TestIsRunning:
    """Test running status detection."""

    def test_not_running_no_pid(self, tmp_path):
        mgr = OllamaManager(ollama_dir=tmp_path / "ollama")
        assert mgr.is_running() is False

    def test_not_running_dead_process(self, tmp_path):
        mgr = OllamaManager(ollama_dir=tmp_path / "ollama")
        (tmp_path / "ollama").mkdir(parents=True)
        (tmp_path / "ollama" / "ollama.pid").write_text("99999")

        def mock_kill(pid, sig):
            raise OSError("No such process")

        with patch("stateloom.local.manager.os.kill", side_effect=mock_kill):
            assert mgr.is_running() is False

        # Stale PID file cleaned up
        assert not (tmp_path / "ollama" / "ollama.pid").exists()

    def test_running_process_alive_http_ok(self, tmp_path):
        mgr = OllamaManager(ollama_dir=tmp_path / "ollama")
        (tmp_path / "ollama").mkdir(parents=True)
        (tmp_path / "ollama" / "ollama.pid").write_text("12345")

        with (
            patch("stateloom.local.manager.os.kill"),  # no exception = process alive
            patch.object(mgr, "_http_health_check", return_value=True),
        ):
            assert mgr.is_running() is True

    def test_running_process_alive_http_fail(self, tmp_path):
        mgr = OllamaManager(ollama_dir=tmp_path / "ollama")
        (tmp_path / "ollama").mkdir(parents=True)
        (tmp_path / "ollama" / "ollama.pid").write_text("12345")

        with (
            patch("stateloom.local.manager.os.kill"),  # process alive
            patch.object(mgr, "_http_health_check", return_value=False),
        ):
            assert mgr.is_running() is False


# ---------------------------------------------------------------------------
# ensure_running
# ---------------------------------------------------------------------------


class TestEnsureRunning:
    """Test the ensure_running convenience method."""

    def test_already_running_noop(self, tmp_path):
        mgr = OllamaManager(ollama_dir=tmp_path / "ollama")

        with (
            patch.object(mgr, "is_running", return_value=True),
            patch.object(mgr, "start") as mock_start,
        ):
            mgr.ensure_running()

        mock_start.assert_not_called()

    def test_not_running_starts(self, tmp_path):
        mgr = OllamaManager(ollama_dir=tmp_path / "ollama")

        with (
            patch.object(mgr, "is_running", return_value=False),
            patch.object(mgr, "start", return_value=12345) as mock_start,
        ):
            mgr.ensure_running(port=11435)

        mock_start.assert_called_once_with(port=11435)

    def test_not_installed_raises(self, tmp_path):
        mgr = OllamaManager(ollama_dir=tmp_path / "ollama")

        with patch.object(mgr, "is_running", return_value=False):
            with pytest.raises(RuntimeError, match="not installed"):
                mgr.ensure_running()


# ---------------------------------------------------------------------------
# ensure_model
# ---------------------------------------------------------------------------


class TestEnsureModel:
    """Test model availability checking and pulling."""

    def test_model_present_noop(self, tmp_path):
        mgr = OllamaManager(ollama_dir=tmp_path / "ollama")

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {
            "models": [{"name": "llama3.2:latest", "size": 2000000000}],
        }

        with (
            patch("stateloom.local.manager.httpx.Client") as MockClient,
            patch.object(mgr, "_pull_model") as mock_pull,
        ):
            mock_client = MagicMock()
            MockClient.return_value.__enter__ = MagicMock(return_value=mock_client)
            MockClient.return_value.__exit__ = MagicMock(return_value=False)
            mock_client.get.return_value = mock_resp

            mgr.ensure_model("llama3.2", port=11435)

        mock_pull.assert_not_called()

    def test_model_missing_pulls(self, tmp_path):
        mgr = OllamaManager(ollama_dir=tmp_path / "ollama")

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {"models": []}

        with (
            patch("stateloom.local.manager.httpx.Client") as MockClient,
            patch.object(mgr, "_pull_model") as mock_pull,
        ):
            mock_client = MagicMock()
            MockClient.return_value.__enter__ = MagicMock(return_value=mock_client)
            MockClient.return_value.__exit__ = MagicMock(return_value=False)
            mock_client.get.return_value = mock_resp

            mgr.ensure_model("llama3.2", port=11435)

        mock_pull.assert_called_once_with("llama3.2", port=11435, progress=None)

    def test_model_exact_name_match(self, tmp_path):
        mgr = OllamaManager(ollama_dir=tmp_path / "ollama")

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {
            "models": [{"name": "llama3.2", "size": 2000000000}],
        }

        with (
            patch("stateloom.local.manager.httpx.Client") as MockClient,
            patch.object(mgr, "_pull_model") as mock_pull,
        ):
            mock_client = MagicMock()
            MockClient.return_value.__enter__ = MagicMock(return_value=mock_client)
            MockClient.return_value.__exit__ = MagicMock(return_value=False)
            mock_client.get.return_value = mock_resp

            mgr.ensure_model("llama3.2", port=11435)

        mock_pull.assert_not_called()

    def test_model_check_failure_still_pulls(self, tmp_path):
        mgr = OllamaManager(ollama_dir=tmp_path / "ollama")

        with (
            patch("stateloom.local.manager.httpx.Client") as MockClient,
            patch.object(mgr, "_pull_model") as mock_pull,
        ):
            mock_client = MagicMock()
            MockClient.return_value.__enter__ = MagicMock(return_value=mock_client)
            MockClient.return_value.__exit__ = MagicMock(return_value=False)
            mock_client.get.side_effect = httpx.ConnectError("fail")

            mgr.ensure_model("llama3.2", port=11435)

        mock_pull.assert_called_once()

    def test_ensure_model_with_progress(self, tmp_path):
        mgr = OllamaManager(ollama_dir=tmp_path / "ollama")
        progress_fn = MagicMock()

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {"models": []}

        with (
            patch("stateloom.local.manager.httpx.Client") as MockClient,
            patch.object(mgr, "_pull_model") as mock_pull,
        ):
            mock_client = MagicMock()
            MockClient.return_value.__enter__ = MagicMock(return_value=mock_client)
            MockClient.return_value.__exit__ = MagicMock(return_value=False)
            mock_client.get.return_value = mock_resp

            mgr.ensure_model("llama3.2", port=11435, progress=progress_fn)

        mock_pull.assert_called_once_with("llama3.2", port=11435, progress=progress_fn)


# ---------------------------------------------------------------------------
# HTTP health check
# ---------------------------------------------------------------------------


class TestHttpHealthCheck:
    """Test the HTTP health check helper."""

    def test_health_check_success(self, tmp_path):
        mgr = OllamaManager(ollama_dir=tmp_path / "ollama")

        mock_resp = MagicMock()
        mock_resp.status_code = 200

        with patch("stateloom.local.manager.httpx.Client") as MockClient:
            mock_client = MagicMock()
            MockClient.return_value.__enter__ = MagicMock(return_value=mock_client)
            MockClient.return_value.__exit__ = MagicMock(return_value=False)
            mock_client.get.return_value = mock_resp

            assert mgr._http_health_check(11435) is True

    def test_health_check_connection_error(self, tmp_path):
        mgr = OllamaManager(ollama_dir=tmp_path / "ollama")

        with patch("stateloom.local.manager.httpx.Client") as MockClient:
            mock_client = MagicMock()
            MockClient.return_value.__enter__ = MagicMock(return_value=mock_client)
            MockClient.return_value.__exit__ = MagicMock(return_value=False)
            mock_client.get.side_effect = httpx.ConnectError("refused")

            assert mgr._http_health_check(11435) is False

    def test_health_check_non_200(self, tmp_path):
        mgr = OllamaManager(ollama_dir=tmp_path / "ollama")

        mock_resp = MagicMock()
        mock_resp.status_code = 500

        with patch("stateloom.local.manager.httpx.Client") as MockClient:
            mock_client = MagicMock()
            MockClient.return_value.__enter__ = MagicMock(return_value=mock_client)
            MockClient.return_value.__exit__ = MagicMock(return_value=False)
            mock_client.get.return_value = mock_resp

            assert mgr._http_health_check(11435) is False


# ---------------------------------------------------------------------------
# Wait for health
# ---------------------------------------------------------------------------


class TestWaitForHealth:
    """Test the health-check polling loop."""

    def test_wait_succeeds_immediately(self, tmp_path):
        mgr = OllamaManager(ollama_dir=tmp_path / "ollama")

        with patch.object(mgr, "_http_health_check", return_value=True):
            assert mgr._wait_for_health(11435, timeout=1.0) is True

    def test_wait_succeeds_after_delay(self, tmp_path):
        mgr = OllamaManager(ollama_dir=tmp_path / "ollama")
        call_count = [0]

        def health_check(port):
            call_count[0] += 1
            return call_count[0] >= 3

        with (
            patch.object(mgr, "_http_health_check", side_effect=health_check),
            patch("stateloom.local.manager.time.sleep"),
        ):
            assert mgr._wait_for_health(11435, timeout=10.0) is True

    def test_wait_times_out(self, tmp_path):
        mgr = OllamaManager(ollama_dir=tmp_path / "ollama")

        with (
            patch.object(mgr, "_http_health_check", return_value=False),
            patch("stateloom.local.manager.time.monotonic", side_effect=[0.0, 0.5, 1.0, 11.0]),
            patch("stateloom.local.manager.time.sleep"),
        ):
            assert mgr._wait_for_health(11435, timeout=10.0) is False


# ---------------------------------------------------------------------------
# Read PID
# ---------------------------------------------------------------------------


class TestReadPid:
    """Test PID file reading."""

    def test_no_pid_file(self, tmp_path):
        mgr = OllamaManager(ollama_dir=tmp_path / "ollama")
        assert mgr._read_pid() is None

    def test_valid_pid(self, tmp_path):
        mgr = OllamaManager(ollama_dir=tmp_path / "ollama")
        (tmp_path / "ollama").mkdir(parents=True)
        (tmp_path / "ollama" / "ollama.pid").write_text("12345")

        with patch("stateloom.local.manager.os.kill"):  # process alive
            assert mgr._read_pid() == 12345

    def test_stale_pid_cleaned(self, tmp_path):
        mgr = OllamaManager(ollama_dir=tmp_path / "ollama")
        (tmp_path / "ollama").mkdir(parents=True)
        pid_file = tmp_path / "ollama" / "ollama.pid"
        pid_file.write_text("12345")

        with patch("stateloom.local.manager.os.kill", side_effect=OSError("gone")):
            assert mgr._read_pid() is None

        assert not pid_file.exists()

    def test_invalid_pid_content(self, tmp_path):
        mgr = OllamaManager(ollama_dir=tmp_path / "ollama")
        (tmp_path / "ollama").mkdir(parents=True)
        (tmp_path / "ollama" / "ollama.pid").write_text("not-a-number")
        assert mgr._read_pid() is None


# ---------------------------------------------------------------------------
# Gate integration
# ---------------------------------------------------------------------------


class TestGateIntegration:
    """Test that Gate starts and stops managed Ollama correctly."""

    def test_gate_shutdown_stops_manager(self):
        """Gate.shutdown() should call _ollama_manager.stop()."""
        from stateloom.core.config import StateLoomConfig
        from stateloom.gate import Gate

        config = StateLoomConfig(
            auto_patch=False,
            dashboard=False,
            store_backend="memory",
            console_output=False,
        )

        gate = Gate(config)
        mock_manager = MagicMock()
        gate._ollama_manager = mock_manager

        gate.shutdown()

        mock_manager.stop.assert_called_once()

    def test_gate_init_with_managed_ollama(self):
        """Gate.__init__() with ollama_managed=True should start the manager."""
        from stateloom.core.config import StateLoomConfig

        config = StateLoomConfig(
            auto_patch=False,
            dashboard=False,
            store_backend="memory",
            console_output=False,
            ollama_managed=True,
            ollama_managed_port=11435,
            local_model_default="llama3.2",
            ollama_auto_pull=True,
        )

        mock_mgr = MagicMock()

        with patch("stateloom.local.manager.OllamaManager", return_value=mock_mgr):
            from stateloom.gate import Gate

            gate = Gate(config)

        mock_mgr.ensure_running.assert_called_once_with(port=11435)
        mock_mgr.ensure_model.assert_called_once_with("llama3.2", port=11435)
        assert config.local_model_host == "http://127.0.0.1:11435"
        assert config.local_model_enabled is True

        # Cleanup
        gate._ollama_manager = None
        gate.shutdown()

    def test_gate_init_managed_no_auto_pull(self):
        """Gate.__init__() with ollama_auto_pull=False skips model pull."""
        from stateloom.core.config import StateLoomConfig

        config = StateLoomConfig(
            auto_patch=False,
            dashboard=False,
            store_backend="memory",
            console_output=False,
            ollama_managed=True,
            ollama_managed_port=11435,
            local_model_default="llama3.2",
            ollama_auto_pull=False,
        )

        mock_mgr = MagicMock()

        with patch("stateloom.local.manager.OllamaManager", return_value=mock_mgr):
            from stateloom.gate import Gate

            gate = Gate(config)

        mock_mgr.ensure_running.assert_called_once()
        mock_mgr.ensure_model.assert_not_called()

        gate._ollama_manager = None
        gate.shutdown()

    def test_gate_init_managed_no_default_model(self):
        """Gate.__init__() with no local_model_default skips auto-pull."""
        from stateloom.core.config import StateLoomConfig

        config = StateLoomConfig(
            auto_patch=False,
            dashboard=False,
            store_backend="memory",
            console_output=False,
            ollama_managed=True,
            ollama_managed_port=11435,
            ollama_auto_pull=True,
        )

        mock_mgr = MagicMock()

        with patch("stateloom.local.manager.OllamaManager", return_value=mock_mgr):
            from stateloom.gate import Gate

            gate = Gate(config)

        mock_mgr.ensure_running.assert_called_once()
        mock_mgr.ensure_model.assert_not_called()

        gate._ollama_manager = None
        gate.shutdown()


# ---------------------------------------------------------------------------
# Atexit cleanup
# ---------------------------------------------------------------------------


class TestAtexitCleanup:
    """Test the atexit handler."""

    def test_atexit_calls_stop(self, tmp_path):
        mgr = OllamaManager(ollama_dir=tmp_path / "ollama")

        with patch.object(mgr, "stop") as mock_stop:
            mgr._atexit_cleanup()

        mock_stop.assert_called_once()

    def test_atexit_swallows_exceptions(self, tmp_path):
        mgr = OllamaManager(ollama_dir=tmp_path / "ollama")

        with patch.object(mgr, "stop", side_effect=Exception("boom")):
            # Should not raise
            mgr._atexit_cleanup()


# ---------------------------------------------------------------------------
# Download binary
# ---------------------------------------------------------------------------


class TestDownloadAndExtract:
    """Test archive download and extraction."""

    def _make_tgz(self, files: dict[str, bytes]) -> bytes:
        """Create an in-memory .tgz archive with the given files."""
        import io
        import tarfile

        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w:gz") as tf:
            for name, content in files.items():
                info = tarfile.TarInfo(name=name)
                info.size = len(content)
                tf.addfile(info, io.BytesIO(content))
        return buf.getvalue()

    def test_download_extracts_tgz(self, tmp_path):
        mgr = OllamaManager(ollama_dir=tmp_path / "ollama")
        dest_dir = tmp_path / "extracted"
        dest_dir.mkdir()

        archive = self._make_tgz({"ollama": b"fake-binary", "libfoo.dylib": b"lib"})

        mock_resp = MagicMock()
        mock_resp.headers = {"content-length": str(len(archive))}
        mock_resp.raise_for_status = MagicMock()
        mock_resp.iter_bytes.return_value = [archive]

        with patch("stateloom.local.manager.httpx.Client") as MockClient:
            mock_client = MagicMock()
            MockClient.return_value.__enter__ = MagicMock(return_value=mock_client)
            MockClient.return_value.__exit__ = MagicMock(return_value=False)
            mock_client.stream.return_value.__enter__ = MagicMock(return_value=mock_resp)
            mock_client.stream.return_value.__exit__ = MagicMock(return_value=False)

            mgr._download_and_extract("https://example.com/ollama-darwin.tgz", dest_dir)

        assert (dest_dir / "ollama").exists()
        assert (dest_dir / "ollama").read_bytes() == b"fake-binary"
        assert (dest_dir / "libfoo.dylib").exists()

    def test_download_with_progress_callback(self, tmp_path):
        mgr = OllamaManager(ollama_dir=tmp_path / "ollama")
        dest_dir = tmp_path / "extracted"
        dest_dir.mkdir()
        progress_calls = []

        archive = self._make_tgz({"ollama": b"fake-binary"})
        half = len(archive) // 2

        mock_resp = MagicMock()
        mock_resp.headers = {"content-length": str(len(archive))}
        mock_resp.raise_for_status = MagicMock()
        mock_resp.iter_bytes.return_value = [archive[:half], archive[half:]]

        with patch("stateloom.local.manager.httpx.Client") as MockClient:
            mock_client = MagicMock()
            MockClient.return_value.__enter__ = MagicMock(return_value=mock_client)
            MockClient.return_value.__exit__ = MagicMock(return_value=False)
            mock_client.stream.return_value.__enter__ = MagicMock(return_value=mock_resp)
            mock_client.stream.return_value.__exit__ = MagicMock(return_value=False)

            mgr._download_and_extract(
                "https://example.com/ollama-darwin.tgz",
                dest_dir,
                progress=lambda d, t: progress_calls.append((d, t)),
            )

        assert len(progress_calls) == 2
        assert progress_calls[0] == (half, len(archive))
        assert progress_calls[1] == (len(archive), len(archive))


# ---------------------------------------------------------------------------
# Config YAML
# ---------------------------------------------------------------------------


class TestConfigYaml:
    """Test that config YAML section maps correctly."""

    def test_local_managed_from_yaml(self, tmp_path):
        from stateloom.core.config import StateLoomConfig

        yaml_file = tmp_path / "config.yaml"
        yaml_file.write_text(
            "local:\n"
            "  managed: true\n"
            "  managed_port: 22222\n"
            "  auto_pull: false\n"
            "  default: llama3.2\n"
        )
        config = StateLoomConfig.from_yaml(yaml_file)
        assert config.ollama_managed is True
        assert config.ollama_managed_port == 22222
        assert config.ollama_auto_pull is False
        assert config.local_model_default == "llama3.2"
