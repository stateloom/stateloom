"""Tests for the stateloom doctor command."""

from __future__ import annotations

import importlib
import json
from unittest.mock import MagicMock, patch

from click.testing import CliRunner

from stateloom.cli import main
from stateloom.cli.doctor_command import _run_checks, doctor

# SDK module names checked by doctor
_SDK_MODULES = {
    "openai",
    "anthropic",
    "google.generativeai",
    "google.genai",
    "mistralai",
    "cohere",
    "litellm",
    "sentence_transformers",
    "faiss",
    "prometheus_client",
}

_real_import = importlib.import_module


def _sdk_import_error(name):
    """Raise ImportError only for SDK modules, delegate to real import otherwise."""
    if name in _SDK_MODULES:
        raise ImportError(name)
    return _real_import(name)


class TestDoctorHelp:
    def test_help(self):
        runner = CliRunner()
        result = runner.invoke(main, ["doctor", "--help"])
        assert result.exit_code == 0
        assert "--port" in result.output
        assert "--json" in result.output
        assert "diagnostic" in result.output.lower()


class TestRunChecks:
    def test_returns_list_of_dicts(self):
        """_run_checks returns a list with expected keys."""
        with patch(
            "stateloom.cli.doctor_command.importlib.import_module",
            side_effect=_sdk_import_error,
        ):
            results = _run_checks(port=19999)
        assert isinstance(results, list)
        assert len(results) > 0
        for r in results:
            assert "category" in r
            assert "name" in r
            assert "status" in r
            assert "detail" in r
            assert r["status"] in ("pass", "warn", "fail")

    def test_sdk_check_pass(self):
        """SDK checks report pass when module is importable."""
        mock_mod = MagicMock()
        mock_mod.__version__ = "1.2.3"

        def mock_import(name):
            if name == "openai":
                return mock_mod
            return _sdk_import_error(name)

        with patch(
            "stateloom.cli.doctor_command.importlib.import_module",
            side_effect=mock_import,
        ):
            results = _run_checks(port=19999)

        openai_checks = [r for r in results if "OpenAI SDK" in r["name"]]
        assert len(openai_checks) == 1
        assert openai_checks[0]["status"] == "pass"
        assert "1.2.3" in openai_checks[0]["detail"]

    def test_sdk_check_fail(self):
        """SDK checks report fail when module not importable."""
        with patch(
            "stateloom.cli.doctor_command.importlib.import_module",
            side_effect=_sdk_import_error,
        ):
            results = _run_checks(port=19999)

        sdk_checks = [r for r in results if r["category"] == "SDKs"]
        assert all(r["status"] == "fail" for r in sdk_checks)

    def test_provider_key_pass(self, monkeypatch):
        """Provider key checks report pass when env vars are set."""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
        monkeypatch.setenv("GOOGLE_API_KEY", "AIzaSy-test")

        with patch(
            "stateloom.cli.doctor_command.importlib.import_module",
            side_effect=_sdk_import_error,
        ):
            results = _run_checks(port=19999)

        provider_checks = [r for r in results if r["category"] == "Providers"]
        assert all(r["status"] == "pass" for r in provider_checks)

    def test_provider_key_warn(self, monkeypatch):
        """Provider key checks report warn when env vars are missing."""
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)

        with patch(
            "stateloom.cli.doctor_command.importlib.import_module",
            side_effect=_sdk_import_error,
        ):
            results = _run_checks(port=19999)

        provider_checks = [r for r in results if r["category"] == "Providers"]
        assert all(r["status"] == "warn" for r in provider_checks)

    def test_hardware_checks_present(self):
        """Hardware checks are included in results."""
        with patch(
            "stateloom.cli.doctor_command.importlib.import_module",
            side_effect=_sdk_import_error,
        ):
            results = _run_checks(port=19999)

        hw_checks = [r for r in results if r["category"] == "Hardware"]
        assert len(hw_checks) >= 2  # RAM + disk at minimum

    def test_ollama_not_reachable(self):
        """Ollama check reports fail when not reachable."""
        mock_client = MagicMock()
        mock_client.is_available.return_value = False

        with (
            patch(
                "stateloom.cli.doctor_command.importlib.import_module",
                side_effect=_sdk_import_error,
            ),
            patch(
                "stateloom.local.client.OllamaClient",
                return_value=mock_client,
            ),
            patch("stateloom.cli.doctor_command.httpx.get", side_effect=Exception("no server")),
        ):
            results = _run_checks(port=19999)

        ollama_checks = [r for r in results if "Ollama reachable" in r["name"]]
        assert len(ollama_checks) == 1
        assert ollama_checks[0]["status"] == "fail"

    def test_ollama_reachable(self):
        """Ollama check reports pass with model count."""
        mock_client = MagicMock()
        mock_client.is_available.return_value = True
        mock_client.list_models.return_value = [{"name": "llama3"}, {"name": "phi3"}]

        with (
            patch(
                "stateloom.cli.doctor_command.importlib.import_module",
                side_effect=_sdk_import_error,
            ),
            patch(
                "stateloom.local.client.OllamaClient",
                return_value=mock_client,
            ),
            patch("stateloom.cli.doctor_command.httpx.get", side_effect=Exception("no server")),
        ):
            results = _run_checks(port=19999)

        ollama_checks = [r for r in results if "Ollama reachable" in r["name"]]
        assert len(ollama_checks) == 1
        assert ollama_checks[0]["status"] == "pass"
        assert "2 model" in ollama_checks[0]["detail"]

    def test_server_reachable(self):
        """Server check reports pass when dashboard responds."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200

        with (
            patch(
                "stateloom.cli.doctor_command.importlib.import_module",
                side_effect=_sdk_import_error,
            ),
            patch("stateloom.cli.doctor_command.httpx.get", return_value=mock_resp),
        ):
            results = _run_checks(port=4782)

        server_checks = [r for r in results if "Dashboard reachable" in r["name"]]
        assert len(server_checks) == 1
        assert server_checks[0]["status"] == "pass"

    def test_server_not_reachable(self):
        """Server check reports fail when dashboard is down."""
        with (
            patch(
                "stateloom.cli.doctor_command.importlib.import_module",
                side_effect=_sdk_import_error,
            ),
            patch(
                "stateloom.cli.doctor_command.httpx.get",
                side_effect=Exception("Connection refused"),
            ),
        ):
            results = _run_checks(port=19999)

        server_checks = [r for r in results if "Dashboard reachable" in r["name"]]
        assert len(server_checks) == 1
        assert server_checks[0]["status"] == "fail"


class TestDoctorCommand:
    def test_json_output(self):
        """--json outputs valid JSON list."""
        runner = CliRunner()
        with patch(
            "stateloom.cli.doctor_command._run_checks",
            return_value=[{"category": "Test", "name": "check1", "status": "pass", "detail": "ok"}],
        ):
            result = runner.invoke(doctor, ["--json"])
        data = json.loads(result.output)
        assert isinstance(data, list)
        assert data[0]["name"] == "check1"

    def test_formatted_output(self):
        """Default output shows formatted results."""
        runner = CliRunner()
        with patch(
            "stateloom.cli.doctor_command._run_checks",
            return_value=[
                {"category": "SDKs", "name": "Test SDK", "status": "pass", "detail": "v1.0"},
                {"category": "Local", "name": "Ollama", "status": "fail", "detail": "down"},
            ],
        ):
            result = runner.invoke(doctor, [])
        assert "StateLoom Doctor" in result.output
        assert "Summary" in result.output

    def test_exit_code_on_failure(self):
        """Exit code 1 when any check fails."""
        runner = CliRunner()
        with patch(
            "stateloom.cli.doctor_command._run_checks",
            return_value=[
                {"category": "Test", "name": "fail_check", "status": "fail", "detail": "bad"},
            ],
        ):
            result = runner.invoke(doctor, [])
        assert result.exit_code == 1

    def test_exit_code_on_success(self):
        """Exit code 0 when all checks pass or warn."""
        runner = CliRunner()
        with patch(
            "stateloom.cli.doctor_command._run_checks",
            return_value=[
                {"category": "Test", "name": "ok_check", "status": "pass", "detail": "ok"},
                {"category": "Test", "name": "warn_check", "status": "warn", "detail": "meh"},
            ],
        ):
            result = runner.invoke(doctor, [])
        assert result.exit_code == 0

    def test_custom_port(self):
        """--port is passed to _run_checks."""
        runner = CliRunner()
        with patch("stateloom.cli.doctor_command._run_checks", return_value=[]) as mock_run:
            runner.invoke(doctor, ["--port", "9999", "--json"])
        mock_run.assert_called_once_with(9999)
