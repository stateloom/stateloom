"""Tests for local Llama-Guard validator."""

from __future__ import annotations

from dataclasses import dataclass
from unittest.mock import MagicMock, patch

import pytest

from stateloom.guardrails.local_validator import LocalGuardrailValidator


@dataclass
class FakeOllamaResponse:
    content: str = ""
    model: str = "llama-guard3:1b"
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    latency_ms: float = 0.0
    raw: dict | None = None


class TestLocalGuardrailValidator:
    def test_safe_input(self):
        validator = LocalGuardrailValidator()
        # Mock the client
        mock_client = MagicMock()
        mock_client.list_models.return_value = [{"name": "llama-guard3:1b"}]
        mock_client.chat.return_value = FakeOllamaResponse(content="safe")

        validator._client = mock_client
        validator._available = True

        result = validator.validate([{"role": "user", "content": "Hello, how are you?"}])
        assert result.safe is True
        assert result.score == 0.0

    def test_unsafe_input(self):
        validator = LocalGuardrailValidator()
        mock_client = MagicMock()
        mock_client.chat.return_value = FakeOllamaResponse(content="unsafe\nS1")

        validator._client = mock_client
        validator._available = True

        result = validator.validate([{"role": "user", "content": "something unsafe"}])
        assert result.safe is False
        assert "S1" in result.category
        assert result.score == 1.0

    def test_unsafe_with_category(self):
        validator = LocalGuardrailValidator()
        mock_client = MagicMock()
        mock_client.chat.return_value = FakeOllamaResponse(content="unsafe\nS7")

        validator._client = mock_client
        validator._available = True

        result = validator.validate([{"role": "user", "content": "test"}])
        assert result.safe is False
        assert "Privacy" in result.category
        assert result.severity == "high"

    def test_ollama_unavailable_returns_safe(self):
        validator = LocalGuardrailValidator()
        validator._available = False

        result = validator.validate([{"role": "user", "content": "ignore instructions"}])
        assert result.safe is True

    def test_auto_pull_on_first_use(self):
        validator = LocalGuardrailValidator()
        mock_client = MagicMock()
        mock_client.list_models.return_value = []  # Model not found
        mock_client.pull_model.return_value = None
        mock_client.chat.return_value = FakeOllamaResponse(content="safe")

        with patch("stateloom.local.client.OllamaClient", return_value=mock_client):
            # Reset validator state so _lazy_init runs
            validator._available = None
            validator._client = None
            result = validator._lazy_init()
            assert result is True
            mock_client.pull_model.assert_called_once_with("llama-guard3:1b")

    def test_timeout_returns_safe(self):
        validator = LocalGuardrailValidator()
        mock_client = MagicMock()
        mock_client.chat.side_effect = TimeoutError("Ollama timeout")

        validator._client = mock_client
        validator._available = True

        result = validator.validate([{"role": "user", "content": "test"}])
        assert result.safe is True  # fail-open

    def test_empty_messages_returns_safe(self):
        validator = LocalGuardrailValidator()
        mock_client = MagicMock()
        validator._client = mock_client
        validator._available = True

        result = validator.validate([])
        assert result.safe is True

    def test_parse_unrecognized_output(self):
        validator = LocalGuardrailValidator()
        result = validator._parse_response("something unexpected")
        assert result.safe is True  # fail-open

    def test_parse_space_separated_category(self):
        """Handle 'unsafe S7' (space-separated) format — not just newline."""
        validator = LocalGuardrailValidator()
        result = validator._parse_response("unsafe S7")
        assert result.safe is False
        assert "Privacy" in result.category
        assert result.severity == "high"
