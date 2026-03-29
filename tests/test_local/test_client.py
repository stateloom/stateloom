"""Tests for the Ollama client and request translator."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import httpx
import pytest

from stateloom.local.client import OllamaClient, OllamaResponse, RequestTranslator


class TestRequestTranslator:
    """Test request translation from various provider formats."""

    def test_translate_openai_simple(self):
        result = RequestTranslator.translate(
            "openai",
            "llama3.2",
            {"messages": [{"role": "user", "content": "hello"}]},
        )
        assert result["model"] == "llama3.2"
        assert result["stream"] is False
        assert result["messages"] == [{"role": "user", "content": "hello"}]

    def test_translate_anthropic_system(self):
        result = RequestTranslator.translate(
            "anthropic",
            "llama3.2",
            {
                "system": "You are helpful.",
                "messages": [{"role": "user", "content": "hello"}],
            },
        )
        assert len(result["messages"]) == 2
        assert result["messages"][0] == {"role": "system", "content": "You are helpful."}
        assert result["messages"][1] == {"role": "user", "content": "hello"}

    def test_translate_anthropic_system_blocks(self):
        result = RequestTranslator.translate(
            "anthropic",
            "llama3.2",
            {
                "system": [{"type": "text", "text": "Be helpful."}],
                "messages": [{"role": "user", "content": "hi"}],
            },
        )
        assert result["messages"][0] == {"role": "system", "content": "Be helpful."}

    def test_translate_options_temperature(self):
        result = RequestTranslator.translate(
            "openai",
            "llama3.2",
            {
                "messages": [{"role": "user", "content": "hi"}],
                "temperature": 0.7,
                "top_p": 0.9,
            },
        )
        assert result["options"]["temperature"] == 0.7
        assert result["options"]["top_p"] == 0.9

    def test_translate_max_tokens_to_num_predict(self):
        result = RequestTranslator.translate(
            "openai",
            "llama3.2",
            {
                "messages": [{"role": "user", "content": "hi"}],
                "max_tokens": 100,
            },
        )
        assert result["options"]["num_predict"] == 100

    def test_translate_max_completion_tokens(self):
        result = RequestTranslator.translate(
            "openai",
            "llama3.2",
            {
                "messages": [{"role": "user", "content": "hi"}],
                "max_completion_tokens": 200,
            },
        )
        assert result["options"]["num_predict"] == 200

    def test_translate_seed(self):
        result = RequestTranslator.translate(
            "openai",
            "llama3.2",
            {
                "messages": [{"role": "user", "content": "hi"}],
                "seed": 42,
            },
        )
        assert result["options"]["seed"] == 42

    def test_translate_strips_unsupported_keys(self):
        """Unsupported keys should not appear in the translated payload."""
        result = RequestTranslator.translate(
            "openai",
            "llama3.2",
            {
                "messages": [{"role": "user", "content": "hi"}],
                "response_format": {"type": "json_object"},
                "tools": [{"type": "function"}],
                "logprobs": True,
            },
        )
        assert "response_format" not in result
        assert "tools" not in result
        assert "logprobs" not in result

    def test_translate_content_blocks(self):
        """Messages with content blocks should be flattened to text."""
        result = RequestTranslator.translate(
            "openai",
            "llama3.2",
            {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Hello"},
                            {"type": "text", "text": "World"},
                        ],
                    }
                ],
            },
        )
        assert result["messages"][0]["content"] == "Hello\nWorld"

    def test_translate_no_options_when_empty(self):
        result = RequestTranslator.translate(
            "openai",
            "llama3.2",
            {"messages": [{"role": "user", "content": "hi"}]},
        )
        assert "options" not in result


class TestOllamaResponse:
    def test_defaults(self):
        resp = OllamaResponse()
        assert resp.model == ""
        assert resp.content == ""
        assert resp.prompt_tokens == 0
        assert resp.completion_tokens == 0
        assert resp.total_tokens == 0
        assert resp.latency_ms == 0.0

    def test_with_data(self):
        resp = OllamaResponse(
            model="llama3.2",
            content="Hello!",
            prompt_tokens=10,
            completion_tokens=5,
            total_tokens=15,
            latency_ms=100.0,
        )
        assert resp.total_tokens == 15


class TestOllamaClient:
    def test_parse_response(self):
        client = OllamaClient()
        data = {
            "model": "llama3.2",
            "message": {"role": "assistant", "content": "Hi there!"},
            "prompt_eval_count": 10,
            "eval_count": 5,
        }
        resp = client._parse_response(data, 150.0)
        assert resp.model == "llama3.2"
        assert resp.content == "Hi there!"
        assert resp.prompt_tokens == 10
        assert resp.completion_tokens == 5
        assert resp.total_tokens == 15
        assert resp.latency_ms == 150.0

    def test_parse_response_missing_fields(self):
        client = OllamaClient()
        data = {"model": "test", "message": {}}
        resp = client._parse_response(data, 50.0)
        assert resp.content == ""
        assert resp.prompt_tokens == 0
        assert resp.completion_tokens == 0

    def test_is_available_returns_false_on_error(self):
        client = OllamaClient(host="http://localhost:99999")
        assert client.is_available() is False

    def test_custom_host(self):
        client = OllamaClient(host="http://myhost:12345/")
        assert client._host == "http://myhost:12345"

    def test_close_idempotent(self):
        client = OllamaClient()
        client.close()
        client.close()  # Should not raise
