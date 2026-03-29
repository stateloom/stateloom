"""Tests for the Ollama provider adapter."""

from __future__ import annotations

from stateloom.core.types import Provider
from stateloom.local.adapter import OllamaAdapter
from stateloom.local.client import OllamaResponse


class TestOllamaAdapter:
    def setup_method(self):
        self.adapter = OllamaAdapter()

    def test_name_is_local(self):
        assert self.adapter.name == Provider.LOCAL
        assert self.adapter.name == "local"

    def test_method_label(self):
        assert self.adapter.method_label == "chat"

    def test_no_patch_targets(self):
        assert self.adapter.get_patch_targets() == []

    def test_extract_tokens_from_ollama_response(self):
        resp = OllamaResponse(
            prompt_tokens=10,
            completion_tokens=20,
            total_tokens=30,
        )
        tokens = self.adapter.extract_tokens(resp)
        assert tokens == (10, 20, 30)

    def test_extract_tokens_unknown_response(self):
        assert self.adapter.extract_tokens(None) == (0, 0, 0)
        assert self.adapter.extract_tokens({"usage": {}}) == (0, 0, 0)

    def test_extract_model(self):
        assert self.adapter.extract_model(None, (), {"model": "llama3.2"}) == "llama3.2"
        assert self.adapter.extract_model(None, (), {}) == "unknown"

    def test_is_streaming_always_false(self):
        assert self.adapter.is_streaming({"stream": True}) is False

    def test_apply_system_prompt_insert(self):
        kwargs = {"messages": [{"role": "user", "content": "hi"}]}
        self.adapter.apply_system_prompt(kwargs, "Be helpful.")
        assert kwargs["messages"][0] == {"role": "system", "content": "Be helpful."}

    def test_apply_system_prompt_replace(self):
        kwargs = {
            "messages": [{"role": "system", "content": "old"}, {"role": "user", "content": "hi"}]
        }
        self.adapter.apply_system_prompt(kwargs, "new")
        assert kwargs["messages"][0]["content"] == "new"
