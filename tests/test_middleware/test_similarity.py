"""Tests for the similarity scoring module."""

from __future__ import annotations

from dataclasses import dataclass
from unittest.mock import MagicMock

import pytest

from stateloom.middleware.similarity import (
    SimilarityResult,
    compute_similarity,
    extract_response_text,
)

# --- extract_response_text ---


class TestExtractResponseText:
    def test_none_response(self):
        assert extract_response_text(None, "openai") == ""

    def test_openai_response(self):
        """OpenAI: response.choices[0].message.content"""
        msg = MagicMock()
        msg.content = "Hello from OpenAI"
        choice = MagicMock()
        choice.message = msg
        response = MagicMock()
        response.choices = [choice]
        # Make sure .content is not a string on the response itself
        response.content = [choice]  # list, not str

        assert extract_response_text(response, "openai") == "Hello from OpenAI"

    def test_anthropic_response(self):
        """Anthropic: response.content[0].text"""
        block = MagicMock()
        block.text = "Hello from Anthropic"
        response = MagicMock(spec=[])

        # Set up content as a list of blocks (no .choices attribute)
        response.content = [block]

        assert extract_response_text(response, "anthropic") == "Hello from Anthropic"

    def test_gemini_response(self):
        """Gemini: response.text"""
        response = MagicMock(spec=[])
        response.text = "Hello from Gemini"

        assert extract_response_text(response, "gemini") == "Hello from Gemini"

    def test_ollama_response(self):
        """Ollama: response.content as str"""
        response = MagicMock(spec=[])
        response.content = "Hello from Ollama"

        assert extract_response_text(response, "local") == "Hello from Ollama"

    def test_ollama_response_dataclass(self):
        """Ollama OllamaResponse dataclass."""
        from stateloom.local.client import OllamaResponse

        response = OllamaResponse(
            model="llama3.2",
            content="Hello local",
            prompt_tokens=5,
            completion_tokens=3,
            total_tokens=8,
            latency_ms=100.0,
        )
        assert extract_response_text(response, "local") == "Hello local"

    def test_gemini_valueerror_with_candidates_fallback(self):
        """Gemini .text raises ValueError, but candidates are accessible."""

        class _FakePart:
            def __init__(self, text: str) -> None:
                self.text = text

        class _FakeContent:
            def __init__(self, text: str) -> None:
                self.parts = [_FakePart(text)]

        class _FakeCandidate:
            def __init__(self, text: str) -> None:
                self.content = _FakeContent(text)

        class _FakeGeminiResponse:
            @property
            def text(self):
                raise ValueError("Multiple candidates")

            candidates = [_FakeCandidate("Gemini fallback text")]

        assert extract_response_text(_FakeGeminiResponse(), "gemini") == "Gemini fallback text"

    def test_gemini_valueerror_no_candidates(self):
        """Gemini .text raises ValueError and no candidates → empty string."""

        class _FakeGeminiResponse:
            @property
            def text(self):
                raise ValueError("No candidates")

            candidates = []

        assert extract_response_text(_FakeGeminiResponse(), "gemini") == ""

    def test_unknown_response(self):
        """Unknown object returns empty string."""
        response = object()
        assert extract_response_text(response, "unknown") == ""

    def test_dict_response(self):
        """Plain dict returns empty string (no attribute access)."""
        assert extract_response_text({"text": "hi"}, "openai") == ""


# --- compute_similarity ---


class TestComputeSimilarity:
    def test_identical_texts(self):
        result = compute_similarity("Hello world", "Hello world")
        assert result is not None
        assert result.score == 1.0
        assert result.method == "difflib"
        assert result.length_ratio == 1.0

    def test_completely_different(self):
        result = compute_similarity("aaa", "zzz")
        assert result is not None
        assert result.score < 0.5

    def test_similar_texts(self):
        result = compute_similarity(
            "The quick brown fox jumps over the lazy dog",
            "The quick brown fox leaps over the lazy dog",
        )
        assert result is not None
        assert result.score > 0.8

    def test_empty_cloud_text(self):
        assert compute_similarity("", "hello") is None

    def test_empty_local_text(self):
        assert compute_similarity("hello", "") is None

    def test_both_empty(self):
        assert compute_similarity("", "") is None

    def test_case_insensitive(self):
        """Score should be the same regardless of case."""
        result1 = compute_similarity("Hello World", "hello world")
        assert result1 is not None
        assert result1.score == 1.0

    def test_preview_truncation(self):
        long_text = "A" * 500
        result = compute_similarity(long_text, "B" * 500, preview_length=200)
        assert result is not None
        assert len(result.cloud_preview) == 200
        assert len(result.local_preview) == 200

    def test_preview_short_text(self):
        result = compute_similarity("short", "text")
        assert result is not None
        assert result.cloud_preview == "short"
        assert result.local_preview == "text"

    def test_length_ratio(self):
        result = compute_similarity("ab", "abcd")
        assert result is not None
        assert result.length_ratio == 0.5
        assert result.cloud_length == 2
        assert result.local_length == 4

    def test_returns_similarity_result(self):
        result = compute_similarity("hello", "hello")
        assert isinstance(result, SimilarityResult)
        assert result.score == 1.0
        assert result.method == "difflib"
        assert result.cloud_preview == "hello"
        assert result.local_preview == "hello"
        assert result.cloud_length == 5
        assert result.local_length == 5
        assert result.length_ratio == 1.0

    def test_custom_preview_length(self):
        result = compute_similarity("abcdef", "ghijkl", preview_length=3)
        assert result is not None
        assert result.cloud_preview == "abc"
        assert result.local_preview == "ghi"
