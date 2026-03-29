"""Tests for durable streaming — record & replay stream chunks."""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any

import pytest
from pydantic import BaseModel

from stateloom.replay.schema import (
    CachedStreamChunks,
    deserialize_response,
    deserialize_stream_chunks,
    serialize_stream_chunks,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class MockChunk(BaseModel):
    """Mock streaming chunk with model_dump support."""

    id: str = "chunk-1"
    delta: str = ""
    index: int = 0


class DictChunk:
    """Chunk that only supports dict() conversion."""

    def __init__(self, text: str) -> None:
        self.text = text

    def dict(self) -> dict[str, Any]:
        return {"text": self.text}


# ---------------------------------------------------------------------------
# Unit: CachedStreamChunks
# ---------------------------------------------------------------------------


class TestCachedStreamChunks:
    def test_sync_iteration(self):
        chunks = [{"id": 1}, {"id": 2}, {"id": 3}]
        cached = CachedStreamChunks(chunks)
        result = list(cached)
        assert result == chunks

    def test_sync_iteration_empty(self):
        cached = CachedStreamChunks([])
        result = list(cached)
        assert result == []

    @pytest.mark.asyncio
    async def test_async_iteration(self):
        chunks = [{"id": 1}, {"id": 2}]
        cached = CachedStreamChunks(chunks)
        result = []
        async for chunk in cached:
            result.append(chunk)
        assert result == chunks

    @pytest.mark.asyncio
    async def test_async_iteration_empty(self):
        cached = CachedStreamChunks([])
        result = []
        async for chunk in cached:
            result.append(chunk)
        assert result == []

    def test_sync_with_delay(self):
        chunks = [{"a": 1}, {"a": 2}]
        cached = CachedStreamChunks(chunks, delay_ms=50)
        start = time.monotonic()
        result = list(cached)
        elapsed = time.monotonic() - start
        assert result == chunks
        # 2 chunks with 50ms delay each → at least ~100ms
        assert elapsed >= 0.08  # allow some slack

    @pytest.mark.asyncio
    async def test_async_with_delay(self):
        chunks = [{"a": 1}, {"a": 2}]
        cached = CachedStreamChunks(chunks, delay_ms=50)
        start = time.monotonic()
        result = []
        async for chunk in cached:
            result.append(chunk)
        elapsed = time.monotonic() - start
        assert result == chunks
        assert elapsed >= 0.08

    def test_zero_delay_no_sleep(self):
        """delay_ms=0 should not introduce any delay."""
        chunks = list(range(100))
        cached = CachedStreamChunks(chunks, delay_ms=0)
        start = time.monotonic()
        result = list(cached)
        elapsed = time.monotonic() - start
        assert result == chunks
        assert elapsed < 0.5  # should be near-instant


# ---------------------------------------------------------------------------
# Unit: serialize_stream_chunks
# ---------------------------------------------------------------------------


class TestSerializeStreamChunks:
    def test_serialize_pydantic_chunks(self):
        chunks = [MockChunk(id="c1", delta="Hello"), MockChunk(id="c2", delta=" world")]
        result = serialize_stream_chunks(chunks, "openai")
        parsed = json.loads(result)
        assert parsed["_type"] == "stream"
        assert parsed["provider"] == "openai"
        assert len(parsed["chunks"]) == 2
        assert parsed["chunks"][0]["id"] == "c1"
        assert parsed["chunks"][1]["delta"] == " world"

    def test_serialize_dict_chunks(self):
        chunks = [{"text": "a"}, {"text": "b"}]
        result = serialize_stream_chunks(chunks, "anthropic")
        parsed = json.loads(result)
        assert parsed["_type"] == "stream"
        assert parsed["provider"] == "anthropic"
        assert parsed["chunks"] == [{"text": "a"}, {"text": "b"}]

    def test_serialize_dict_method_chunks(self):
        chunks = [DictChunk("hello"), DictChunk("world")]
        result = serialize_stream_chunks(chunks, "openai")
        parsed = json.loads(result)
        assert parsed["chunks"][0] == {"text": "hello"}

    def test_serialize_empty(self):
        result = serialize_stream_chunks([], "openai")
        parsed = json.loads(result)
        assert parsed["_type"] == "stream"
        assert parsed["chunks"] == []

    def test_serialize_non_serializable_fallback(self):
        """Non-serializable objects fall back to str()."""

        class Weird:
            def __str__(self):
                return "weird-obj"

        result = serialize_stream_chunks([Weird()], "openai")
        parsed = json.loads(result)
        assert parsed["chunks"][0] == "weird-obj"


# ---------------------------------------------------------------------------
# Unit: deserialize_stream_chunks
# ---------------------------------------------------------------------------


class TestDeserializeStreamChunks:
    def test_deserialize_dict_chunks(self):
        data = {
            "_type": "stream",
            "provider": "unknown",
            "chunks": [{"text": "a"}, {"text": "b"}],
        }
        cached = deserialize_stream_chunks(data)
        assert isinstance(cached, CachedStreamChunks)
        result = list(cached)
        assert result == [{"text": "a"}, {"text": "b"}]

    def test_deserialize_with_delay(self):
        data = {"_type": "stream", "provider": "openai", "chunks": [{"x": 1}]}
        cached = deserialize_stream_chunks(data, delay_ms=100)
        assert cached.delay_ms == 100

    def test_deserialize_empty_chunks(self):
        data = {"_type": "stream", "provider": "openai", "chunks": []}
        cached = deserialize_stream_chunks(data)
        assert list(cached) == []

    def test_roundtrip_dict_chunks(self):
        original = [{"id": 1, "text": "hello"}, {"id": 2, "text": "world"}]
        serialized = serialize_stream_chunks(original, "openai")
        parsed = json.loads(serialized)
        cached = deserialize_stream_chunks(parsed)
        result = list(cached)
        assert result == original

    def test_roundtrip_pydantic_chunks(self):
        original = [MockChunk(id="c1", delta="Hi"), MockChunk(id="c2", delta="!")]
        serialized = serialize_stream_chunks(original, "unknown")
        parsed = json.loads(serialized)
        cached = deserialize_stream_chunks(parsed)
        result = list(cached)
        # Pydantic → dict → dict (unknown provider, no reconstruction)
        assert result[0]["id"] == "c1"
        assert result[1]["delta"] == "!"


# ---------------------------------------------------------------------------
# Unit: deserialize_response detects stream marker
# ---------------------------------------------------------------------------


class TestDeserializeResponseStreamDetection:
    def test_stream_marker_returns_cached_stream_chunks(self):
        json_str = json.dumps(
            {
                "_type": "stream",
                "provider": "openai",
                "chunks": [{"id": "chunk-1"}, {"id": "chunk-2"}],
            }
        )
        result = deserialize_response(json_str)
        assert isinstance(result, CachedStreamChunks)
        chunks = list(result)
        assert len(chunks) == 2

    def test_stream_marker_with_delay(self):
        json_str = json.dumps(
            {
                "_type": "stream",
                "provider": "openai",
                "chunks": [{"id": "c1"}],
            }
        )
        result = deserialize_response(json_str, delay_ms=50)
        assert isinstance(result, CachedStreamChunks)
        assert result.delay_ms == 50

    def test_non_stream_dict_unchanged(self):
        """Regular dict responses should not be affected."""
        json_str = json.dumps({"id": "resp-1", "content": "hello"})
        result = deserialize_response(json_str)
        assert isinstance(result, dict)
        assert result["id"] == "resp-1"

    def test_gemini_type_not_confused_with_stream(self):
        """_type: gemini should still work correctly."""
        json_str = json.dumps(
            {
                "_type": "gemini",
                "text": "hello",
                "prompt_tokens": 5,
                "completion_tokens": 3,
            }
        )
        result = deserialize_response(json_str)
        assert not isinstance(result, CachedStreamChunks)
        assert hasattr(result, "text")

    def test_none_input_still_returns_none(self):
        result = deserialize_response(None)
        assert result is None


# ---------------------------------------------------------------------------
# Integration: full serialize → deserialize → iterate roundtrip
# ---------------------------------------------------------------------------


class TestEndToEndRoundtrip:
    def test_full_sync_roundtrip(self):
        """Simulate: record chunks → serialize → store → deserialize → replay."""
        # Simulate streaming chunks
        chunks = [
            {"id": "chatcmpl-1", "choices": [{"delta": {"content": "Hello"}}]},
            {"id": "chatcmpl-1", "choices": [{"delta": {"content": " world"}}]},
            {"id": "chatcmpl-1", "choices": [{"delta": {}, "finish_reason": "stop"}]},
        ]

        # Serialize (as stream wrapper would)
        json_str = serialize_stream_chunks(chunks, "openai")

        # Deserialize (as replay engine would)
        cached = deserialize_response(json_str, provider="openai")
        assert isinstance(cached, CachedStreamChunks)

        # Iterate (as user code would)
        replayed = list(cached)
        assert len(replayed) == 3
        assert replayed[0]["choices"][0]["delta"]["content"] == "Hello"
        assert replayed[2]["choices"][0]["finish_reason"] == "stop"

    @pytest.mark.asyncio
    async def test_full_async_roundtrip(self):
        chunks = [{"delta": "Hi"}, {"delta": "!"}]
        json_str = serialize_stream_chunks(chunks, "anthropic")
        cached = deserialize_response(json_str, provider="anthropic")
        assert isinstance(cached, CachedStreamChunks)

        replayed = []
        async for chunk in cached:
            replayed.append(chunk)
        assert len(replayed) == 2
        assert replayed[0]["delta"] == "Hi"
