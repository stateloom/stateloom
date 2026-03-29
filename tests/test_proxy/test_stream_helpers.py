"""Tests for the passthrough stream relay helper."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest

from stateloom.proxy.passthrough import UpstreamStreamError
from stateloom.proxy.stream_helpers import SSE_HEADERS, passthrough_stream_relay


class _FakePassthrough:
    """Minimal passthrough stub that yields pre-defined chunks."""

    def __init__(
        self,
        chunks: list[bytes],
        error: Exception | None = None,
        error_before_yield: bool = False,
    ) -> None:
        self._chunks = chunks
        self._error = error
        self._error_before_yield = error_before_yield

    async def forward_stream(self, upstream_url: str, body: bytes, headers: dict[str, str]):
        if self._error_before_yield and self._error is not None:
            raise self._error
        for chunk in self._chunks:
            yield chunk
        if self._error is not None:
            raise self._error


def _make_ctx() -> SimpleNamespace:
    """Create a minimal ctx-like namespace with _on_stream_complete."""
    return SimpleNamespace(_on_stream_complete=[])


class TestPassthroughStreamRelay:
    async def test_chunks_forwarded(self):
        """All chunks from upstream are forwarded to the client."""
        chunks = [b"data: {}\n\n", b"data: [DONE]\n\n"]
        pt = _FakePassthrough(chunks)
        ctx = _make_ctx()

        resp = await passthrough_stream_relay(pt, "https://up/v1", b"{}", {}, ctx=ctx)
        body_parts: list[bytes] = []
        async for part in resp.body_iterator:
            body_parts.append(part)

        assert body_parts == chunks

    async def test_usage_tracker_called(self):
        """The track_usage callback is called for each chunk."""
        chunks = [b"chunk1\n\n", b"chunk2\n\n"]
        pt = _FakePassthrough(chunks)
        ctx = _make_ctx()
        tracker_calls: list[str] = []

        def tracker(chunk_str: str, ctx: Any) -> None:
            tracker_calls.append(chunk_str)

        resp = await passthrough_stream_relay(
            pt, "https://up/v1", b"{}", {}, ctx=ctx, track_usage=tracker
        )
        async for _ in resp.body_iterator:
            pass

        assert len(tracker_calls) == 2
        assert tracker_calls[0] == "chunk1\n\n"

    async def test_error_formatter_called(self):
        """On mid-stream exception, the format_error callback produces error bytes."""
        # Error occurs after one chunk is yielded (mid-stream)
        pt = _FakePassthrough([b"data: ok\n\n"], error=RuntimeError("boom"))
        ctx = _make_ctx()

        def fmt(exc: Exception) -> bytes:
            return b'data: {"error": "boom"}\n\n'

        resp = await passthrough_stream_relay(
            pt, "https://up/v1", b"{}", {}, ctx=ctx, format_error=fmt
        )
        parts: list[bytes] = []
        async for part in resp.body_iterator:
            parts.append(part)

        assert any(b"error" in p for p in parts)

    async def test_upstream_error_returns_proper_status(self):
        """UpstreamStreamError returns a proper HTTP error, not a 200 SSE stream."""
        pt = _FakePassthrough(
            [],
            error=UpstreamStreamError(429, b'{"error": "rate limited"}', "application/json"),
            error_before_yield=True,
        )
        ctx = _make_ctx()

        resp = await passthrough_stream_relay(pt, "https://up/v1", b"{}", {}, ctx=ctx)
        assert resp.status_code == 429
        assert resp.body == b'{"error": "rate limited"}'

    async def test_stream_complete_callbacks_fire(self):
        """ctx._on_stream_complete callbacks are invoked in the finally block."""
        pt = _FakePassthrough([b"data: ok\n\n"])
        fired: list[bool] = []
        ctx = SimpleNamespace(_on_stream_complete=[lambda: fired.append(True)])

        resp = await passthrough_stream_relay(pt, "https://up/v1", b"{}", {}, ctx=ctx)
        async for _ in resp.body_iterator:
            pass

        assert fired == [True]

    async def test_rate_limiter_slot_released(self):
        """Proxy rate limiter on_request_complete is called in finally."""
        pt = _FakePassthrough([b"data: ok\n\n"])
        ctx = _make_ctx()
        limiter = MagicMock()

        resp = await passthrough_stream_relay(
            pt,
            "https://up/v1",
            b"{}",
            {},
            ctx=ctx,
            proxy_rate_limiter=limiter,
            vk_id="vk-123",
        )
        async for _ in resp.body_iterator:
            pass

        limiter.on_request_complete.assert_called_once_with("vk-123")

    async def test_no_ctx_no_tracker(self):
        """Works without ctx or track_usage (no errors)."""
        pt = _FakePassthrough([b"data: hi\n\n"])
        resp = await passthrough_stream_relay(pt, "https://up/v1", b"{}", {})
        parts: list[bytes] = []
        async for part in resp.body_iterator:
            parts.append(part)
        assert parts == [b"data: hi\n\n"]


class TestSSEHeaders:
    def test_sse_headers_contents(self):
        assert SSE_HEADERS["Cache-Control"] == "no-cache"
        assert SSE_HEADERS["Connection"] == "keep-alive"
        assert SSE_HEADERS["X-Accel-Buffering"] == "no"
        assert len(SSE_HEADERS) == 3
