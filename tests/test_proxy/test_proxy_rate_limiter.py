"""Tests for per-virtual-key rate limiting in the proxy gateway."""

from __future__ import annotations

import asyncio
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from stateloom.core.config import StateLoomConfig
from stateloom.core.errors import StateLoomRateLimitError
from stateloom.core.event import RateLimitEvent
from stateloom.proxy.rate_limiter import ProxyRateLimiter
from stateloom.proxy.virtual_key import (
    VirtualKey,
    generate_virtual_key,
    make_key_preview,
    make_virtual_key_id,
)
from stateloom.store.memory_store import MemoryStore


def _make_vk(
    *,
    rate_limit_tps: float | None = None,
    rate_limit_max_queue: int = 100,
    rate_limit_queue_timeout: float = 30.0,
) -> VirtualKey:
    """Create a VirtualKey with rate limit fields."""
    full_key, key_hash = generate_virtual_key()
    return VirtualKey(
        id=make_virtual_key_id(),
        key_hash=key_hash,
        key_preview=make_key_preview(full_key),
        team_id="team-1",
        org_id="org-1",
        name="test-key",
        rate_limit_tps=rate_limit_tps,
        rate_limit_max_queue=rate_limit_max_queue,
        rate_limit_queue_timeout=rate_limit_queue_timeout,
    )


class TestProxyRateLimiterNoLimit:
    """VK without rate_limit_tps passes through."""

    async def test_no_limit_passes_through(self):
        limiter = ProxyRateLimiter()
        vk = _make_vk(rate_limit_tps=None)
        # Should not raise
        await limiter.check(vk)

    async def test_no_limit_no_bucket_created(self):
        limiter = ProxyRateLimiter()
        vk = _make_vk(rate_limit_tps=None)
        await limiter.check(vk)
        assert limiter.get_status() == {"keys": {}}


class TestProxyRateLimiterAcquire:
    """TPS=1 allows first request, blocks second within window."""

    async def test_first_request_passes(self):
        limiter = ProxyRateLimiter()
        vk = _make_vk(rate_limit_tps=1.0, rate_limit_max_queue=0)
        await limiter.check(vk)
        # Bucket should exist now
        status = limiter.get_status()
        assert vk.id in status["keys"]

    async def test_second_request_rejected_when_queue_zero(self):
        limiter = ProxyRateLimiter()
        vk = _make_vk(rate_limit_tps=1.0, rate_limit_max_queue=0)
        await limiter.check(vk)
        with pytest.raises(StateLoomRateLimitError):
            await limiter.check(vk)

    async def test_request_after_release_passes(self):
        limiter = ProxyRateLimiter()
        vk = _make_vk(rate_limit_tps=1.0, rate_limit_max_queue=0)
        await limiter.check(vk)
        limiter.on_request_complete(vk.id)
        # After sufficient time for refill
        await asyncio.sleep(1.05)
        await limiter.check(vk)


class TestProxyRateLimiterQueue:
    """Request queues when bucket empty, releases on on_request_complete()."""

    async def test_queued_request_released(self):
        limiter = ProxyRateLimiter()
        vk = _make_vk(rate_limit_tps=1.0, rate_limit_max_queue=10, rate_limit_queue_timeout=5.0)
        # First request acquires the token
        await limiter.check(vk)

        # Second request should queue, then be released when first completes
        async def release_after_delay():
            await asyncio.sleep(0.1)
            limiter.on_request_complete(vk.id)

        task = asyncio.create_task(release_after_delay())
        # This should wait in the queue and then be released
        # Need to wait for refill (1s) or for on_request_complete + refill
        # Actually on_request_complete calls release_next_waiter which needs tokens
        # Let's wait enough for refill
        start = time.monotonic()
        await limiter.check(vk)
        elapsed = time.monotonic() - start
        assert elapsed < 5.0  # Didn't time out
        await task


class TestProxyRateLimiterQueueFull:
    """Raises error when queue at capacity."""

    async def test_queue_full_rejected(self):
        limiter = ProxyRateLimiter()
        vk = _make_vk(rate_limit_tps=1.0, rate_limit_max_queue=1, rate_limit_queue_timeout=5.0)
        # Acquire the token
        await limiter.check(vk)

        # Queue one request (blocks in background)
        async def queued_request():
            try:
                await limiter.check(vk)
            except StateLoomRateLimitError:
                pass

        task = asyncio.create_task(queued_request())
        await asyncio.sleep(0.05)  # Let it enter the queue

        # Third request should be rejected — queue is full
        with pytest.raises(StateLoomRateLimitError, match="queue full"):
            await limiter.check(vk)

        # Clean up
        limiter.on_request_complete(vk.id)
        await asyncio.sleep(1.1)  # Wait for refill
        await task


class TestProxyRateLimiterTimeout:
    """Raises error on queue timeout."""

    async def test_queue_timeout(self):
        limiter = ProxyRateLimiter()
        vk = _make_vk(
            rate_limit_tps=100.0,  # High TPS so bucket drains
            rate_limit_max_queue=10,
            rate_limit_queue_timeout=0.5,  # Short timeout
        )
        # Exhaust all tokens
        for _ in range(100):
            await limiter.check(vk)

        # Next request should queue and timeout
        with pytest.raises(StateLoomRateLimitError, match="queue timeout"):
            await limiter.check(vk)


class TestProxyRateLimiterHotReload:
    """Changing VK TPS takes effect immediately."""

    async def test_tps_change_takes_effect(self):
        limiter = ProxyRateLimiter()
        vk = _make_vk(rate_limit_tps=1.0, rate_limit_max_queue=0)
        await limiter.check(vk)
        # Now vk at tps=1 is exhausted
        with pytest.raises(StateLoomRateLimitError):
            await limiter.check(vk)

        # "Hot reload" — increase TPS
        vk.rate_limit_tps = 100.0
        # Next check should hot-reload the TPS on the bucket
        # Need a moment for token refill
        await asyncio.sleep(0.02)
        await limiter.check(vk)  # Should succeed with new TPS


class TestProxyRateLimiterRemoveBucket:
    """remove_bucket removes a key's bucket."""

    async def test_remove_bucket(self):
        limiter = ProxyRateLimiter()
        vk = _make_vk(rate_limit_tps=1.0)
        await limiter.check(vk)
        assert vk.id in limiter.get_status()["keys"]
        limiter.remove_bucket(vk.id)
        assert vk.id not in limiter.get_status()["keys"]


class TestProxyRateLimiterGetStatus:
    """get_status returns state for all keys."""

    async def test_status_shape(self):
        limiter = ProxyRateLimiter()
        vk = _make_vk(rate_limit_tps=5.0)
        await limiter.check(vk)
        status = limiter.get_status()
        key_status = status["keys"][vk.id]
        assert key_status["tps"] == 5.0
        assert "tokens_available" in key_status
        assert "queue_size" in key_status
        assert "active_requests" in key_status
        assert key_status["active_requests"] == 1


class TestSQLiteVKRateLimitRoundTrip:
    """VK rate limit fields persist and load correctly in SQLite."""

    def test_roundtrip(self, tmp_path):
        from stateloom.store.sqlite_store import SQLiteStore

        store = SQLiteStore(str(tmp_path / "test.db"))
        vk = _make_vk(rate_limit_tps=5.0, rate_limit_max_queue=50, rate_limit_queue_timeout=10.0)
        store.save_virtual_key(vk)

        loaded = store.get_virtual_key_by_hash(vk.key_hash)
        assert loaded is not None
        assert loaded.rate_limit_tps == 5.0
        assert loaded.rate_limit_max_queue == 50
        assert loaded.rate_limit_queue_timeout == 10.0

    def test_roundtrip_no_rate_limit(self, tmp_path):
        from stateloom.store.sqlite_store import SQLiteStore

        store = SQLiteStore(str(tmp_path / "test.db"))
        vk = _make_vk(rate_limit_tps=None)
        store.save_virtual_key(vk)

        loaded = store.get_virtual_key_by_hash(vk.key_hash)
        assert loaded is not None
        assert loaded.rate_limit_tps is None
        assert loaded.rate_limit_max_queue == 100
        assert loaded.rate_limit_queue_timeout == 30.0

    def test_get_virtual_key_by_id(self, tmp_path):
        from stateloom.store.sqlite_store import SQLiteStore

        store = SQLiteStore(str(tmp_path / "test.db"))
        vk = _make_vk(rate_limit_tps=2.0)
        store.save_virtual_key(vk)

        loaded = store.get_virtual_key(vk.id)
        assert loaded is not None
        assert loaded.id == vk.id
        assert loaded.rate_limit_tps == 2.0


class TestCreateVirtualKeyWithRateLimit:
    """create_virtual_key() accepts and stores rate limit params."""

    def test_create_with_rate_limit(self):
        import stateloom

        stateloom.init(console_output=False, dashboard=False)
        try:
            stateloom.create_team("test-org", "team-rl")
        except Exception:
            pass

        result = stateloom.create_virtual_key(
            team_id="team-rl",
            name="rate-limited-key",
            rate_limit_tps=10.0,
            rate_limit_max_queue=50,
            rate_limit_queue_timeout=15.0,
        )
        assert "id" in result
        assert "key" in result

        # Verify stored
        gate = stateloom.get_gate()
        vk = gate.store.get_virtual_key(result["id"])
        assert vk is not None
        assert vk.rate_limit_tps == 10.0
        assert vk.rate_limit_max_queue == 50
        assert vk.rate_limit_queue_timeout == 15.0


class TestRateLimitEventVirtualKeyId:
    """RateLimitEvent has virtual_key_id field."""

    def test_default_empty(self):
        event = RateLimitEvent(session_id="s1", step=1)
        assert event.virtual_key_id == ""

    def test_set_virtual_key_id(self):
        event = RateLimitEvent(session_id="s1", step=1, virtual_key_id="vk-123")
        assert event.virtual_key_id == "vk-123"


class TestRouterRateLimitIntegration:
    """429 returned when key rate limit exceeded (mock Client)."""

    def test_rate_limit_429(self):
        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        from stateloom.proxy.router import create_proxy_router

        gate = MagicMock()
        gate.store = MemoryStore()
        gate.config = StateLoomConfig(
            proxy_enabled=True,
            proxy_require_virtual_key=True,
            console_output=False,
            dashboard=False,
        )
        gate.pricing = MagicMock()
        gate.pricing._prices = {"gpt-4": MagicMock()}

        app = FastAPI()
        router = create_proxy_router(gate)
        app.include_router(router, prefix="/v1")
        client = TestClient(app)

        # Create a VK with TPS=1 and max_queue=0
        full_key, key_hash = generate_virtual_key()
        vk = VirtualKey(
            id=make_virtual_key_id(),
            key_hash=key_hash,
            key_preview=make_key_preview(full_key),
            team_id="team-1",
            org_id="org-1",
            name="limited-key",
            rate_limit_tps=1.0,
            rate_limit_max_queue=0,
        )
        gate.store.save_virtual_key(vk)

        body = {"model": "gpt-4", "messages": [{"role": "user", "content": "hi"}]}
        headers = {"Authorization": f"Bearer {full_key}"}

        # Mock _handle_provider_sdk to avoid actual LLM calls
        from fastapi.responses import JSONResponse as _JSONResponse

        mock_json = _JSONResponse(
            content={
                "id": "test",
                "choices": [{"message": {"content": "hello"}, "finish_reason": "stop"}],
                "model": "gpt-4",
            }
        )

        with patch(
            "stateloom.proxy.router._handle_provider_sdk",
            new_callable=AsyncMock,
            return_value=mock_json,
        ):
            # First request — should pass
            resp1 = client.post("/v1/chat/completions", json=body, headers=headers)
            assert resp1.status_code == 200

            # Second request — should be rate limited (429)
            resp2 = client.post("/v1/chat/completions", json=body, headers=headers)
            assert resp2.status_code == 429
            data = resp2.json()
            assert data["error"]["code"] == "key_rate_limit_exceeded"


class TestProxyRateLimiterMetrics:
    """Metrics are recorded at each rate limit decision point."""

    async def test_passed_records_metric(self):
        metrics = MagicMock()
        limiter = ProxyRateLimiter(metrics=metrics)
        vk = _make_vk(rate_limit_tps=10.0, rate_limit_max_queue=0)
        await limiter.check(vk)
        metrics.record_rate_limit.assert_called_once_with(
            team_id="",
            virtual_key_id=vk.id,
            outcome="passed",
            wait_ms=0.0,
        )

    async def test_rejected_records_metric(self):
        metrics = MagicMock()
        limiter = ProxyRateLimiter(metrics=metrics)
        vk = _make_vk(rate_limit_tps=1.0, rate_limit_max_queue=0)
        await limiter.check(vk)  # First passes
        metrics.record_rate_limit.assert_called_with(
            team_id="",
            virtual_key_id=vk.id,
            outcome="passed",
            wait_ms=0.0,
        )
        with pytest.raises(StateLoomRateLimitError):
            await limiter.check(vk)  # Second rejected
        assert metrics.record_rate_limit.call_count == 2
        metrics.record_rate_limit.assert_called_with(
            team_id="",
            virtual_key_id=vk.id,
            outcome="rejected",
            wait_ms=0.0,
        )

    async def test_no_metrics_when_none(self):
        limiter = ProxyRateLimiter(metrics=None)
        vk = _make_vk(rate_limit_tps=10.0, rate_limit_max_queue=0)
        # Should not raise
        await limiter.check(vk)

    async def test_queued_records_metric_with_wait_ms(self):
        metrics = MagicMock()
        limiter = ProxyRateLimiter(metrics=metrics)
        vk = _make_vk(rate_limit_tps=1.0, rate_limit_max_queue=10, rate_limit_queue_timeout=5.0)
        await limiter.check(vk)  # Acquire token

        # Release after short delay so queued request succeeds
        async def release():
            await asyncio.sleep(0.1)
            limiter.on_request_complete(vk.id)

        task = asyncio.create_task(release())
        await limiter.check(vk)  # Queued then released
        await task

        # Last call should be "queued" with positive wait_ms
        last_call = metrics.record_rate_limit.call_args
        assert last_call.kwargs["outcome"] == "queued"
        assert last_call.kwargs["wait_ms"] > 0
        assert last_call.kwargs["virtual_key_id"] == vk.id


class TestMemoryStoreGetVirtualKey:
    """MemoryStore.get_virtual_key works correctly."""

    def test_get_by_id(self):
        store = MemoryStore()
        vk = _make_vk(rate_limit_tps=3.0)
        store.save_virtual_key(vk)

        loaded = store.get_virtual_key(vk.id)
        assert loaded is not None
        assert loaded.id == vk.id
        assert loaded.rate_limit_tps == 3.0

    def test_get_missing_returns_none(self):
        store = MemoryStore()
        assert store.get_virtual_key("nonexistent") is None
