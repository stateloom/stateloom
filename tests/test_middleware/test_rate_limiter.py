"""Tests for the rate limiter middleware."""

from __future__ import annotations

import asyncio
import threading
import time

import pytest

from stateloom.core.config import StateLoomConfig
from stateloom.core.errors import StateLoomRateLimitError
from stateloom.core.event import RateLimitEvent
from stateloom.core.organization import Team
from stateloom.core.session import Session
from stateloom.middleware.base import MiddlewareContext
from stateloom.middleware.rate_limiter import RateLimiterMiddleware, _TokenBucket
from stateloom.store.memory_store import MemoryStore


def _make_config(**overrides):
    defaults = {
        "auto_patch": False,
        "dashboard": False,
        "console_output": False,
        "store_backend": "memory",
    }
    defaults.update(overrides)
    return StateLoomConfig(**defaults)


def _make_team(
    team_id: str = "team-1",
    tps: float | None = None,
    priority: int = 0,
    max_queue: int = 100,
    queue_timeout: float = 30.0,
) -> Team:
    return Team(
        id=team_id,
        org_id="org-1",
        name="Test Team",
        rate_limit_tps=tps,
        rate_limit_priority=priority,
        rate_limit_max_queue=max_queue,
        rate_limit_queue_timeout=queue_timeout,
    )


def _make_ctx(
    session: Session | None = None,
    team_id: str = "team-1",
    model: str = "gpt-4",
) -> MiddlewareContext:
    if session is None:
        session = Session(id="sess-1", team_id=team_id)
    config = _make_config()
    return MiddlewareContext(
        session=session,
        config=config,
        provider="openai",
        model=model,
        request_kwargs={"messages": [{"role": "user", "content": "hello"}]},
    )


class TestPassthrough:
    """Requests without team_id or without TPS config pass through unthrottled."""

    async def test_passthrough_no_team_id(self):
        teams: dict[str, Team] = {}
        mw = RateLimiterMiddleware(team_lookup=lambda tid: teams.get(tid))

        session = Session(id="sess-1", team_id="")
        ctx = _make_ctx(session=session, team_id="")

        async def call_next(c):
            return "response"

        result = await mw.process(ctx, call_next)
        assert result == "response"

    async def test_passthrough_no_tps(self):
        team = _make_team(tps=None)
        teams = {team.id: team}
        mw = RateLimiterMiddleware(team_lookup=lambda tid: teams.get(tid))

        ctx = _make_ctx()

        async def call_next(c):
            return "response"

        result = await mw.process(ctx, call_next)
        assert result == "response"

    async def test_passthrough_team_not_found(self):
        mw = RateLimiterMiddleware(team_lookup=lambda tid: None)
        ctx = _make_ctx()

        async def call_next(c):
            return "response"

        result = await mw.process(ctx, call_next)
        assert result == "response"


class TestImmediateAcquire:
    """Within TPS limit — no queuing, immediate pass-through."""

    async def test_immediate_acquire(self):
        team = _make_team(tps=10.0)
        teams = {team.id: team}
        mw = RateLimiterMiddleware(team_lookup=lambda tid: teams.get(tid))

        ctx = _make_ctx()

        async def call_next(c):
            return "response"

        result = await mw.process(ctx, call_next)
        assert result == "response"
        # Should have a rate limit event with queued=False
        rl_events = [e for e in ctx.events if isinstance(e, RateLimitEvent)]
        assert len(rl_events) == 1
        assert rl_events[0].queued is False
        assert rl_events[0].rejected is False
        assert rl_events[0].timed_out is False

    async def test_multiple_within_tps(self):
        team = _make_team(tps=5.0)
        teams = {team.id: team}
        mw = RateLimiterMiddleware(team_lookup=lambda tid: teams.get(tid))

        async def call_next(c):
            return "ok"

        for i in range(5):
            ctx = _make_ctx(session=Session(id=f"sess-{i}", team_id="team-1"))
            result = await mw.process(ctx, call_next)
            assert result == "ok"


class TestQueueAndRelease:
    """Exceed TPS → queued → released on completion."""

    async def test_queue_and_release(self):
        team = _make_team(tps=1.0, queue_timeout=5.0)
        teams = {team.id: team}
        mw = RateLimiterMiddleware(team_lookup=lambda tid: teams.get(tid))

        order: list[str] = []

        async def call_next(c):
            order.append(c.session.id)
            return "ok"

        async def slow_next(c):
            order.append(c.session.id)
            await asyncio.sleep(0.3)
            return "ok"

        # First request: takes the only token, completes slowly
        ctx1 = _make_ctx(session=Session(id="first", team_id="team-1"))
        task1 = asyncio.create_task(mw.process(ctx1, slow_next))

        # Small delay to ensure first request acquires the token
        await asyncio.sleep(0.05)

        # Second request: should be queued then released after first completes
        ctx2 = _make_ctx(session=Session(id="second", team_id="team-1"))
        task2 = asyncio.create_task(mw.process(ctx2, call_next))

        r1 = await task1
        r2 = await task2
        assert r1 == "ok"
        assert r2 == "ok"

        # Second request should have been queued
        rl_events = [e for e in ctx2.events if isinstance(e, RateLimitEvent)]
        assert len(rl_events) == 1
        assert rl_events[0].queued is True
        assert rl_events[0].wait_ms > 0


class TestPriorityOrdering:
    """Higher-priority team's queued requests are served first."""

    async def test_priority_ordering(self):
        team = _make_team(tps=1.0, priority=0)
        mw = RateLimiterMiddleware(team_lookup=lambda tid: team if tid == team.id else None)
        bucket = mw._get_or_create_bucket(team)

        from collections import deque

        from stateloom.middleware.rate_limiter import _QueuedRequest

        req_low = _QueuedRequest(event=threading.Event(), priority=0, enqueued_at=time.monotonic())
        req_high = _QueuedRequest(
            event=threading.Event(), priority=10, enqueued_at=time.monotonic()
        )

        with bucket.lock:
            bucket.queue[0] = deque([req_low])
            bucket.queue[10] = deque([req_high])
            bucket.queue_size = 2
            # Force tokens available
            bucket.tokens = 1.0
            bucket.release_next_waiter()

        # High-priority request should be released first
        assert req_high.released is True
        assert req_low.released is False


class TestFIFOWithinPriority:
    """Same priority → FIFO order."""

    async def test_fifo_within_priority(self):
        team = _make_team(tps=1.0, priority=5)
        mw = RateLimiterMiddleware(team_lookup=lambda tid: team if tid == team.id else None)
        bucket = mw._get_or_create_bucket(team)

        from collections import deque

        from stateloom.middleware.rate_limiter import _QueuedRequest

        req_first = _QueuedRequest(
            event=threading.Event(), priority=5, enqueued_at=time.monotonic()
        )
        req_second = _QueuedRequest(
            event=threading.Event(), priority=5, enqueued_at=time.monotonic() + 0.1
        )

        with bucket.lock:
            bucket.queue[5] = deque([req_first, req_second])
            bucket.queue_size = 2
            bucket.tokens = 1.0
            bucket.release_next_waiter()

        assert req_first.released is True
        assert req_second.released is False


class TestQueueFullRejection:
    """Queue at max → immediate StateLoomRateLimitError."""

    async def test_queue_full_rejection(self):
        team = _make_team(tps=1.0, max_queue=2, queue_timeout=5.0)
        teams = {team.id: team}
        mw = RateLimiterMiddleware(team_lookup=lambda tid: teams.get(tid))

        async def slow_next(c):
            await asyncio.sleep(10)
            return "ok"

        # Exhaust the token
        ctx1 = _make_ctx(session=Session(id="s1", team_id="team-1"))
        task1 = asyncio.create_task(mw.process(ctx1, slow_next))
        await asyncio.sleep(0.05)

        # Fill the queue (2 slots)
        ctx2 = _make_ctx(session=Session(id="s2", team_id="team-1"))
        task2 = asyncio.create_task(mw.process(ctx2, slow_next))
        await asyncio.sleep(0.05)

        ctx3 = _make_ctx(session=Session(id="s3", team_id="team-1"))
        task3 = asyncio.create_task(mw.process(ctx3, slow_next))
        await asyncio.sleep(0.05)

        # Third queued request should be rejected (queue is full at 2)
        ctx4 = _make_ctx(session=Session(id="s4", team_id="team-1"))
        with pytest.raises(StateLoomRateLimitError, match="queue full"):
            await mw.process(ctx4, slow_next)

        # Check rejection event
        rl_events = [e for e in ctx4.events if isinstance(e, RateLimitEvent)]
        assert len(rl_events) == 1
        assert rl_events[0].rejected is True

        # Cancel background tasks
        task1.cancel()
        task2.cancel()
        task3.cancel()
        await asyncio.sleep(0.1)


class TestQueueTimeout:
    """Timeout → StateLoomRateLimitError."""

    async def test_queue_timeout(self):
        team = _make_team(tps=1.0, queue_timeout=0.3)
        teams = {team.id: team}
        mw = RateLimiterMiddleware(team_lookup=lambda tid: teams.get(tid))

        async def slow_next(c):
            await asyncio.sleep(10)
            return "ok"

        # Exhaust the token
        ctx1 = _make_ctx(session=Session(id="s1", team_id="team-1"))
        task1 = asyncio.create_task(mw.process(ctx1, slow_next))
        await asyncio.sleep(0.05)

        # Second request should timeout
        ctx2 = _make_ctx(session=Session(id="s2", team_id="team-1"))
        with pytest.raises(StateLoomRateLimitError, match="queue timeout"):
            await mw.process(ctx2, slow_next)

        rl_events = [e for e in ctx2.events if isinstance(e, RateLimitEvent)]
        assert len(rl_events) == 1
        assert rl_events[0].timed_out is True
        assert rl_events[0].queued is True

        task1.cancel()
        await asyncio.sleep(0.1)


class TestTokenRefill:
    """Tokens refill after waiting."""

    async def test_token_refill(self):
        bucket = _TokenBucket(tps=10.0)
        # Drain all tokens
        with bucket.lock:
            bucket.tokens = 0.0
            bucket.last_refill = time.monotonic()

        # Wait a bit for refill
        await asyncio.sleep(0.2)

        with bucket.lock:
            bucket.refill()
            assert bucket.tokens >= 1.0

    async def test_refill_capped_at_tps(self):
        bucket = _TokenBucket(tps=5.0)
        with bucket.lock:
            bucket.tokens = 0.0
            bucket.last_refill = time.monotonic() - 10.0  # Long ago
            bucket.refill()
            # Should be capped at tps
            assert bucket.tokens == 5.0


class TestHotReloadConfig:
    """Dashboard TPS change takes effect on next request."""

    async def test_hot_reload_config(self):
        team = _make_team(tps=100.0)
        teams = {team.id: team}
        mw = RateLimiterMiddleware(team_lookup=lambda tid: teams.get(tid))

        async def call_next(c):
            return "ok"

        # First request with TPS=100
        ctx = _make_ctx()
        await mw.process(ctx, call_next)

        # Change TPS via team object (simulating dashboard update)
        team.rate_limit_tps = 1.0
        team.rate_limit_priority = 5

        # Next request should use new TPS
        ctx2 = _make_ctx(session=Session(id="sess-2", team_id="team-1"))
        await mw.process(ctx2, call_next)

        bucket = mw._get_or_create_bucket(team)
        with bucket.lock:
            assert bucket.tps == 1.0
            assert bucket.priority == 5


class TestEventRecorded:
    """RateLimitEvent created with correct fields."""

    async def test_event_recorded(self):
        team = _make_team(tps=10.0)
        teams = {team.id: team}
        mw = RateLimiterMiddleware(team_lookup=lambda tid: teams.get(tid))

        ctx = _make_ctx()

        async def call_next(c):
            return "ok"

        await mw.process(ctx, call_next)

        rl_events = [e for e in ctx.events if isinstance(e, RateLimitEvent)]
        assert len(rl_events) == 1
        event = rl_events[0]
        assert event.team_id == "team-1"
        assert event.queued is False
        assert event.rejected is False
        assert event.timed_out is False
        assert event.session_id == "sess-1"


class TestGetStatus:
    """Returns queue depths + token counts."""

    async def test_get_status(self):
        team = _make_team(tps=5.0, priority=3)
        teams = {team.id: team}
        mw = RateLimiterMiddleware(team_lookup=lambda tid: teams.get(tid))

        async def call_next(c):
            return "ok"

        ctx = _make_ctx()
        await mw.process(ctx, call_next)

        status = mw.get_status()
        assert "teams" in status
        assert "team-1" in status["teams"]
        team_status = status["teams"]["team-1"]
        assert team_status["tps"] == 5.0
        assert team_status["priority"] == 3
        assert team_status["tokens_available"] <= 5.0
        assert team_status["queue_size"] == 0

    async def test_get_status_empty(self):
        mw = RateLimiterMiddleware(team_lookup=lambda tid: None)
        status = mw.get_status()
        assert status == {"teams": {}}


class TestRemoveBucket:
    """Cleanup after TPS removed."""

    async def test_remove_bucket(self):
        team = _make_team(tps=5.0)
        teams = {team.id: team}
        mw = RateLimiterMiddleware(team_lookup=lambda tid: teams.get(tid))

        async def call_next(c):
            return "ok"

        ctx = _make_ctx()
        await mw.process(ctx, call_next)

        assert "team-1" in mw.get_status()["teams"]
        mw.remove_bucket("team-1")
        assert "team-1" not in mw.get_status()["teams"]

    async def test_remove_nonexistent_bucket(self):
        mw = RateLimiterMiddleware(team_lookup=lambda tid: None)
        # Should not raise
        mw.remove_bucket("nonexistent")


class TestBlastRadiusExclusion:
    """StateLoomRateLimitError should not count as a failure in blast radius."""

    async def test_blast_radius_excludes_rate_limit(self):
        from stateloom.middleware.blast_radius import BlastRadiusMiddleware

        config = _make_config(
            blast_radius_enabled=True,
            blast_radius_consecutive_failures=3,
        )
        store = MemoryStore()
        blast_mw = BlastRadiusMiddleware(config, store)

        session = Session(id="test-session", team_id="team-1")
        ctx = MiddlewareContext(
            session=session,
            config=config,
            provider="openai",
            model="gpt-4",
            request_kwargs={"messages": [{"role": "user", "content": "hi"}]},
        )

        async def rate_limit_next(c):
            raise StateLoomRateLimitError(team_id="team-1", tps=1.0, queue_size=100)

        # Should re-raise without counting as a failure
        with pytest.raises(StateLoomRateLimitError):
            await blast_mw.process(ctx, rate_limit_next)

        # Failure count should not have increased
        assert blast_mw._failure_counts.get("test-session", 0) == 0
