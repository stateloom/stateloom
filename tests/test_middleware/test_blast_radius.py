"""Tests for the blast radius containment middleware."""

from __future__ import annotations

import time
from unittest.mock import patch

import pytest

from stateloom.core.config import StateLoomConfig
from stateloom.core.errors import (
    StateLoomBlastRadiusError,
    StateLoomBudgetError,
    StateLoomKillSwitchError,
)
from stateloom.core.event import BlastRadiusEvent
from stateloom.core.session import Session
from stateloom.core.types import EventType, SessionStatus
from stateloom.middleware.base import MiddlewareContext
from stateloom.middleware.blast_radius import BlastRadiusMiddleware
from stateloom.store.memory_store import MemoryStore


def _make_config(**overrides) -> StateLoomConfig:
    defaults = {
        "blast_radius_enabled": True,
        "blast_radius_consecutive_failures": 3,
        "blast_radius_budget_violations_per_hour": 5,
        "console_output": False,
    }
    defaults.update(overrides)
    return StateLoomConfig(**defaults)


def _make_ctx(
    session_id: str = "test-session",
    model: str = "gpt-4",
    agent_name: str | None = None,
    config: StateLoomConfig | None = None,
) -> MiddlewareContext:
    session = Session(id=session_id)
    if agent_name:
        session.agent_name = agent_name
        session.metadata["agent_name"] = agent_name
    return MiddlewareContext(
        session=session,
        config=config or _make_config(),
        provider="openai",
        model=model,
        request_kwargs={"messages": [{"role": "user", "content": "hello"}]},
    )


class TestPassThrough:
    """Successful calls pass through without issue."""

    async def test_success_passes_through(self):
        config = _make_config()
        mw = BlastRadiusMiddleware(config)
        ctx = _make_ctx()

        async def call_next(c):
            return "response"

        result = await mw.process(ctx, call_next)
        assert result == "response"

    async def test_success_resets_failure_counter(self):
        config = _make_config(blast_radius_consecutive_failures=3)
        mw = BlastRadiusMiddleware(config)
        ctx = _make_ctx()

        # Record 2 failures (below threshold)
        for _ in range(2):
            with pytest.raises(RuntimeError):

                async def fail_next(c):
                    raise RuntimeError("fail")

                await mw.process(ctx, fail_next)

        assert mw._failure_counts["test-session"] == 2

        # Now a success
        async def ok_next(c):
            return "ok"

        await mw.process(ctx, ok_next)
        assert "test-session" not in mw._failure_counts


class TestConsecutiveFailures:
    """Session is paused after N consecutive failures."""

    async def test_below_threshold_does_not_pause(self):
        config = _make_config(blast_radius_consecutive_failures=5)
        mw = BlastRadiusMiddleware(config)

        for i in range(4):
            ctx = _make_ctx()
            with pytest.raises(RuntimeError):

                async def fail_next(c):
                    raise RuntimeError("fail")

                await mw.process(ctx, fail_next)

        assert "test-session" not in mw._paused_sessions

    async def test_threshold_pauses_session(self):
        config = _make_config(blast_radius_consecutive_failures=3)
        store = MemoryStore()
        mw = BlastRadiusMiddleware(config, store)

        for i in range(2):
            ctx = _make_ctx()
            with pytest.raises(RuntimeError):

                async def fail_next(c):
                    raise RuntimeError("fail")

                await mw.process(ctx, fail_next)

        # 3rd failure should trigger pause
        ctx = _make_ctx()
        with pytest.raises(StateLoomBlastRadiusError):

            async def fail_next(c):
                raise RuntimeError("fail")

            await mw.process(ctx, fail_next)

        assert "test-session" in mw._paused_sessions

    async def test_paused_session_blocks_immediately(self):
        config = _make_config(blast_radius_consecutive_failures=3)
        mw = BlastRadiusMiddleware(config)

        # Pause the session
        for i in range(3):
            ctx = _make_ctx()
            try:

                async def fail_next(c):
                    raise RuntimeError("fail")

                await mw.process(ctx, fail_next)
            except (RuntimeError, StateLoomBlastRadiusError):
                pass

        # Now any new call should be blocked immediately
        ctx = _make_ctx()
        with pytest.raises(StateLoomBlastRadiusError):

            async def ok_next(c):
                return "ok"

            await mw.process(ctx, ok_next)

    async def test_event_saved_on_pause(self):
        config = _make_config(blast_radius_consecutive_failures=2)
        store = MemoryStore()
        mw = BlastRadiusMiddleware(config, store)

        # Trigger pause
        for i in range(2):
            ctx = _make_ctx()
            try:

                async def fail_next(c):
                    raise RuntimeError("fail")

                await mw.process(ctx, fail_next)
            except (RuntimeError, StateLoomBlastRadiusError):
                pass

        events = store.get_session_events("test-session", event_type="blast_radius")
        assert len(events) == 1
        event = events[0]
        assert isinstance(event, BlastRadiusEvent)
        assert event.trigger == "consecutive_failures"
        assert event.count == 2
        assert event.threshold == 2

    async def test_session_status_set_to_paused(self):
        config = _make_config(blast_radius_consecutive_failures=1)
        store = MemoryStore()
        mw = BlastRadiusMiddleware(config, store)
        ctx = _make_ctx()

        with pytest.raises(StateLoomBlastRadiusError):

            async def fail_next(c):
                raise RuntimeError("fail")

            await mw.process(ctx, fail_next)

        assert ctx.session.status == SessionStatus.PAUSED


class TestBudgetViolations:
    """Budget violations tracked in sliding window."""

    async def test_budget_violation_below_threshold(self):
        config = _make_config(blast_radius_budget_violations_per_hour=5)
        mw = BlastRadiusMiddleware(config)

        for _ in range(4):
            ctx = _make_ctx()
            with pytest.raises(StateLoomBudgetError):

                async def budget_fail(c):
                    raise StateLoomBudgetError(limit=1.0, spent=2.0, session_id="test-session")

                await mw.process(ctx, budget_fail)

        assert "test-session" not in mw._paused_sessions

    async def test_budget_violation_at_threshold(self):
        config = _make_config(blast_radius_budget_violations_per_hour=3)
        mw = BlastRadiusMiddleware(config)

        for i in range(2):
            ctx = _make_ctx()
            with pytest.raises(StateLoomBudgetError):

                async def budget_fail(c):
                    raise StateLoomBudgetError(limit=1.0, spent=2.0, session_id="test-session")

                await mw.process(ctx, budget_fail)

        # 3rd should trigger pause
        ctx = _make_ctx()
        with pytest.raises(StateLoomBlastRadiusError):

            async def budget_fail(c):
                raise StateLoomBudgetError(limit=1.0, spent=2.0, session_id="test-session")

            await mw.process(ctx, budget_fail)

        assert "test-session" in mw._paused_sessions

    async def test_budget_violations_expire_after_one_hour(self):
        config = _make_config(blast_radius_budget_violations_per_hour=3)
        mw = BlastRadiusMiddleware(config)

        # Add old timestamps (more than 1 hour ago)
        old_time = time.time() - 3700
        mw._budget_violations["test-session"] = [old_time, old_time]

        # New violation — only 1 in window (old ones expired)
        ctx = _make_ctx()
        with pytest.raises(StateLoomBudgetError):

            async def budget_fail(c):
                raise StateLoomBudgetError(limit=1.0, spent=2.0, session_id="test-session")

            await mw.process(ctx, budget_fail)

        assert "test-session" not in mw._paused_sessions
        # Only the fresh timestamp remains
        assert len(mw._budget_violations["test-session"]) == 1


class TestExcludedErrors:
    """Kill switch and blast radius errors don't count as failures."""

    async def test_kill_switch_error_not_counted(self):
        config = _make_config(blast_radius_consecutive_failures=1)
        mw = BlastRadiusMiddleware(config)
        ctx = _make_ctx()

        with pytest.raises(StateLoomKillSwitchError):

            async def ks_fail(c):
                raise StateLoomKillSwitchError("killed")

            await mw.process(ctx, ks_fail)

        assert mw._failure_counts.get("test-session", 0) == 0
        assert "test-session" not in mw._paused_sessions

    async def test_blast_radius_error_not_counted(self):
        config = _make_config(blast_radius_consecutive_failures=1)
        mw = BlastRadiusMiddleware(config)
        ctx = _make_ctx()

        with pytest.raises(StateLoomBlastRadiusError):

            async def br_fail(c):
                raise StateLoomBlastRadiusError(session_id="test-session", trigger="test")

            await mw.process(ctx, br_fail)

        assert mw._failure_counts.get("test-session", 0) == 0


class TestWebhook:
    """Webhook fired on pause."""

    async def test_webhook_fired_on_pause(self):
        config = _make_config(
            blast_radius_consecutive_failures=1,
            blast_radius_webhook_url="https://hooks.example.com/blast",
        )
        store = MemoryStore()
        mw = BlastRadiusMiddleware(config, store)
        ctx = _make_ctx()

        with patch.object(BlastRadiusMiddleware, "_fire_webhook") as mock_webhook:
            with pytest.raises(StateLoomBlastRadiusError):

                async def fail_next(c):
                    raise RuntimeError("fail")

                await mw.process(ctx, fail_next)

            # Webhook is fired in a thread, but we patched the static method
            # The event should record webhook_fired=True
            events = store.get_session_events("test-session", event_type="blast_radius")
            assert len(events) == 1
            assert events[0].webhook_fired is True
            assert events[0].webhook_url == "https://hooks.example.com/blast"


class TestAgentTracking:
    """Cross-session agent tracking."""

    async def test_agent_identity_from_metadata(self):
        config = _make_config()
        mw = BlastRadiusMiddleware(config)
        ctx = _make_ctx(agent_name="ticket-agent")
        agent_id = mw._get_agent_identity(ctx)
        assert agent_id == "agent:ticket-agent"

    async def test_agent_identity_fallback_to_model(self):
        config = _make_config()
        mw = BlastRadiusMiddleware(config)
        ctx = _make_ctx(model="gpt-4")
        agent_id = mw._get_agent_identity(ctx)
        assert agent_id == "model:gpt-4"

    async def test_cross_session_agent_pause(self):
        """Same agent identity across different sessions triggers agent pause."""
        config = _make_config(blast_radius_consecutive_failures=3)
        mw = BlastRadiusMiddleware(config)

        # 3 failures from different sessions but same agent
        for i in range(3):
            ctx = _make_ctx(session_id=f"session-{i}", agent_name="bad-agent")
            try:

                async def fail_next(c):
                    raise RuntimeError("fail")

                await mw.process(ctx, fail_next)
            except (RuntimeError, StateLoomBlastRadiusError):
                pass

        assert "agent:bad-agent" in mw._paused_agents

    async def test_agent_pause_blocks_all_sessions(self):
        """Once an agent is paused, any session with that agent is blocked."""
        config = _make_config(blast_radius_consecutive_failures=2)
        mw = BlastRadiusMiddleware(config)

        # Pause the agent
        for i in range(2):
            ctx = _make_ctx(session_id=f"session-{i}", agent_name="bad-agent")
            try:

                async def fail_next(c):
                    raise RuntimeError("fail")

                await mw.process(ctx, fail_next)
            except (RuntimeError, StateLoomBlastRadiusError):
                pass

        # New session with same agent should be blocked
        ctx = _make_ctx(session_id="session-new", agent_name="bad-agent")
        with pytest.raises(StateLoomBlastRadiusError):

            async def ok_next(c):
                return "ok"

            await mw.process(ctx, ok_next)

    async def test_agent_event_includes_agent_id(self):
        config = _make_config(blast_radius_consecutive_failures=1)
        store = MemoryStore()
        mw = BlastRadiusMiddleware(config, store)
        ctx = _make_ctx(agent_name="my-agent")

        with pytest.raises(StateLoomBlastRadiusError):

            async def fail_next(c):
                raise RuntimeError("fail")

            await mw.process(ctx, fail_next)

        events = store.get_session_events("test-session", event_type="blast_radius")
        assert len(events) == 1
        assert events[0].agent_id == "agent:my-agent"


class TestUnpause:
    """Unpause session and agent methods."""

    async def test_unpause_session(self):
        config = _make_config(blast_radius_consecutive_failures=1)
        mw = BlastRadiusMiddleware(config)

        # Pause the session
        ctx = _make_ctx()
        try:

            async def fail_next(c):
                raise RuntimeError("fail")

            await mw.process(ctx, fail_next)
        except (RuntimeError, StateLoomBlastRadiusError):
            pass

        assert "test-session" in mw._paused_sessions

        # Unpause
        result = mw.unpause_session("test-session")
        assert result is True
        assert "test-session" not in mw._paused_sessions

        # Unpause again returns False
        result = mw.unpause_session("test-session")
        assert result is False

    async def test_unpause_agent(self):
        config = _make_config(blast_radius_consecutive_failures=2)
        mw = BlastRadiusMiddleware(config)

        # Pause the agent via cross-session failures
        for i in range(2):
            ctx = _make_ctx(session_id=f"s-{i}", agent_name="my-agent")
            try:

                async def fail_next(c):
                    raise RuntimeError("fail")

                await mw.process(ctx, fail_next)
            except (RuntimeError, StateLoomBlastRadiusError):
                pass

        assert "agent:my-agent" in mw._paused_agents

        # Unpause
        result = mw.unpause_agent("agent:my-agent")
        assert result is True
        assert "agent:my-agent" not in mw._paused_agents

    async def test_unpause_nonexistent_returns_false(self):
        config = _make_config()
        mw = BlastRadiusMiddleware(config)
        assert mw.unpause_session("nonexistent") is False
        assert mw.unpause_agent("nonexistent") is False


class TestGetStatus:
    """get_status() returns correct state."""

    async def test_empty_status(self):
        config = _make_config()
        mw = BlastRadiusMiddleware(config)
        status = mw.get_status()
        assert status["paused_sessions"] == []
        assert status["paused_agents"] == []
        assert status["session_failure_counts"] == {}
        assert status["agent_failure_counts"] == {}

    async def test_status_after_failures(self):
        config = _make_config(blast_radius_consecutive_failures=5)
        mw = BlastRadiusMiddleware(config)

        # Record 2 failures
        for _ in range(2):
            ctx = _make_ctx(agent_name="my-agent")
            with pytest.raises(RuntimeError):

                async def fail_next(c):
                    raise RuntimeError("fail")

                await mw.process(ctx, fail_next)

        status = mw.get_status()
        assert status["session_failure_counts"]["test-session"] == 2
        assert status["agent_failure_counts"]["agent:my-agent"] == 2
        assert status["paused_sessions"] == []
        assert status["paused_agents"] == []

    async def test_status_after_pause(self):
        config = _make_config(blast_radius_consecutive_failures=1)
        mw = BlastRadiusMiddleware(config)
        ctx = _make_ctx()

        try:

            async def fail_next(c):
                raise RuntimeError("fail")

            await mw.process(ctx, fail_next)
        except (RuntimeError, StateLoomBlastRadiusError):
            pass

        status = mw.get_status()
        assert "test-session" in status["paused_sessions"]
