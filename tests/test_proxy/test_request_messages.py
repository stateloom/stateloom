"""Tests for request messages persistence on LLMCallEvents."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock

import pytest
from stateloom.core.config import StateLoomConfig
from stateloom.core.event import LLMCallEvent
from stateloom.middleware.base import MiddlewareContext
from stateloom.middleware.cost_tracker import CostTracker
from stateloom.pricing.registry import PricingRegistry
from stateloom.store.memory_store import MemoryStore
from stateloom.store.sqlite_store import SQLiteStore

# ── CostTracker population tests ────────────────────────────────────


class TestCostTrackerRequestMessages:
    """Test that CostTracker populates request_messages_json."""

    def _make_ctx(
        self,
        *,
        store_payloads: bool = True,
        messages: list | None = None,
        zero_retention: bool = False,
    ) -> MiddlewareContext:
        config = StateLoomConfig(store_payloads=store_payloads)
        session = MagicMock()
        session.id = "test-session"
        session.step_counter = 1
        session.durable = False
        session.metadata = {}
        if zero_retention:
            session.metadata["_compliance_zero_retention"] = True
        session.add_cost = MagicMock()

        ctx = MiddlewareContext(
            provider="openai",
            model="gpt-4o",
            request_kwargs={"messages": messages or [{"role": "user", "content": "hello"}]},
            session=session,
            config=config,
        )
        ctx.response = MagicMock()
        ctx.response.usage = MagicMock()
        ctx.response.usage.prompt_tokens = 10
        ctx.response.usage.completion_tokens = 5
        return ctx

    @pytest.mark.asyncio
    async def test_messages_stored_when_payloads_enabled(self) -> None:
        pricing = PricingRegistry()
        pricing.register("gpt-4o", input_per_token=0.0, output_per_token=0.0)
        tracker = CostTracker(pricing)

        messages = [
            {"role": "system", "content": "You are helpful"},
            {"role": "user", "content": "What is 2+2?"},
        ]
        ctx = self._make_ctx(store_payloads=True, messages=messages)

        async def noop(c: MiddlewareContext) -> None:
            return None

        await tracker.process(ctx, noop)

        assert len(ctx.events) == 1
        event = ctx.events[0]
        assert isinstance(event, LLMCallEvent)
        assert event.request_messages_json is not None
        parsed = json.loads(event.request_messages_json)
        assert len(parsed) == 2
        assert parsed[1]["content"] == "What is 2+2?"

    @pytest.mark.asyncio
    async def test_messages_not_stored_when_payloads_disabled(self) -> None:
        pricing = PricingRegistry()
        pricing.register("gpt-4o", input_per_token=0.0, output_per_token=0.0)
        tracker = CostTracker(pricing)

        ctx = self._make_ctx(store_payloads=False)

        async def noop(c: MiddlewareContext) -> None:
            return None

        await tracker.process(ctx, noop)

        event = ctx.events[0]
        assert event.request_messages_json is None

    @pytest.mark.asyncio
    async def test_messages_not_stored_for_tool_continuation(self) -> None:
        pricing = PricingRegistry()
        pricing.register("gpt-4o", input_per_token=0.0, output_per_token=0.0)
        tracker = CostTracker(pricing)

        # Tool continuation: last message has role="tool"
        messages = [
            {"role": "user", "content": "call the tool"},
            {"role": "assistant", "content": "", "tool_calls": [{"id": "c1"}]},
            {"role": "tool", "tool_call_id": "c1", "content": "result"},
        ]
        ctx = self._make_ctx(store_payloads=True, messages=messages)

        async def noop(c: MiddlewareContext) -> None:
            return None

        await tracker.process(ctx, noop)

        event = ctx.events[0]
        assert isinstance(event, LLMCallEvent)
        assert event.request_messages_json is None

    @pytest.mark.asyncio
    async def test_messages_not_stored_with_zero_retention(self) -> None:
        pricing = PricingRegistry()
        pricing.register("gpt-4o", input_per_token=0.0, output_per_token=0.0)
        tracker = CostTracker(pricing)

        ctx = self._make_ctx(store_payloads=True, zero_retention=True)

        async def noop(c: MiddlewareContext) -> None:
            return None

        await tracker.process(ctx, noop)

        event = ctx.events[0]
        assert event.request_messages_json is None


# ── Store layer tests ────────────────────────────────────────────────


class TestMemoryStoreRequestMessages:
    """Test MemoryStore get_event_messages and cleanup_request_messages."""

    def test_get_event_messages(self) -> None:
        store = MemoryStore()
        event = LLMCallEvent(
            id="evt-1",
            session_id="s1",
            request_messages_json='[{"role":"user","content":"hi"}]',
        )
        store.save_event(event)
        raw = store.get_event_messages("evt-1")
        assert raw is not None
        assert json.loads(raw) == [{"role": "user", "content": "hi"}]

    def test_get_event_messages_not_found(self) -> None:
        store = MemoryStore()
        assert store.get_event_messages("nonexistent") is None

    def test_get_event_messages_none_field(self) -> None:
        store = MemoryStore()
        event = LLMCallEvent(id="evt-2", session_id="s1")
        store.save_event(event)
        assert store.get_event_messages("evt-2") is None

    def test_cleanup_request_messages(self) -> None:
        store = MemoryStore()
        old_ts = datetime.now(timezone.utc) - timedelta(hours=48)
        event_old = LLMCallEvent(
            id="evt-old",
            session_id="s1",
            timestamp=old_ts,
            request_messages_json='[{"role":"user","content":"old"}]',
        )
        event_new = LLMCallEvent(
            id="evt-new",
            session_id="s1",
            request_messages_json='[{"role":"user","content":"new"}]',
        )
        store.save_event(event_old)
        store.save_event(event_new)

        cleaned = store.cleanup_request_messages(retention_hours=24)
        assert cleaned == 1
        assert store.get_event_messages("evt-old") is None
        assert store.get_event_messages("evt-new") is not None


class TestSQLiteStoreRequestMessages:
    """Test SQLiteStore get_event_messages and cleanup_request_messages."""

    def test_get_event_messages(self, tmp_path: object) -> None:
        store = SQLiteStore(str(tmp_path) + "/test.db")  # type: ignore[arg-type]
        event = LLMCallEvent(
            id="evt-1",
            session_id="s1",
            request_messages_json='[{"role":"user","content":"hello"}]',
        )
        store.save_event(event)
        raw = store.get_event_messages("evt-1")
        assert raw is not None
        assert json.loads(raw) == [{"role": "user", "content": "hello"}]

    def test_get_event_messages_not_found(self, tmp_path: object) -> None:
        store = SQLiteStore(str(tmp_path) + "/test.db")  # type: ignore[arg-type]
        assert store.get_event_messages("nonexistent") is None

    def test_get_event_messages_excluded_from_bulk_query(self, tmp_path: object) -> None:
        """Full request_messages_json payload should NOT appear in get_session_events;
        only a lightweight boolean sentinel '1' is returned."""
        store = SQLiteStore(str(tmp_path) + "/test.db")  # type: ignore[arg-type]
        event = LLMCallEvent(
            id="evt-bulk",
            session_id="s-bulk",
            request_messages_json='[{"role":"user","content":"big payload"}]',
        )
        store.save_event(event)
        events = store.get_session_events("s-bulk")
        assert len(events) == 1
        # The field contains a lightweight sentinel, not the full payload
        assert events[0].request_messages_json == "1"
        # Full payload is available via lazy-load
        raw = store.get_event_messages("evt-bulk")
        assert raw is not None
        assert json.loads(raw) == [{"role": "user", "content": "big payload"}]

    def test_cleanup_request_messages(self, tmp_path: object) -> None:
        store = SQLiteStore(str(tmp_path) + "/test.db")  # type: ignore[arg-type]
        old_ts = datetime.now(timezone.utc) - timedelta(hours=48)
        event_old = LLMCallEvent(
            id="evt-old",
            session_id="s1",
            timestamp=old_ts,
            request_messages_json='[{"role":"user","content":"old"}]',
        )
        event_new = LLMCallEvent(
            id="evt-new",
            session_id="s1",
            request_messages_json='[{"role":"user","content":"new"}]',
        )
        store.save_event(event_old)
        store.save_event(event_new)

        cleaned = store.cleanup_request_messages(retention_hours=24)
        assert cleaned == 1
        assert store.get_event_messages("evt-old") is None
        assert store.get_event_messages("evt-new") is not None


# ── LLMCallEvent field tests ─────────────────────────────────────────


class TestLLMCallEventField:
    """Test the request_messages_json field on LLMCallEvent."""

    def test_default_none(self) -> None:
        event = LLMCallEvent()
        assert event.request_messages_json is None

    def test_roundtrip(self) -> None:
        msgs = [{"role": "user", "content": "test"}]
        event = LLMCallEvent(request_messages_json=json.dumps(msgs))
        assert event.request_messages_json is not None
        assert json.loads(event.request_messages_json) == msgs

    def test_model_dump_includes_field(self) -> None:
        event = LLMCallEvent(request_messages_json='[{"role":"user","content":"x"}]')
        d = event.model_dump()
        assert "request_messages_json" in d
        assert d["request_messages_json"] is not None

    def test_model_dump_excludes_when_none(self) -> None:
        event = LLMCallEvent()
        d = event.model_dump()
        # Field exists but is None
        assert d["request_messages_json"] is None


# ── Gate auto-cleanup tests ──────────────────────────────────────────


class TestGatePromptCleanup:
    """Test gate._maybe_cleanup_prompt_payloads throttling."""

    def test_cleanup_skipped_when_payloads_disabled(self) -> None:
        from stateloom.gate import Gate

        config = StateLoomConfig(store_payloads=False, store_backend="memory", dashboard=False)
        gate = Gate(config)
        gate._setup_middleware()

        gate.store.cleanup_request_messages = MagicMock(return_value=0)  # type: ignore[attr-defined]
        gate._maybe_cleanup_prompt_payloads()
        gate.store.cleanup_request_messages.assert_not_called()  # type: ignore[attr-defined]

    def test_cleanup_throttled(self) -> None:
        from stateloom.gate import Gate

        config = StateLoomConfig(store_payloads=True, store_backend="memory", dashboard=False)
        gate = Gate(config)
        gate._setup_middleware()

        gate.store.cleanup_request_messages = MagicMock(return_value=0)  # type: ignore[attr-defined]

        # First call should run
        gate._last_prompt_cleanup = 0.0
        gate._maybe_cleanup_prompt_payloads()
        assert gate.store.cleanup_request_messages.call_count == 1  # type: ignore[attr-defined]

        # Second call should be throttled
        gate._maybe_cleanup_prompt_payloads()
        assert gate.store.cleanup_request_messages.call_count == 1  # type: ignore[attr-defined]

    def test_cleanup_prompt_payloads_public(self) -> None:
        from stateloom.gate import Gate

        config = StateLoomConfig(
            store_payloads=True,
            store_prompt_retention_hours=12,
            store_backend="memory",
            dashboard=False,
        )
        gate = Gate(config)
        gate._setup_middleware()

        gate.store.cleanup_request_messages = MagicMock(return_value=5)  # type: ignore[attr-defined]
        result = gate.cleanup_prompt_payloads()
        assert result == 5
        gate.store.cleanup_request_messages.assert_called_once_with(retention_hours=12)  # type: ignore[attr-defined]
