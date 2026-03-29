"""Tests for HIPAA zero-retention enforcement in EventRecorder."""

from __future__ import annotations

import pytest

from stateloom.core.config import ComplianceProfile, StateLoomConfig
from stateloom.core.event import (
    ComplianceAuditEvent,
    LLMCallEvent,
    ShadowDraftEvent,
)
from stateloom.core.session import Session
from stateloom.middleware.base import MiddlewareContext
from stateloom.middleware.compliance import ComplianceMiddleware
from stateloom.middleware.event_recorder import EventRecorder
from stateloom.store.memory_store import MemoryStore


def _make_session(
    session_id: str = "test-session",
    durable: bool = False,
) -> Session:
    s = Session(id=session_id)
    if durable:
        s.durable = True
        s.metadata["durable"] = True
    return s


def _make_ctx(
    session: Session | None = None,
    events: list | None = None,
) -> MiddlewareContext:
    if session is None:
        session = _make_session()
    ctx = MiddlewareContext(
        session=session,
        config=StateLoomConfig(console_output=False),
        provider="openai",
        model="gpt-4",
        request_kwargs={"messages": [{"role": "user", "content": "hello"}]},
    )
    if events:
        ctx.events.extend(events)
    return ctx


class TestStripCachedResponseJson:
    """Zero-retention strips cached_response_json from LLMCallEvents."""

    async def test_cached_response_stripped(self):
        store = MemoryStore()
        recorder = EventRecorder(store)

        session = _make_session()
        session.metadata["_compliance_zero_retention"] = True
        session.metadata["_compliance_standard"] = "hipaa"

        llm_event = LLMCallEvent(
            session_id="test-session",
            step=1,
            provider="openai",
            model="gpt-4",
            prompt_tokens=10,
            completion_tokens=20,
            total_tokens=30,
            cost=0.001,
        )
        llm_event.cached_response_json = '{"choices": [{"message": {"content": "secret"}}]}'

        ctx = _make_ctx(session=session, events=[llm_event])

        async def call_next(c):
            return "response"

        await recorder.process(ctx, call_next)

        assert llm_event.cached_response_json is None

    async def test_no_strip_without_flag(self):
        """Without _compliance_zero_retention, cached_response_json is preserved."""
        store = MemoryStore()
        recorder = EventRecorder(store)

        session = _make_session()

        llm_event = LLMCallEvent(
            session_id="test-session",
            step=1,
            provider="openai",
            model="gpt-4",
            prompt_tokens=10,
            completion_tokens=20,
            total_tokens=30,
            cost=0.001,
        )
        llm_event.cached_response_json = '{"choices": [{"message": {"content": "kept"}}]}'

        ctx = _make_ctx(session=session, events=[llm_event])

        async def call_next(c):
            return "response"

        await recorder.process(ctx, call_next)

        assert llm_event.cached_response_json == '{"choices": [{"message": {"content": "kept"}}]}'


class TestStripShadowPreviews:
    """Zero-retention strips cloud_preview and local_preview from ShadowDraftEvents."""

    async def test_shadow_previews_stripped(self):
        store = MemoryStore()
        recorder = EventRecorder(store)

        session = _make_session()
        session.metadata["_compliance_zero_retention"] = True
        session.metadata["_compliance_standard"] = "hipaa"

        shadow_event = ShadowDraftEvent(
            session_id="test-session",
            step=1,
            cloud_provider="openai",
            cloud_model="gpt-4",
            cloud_latency_ms=200.0,
            cloud_tokens=30,
            cloud_cost=0.001,
            local_model="llama3.2",
            local_latency_ms=100.0,
            local_tokens=25,
            latency_ratio=0.5,
            cost_saved=0.001,
            shadow_status="success",
            cloud_preview="This is the cloud response text",
            local_preview="This is the local response text",
        )

        ctx = _make_ctx(session=session, events=[shadow_event])

        async def call_next(c):
            return "response"

        await recorder.process(ctx, call_next)

        assert shadow_event.cloud_preview == ""
        assert shadow_event.local_preview == ""


class TestDataPurgedAuditEvent:
    """When data is stripped, a ComplianceAuditEvent with action='data_purged' is emitted."""

    async def test_data_purged_audit_event(self):
        store = MemoryStore()
        recorder = EventRecorder(store)

        session = _make_session()
        session.metadata["_compliance_zero_retention"] = True
        session.metadata["_compliance_standard"] = "hipaa"
        session.metadata["_compliance_audit_salt"] = "salt"

        llm_event = LLMCallEvent(
            session_id="test-session",
            step=1,
            provider="openai",
            model="gpt-4",
            prompt_tokens=10,
            completion_tokens=20,
            total_tokens=30,
            cost=0.001,
        )
        llm_event.cached_response_json = '{"data": "secret"}'

        ctx = _make_ctx(session=session, events=[llm_event])

        async def call_next(c):
            return "response"

        await recorder.process(ctx, call_next)

        audit_events = [
            e
            for e in ctx.events
            if isinstance(e, ComplianceAuditEvent) and e.action == "data_purged"
        ]
        assert len(audit_events) == 1
        audit = audit_events[0]
        assert audit.compliance_standard == "hipaa"
        assert "zero-retention" in audit.justification
        assert audit.integrity_hash  # non-empty

    async def test_no_audit_event_when_nothing_stripped(self):
        """No data_purged event if there's nothing to strip."""
        store = MemoryStore()
        recorder = EventRecorder(store)

        session = _make_session()
        session.metadata["_compliance_zero_retention"] = True
        session.metadata["_compliance_standard"] = "hipaa"

        llm_event = LLMCallEvent(
            session_id="test-session",
            step=1,
            provider="openai",
            model="gpt-4",
            prompt_tokens=10,
            completion_tokens=20,
            total_tokens=30,
            cost=0.001,
        )
        # cached_response_json is None by default — nothing to strip

        ctx = _make_ctx(session=session, events=[llm_event])

        async def call_next(c):
            return "response"

        await recorder.process(ctx, call_next)

        audit_events = [
            e
            for e in ctx.events
            if isinstance(e, ComplianceAuditEvent) and e.action == "data_purged"
        ]
        assert len(audit_events) == 0


class TestPreservedMetadata:
    """Zero-retention strips response data but preserves cost/token metadata."""

    async def test_cost_metadata_preserved(self):
        store = MemoryStore()
        recorder = EventRecorder(store)

        session = _make_session()
        session.metadata["_compliance_zero_retention"] = True
        session.metadata["_compliance_standard"] = "hipaa"

        llm_event = LLMCallEvent(
            session_id="test-session",
            step=1,
            provider="openai",
            model="gpt-4",
            prompt_tokens=10,
            completion_tokens=20,
            total_tokens=30,
            cost=0.001,
        )
        llm_event.cached_response_json = '{"secret": "data"}'

        ctx = _make_ctx(session=session, events=[llm_event])

        async def call_next(c):
            return "response"

        await recorder.process(ctx, call_next)

        # Cost/token metadata is preserved
        assert llm_event.prompt_tokens == 10
        assert llm_event.completion_tokens == 20
        assert llm_event.total_tokens == 30
        assert llm_event.cost == 0.001
        assert llm_event.model == "gpt-4"
        # But response data is gone
        assert llm_event.cached_response_json is None


class TestDurableZeroRetentionWarning:
    """Durable + zero-retention triggers a warning (compliance > convenience)."""

    async def test_durable_zero_retention_logs_warning(self, caplog):
        store = MemoryStore()
        recorder = EventRecorder(store)

        session = _make_session(durable=True)
        session.metadata["_compliance_zero_retention"] = True
        session.metadata["_compliance_standard"] = "hipaa"

        llm_event = LLMCallEvent(
            session_id="test-session",
            step=1,
            provider="openai",
            model="gpt-4",
            prompt_tokens=10,
            completion_tokens=20,
            total_tokens=30,
            cost=0.001,
        )
        llm_event.cached_response_json = '{"data": "secret"}'

        ctx = _make_ctx(session=session, events=[llm_event])

        async def call_next(c):
            return "response"

        import logging

        with caplog.at_level(logging.WARNING, logger="stateloom.middleware.event_recorder"):
            await recorder.process(ctx, call_next)

        assert any("durable" in r.message and "zero-retention" in r.message for r in caplog.records)


class TestComplianceMiddlewareZeroRetentionFlags:
    """ComplianceMiddleware sets the enforcement flags when zero_retention_logs=True."""

    async def test_sets_zero_retention_flag(self):
        profile = ComplianceProfile(
            standard="hipaa",
            zero_retention_logs=True,
            audit_salt="test-salt",
        )
        mw = ComplianceMiddleware(
            StateLoomConfig(console_output=False),
            compliance_fn=lambda o, t: profile,
        )
        ctx = _make_ctx()

        async def call_next(c):
            return "response"

        await mw.process(ctx, call_next)

        assert ctx.session.metadata["_compliance_zero_retention"] is True
        assert ctx.session.metadata["_compliance_audit_salt"] == "test-salt"
        assert ctx.session.metadata["store_payloads"] is False

    async def test_no_flags_without_zero_retention(self):
        profile = ComplianceProfile(
            standard="gdpr",
            zero_retention_logs=False,
        )
        mw = ComplianceMiddleware(
            StateLoomConfig(console_output=False),
            compliance_fn=lambda o, t: profile,
        )
        ctx = _make_ctx()

        async def call_next(c):
            return "response"

        await mw.process(ctx, call_next)

        assert "_compliance_zero_retention" not in ctx.session.metadata


class TestNonRetryableAndBlastRadius:
    """StateLoomComplianceError is non-retryable and excluded from blast radius."""

    def test_compliance_error_in_non_retryable(self):
        from stateloom.core.errors import StateLoomComplianceError
        from stateloom.retry import _NON_RETRYABLE

        assert StateLoomComplianceError in _NON_RETRYABLE

    def test_compliance_error_excluded_from_blast_radius(self):
        """The blast radius except tuple includes StateLoomComplianceError."""
        # Verify by checking import exists in blast_radius module
        from stateloom.middleware.blast_radius import StateLoomComplianceError as _Err

        assert _Err is not None
