"""Tests for the ComplianceMiddleware."""

from __future__ import annotations

import pytest

from stateloom.core.config import ComplianceProfile, PIIRule, StateLoomConfig
from stateloom.core.event import ComplianceAuditEvent, PIIDetectionEvent
from stateloom.core.session import Session
from stateloom.core.types import PIIMode
from stateloom.middleware.base import MiddlewareContext
from stateloom.middleware.compliance import ComplianceMiddleware
from stateloom.store.memory_store import MemoryStore


def _make_ctx(
    org_id: str = "",
    team_id: str = "",
    **config_overrides,
) -> MiddlewareContext:
    defaults = {"console_output": False}
    defaults.update(config_overrides)
    return MiddlewareContext(
        session=Session(id="test-session", org_id=org_id, team_id=team_id),
        config=StateLoomConfig(**defaults),
        provider="openai",
        model="gpt-4",
        request_kwargs={"messages": [{"role": "user", "content": "hello"}]},
    )


class TestNoProfilePassthrough:
    async def test_no_profile_passes_through(self):
        """No compliance profile = call_next is called normally."""
        mw = ComplianceMiddleware(StateLoomConfig(console_output=False))
        ctx = _make_ctx()
        called = False

        async def call_next(c):
            nonlocal called
            called = True
            return "response"

        result = await mw.process(ctx, call_next)
        assert result == "response"
        assert called

    async def test_none_standard_passes_through(self):
        config = StateLoomConfig(
            console_output=False,
            compliance_profile=ComplianceProfile(standard="none"),
        )
        mw = ComplianceMiddleware(config)
        ctx = _make_ctx()

        async def call_next(c):
            return "response"

        result = await mw.process(ctx, call_next)
        assert result == "response"


class TestComplianceMetadata:
    async def test_sets_compliance_standard(self):
        config = StateLoomConfig(
            console_output=False,
            compliance_profile=ComplianceProfile(standard="gdpr", region="eu"),
        )
        mw = ComplianceMiddleware(config)
        ctx = _make_ctx()

        async def call_next(c):
            return "response"

        await mw.process(ctx, call_next)
        assert ctx.session.metadata["_compliance_standard"] == "gdpr"
        assert ctx.session.metadata["_compliance_region"] == "eu"


class TestHIPAAZeroRetention:
    async def test_sets_store_payloads_false(self):
        config = StateLoomConfig(
            console_output=False,
            compliance_profile=ComplianceProfile(standard="hipaa", zero_retention_logs=True),
        )
        mw = ComplianceMiddleware(config)
        ctx = _make_ctx()

        async def call_next(c):
            return "response"

        await mw.process(ctx, call_next)
        assert ctx.session.metadata["store_payloads"] is False


class TestAuditEventsForPII:
    async def test_emits_audit_event_for_pii_block(self):
        config = StateLoomConfig(
            console_output=False,
            compliance_profile=ComplianceProfile(standard="gdpr"),
        )
        mw = ComplianceMiddleware(config)
        ctx = _make_ctx()
        ctx.session.metadata["gdpr_consent"] = True

        # Simulate downstream PII event
        pii_event = PIIDetectionEvent(
            session_id="test-session",
            pii_type="ssn",
            mode="block",
            action_taken="blocked",
        )

        async def call_next(c):
            c.events.append(pii_event)
            return "response"

        await mw.process(ctx, call_next)
        audit_events = [e for e in ctx.events if isinstance(e, ComplianceAuditEvent)]
        assert len(audit_events) == 1
        audit = audit_events[0]
        assert audit.compliance_standard == "gdpr"
        assert audit.action == "pii_blocked"
        assert "GDPR-Art-32" in audit.legal_rule
        assert audit.integrity_hash != ""

    async def test_emits_audit_event_for_pii_redact(self):
        config = StateLoomConfig(
            console_output=False,
            compliance_profile=ComplianceProfile(standard="gdpr"),
        )
        mw = ComplianceMiddleware(config)
        ctx = _make_ctx()
        ctx.session.metadata["gdpr_consent"] = True

        pii_event = PIIDetectionEvent(
            session_id="test-session",
            pii_type="email",
            mode="redact",
            action_taken="redacted",
        )

        async def call_next(c):
            c.events.append(pii_event)
            return "response"

        await mw.process(ctx, call_next)
        audit_events = [e for e in ctx.events if isinstance(e, ComplianceAuditEvent)]
        assert len(audit_events) == 1
        assert audit_events[0].action == "pii_redacted"


class TestComplianceCallback:
    async def test_uses_callback_over_config(self):
        """compliance_fn callback takes precedence over config."""
        config = StateLoomConfig(
            console_output=False,
            compliance_profile=ComplianceProfile(standard="gdpr"),
        )

        callback_profile = ComplianceProfile(standard="hipaa", zero_retention_logs=True)

        def compliance_fn(org_id, team_id):
            return callback_profile

        mw = ComplianceMiddleware(config, compliance_fn=compliance_fn)
        ctx = _make_ctx()

        async def call_next(c):
            return "response"

        await mw.process(ctx, call_next)
        assert ctx.session.metadata["_compliance_standard"] == "hipaa"

    async def test_callback_returns_none_falls_to_config(self):
        config = StateLoomConfig(
            console_output=False,
            compliance_profile=ComplianceProfile(standard="ccpa"),
        )

        def compliance_fn(org_id, team_id):
            return None

        mw = ComplianceMiddleware(config, compliance_fn=compliance_fn)
        ctx = _make_ctx()

        async def call_next(c):
            return "response"

        await mw.process(ctx, call_next)
        assert ctx.session.metadata["_compliance_standard"] == "ccpa"


class TestIntegrityHash:
    async def test_integrity_hash_with_salt(self):
        config = StateLoomConfig(
            console_output=False,
            compliance_profile=ComplianceProfile(standard="gdpr", audit_salt="enterprise-secret"),
        )
        mw = ComplianceMiddleware(config)
        ctx = _make_ctx()
        ctx.session.metadata["gdpr_consent"] = True

        pii_event = PIIDetectionEvent(
            session_id="test-session",
            pii_type="ssn",
            mode="block",
            action_taken="blocked",
        )

        async def call_next(c):
            c.events.append(pii_event)
            return "response"

        await mw.process(ctx, call_next)
        audit_events = [e for e in ctx.events if isinstance(e, ComplianceAuditEvent)]
        assert len(audit_events) == 1
        assert len(audit_events[0].integrity_hash) == 64


class TestBlockStreaming:
    async def test_block_streaming_sets_stream_false(self):
        """block_streaming profile forces stream=False in request kwargs."""
        config = StateLoomConfig(
            console_output=False,
            compliance_profile=ComplianceProfile(standard="hipaa", block_streaming=True),
        )
        mw = ComplianceMiddleware(config)
        ctx = _make_ctx()
        ctx.request_kwargs["stream"] = True

        async def call_next(c):
            return "response"

        await mw.process(ctx, call_next)
        assert ctx.request_kwargs["stream"] is False

    async def test_block_streaming_emits_audit_event(self):
        """block_streaming emits a streaming_blocked audit event."""
        config = StateLoomConfig(
            console_output=False,
            compliance_profile=ComplianceProfile(standard="hipaa", block_streaming=True),
        )
        mw = ComplianceMiddleware(config)
        ctx = _make_ctx()
        ctx.request_kwargs["stream"] = True

        async def call_next(c):
            return "response"

        await mw.process(ctx, call_next)
        audits = [e for e in ctx.events if isinstance(e, ComplianceAuditEvent)]
        streaming_audits = [a for a in audits if a.action == "streaming_blocked"]
        assert len(streaming_audits) == 1
        assert "HIPAA" in streaming_audits[0].justification

    async def test_no_block_streaming_preserves_stream(self):
        """Without block_streaming, stream=True is preserved."""
        config = StateLoomConfig(
            console_output=False,
            compliance_profile=ComplianceProfile(standard="hipaa", block_streaming=False),
        )
        mw = ComplianceMiddleware(config)
        ctx = _make_ctx()
        ctx.request_kwargs["stream"] = True

        async def call_next(c):
            return "response"

        await mw.process(ctx, call_next)
        assert ctx.request_kwargs["stream"] is True

    async def test_block_streaming_no_audit_when_not_streaming(self):
        """No audit event when request is not streaming."""
        config = StateLoomConfig(
            console_output=False,
            compliance_profile=ComplianceProfile(standard="hipaa", block_streaming=True),
        )
        mw = ComplianceMiddleware(config)
        ctx = _make_ctx()
        # stream not set in kwargs

        async def call_next(c):
            return "response"

        await mw.process(ctx, call_next)
        audits = [e for e in ctx.events if isinstance(e, ComplianceAuditEvent)]
        streaming_audits = [a for a in audits if a.action == "streaming_blocked"]
        assert len(streaming_audits) == 0


class TestGDPRConsent:
    async def test_consent_missing_emits_warning(self):
        """GDPR without consent metadata emits consent_missing audit event."""
        config = StateLoomConfig(
            console_output=False,
            compliance_profile=ComplianceProfile(standard="gdpr"),
        )
        mw = ComplianceMiddleware(config)
        ctx = _make_ctx()

        async def call_next(c):
            return "response"

        await mw.process(ctx, call_next)
        audits = [e for e in ctx.events if isinstance(e, ComplianceAuditEvent)]
        consent_audits = [a for a in audits if a.action == "consent_missing"]
        assert len(consent_audits) == 1
        assert "GDPR-Art-6" in consent_audits[0].legal_rule
        assert "consent" in consent_audits[0].justification.lower()

    async def test_consent_present_no_warning(self):
        """GDPR with consent metadata does not emit consent_missing event."""
        config = StateLoomConfig(
            console_output=False,
            compliance_profile=ComplianceProfile(standard="gdpr"),
        )
        mw = ComplianceMiddleware(config)
        ctx = _make_ctx()
        ctx.session.metadata["gdpr_consent"] = True

        async def call_next(c):
            return "response"

        await mw.process(ctx, call_next)
        audits = [e for e in ctx.events if isinstance(e, ComplianceAuditEvent)]
        consent_audits = [a for a in audits if a.action == "consent_missing"]
        assert len(consent_audits) == 0

    async def test_consent_not_checked_for_hipaa(self):
        """HIPAA profile does not emit consent_missing events."""
        config = StateLoomConfig(
            console_output=False,
            compliance_profile=ComplianceProfile(standard="hipaa"),
        )
        mw = ComplianceMiddleware(config)
        ctx = _make_ctx()

        async def call_next(c):
            return "response"

        await mw.process(ctx, call_next)
        audits = [e for e in ctx.events if isinstance(e, ComplianceAuditEvent)]
        consent_audits = [a for a in audits if a.action == "consent_missing"]
        assert len(consent_audits) == 0

    async def test_default_consent_suppresses_warning(self):
        """default_consent=True with no metadata key suppresses consent_missing."""
        config = StateLoomConfig(
            console_output=False,
            compliance_profile=ComplianceProfile(standard="gdpr", default_consent=True),
        )
        mw = ComplianceMiddleware(config)
        ctx = _make_ctx()

        async def call_next(c):
            return "response"

        await mw.process(ctx, call_next)
        audits = [e for e in ctx.events if isinstance(e, ComplianceAuditEvent)]
        consent_audits = [a for a in audits if a.action == "consent_missing"]
        assert len(consent_audits) == 0

    async def test_explicit_false_overrides_default_consent(self):
        """Explicit gdpr_consent=False overrides default_consent=True."""
        config = StateLoomConfig(
            console_output=False,
            compliance_profile=ComplianceProfile(standard="gdpr", default_consent=True),
        )
        mw = ComplianceMiddleware(config)
        ctx = _make_ctx()
        ctx.session.metadata["gdpr_consent"] = False

        async def call_next(c):
            return "response"

        await mw.process(ctx, call_next)
        audits = [e for e in ctx.events if isinstance(e, ComplianceAuditEvent)]
        consent_audits = [a for a in audits if a.action == "consent_missing"]
        assert len(consent_audits) == 1

    async def test_explicit_true_with_default_consent(self):
        """Explicit gdpr_consent=True with default_consent=True emits no warning."""
        config = StateLoomConfig(
            console_output=False,
            compliance_profile=ComplianceProfile(standard="gdpr", default_consent=True),
        )
        mw = ComplianceMiddleware(config)
        ctx = _make_ctx()
        ctx.session.metadata["gdpr_consent"] = True

        async def call_next(c):
            return "response"

        await mw.process(ctx, call_next)
        audits = [e for e in ctx.events if isinstance(e, ComplianceAuditEvent)]
        consent_audits = [a for a in audits if a.action == "consent_missing"]
        assert len(consent_audits) == 0

    async def test_default_consent_false_still_emits_warning(self):
        """default_consent=False with no metadata key still emits consent_missing."""
        config = StateLoomConfig(
            console_output=False,
            compliance_profile=ComplianceProfile(standard="gdpr", default_consent=False),
        )
        mw = ComplianceMiddleware(config)
        ctx = _make_ctx()

        async def call_next(c):
            return "response"

        await mw.process(ctx, call_next)
        audits = [e for e in ctx.events if isinstance(e, ComplianceAuditEvent)]
        consent_audits = [a for a in audits if a.action == "consent_missing"]
        assert len(consent_audits) == 1
