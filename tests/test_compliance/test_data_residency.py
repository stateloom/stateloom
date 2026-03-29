"""Tests for data residency enforcement via allowed_endpoints."""

from __future__ import annotations

import pytest

from stateloom.compliance.legal_rules import get_legal_rule
from stateloom.compliance.profiles import gdpr_profile
from stateloom.core.config import ComplianceProfile, StateLoomConfig
from stateloom.core.errors import StateLoomComplianceError
from stateloom.core.session import Session
from stateloom.middleware.base import MiddlewareContext
from stateloom.middleware.compliance import ComplianceMiddleware
from stateloom.store.memory_store import MemoryStore


def _make_ctx(
    provider_base_url: str = "",
    org_id: str = "",
    team_id: str = "",
) -> MiddlewareContext:
    return MiddlewareContext(
        session=Session(id="test-session", org_id=org_id, team_id=team_id),
        config=StateLoomConfig(console_output=False),
        provider="openai",
        model="gpt-4",
        request_kwargs={"messages": [{"role": "user", "content": "hello"}]},
        provider_base_url=provider_base_url,
    )


async def _noop_call_next(ctx: MiddlewareContext):
    return "response"


class TestEndpointBlocking:
    """Endpoint not in allowed_endpoints raises StateLoomComplianceError."""

    async def test_non_matching_endpoint_blocked(self):
        profile = ComplianceProfile(
            standard="gdpr",
            region="eu",
            allowed_endpoints=[r"https://.*\.openai\.com/.*"],
        )
        store = MemoryStore()
        mw = ComplianceMiddleware(
            StateLoomConfig(console_output=False),
            store=store,
            compliance_fn=lambda o, t: profile,
        )
        ctx = _make_ctx(provider_base_url="https://evil-proxy.example.com/v1")

        with pytest.raises(StateLoomComplianceError) as exc_info:
            await mw.process(ctx, _noop_call_next)

        assert exc_info.value.standard == "gdpr"
        assert exc_info.value.action == "endpoint_blocked"
        assert "evil-proxy.example.com" in str(exc_info.value)

    async def test_matching_endpoint_allowed(self):
        profile = ComplianceProfile(
            standard="gdpr",
            region="eu",
            allowed_endpoints=[r"https://.*\.openai\.com/.*"],
        )
        mw = ComplianceMiddleware(
            StateLoomConfig(console_output=False),
            compliance_fn=lambda o, t: profile,
        )
        ctx = _make_ctx(provider_base_url="https://api.openai.com/v1")

        result = await mw.process(ctx, _noop_call_next)
        assert result == "response"

    async def test_multiple_patterns_any_match(self):
        profile = ComplianceProfile(
            standard="gdpr",
            region="eu",
            allowed_endpoints=[
                r"https://.*\.openai\.com/.*",
                r"https://.*\.anthropic\.com",
            ],
        )
        mw = ComplianceMiddleware(
            StateLoomConfig(console_output=False),
            compliance_fn=lambda o, t: profile,
        )
        ctx = _make_ctx(provider_base_url="https://api.anthropic.com")

        result = await mw.process(ctx, _noop_call_next)
        assert result == "response"

    async def test_empty_allowed_endpoints_no_check(self):
        """No allowed_endpoints = no endpoint enforcement."""
        profile = ComplianceProfile(
            standard="gdpr",
            region="eu",
            allowed_endpoints=[],
        )
        mw = ComplianceMiddleware(
            StateLoomConfig(console_output=False),
            compliance_fn=lambda o, t: profile,
        )
        ctx = _make_ctx(provider_base_url="https://anything.example.com")

        result = await mw.process(ctx, _noop_call_next)
        assert result == "response"

    async def test_empty_base_url_skips_check(self):
        """Empty provider_base_url with allowed_endpoints skips the check.

        Wrapper libraries (LiteLLM, etc.) don't expose a base URL.
        The request is allowed but an audit event is recorded.
        """
        profile = ComplianceProfile(
            standard="gdpr",
            region="eu",
            allowed_endpoints=[r"https://.*\.openai\.com/.*"],
        )
        mw = ComplianceMiddleware(
            StateLoomConfig(console_output=False),
            compliance_fn=lambda o, t: profile,
        )
        ctx = _make_ctx(provider_base_url="")

        result = await mw.process(ctx, _noop_call_next)
        assert result == "response"

        # An audit event should be recorded for visibility
        audit_events = [
            e for e in ctx.events
            if hasattr(e, "action") and e.action == "endpoint_unknown"
        ]
        assert len(audit_events) == 1
        assert "skipped" in audit_events[0].justification


class TestAuditEventOnBlock:
    """Blocked endpoint persists a ComplianceAuditEvent directly to the store."""

    async def test_audit_event_persisted(self):
        profile = ComplianceProfile(
            standard="gdpr",
            region="eu",
            allowed_endpoints=[r"https://.*\.openai\.com/.*"],
            audit_salt="test-salt",
        )
        store = MemoryStore()
        mw = ComplianceMiddleware(
            StateLoomConfig(console_output=False),
            store=store,
            compliance_fn=lambda o, t: profile,
        )
        ctx = _make_ctx(provider_base_url="https://bad-endpoint.example.com")

        with pytest.raises(StateLoomComplianceError):
            await mw.process(ctx, _noop_call_next)

        events = store.get_session_events("test-session")
        audit_events = [
            e for e in events if hasattr(e, "action") and e.action == "endpoint_blocked"
        ]
        assert len(audit_events) == 1
        audit = audit_events[0]
        assert audit.compliance_standard == "gdpr"
        assert audit.legal_rule == "GDPR-Art-44 — Transfer restrictions"
        assert audit.integrity_hash  # non-empty


class TestGDPRPresetDefaults:
    """GDPR profile preset includes broad allowed_endpoints."""

    def test_gdpr_has_allowed_endpoints(self):
        p = gdpr_profile()
        assert len(p.allowed_endpoints) > 0

    def test_gdpr_allows_openai(self):
        import re

        p = gdpr_profile()
        url = "https://api.openai.com/v1"
        assert any(re.fullmatch(pat, url) for pat in p.allowed_endpoints)

    def test_gdpr_allows_anthropic(self):
        import re

        p = gdpr_profile()
        url = "https://api.anthropic.com/v1"
        assert any(re.fullmatch(pat, url) for pat in p.allowed_endpoints)

    def test_gdpr_allows_googleapis(self):
        import re

        p = gdpr_profile()
        url = "https://generativelanguage.googleapis.com/v1"
        assert any(re.fullmatch(pat, url) for pat in p.allowed_endpoints)

    def test_gdpr_allows_azure(self):
        import re

        p = gdpr_profile()
        url = "https://my-resource.openai.azure.com/v1"
        assert any(re.fullmatch(pat, url) for pat in p.allowed_endpoints)


class TestLegalRules:
    """Legal rule mappings for endpoint_blocked exist."""

    def test_gdpr_endpoint_blocked_rule(self):
        rule = get_legal_rule("gdpr", "endpoint_blocked")
        assert "GDPR-Art-44" in rule

    def test_hipaa_endpoint_blocked_rule(self):
        rule = get_legal_rule("hipaa", "endpoint_blocked")
        assert "HIPAA-164.312(e)" in rule


class TestComplianceErrorAttributes:
    """StateLoomComplianceError carries standard and action."""

    def test_error_attributes(self):
        err = StateLoomComplianceError("blocked", standard="gdpr", action="endpoint_blocked")
        assert err.standard == "gdpr"
        assert err.action == "endpoint_blocked"
        assert err.error_code == "COMPLIANCE_BLOCKED"

    def test_error_is_stateloom_error(self):
        from stateloom.core.errors import StateLoomError

        err = StateLoomComplianceError("test")
        assert isinstance(err, StateLoomError)
