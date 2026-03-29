"""Compliance middleware — enforces declarative compliance profiles."""

from __future__ import annotations

import logging
import re
from collections.abc import Awaitable, Callable
from typing import Any

from stateloom.compliance.audit import compute_audit_hash
from stateloom.compliance.legal_rules import get_legal_rule
from stateloom.core.config import ComplianceProfile, StateLoomConfig
from stateloom.core.errors import StateLoomComplianceError
from stateloom.core.event import (
    BlastRadiusEvent,
    BudgetEnforcementEvent,
    CacheHitEvent,
    ComplianceAuditEvent,
    KillSwitchEvent,
    PIIDetectionEvent,
    RateLimitEvent,
)
from stateloom.middleware.base import MiddlewareContext

logger = logging.getLogger("stateloom.middleware.compliance")


class ComplianceMiddleware:
    """Resolves and enforces compliance profiles for each request.

    Sits early in the chain (after KillSwitch, before BlastRadius).
    Responsibilities:
    1. Resolve active compliance profile (team > org > global) via callback
    2. Store profile metadata on session for downstream middleware
    3. Set zero-retention flags for HIPAA
    4. After pipeline completes, emit ComplianceAuditEvent for PII events
    """

    def __init__(
        self,
        config: StateLoomConfig,
        store: Any = None,
        compliance_fn: Callable[[str, str], ComplianceProfile | None] | None = None,
    ) -> None:
        self._config = config
        self._store = store
        self._compliance_fn = compliance_fn

    async def process(
        self,
        ctx: MiddlewareContext,
        call_next: Callable[[MiddlewareContext], Awaitable[Any]],
    ) -> Any:
        profile = self._resolve_profile(ctx)
        if not profile or profile.standard == "none":
            return await call_next(ctx)

        # Store compliance metadata on session for downstream middleware
        ctx.session.metadata["_compliance_standard"] = profile.standard
        ctx.session.metadata["_compliance_region"] = profile.region

        if profile.zero_retention_logs:
            ctx.session.metadata["store_payloads"] = False
            ctx.session.metadata["_compliance_zero_retention"] = True
            ctx.session.metadata["_compliance_audit_salt"] = profile.audit_salt

        # Force non-streaming for regulated profiles
        if profile.block_streaming and ctx.request_kwargs.get("stream"):
            ctx.request_kwargs["stream"] = False
            audit = ComplianceAuditEvent(
                session_id=ctx.session.id,
                step=ctx.session.step_counter,
                compliance_standard=profile.standard,
                action="streaming_blocked",
                legal_rule=get_legal_rule(profile.standard, "streaming_blocked"),
                justification=(
                    f"Streaming disabled per {profile.standard.upper()} compliance — "
                    f"all requests routed through full middleware pipeline"
                ),
                target_type="session",
                target_id=ctx.session.id,
                org_id=ctx.session.org_id,
                team_id=ctx.session.team_id,
            )
            audit.integrity_hash = compute_audit_hash(audit, profile.audit_salt)
            ctx.events.append(audit)

        # GDPR consent check
        if profile.standard == "gdpr":
            if "gdpr_consent" in ctx.session.metadata:
                consent = ctx.session.metadata["gdpr_consent"]
            else:
                consent = profile.default_consent
            if not consent:
                audit = ComplianceAuditEvent(
                    session_id=ctx.session.id,
                    step=ctx.session.step_counter,
                    compliance_standard="gdpr",
                    action="consent_missing",
                    legal_rule=get_legal_rule("gdpr", "consent_missing"),
                    justification=(
                        "GDPR Article 6 requires lawful basis; no consent metadata found"
                    ),
                    target_type="session",
                    target_id=ctx.session.id,
                    org_id=ctx.session.org_id,
                    team_id=ctx.session.team_id,
                )
                audit.integrity_hash = compute_audit_hash(audit, profile.audit_salt)
                ctx.events.append(audit)

        # Data residency: check provider endpoint against allowed patterns
        if profile.allowed_endpoints:
            if not ctx.provider_base_url:
                # Wrapper libraries (LiteLLM, etc.) don't expose a base URL.
                # Record an audit event for visibility but allow the request —
                # blocking every call from URL-less providers makes compliance
                # unusable with them.
                logger.warning(
                    "Compliance: provider_base_url is empty but allowed_endpoints is set "
                    "for %s profile — endpoint check skipped (provider does not expose URL)",
                    profile.standard.upper(),
                )
                audit = ComplianceAuditEvent(
                    session_id=ctx.session.id,
                    step=ctx.session.step_counter,
                    compliance_standard=profile.standard,
                    action="endpoint_unknown",
                    legal_rule=get_legal_rule(profile.standard, "endpoint_blocked"),
                    justification=(
                        f"Provider endpoint URL not available — data residency "
                        f"check skipped per {profile.standard.upper()} compliance policy"
                    ),
                    target_type="session",
                    target_id=ctx.session.id,
                    org_id=ctx.session.org_id,
                    team_id=ctx.session.team_id,
                )
                audit.integrity_hash = compute_audit_hash(audit, profile.audit_salt)
                ctx.events.append(audit)

            elif not any(re.fullmatch(p, ctx.provider_base_url) for p in profile.allowed_endpoints):
                audit = ComplianceAuditEvent(
                    session_id=ctx.session.id,
                    step=ctx.session.step_counter,
                    compliance_standard=profile.standard,
                    action="endpoint_blocked",
                    legal_rule=get_legal_rule(profile.standard, "endpoint_blocked"),
                    justification=(
                        f"Endpoint '{ctx.provider_base_url}' not in allowed list "
                        f"per {profile.standard.upper()} data residency policy"
                    ),
                    target_type="session",
                    target_id=ctx.session.id,
                    org_id=ctx.session.org_id,
                    team_id=ctx.session.team_id,
                )
                audit.integrity_hash = compute_audit_hash(audit, profile.audit_salt)
                # Persist directly — EventRecorder won't run after raise
                if self._store:
                    try:
                        self._store.save_event(audit)
                    except Exception:
                        logger.debug("Failed to persist compliance audit event", exc_info=True)
                raise StateLoomComplianceError(
                    f"Endpoint '{ctx.provider_base_url}' blocked by "
                    f"{profile.standard.upper()} data residency policy",
                    standard=profile.standard,
                    action="endpoint_blocked",
                )

        result = await call_next(ctx)

        # Emit audit events for observable actions
        for event in list(ctx.events):
            audit_action = ""
            justification = ""

            if isinstance(event, PIIDetectionEvent):
                audit_action = "pii_blocked" if event.mode == "block" else "pii_redacted"
                justification = (
                    f"PII type '{event.pii_type}' {event.action_taken} "
                    f"per {profile.standard.upper()} compliance"
                )
            elif isinstance(event, BudgetEnforcementEvent):
                audit_action = "budget_enforced"
                justification = (
                    f"Budget enforcement: {event.action} "
                    f"(spent ${event.spent:.2f}, limit ${event.limit:.2f})"
                )
            elif isinstance(event, CacheHitEvent):
                audit_action = "cache_hit"
                justification = (
                    f"Cache hit for model '{event.original_model}' (saved ${event.saved_cost:.4f})"
                )
            elif isinstance(event, RateLimitEvent):
                if event.rejected or event.timed_out:
                    audit_action = "rate_limit_enforced"
                    justification = (
                        f"Rate limit enforced for team '{event.team_id}' "
                        f"(rejected={event.rejected}, timed_out={event.timed_out})"
                    )
            elif isinstance(event, BlastRadiusEvent):
                audit_action = "blast_radius_triggered"
                justification = (
                    f"Blast radius containment: {event.trigger} "
                    f"(count={event.count}, threshold={event.threshold})"
                )
            elif isinstance(event, KillSwitchEvent):
                audit_action = "kill_switch_activated"
                justification = f"Kill switch activated: {event.reason}"

            if audit_action:
                legal_rule = get_legal_rule(profile.standard, audit_action)
                audit = ComplianceAuditEvent(
                    session_id=ctx.session.id,
                    step=ctx.session.step_counter,
                    compliance_standard=profile.standard,
                    action=audit_action,
                    legal_rule=legal_rule,
                    justification=justification,
                    target_type="session",
                    target_id=ctx.session.id,
                    org_id=ctx.session.org_id,
                    team_id=ctx.session.team_id,
                )
                audit.integrity_hash = compute_audit_hash(audit, profile.audit_salt)
                ctx.events.append(audit)

        return result

    def _resolve_profile(self, ctx: MiddlewareContext) -> ComplianceProfile | None:
        """Resolve the active compliance profile for this request."""
        # Try callback first (team > org > global)
        if self._compliance_fn:
            profile = self._compliance_fn(ctx.session.org_id, ctx.session.team_id)
            if profile:
                return profile

        # Fall back to global config
        return self._config.compliance_profile
