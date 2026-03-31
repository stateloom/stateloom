"""Compliance profile presets — factory functions for GDPR, HIPAA, CCPA."""

from __future__ import annotations

from collections.abc import Callable

from stateloom.core.config import ComplianceProfile, PIIRule
from stateloom.core.types import FailureAction, PIIMode


def gdpr_profile() -> ComplianceProfile:
    """GDPR compliance profile — EU data protection."""
    return ComplianceProfile(
        standard="gdpr",
        region="eu",
        session_ttl_days=30,
        cache_ttl_seconds=2592000,  # 30 days
        block_local_routing=True,
        block_shadow=True,
        block_streaming=True,
        allowed_endpoints=[
            r"https://.*\.openai\.com(/.*)?",
            r"https://.*\.anthropic\.com(/.*)?",
            r"https://.*\.googleapis\.com(/.*)?",
            r"https://.*\.azure\.com(/.*)?",
        ],
        pii_rules=[
            PIIRule(
                pattern="vat_id",
                mode=PIIMode.BLOCK,
                on_middleware_failure=FailureAction.BLOCK,
            ),
            PIIRule(
                pattern="national_id_eu",
                mode=PIIMode.BLOCK,
                on_middleware_failure=FailureAction.BLOCK,
            ),
            PIIRule(pattern="iban", mode=PIIMode.BLOCK, on_middleware_failure=FailureAction.BLOCK),
            PIIRule(pattern="ssn", mode=PIIMode.BLOCK, on_middleware_failure=FailureAction.BLOCK),
            PIIRule(pattern="email", mode=PIIMode.REDACT),
            PIIRule(pattern="phone", mode=PIIMode.REDACT),
        ],
    )


def hipaa_profile() -> ComplianceProfile:
    """HIPAA compliance profile — US health data protection."""
    return ComplianceProfile(
        standard="hipaa",
        zero_retention_logs=True,
        block_local_routing=True,
        block_shadow=True,
        block_streaming=True,
        cache_ttl_seconds=0,
        pii_rules=[
            PIIRule(
                pattern="medical_record_number",
                mode=PIIMode.BLOCK,
                on_middleware_failure=FailureAction.BLOCK,
            ),
            PIIRule(
                pattern="health_plan_id",
                mode=PIIMode.BLOCK,
                on_middleware_failure=FailureAction.BLOCK,
            ),
            PIIRule(pattern="npi", mode=PIIMode.BLOCK, on_middleware_failure=FailureAction.BLOCK),
            PIIRule(pattern="ssn", mode=PIIMode.BLOCK, on_middleware_failure=FailureAction.BLOCK),
            PIIRule(pattern="email", mode=PIIMode.REDACT),
            PIIRule(pattern="phone", mode=PIIMode.REDACT),
            PIIRule(pattern="date_of_birth", mode=PIIMode.REDACT),
        ],
    )


def ccpa_profile() -> ComplianceProfile:
    """CCPA compliance profile — California consumer privacy."""
    return ComplianceProfile(
        standard="ccpa",
        session_ttl_days=90,
        cache_ttl_seconds=7776000,  # 90 days
        pii_rules=[
            PIIRule(pattern="ssn", mode=PIIMode.BLOCK, on_middleware_failure=FailureAction.BLOCK),
            PIIRule(
                pattern="california_dl",
                mode=PIIMode.BLOCK,
                on_middleware_failure=FailureAction.BLOCK,
            ),
            PIIRule(
                pattern="credit_card",
                mode=PIIMode.BLOCK,
                on_middleware_failure=FailureAction.BLOCK,
            ),
            PIIRule(pattern="email", mode=PIIMode.REDACT),
        ],
    )


PROFILE_PRESETS: dict[str, Callable[[], ComplianceProfile]] = {
    "gdpr": gdpr_profile,
    "hipaa": hipaa_profile,
    "ccpa": ccpa_profile,
}


def resolve_profile(name_or_profile: str | ComplianceProfile) -> ComplianceProfile:
    """Resolve 'gdpr' -> ComplianceProfile, or return as-is."""
    if isinstance(name_or_profile, ComplianceProfile):
        return name_or_profile
    if name_or_profile in PROFILE_PRESETS:
        return PROFILE_PRESETS[name_or_profile]()
    return ComplianceProfile(standard=name_or_profile)
