"""Legal rule mappings — the 'why column' for compliance audit events."""

from __future__ import annotations

LEGAL_RULES: dict[str, dict[str, str]] = {
    "gdpr": {
        "pii_blocked": "GDPR-Art-32 — Security of processing",
        "pii_redacted": "GDPR-Art-25 — Data protection by design",
        "routing_blocked": "GDPR-Art-44 — Transfer restrictions",
        "endpoint_blocked": "GDPR-Art-44 — Transfer restrictions",
        "data_purged": "GDPR-Art-17 — Right to erasure",
        "session_expired": "GDPR-Art-5(1)(e) — Storage limitation",
        "consent_missing": "GDPR-Art-6(1)(a) — Consent",
        "streaming_blocked": "GDPR-Art-25 — Data protection by design",
    },
    "hipaa": {
        "pii_blocked": "HIPAA-164.502(a) — Uses and disclosures",
        "pii_redacted": "HIPAA-164.514(a) — De-identification",
        "routing_blocked": "HIPAA-164.312(e) — Transmission security",
        "endpoint_blocked": "HIPAA-164.312(e) — Transmission security",
        "zero_retention": "HIPAA-164.530(j) — Retention and destruction",
        "data_purged": "HIPAA-164.530(j)(2) — Retention period",
        "streaming_blocked": "HIPAA-164.312(e) — Transmission security",
    },
    "ccpa": {
        "pii_blocked": "CCPA-1798.100(b) — Right to know",
        "pii_redacted": "CCPA-1798.100(d) — Collection limitation",
        "data_purged": "CCPA-1798.105 — Right to deletion",
    },
}


def get_legal_rule(standard: str, action: str) -> str:
    """Get the legal rule string for a compliance standard and action.

    Returns empty string if the standard or action is not found.
    """
    return LEGAL_RULES.get(standard, {}).get(action, "")
