"""Compliance-as-a-Service — declarative GDPR, HIPAA, CCPA profiles.

Note: Enterprise compliance features are now in stateloom.ee.compliance.
This module provides backward compatibility.
"""

from __future__ import annotations

from stateloom.compliance.audit import compute_audit_hash
from stateloom.compliance.legal_rules import LEGAL_RULES, get_legal_rule
from stateloom.compliance.profiles import PROFILE_PRESETS, resolve_profile
from stateloom.compliance.purge import PurgeEngine, PurgeResult

__all__ = [
    "LEGAL_RULES",
    "PROFILE_PRESETS",
    "PurgeEngine",
    "PurgeResult",
    "compute_audit_hash",
    "get_legal_rule",
    "resolve_profile",
]
