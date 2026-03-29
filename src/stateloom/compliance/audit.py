"""Audit hash utility for tamper-proof compliance event records."""

from __future__ import annotations

import hashlib
import json
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from stateloom.core.event import ComplianceAuditEvent


def compute_audit_hash(event: ComplianceAuditEvent, salt: str = "") -> str:
    """SHA-256 over critical event fields + enterprise salt for tamper-proofing."""
    payload = {
        "id": event.id,
        "session_id": event.session_id,
        "timestamp": event.timestamp.isoformat(),
        "compliance_standard": event.compliance_standard,
        "action": event.action,
        "legal_rule": event.legal_rule,
        "target_type": event.target_type,
        "target_id": event.target_id,
        "org_id": event.org_id,
        "team_id": event.team_id,
    }
    data = salt + json.dumps(payload, sort_keys=True)
    return hashlib.sha256(data.encode()).hexdigest()
