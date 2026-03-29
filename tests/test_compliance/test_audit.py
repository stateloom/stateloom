"""Tests for compliance audit hash utility."""

from __future__ import annotations

from datetime import datetime, timezone

from stateloom.compliance.audit import compute_audit_hash
from stateloom.core.event import ComplianceAuditEvent


class TestComputeAuditHash:
    def _make_event(self, **kwargs) -> ComplianceAuditEvent:
        defaults = {
            "id": "evt-001",
            "session_id": "sess-001",
            "compliance_standard": "gdpr",
            "action": "pii_blocked",
            "legal_rule": "GDPR-Art-32",
            "target_type": "session",
            "target_id": "sess-001",
            "org_id": "org-001",
            "team_id": "team-001",
        }
        defaults.update(kwargs)
        return ComplianceAuditEvent(**defaults)

    def test_returns_hex_string(self):
        event = self._make_event()
        h = compute_audit_hash(event)
        assert isinstance(h, str)
        assert len(h) == 64  # SHA-256 hex

    def test_deterministic(self):
        event = self._make_event()
        h1 = compute_audit_hash(event)
        h2 = compute_audit_hash(event)
        assert h1 == h2

    def test_salt_changes_hash(self):
        event = self._make_event()
        h1 = compute_audit_hash(event, salt="")
        h2 = compute_audit_hash(event, salt="enterprise-secret")
        assert h1 != h2

    def test_different_events_different_hashes(self):
        e1 = self._make_event(action="pii_blocked")
        e2 = self._make_event(action="pii_redacted")
        assert compute_audit_hash(e1) != compute_audit_hash(e2)

    def test_includes_all_critical_fields(self):
        """Changing any critical field changes the hash."""
        base = self._make_event()
        base_hash = compute_audit_hash(base)

        for field, alt_value in [
            ("session_id", "other-session"),
            ("compliance_standard", "hipaa"),
            ("org_id", "other-org"),
        ]:
            altered = self._make_event(**{field: alt_value})
            assert compute_audit_hash(altered) != base_hash, f"Field {field} not in hash"
