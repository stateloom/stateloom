"""Tests for SecurityAuditEvent model."""

from __future__ import annotations

from stateloom.core.event import SecurityAuditEvent
from stateloom.core.types import EventType


def test_event_type():
    event = SecurityAuditEvent()
    assert event.event_type == EventType.SECURITY_AUDIT
    assert event.event_type == "security_audit"


def test_fields():
    event = SecurityAuditEvent(
        session_id="sess-1",
        audit_event="subprocess.Popen",
        action_taken="blocked",
        detail="['rm', '-rf', '/']",
        source="audit_hook",
        severity="high",
        blocked=True,
    )
    assert event.audit_event == "subprocess.Popen"
    assert event.action_taken == "blocked"
    assert event.detail == "['rm', '-rf', '/']"
    assert event.source == "audit_hook"
    assert event.severity == "high"
    assert event.blocked is True


def test_serialization():
    event = SecurityAuditEvent(
        session_id="sess-2",
        audit_event="open",
        action_taken="logged",
        detail="/etc/passwd",
        source="audit_hook",
        severity="medium",
        blocked=False,
    )
    d = event.model_dump(mode="json")
    assert d["event_type"] == "security_audit"
    assert d["audit_event"] == "open"
    assert d["blocked"] is False

    # Roundtrip
    restored = SecurityAuditEvent.model_validate(d)
    assert restored.audit_event == "open"
    assert restored.session_id == "sess-2"
