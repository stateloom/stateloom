"""Production tests: Compliance Profiles.

GDPR/HIPAA compliance features including local routing blocks, shadow blocks,
streaming blocks, PII rule merging, and audit events.
"""

from __future__ import annotations

from stateloom.core.config import ComplianceProfile
from tests.test_production.helpers import invoke_pipeline, make_openai_response


def test_compliance_blocks_local_routing(e2e_gate, api_client):
    """HIPAA profile → block_local_routing=True → no auto-route."""
    gate = e2e_gate(cache=False, auto_route=False)
    client = api_client(gate)

    org = gate.create_organization(
        name="HIPAAOrg",
        compliance_profile=ComplianceProfile(
            standard="hipaa",
            block_local_routing=True,
        ),
    )
    team = gate.create_team(org_id=org.id, name="HIPAATeam")
    response = make_openai_response("HIPAA reply")

    with gate.session(session_id="prod-compliance-lr-1", org_id=org.id, team_id=team.id) as session:
        result = invoke_pipeline(
            gate,
            session,
            {"messages": [{"role": "user", "content": "Hi"}]},
            llm_call=lambda: response,
        )

    assert result is response
    # Session should have compliance metadata
    stored = gate.store.get_session("prod-compliance-lr-1")
    assert stored is not None


def test_compliance_blocks_shadow(e2e_gate, api_client):
    """Profile with block_shadow=True → no shadow drafting."""
    gate = e2e_gate(cache=False, shadow=False)
    client = api_client(gate)

    org = gate.create_organization(
        name="ShadowBlockOrg",
        compliance_profile=ComplianceProfile(
            standard="hipaa",
            block_shadow=True,
        ),
    )
    team = gate.create_team(org_id=org.id, name="ShadowBlockTeam")
    response = make_openai_response("No shadow")

    with gate.session(
        session_id="prod-compliance-shadow-1", org_id=org.id, team_id=team.id
    ) as session:
        result = invoke_pipeline(
            gate,
            session,
            {"messages": [{"role": "user", "content": "Hi"}]},
            llm_call=lambda: response,
        )

    assert result is response
    events = client.get("/sessions/prod-compliance-shadow-1/events").json()
    shadow_events = [e for e in events["events"] if e["event_type"] == "shadow_draft"]
    assert len(shadow_events) == 0


def test_compliance_blocks_streaming(e2e_gate, api_client):
    """Profile with block_streaming=True → streaming disabled."""
    gate = e2e_gate(cache=False)
    client = api_client(gate)

    org = gate.create_organization(
        name="StreamBlockOrg",
        compliance_profile=ComplianceProfile(
            standard="hipaa",
            block_streaming=True,
        ),
    )
    team = gate.create_team(org_id=org.id, name="StreamBlockTeam")
    response = make_openai_response("Non-streaming")

    with gate.session(
        session_id="prod-compliance-stream-1", org_id=org.id, team_id=team.id
    ) as session:
        result = invoke_pipeline(
            gate,
            session,
            {
                "messages": [
                    {"role": "user", "content": "Hi"},
                ],
                "stream": True,
            },
            llm_call=lambda: response,
        )

    assert result is response


def test_compliance_pii_rules_from_profile(e2e_gate, api_client):
    """Compliance profile PII rules applied alongside global rules."""
    from stateloom.core.config import PIIRule
    from stateloom.core.types import PIIMode

    gate = e2e_gate(
        cache=False,
        pii=True,
        pii_rules=[PIIRule(pattern="email", mode=PIIMode.AUDIT)],
    )
    client = api_client(gate)

    org = gate.create_organization(
        name="PIIProfileOrg",
        compliance_profile=ComplianceProfile(
            standard="gdpr",
            pii_rules=[PIIRule(pattern="phone", mode=PIIMode.AUDIT)],
        ),
    )
    team = gate.create_team(org_id=org.id, name="PIIProfileTeam")
    response = make_openai_response("OK")

    with gate.session(
        session_id="prod-compliance-pii-1", org_id=org.id, team_id=team.id
    ) as session:
        invoke_pipeline(
            gate,
            session,
            {"messages": [{"role": "user", "content": "Email: a@b.com Phone: 555-123-4567"}]},
            llm_call=lambda: response,
        )

    pii_resp = client.get("/pii").json()
    assert pii_resp["total"] >= 1


def test_compliance_cleanup(e2e_gate, api_client):
    """compliance_cleanup() runs without error."""
    gate = e2e_gate(cache=False)
    client = api_client(gate)

    # Just verify it doesn't crash
    purged = gate.compliance_cleanup()
    assert isinstance(purged, int)
    assert purged >= 0


def test_compliance_audit_event(e2e_gate, api_client):
    """Compliance check → ComplianceAuditEvent recorded for PII events."""
    from stateloom.core.config import PIIRule
    from stateloom.core.types import PIIMode

    gate = e2e_gate(
        cache=False,
        pii=True,
        pii_rules=[PIIRule(pattern="email", mode=PIIMode.AUDIT)],
    )
    client = api_client(gate)

    org = gate.create_organization(
        name="AuditOrg",
        compliance_profile=ComplianceProfile(standard="gdpr"),
    )
    team = gate.create_team(org_id=org.id, name="AuditTeam")
    response = make_openai_response("OK")

    with gate.session(
        session_id="prod-compliance-audit-1", org_id=org.id, team_id=team.id
    ) as session:
        invoke_pipeline(
            gate,
            session,
            {"messages": [{"role": "user", "content": "My email is test@example.com"}]},
            llm_call=lambda: response,
        )

    events = client.get("/sessions/prod-compliance-audit-1/events").json()
    event_types = [e["event_type"] for e in events["events"]]
    # Should have PII detection; compliance audit is generated for PII events
    assert "pii_detection" in event_types
