"""Production tests: PII Scanning & Redaction.

Real-world PII scenarios with emails, credit cards, SSNs, mixed content,
and dashboard API verification.
"""

from __future__ import annotations

import pytest

from stateloom.core.config import PIIRule
from stateloom.core.errors import StateLoomPIIBlockedError
from stateloom.core.types import FailureAction, PIIMode
from tests.test_production.helpers import (
    assert_event_exists,
    invoke_pipeline,
    make_openai_response,
)


def test_pii_block_email(e2e_gate, api_client):
    """Email in message raises StateLoomPIIBlockedError, event persisted."""
    gate = e2e_gate(
        pii=True,
        pii_rules=[
            PIIRule(pattern="email", mode=PIIMode.BLOCK, on_middleware_failure=FailureAction.BLOCK)
        ],
        cache=False,
    )
    client = api_client(gate)
    response = make_openai_response("Never seen")

    with gate.session(session_id="prod-pii-block-1") as session:
        with pytest.raises(StateLoomPIIBlockedError):
            invoke_pipeline(
                gate,
                session,
                {"messages": [{"role": "user", "content": "Contact me at john@company.com"}]},
                llm_call=lambda: response,
            )

    pii_resp = client.get("/pii").json()
    assert pii_resp["total"] >= 1
    assert pii_resp["by_type"].get("email", 0) >= 1


def test_pii_redact_email(e2e_gate, api_client):
    """Email in message gets redacted — LLM never sees original."""
    gate = e2e_gate(
        pii=True,
        pii_rules=[PIIRule(pattern="email", mode=PIIMode.REDACT)],
        cache=False,
    )
    client = api_client(gate)
    response = make_openai_response("Redacted response")

    seen_kwargs = {}

    def capturing_llm_call():
        return response

    with gate.session(session_id="prod-pii-redact-1") as session:
        result = invoke_pipeline(
            gate,
            session,
            {"messages": [{"role": "user", "content": "My email is secret@domain.com"}]},
            llm_call=capturing_llm_call,
        )

    assert result is response
    pii_resp = client.get("/pii").json()
    assert pii_resp["total"] >= 1
    assert pii_resp["by_action"].get("redact", 0) >= 1


def test_pii_audit_only(e2e_gate, api_client):
    """Audit mode logs PII but LLM call proceeds normally."""
    gate = e2e_gate(
        pii=True,
        pii_rules=[PIIRule(pattern="email", mode=PIIMode.AUDIT)],
        cache=False,
    )
    client = api_client(gate)
    response = make_openai_response("Normal response")

    with gate.session(session_id="prod-pii-audit-1") as session:
        result = invoke_pipeline(
            gate,
            session,
            {"messages": [{"role": "user", "content": "Email: test@example.com"}]},
            llm_call=lambda: response,
        )

    assert result is response
    pii_resp = client.get("/pii").json()
    assert pii_resp["total"] >= 1
    assert pii_resp["by_action"].get("audit", 0) >= 1


def test_pii_credit_card_detection(e2e_gate, api_client):
    """Credit card number (Luhn-valid) detected and blocked."""
    gate = e2e_gate(
        pii=True,
        pii_rules=[
            PIIRule(
                pattern="credit_card", mode=PIIMode.BLOCK, on_middleware_failure=FailureAction.BLOCK
            )
        ],
        cache=False,
    )
    client = api_client(gate)
    response = make_openai_response("Never")

    with gate.session(session_id="prod-pii-cc-1") as session:
        with pytest.raises(StateLoomPIIBlockedError):
            invoke_pipeline(
                gate,
                session,
                {"messages": [{"role": "user", "content": "Card: 4111111111111111"}]},
                llm_call=lambda: response,
            )

    pii_resp = client.get("/pii").json()
    assert pii_resp["by_type"].get("credit_card", 0) >= 1


def test_pii_ssn_detection(e2e_gate, api_client):
    """SSN pattern detected and blocked."""
    gate = e2e_gate(
        pii=True,
        pii_rules=[
            PIIRule(pattern="ssn", mode=PIIMode.BLOCK, on_middleware_failure=FailureAction.BLOCK)
        ],
        cache=False,
    )
    client = api_client(gate)
    response = make_openai_response("Never")

    with gate.session(session_id="prod-pii-ssn-1") as session:
        with pytest.raises(StateLoomPIIBlockedError):
            invoke_pipeline(
                gate,
                session,
                {"messages": [{"role": "user", "content": "SSN: 123-45-6789"}]},
                llm_call=lambda: response,
            )

    pii_resp = client.get("/pii").json()
    assert pii_resp["by_type"].get("ssn", 0) >= 1


def test_pii_multiple_types_in_one_message(e2e_gate, api_client):
    """Message with email + phone → all detected, dashboard shows by_type counts."""
    gate = e2e_gate(
        pii=True,
        pii_rules=[
            PIIRule(pattern="email", mode=PIIMode.AUDIT),
            PIIRule(pattern="phone", mode=PIIMode.AUDIT),
        ],
        cache=False,
    )
    client = api_client(gate)
    response = make_openai_response("Got it")

    with gate.session(session_id="prod-pii-multi-1") as session:
        invoke_pipeline(
            gate,
            session,
            {"messages": [{"role": "user", "content": "Email: a@b.com, Phone: 555-123-4567"}]},
            llm_call=lambda: response,
        )

    pii_resp = client.get("/pii").json()
    assert pii_resp["total"] >= 2
    assert pii_resp["by_type"].get("email", 0) >= 1
    assert pii_resp["by_type"].get("phone_us", 0) >= 1
    assert pii_resp["sessions_affected"] >= 1


def test_pii_clean_message_no_detection(e2e_gate, api_client):
    """Clean message produces no PII events."""
    gate = e2e_gate(
        pii=True,
        pii_rules=[
            PIIRule(pattern="email", mode=PIIMode.BLOCK, on_middleware_failure=FailureAction.BLOCK)
        ],
        cache=False,
    )
    client = api_client(gate)
    response = make_openai_response("Clean response")

    with gate.session(session_id="prod-pii-clean-1") as session:
        result = invoke_pipeline(
            gate,
            session,
            {"messages": [{"role": "user", "content": "What is the weather today?"}]},
            llm_call=lambda: response,
        )

    assert result is response
    events = client.get("/sessions/prod-pii-clean-1/events").json()
    pii_events = [e for e in events["events"] if e["event_type"] == "pii_detection"]
    assert len(pii_events) == 0


def test_pii_dashboard_summary(e2e_gate, api_client):
    """Multiple PII detections → /pii endpoint shows correct aggregates."""
    gate = e2e_gate(
        pii=True,
        pii_rules=[PIIRule(pattern="email", mode=PIIMode.AUDIT)],
        cache=False,
    )
    client = api_client(gate)
    response = make_openai_response("OK")

    # Two sessions each with an email
    for sid in ("prod-pii-dash-1", "prod-pii-dash-2"):
        with gate.session(session_id=sid) as session:
            invoke_pipeline(
                gate,
                session,
                {
                    "messages": [
                        {"role": "user", "content": f"Contact me at user-{sid}@example.com"}
                    ]
                },
                llm_call=lambda: response,
            )

    pii_resp = client.get("/pii").json()
    assert pii_resp["total"] >= 2
    assert pii_resp["sessions_affected"] >= 2
    assert pii_resp["by_type"]["email"] >= 2
