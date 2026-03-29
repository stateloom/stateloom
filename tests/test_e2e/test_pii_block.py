"""E2E: PII block raises error and events are still persisted.

Catches the bug where PII block events were missing from the dashboard
because EventRecorder never runs when PII scanner raises.
"""

from __future__ import annotations

import pytest

from stateloom.core.config import PIIRule
from stateloom.core.errors import StateLoomPIIBlockedError
from stateloom.core.types import FailureAction, PIIMode
from tests.test_e2e.helpers import invoke_pipeline, make_openai_response


def test_pii_block_events_persisted(e2e_gate, api_client):
    gate = e2e_gate(
        pii=True,
        pii_rules=[
            PIIRule(
                pattern="email",
                mode=PIIMode.BLOCK,
                on_middleware_failure=FailureAction.BLOCK,
            ),
        ],
        cache=False,
    )
    client = api_client(gate)

    response = make_openai_response("Should not reach here")
    request_kwargs = {
        "messages": [{"role": "user", "content": "My email is user@example.com"}],
    }

    with gate.session(session_id="e2e-pii-1") as session:
        with pytest.raises(StateLoomPIIBlockedError):
            invoke_pipeline(
                gate,
                session,
                request_kwargs,
                llm_call=lambda: response,
                model="gpt-3.5-turbo",
            )

    # --- Dashboard assertions ---
    pii_resp = client.get("/pii").json()
    assert pii_resp["total"] >= 1, "PII detection events should be persisted even on block"

    # Check the detection details
    detections = pii_resp["detections"]
    email_detections = [d for d in detections if d["pii_type"] == "email"]
    assert len(email_detections) >= 1
    assert email_detections[0]["action"] == "block"
    assert pii_resp["sessions_affected"] >= 1

    # The by_type and by_action summaries should reflect the detection
    assert pii_resp["by_type"].get("email", 0) >= 1
    assert pii_resp["by_action"].get("block", 0) >= 1
