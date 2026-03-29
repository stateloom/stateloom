"""E2E: Single cloud call — no shadow, no auto-route.

Verifies the full pipeline produces correct /stats and /sessions/{id}/events
output for a basic cloud LLM call.
"""

from __future__ import annotations

from tests.test_e2e.helpers import invoke_pipeline, make_openai_response


def test_single_cloud_call(e2e_gate, api_client):
    gate = e2e_gate(cache=False)
    client = api_client(gate)

    response = make_openai_response("Hello!", model="gpt-3.5-turbo")
    request_kwargs = {"messages": [{"role": "user", "content": "Hi"}]}

    with gate.session(session_id="e2e-cloud-1") as session:
        result = invoke_pipeline(
            gate,
            session,
            request_kwargs,
            llm_call=lambda: response,
            model="gpt-3.5-turbo",
        )

    assert result is response

    # --- Dashboard assertions ---
    stats = client.get("/stats").json()
    assert stats["total_calls"] == 1
    assert stats["cloud_calls"] == 1
    assert stats["local_calls"] == 0

    events_resp = client.get("/sessions/e2e-cloud-1/events").json()
    event_types = [e["event_type"] for e in events_resp["events"]]
    assert "llm_call" in event_types
    assert "local_routing" not in event_types
    assert "shadow_draft" not in event_types

    # Exactly one LLM call event with provider=openai
    llm_events = [e for e in events_resp["events"] if e["event_type"] == "llm_call"]
    assert len(llm_events) == 1
    assert llm_events[0]["provider"] == "openai"
    assert llm_events[0]["model"] == "gpt-3.5-turbo"
