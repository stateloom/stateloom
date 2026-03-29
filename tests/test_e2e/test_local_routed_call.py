"""E2E: Single call auto-routed to local model.

Catches the "Local Calls: 2" bug where /stats double-counted
LLMCallEvent + LocalRoutingEvent.

Expected: cloud_calls=0, local_calls=1 (NOT 2).
"""

from __future__ import annotations

import time
from unittest.mock import patch

from tests.test_e2e.helpers import invoke_pipeline, make_ollama_response


def test_local_routed_call(e2e_gate, api_client):
    gate = e2e_gate(
        local_model="llama3.2",
        auto_route=True,
        shadow=False,
        cache=False,
    )
    client = api_client(gate)

    # Patch the auto-router's Ollama client to return a mock response
    router = gate._auto_router
    ollama_resp = make_ollama_response("Local answer", model="llama3.2")

    with (
        patch.object(router._client, "chat", return_value=ollama_resp),
        patch.object(router._client, "is_available", return_value=True),
    ):
        # Bypass the 60s availability cache
        router._ollama_available = True
        router._ollama_check_time = time.monotonic()

        request_kwargs = {"messages": [{"role": "user", "content": "Hi"}]}

        # The llm_call should NOT be called since auto-router intercepts
        def should_not_be_called():
            raise AssertionError("Cloud LLM call should not be made")

        with gate.session(session_id="e2e-local-1") as session:
            invoke_pipeline(
                gate,
                session,
                request_kwargs,
                llm_call=should_not_be_called,
                model="gpt-3.5-turbo",
            )

    # --- Dashboard assertions ---
    stats = client.get("/stats").json()
    assert stats["cloud_calls"] == 0
    assert stats["local_calls"] == 1, (
        f"Expected local_calls=1, got {stats['local_calls']}. "
        "LocalRoutingEvent should NOT be counted as a separate call."
    )

    events_resp = client.get("/sessions/e2e-local-1/events").json()
    event_types = [e["event_type"] for e in events_resp["events"]]

    # Should have exactly 1 LLMCallEvent (provider="local") + 1 LocalRoutingEvent
    llm_events = [e for e in events_resp["events"] if e["event_type"] == "llm_call"]
    routing_events = [e for e in events_resp["events"] if e["event_type"] == "local_routing"]
    assert len(llm_events) == 1
    assert llm_events[0]["provider"] == "local"
    assert len(routing_events) == 1
    assert routing_events[0]["routing_success"] is True
