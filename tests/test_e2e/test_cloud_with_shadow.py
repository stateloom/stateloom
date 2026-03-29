"""E2E: Cloud call with shadow drafting enabled.

Verifies:
- Shadow call does NOT count as cloud or local in /stats
- /shadow/metrics shows the shadow call with correct counts
- Shadow event has similarity_score > 0 (both texts available)
- No duplicate shadow events in the store (MemoryStore upsert)
"""

from __future__ import annotations

from unittest.mock import patch

from tests.test_e2e.helpers import (
    invoke_pipeline,
    make_ollama_response,
    make_openai_response,
    wait_for_shadow_events,
)


def test_cloud_with_shadow(e2e_gate, api_client):
    gate = e2e_gate(
        local_model="llama3.2",
        shadow=True,
        auto_route=False,
        cache=False,
    )
    client = api_client(gate)

    cloud_response = make_openai_response("Cloud answer", model="gpt-3.5-turbo")
    shadow_ollama = make_ollama_response("Local shadow answer", model="llama3.2")

    shadow_mw = gate._shadow_middleware
    with patch.object(shadow_mw._client, "chat", return_value=shadow_ollama):
        request_kwargs = {"messages": [{"role": "user", "content": "Hi"}]}

        with gate.session(session_id="e2e-shadow-1") as session:
            result = invoke_pipeline(
                gate,
                session,
                request_kwargs,
                llm_call=lambda: cloud_response,
                model="gpt-3.5-turbo",
            )

    assert result is cloud_response

    # Wait for shadow thread to complete and persist its event
    shadow_events = wait_for_shadow_events(gate, timeout=3.0)
    assert len(shadow_events) >= 1, "Shadow event should have been persisted"

    # --- Dashboard assertions ---
    stats = client.get("/stats").json()
    assert stats["cloud_calls"] == 1
    assert stats["local_calls"] == 0, "Shadow call should NOT be counted as a local call"

    shadow_resp = client.get("/shadow/metrics").json()
    assert shadow_resp["total_calls"] == 1, (
        f"Expected exactly 1 shadow call, got {shadow_resp['total_calls']}. "
        "Duplicate events from re-save should not inflate the count."
    )
    assert shadow_resp["success_count"] == 1

    # Verify similarity was computed (both cloud and local text were available)
    non_cancelled = [e for e in shadow_events if e.shadow_status != "cancelled"]
    assert len(non_cancelled) == 1
    assert non_cancelled[0].similarity_score is not None, (
        "Similarity score should be set when both cloud and local text are available"
    )
    assert non_cancelled[0].similarity_score > 0, (
        f"Similarity should be > 0 for non-empty texts, got {non_cancelled[0].similarity_score}"
    )

    # Dashboard avg_similarity should also be > 0
    assert shadow_resp["avg_similarity"] is not None
    assert shadow_resp["avg_similarity"] > 0
