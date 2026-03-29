"""E2E: Same call twice — second is served from cache.

Verifies:
- total_calls=1 (cache hit doesn't increment call_count on session)
- total_cache_hits=1
- Events include 1 LLMCallEvent + 1 CacheHitEvent
"""

from __future__ import annotations

from tests.test_e2e.helpers import invoke_pipeline, make_openai_response


def test_cache_hit(e2e_gate, api_client):
    gate = e2e_gate(cache=True)
    client = api_client(gate)

    response = make_openai_response("Cached answer", model="gpt-3.5-turbo")
    request_kwargs = {"messages": [{"role": "user", "content": "What is 2+2?"}]}
    call_count = 0

    def tracked_llm_call():
        nonlocal call_count
        call_count += 1
        return response

    with gate.session(session_id="e2e-cache-1") as session:
        # First call — should hit LLM
        result1 = invoke_pipeline(
            gate,
            session,
            request_kwargs,
            llm_call=tracked_llm_call,
            model="gpt-3.5-turbo",
        )

        # Second call with identical kwargs — should be cached
        result2 = invoke_pipeline(
            gate,
            session,
            request_kwargs,
            llm_call=tracked_llm_call,
            model="gpt-3.5-turbo",
        )

    # The LLM should only have been called once
    assert call_count == 1

    # --- Dashboard assertions ---
    stats = client.get("/stats").json()
    assert stats["total_cache_hits"] == 1

    events_resp = client.get("/sessions/e2e-cache-1/events").json()
    event_types = [e["event_type"] for e in events_resp["events"]]
    assert event_types.count("llm_call") == 1
    assert event_types.count("cache_hit") == 1
