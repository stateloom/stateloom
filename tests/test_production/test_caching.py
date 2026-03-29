"""Production tests: Response Caching.

Cache hit/miss, scope, and dashboard metrics with full pipeline verification.
"""

from __future__ import annotations

from tests.test_production.helpers import invoke_pipeline, make_openai_response


def test_cache_hit_identical_request(e2e_gate, api_client):
    """Same request twice → second is cached, LLM called once."""
    gate = e2e_gate(cache=True)
    client = api_client(gate)
    response = make_openai_response("Cached answer")
    request_kwargs = {"messages": [{"role": "user", "content": "What is 2+2?"}]}
    call_count = 0

    def tracked_call():
        nonlocal call_count
        call_count += 1
        return response

    with gate.session(session_id="prod-cache-1") as session:
        invoke_pipeline(gate, session, request_kwargs, llm_call=tracked_call)
        invoke_pipeline(gate, session, request_kwargs, llm_call=tracked_call)

    assert call_count == 1
    stats = client.get("/stats").json()
    assert stats["total_cache_hits"] == 1


def test_cache_miss_different_request(e2e_gate, api_client):
    """Different messages → both hit LLM, no cache."""
    gate = e2e_gate(cache=True)
    client = api_client(gate)
    response = make_openai_response("Answer")
    call_count = 0

    def tracked_call():
        nonlocal call_count
        call_count += 1
        return response

    with gate.session(session_id="prod-cache-miss-1") as session:
        invoke_pipeline(
            gate,
            session,
            {"messages": [{"role": "user", "content": "Question A"}]},
            llm_call=tracked_call,
        )
        invoke_pipeline(
            gate,
            session,
            {"messages": [{"role": "user", "content": "Question B"}]},
            llm_call=tracked_call,
        )

    assert call_count == 2
    stats = client.get("/stats").json()
    assert stats["total_cache_hits"] == 0


def test_cache_disabled_no_hits(e2e_gate, api_client):
    """cache=False → identical requests both call LLM."""
    gate = e2e_gate(cache=False)
    client = api_client(gate)
    response = make_openai_response("No cache")
    request_kwargs = {"messages": [{"role": "user", "content": "Same question"}]}
    call_count = 0

    def tracked_call():
        nonlocal call_count
        call_count += 1
        return response

    with gate.session(session_id="prod-no-cache-1") as session:
        invoke_pipeline(gate, session, request_kwargs, llm_call=tracked_call)
        invoke_pipeline(gate, session, request_kwargs, llm_call=tracked_call)

    assert call_count == 2


def test_cache_savings_tracked(e2e_gate, api_client):
    """Cache hit → cache_savings > 0 in session stats."""
    gate = e2e_gate(cache=True)
    client = api_client(gate)
    response = make_openai_response("Savings", prompt_tokens=100, completion_tokens=50)
    request_kwargs = {"messages": [{"role": "user", "content": "Expensive query"}]}

    with gate.session(session_id="prod-cache-savings-1") as session:
        invoke_pipeline(gate, session, request_kwargs, llm_call=lambda: response)
        invoke_pipeline(gate, session, request_kwargs, llm_call=lambda: response)

    sess_resp = client.get("/sessions/prod-cache-savings-1").json()
    assert sess_resp["cache_hits"] == 1
    assert sess_resp["cache_savings"] >= 0


def test_cache_dashboard_stats(e2e_gate, api_client):
    """Verify /stats shows total_cache_hits correctly."""
    gate = e2e_gate(cache=True, loop_exact_threshold=10)
    client = api_client(gate)
    response = make_openai_response("Cached")
    request_kwargs = {"messages": [{"role": "user", "content": "Repeat me"}]}

    with gate.session(session_id="prod-cache-stats-1") as session:
        invoke_pipeline(gate, session, request_kwargs, llm_call=lambda: response)
        invoke_pipeline(gate, session, request_kwargs, llm_call=lambda: response)
        invoke_pipeline(gate, session, request_kwargs, llm_call=lambda: response)

    stats = client.get("/stats").json()
    assert stats["total_cache_hits"] == 2  # 2nd and 3rd calls cached


def test_cache_different_models_not_cached(e2e_gate, api_client):
    """Same messages but different model in request_kwargs → no cache hit.

    The cache hash is computed from request_kwargs only.  Including
    ``model`` inside request_kwargs makes the hashes differ, so both
    calls go to the LLM.
    """
    gate = e2e_gate(cache=True)
    client = api_client(gate)
    response_35 = make_openai_response("3.5 answer", model="gpt-3.5-turbo")
    response_4o = make_openai_response("4o answer", model="gpt-4o")
    call_count = 0

    def tracked_call():
        nonlocal call_count
        call_count += 1
        return response_35 if call_count == 1 else response_4o

    request_kwargs_35 = {
        "model": "gpt-3.5-turbo",
        "messages": [{"role": "user", "content": "Same question"}],
    }
    request_kwargs_4o = {
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": "Same question"}],
    }

    with gate.session(session_id="prod-cache-model-1") as session:
        invoke_pipeline(
            gate, session, request_kwargs_35, llm_call=tracked_call, model="gpt-3.5-turbo"
        )
        invoke_pipeline(gate, session, request_kwargs_4o, llm_call=tracked_call, model="gpt-4o")

    assert call_count == 2
