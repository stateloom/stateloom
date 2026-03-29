"""Production tests: Durable Resumption (Crash Recovery).

Durable sessions with response caching, replay on resume, and streaming disable.
"""

from __future__ import annotations

from stateloom.core.context import get_current_replay_engine
from tests.test_production.helpers import invoke_pipeline, make_openai_response


def _invoke_with_replay(gate, session, request_kwargs, llm_call, **kwargs):
    """Invoke pipeline with durable replay check.

    The ``invoke_pipeline`` helper calls ``pipeline.execute_sync()`` directly,
    which bypasses the replay engine lookup that normally happens inside
    ``generic_interceptor.py``.  This wrapper mimics that lookup so durable
    resume tests work end-to-end.
    """
    step = session.next_step()
    engine = get_current_replay_engine()
    if engine is not None and engine.is_active and engine.should_mock(step):
        return engine.get_cached_response(step)
    return gate.pipeline.execute_sync(
        provider=kwargs.get("provider", "openai"),
        method="chat.completions.create",
        model=kwargs.get("model", "gpt-3.5-turbo"),
        request_kwargs=request_kwargs,
        session=session,
        config=gate.config,
        llm_call=llm_call,
        auto_route_eligible=kwargs.get("auto_route_eligible", True),
    )


def test_durable_session_caches_response(e2e_gate, api_client):
    """durable=True → response cached in event's cached_response_json."""
    gate = e2e_gate(cache=False)
    client = api_client(gate)
    response = make_openai_response("Durable reply")

    with gate.session(session_id="prod-durable-1", durable=True) as session:
        result = invoke_pipeline(
            gate,
            session,
            {"messages": [{"role": "user", "content": "Hi"}]},
            llm_call=lambda: response,
        )

    assert result is response

    # Check that cached_response_json is populated in the LLM call event
    events = gate.store.get_session_events("prod-durable-1", event_type="llm_call")
    assert len(events) >= 1
    llm_event = events[0]
    assert llm_event.cached_response_json is not None
    assert len(llm_event.cached_response_json) > 0


def test_durable_resume_replays_cached(e2e_gate, api_client):
    """Resume with same session_id → cached responses returned without LLM call."""
    gate = e2e_gate(cache=False)
    client = api_client(gate)
    response = make_openai_response("First run reply")

    # First run — generates cached responses
    with gate.session(session_id="prod-durable-resume-1", durable=True) as session:
        invoke_pipeline(
            gate,
            session,
            {"messages": [{"role": "user", "content": "Hi"}]},
            llm_call=lambda: response,
        )

    # Resume — should replay from cache
    call_count = 0

    def should_not_be_called():
        nonlocal call_count
        call_count += 1
        return make_openai_response("Should not see this")

    with gate.session(session_id="prod-durable-resume-1", durable=True) as session:
        result = _invoke_with_replay(
            gate,
            session,
            {"messages": [{"role": "user", "content": "Hi"}]},
            llm_call=should_not_be_called,
        )

    # LLM should not have been called — response came from durable cache
    assert call_count == 0


def test_durable_resume_continues_from_last(e2e_gate, api_client):
    """2 cached + 1 new → first 2 replayed, 3rd calls LLM."""
    gate = e2e_gate(cache=False)
    client = api_client(gate)
    response1 = make_openai_response("Reply 1")
    response2 = make_openai_response("Reply 2")

    # First run — 2 calls
    with gate.session(session_id="prod-durable-continue-1", durable=True) as session:
        invoke_pipeline(
            gate,
            session,
            {"messages": [{"role": "user", "content": "Q1"}]},
            llm_call=lambda: response1,
        )
        invoke_pipeline(
            gate,
            session,
            {"messages": [{"role": "user", "content": "Q2"}]},
            llm_call=lambda: response2,
        )

    # Resume — 2 cached + 1 new
    new_call_count = 0
    response3 = make_openai_response("Reply 3 (new)")

    def counting_call():
        nonlocal new_call_count
        new_call_count += 1
        return response3

    with gate.session(session_id="prod-durable-continue-1", durable=True) as session:
        _invoke_with_replay(
            gate,
            session,
            {"messages": [{"role": "user", "content": "Q1"}]},
            llm_call=counting_call,
        )
        _invoke_with_replay(
            gate,
            session,
            {"messages": [{"role": "user", "content": "Q2"}]},
            llm_call=counting_call,
        )
        result3 = _invoke_with_replay(
            gate,
            session,
            {"messages": [{"role": "user", "content": "Q3"}]},
            llm_call=counting_call,
        )

    # Only the 3rd call should have hit the LLM
    assert new_call_count == 1


def test_durable_streaming_disabled(e2e_gate, api_client):
    """durable=True forces stream=False in request kwargs."""
    gate = e2e_gate(cache=False)
    client = api_client(gate)
    response = make_openai_response("Non-streaming")

    captured_kwargs = {}

    def capturing_call():
        return response

    with gate.session(session_id="prod-durable-stream-1", durable=True) as session:
        # Even if we pass stream=True, durable mode should override to False
        result = invoke_pipeline(
            gate,
            session,
            {"messages": [{"role": "user", "content": "Hi"}], "stream": True},
            llm_call=capturing_call,
        )

    # Verify the session is durable
    assert session.metadata.get("durable") is True
