"""Production tests: Durable Streaming — Record & Replay Stream Chunks.

Tests that streaming works in durable sessions: chunks are accumulated
during the first run, serialized into the event store, and replayed as
an iterable on resume.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

from stateloom.core.context import get_current_replay_engine
from stateloom.replay.schema import CachedStreamChunks
from tests.test_production.helpers import make_openai_response


def _invoke_streaming_pipeline(gate, session, request_kwargs, stream_chunks, **kwargs):
    """Invoke the pipeline in streaming mode and consume the stream.

    Mimics what generic_interceptor does for streaming calls: runs
    execute_streaming_sync, then wraps the result via _wrap_stream_sync.

    Returns the list of yielded chunks.
    """
    from stateloom.intercept.generic_interceptor import _wrap_stream_sync
    from stateloom.intercept.provider_adapter import BaseProviderAdapter

    provider = kwargs.get("provider", "openai")
    model = kwargs.get("model", "gpt-3.5-turbo")

    class _StreamAdapter(BaseProviderAdapter):
        @property
        def name(self):
            return provider

        @property
        def method_label(self):
            return "chat.completions.create"

    adapter = _StreamAdapter()
    step = session.next_step()

    ctx = gate.pipeline.execute_streaming_sync(
        provider=provider,
        method="chat.completions.create",
        model=model,
        request_kwargs=request_kwargs,
        session=session,
        config=gate.config,
    )

    if ctx.skip_call and ctx.cached_response is not None:
        return (
            list(ctx.cached_response)
            if hasattr(ctx.cached_response, "__iter__")
            else [ctx.cached_response]
        )

    collected = list(
        _wrap_stream_sync(
            gate,
            adapter,
            iter(stream_chunks),
            session,
            model,
            step,
            {},
            ctx=ctx,
        )
    )
    return collected


def _invoke_with_replay(gate, session, request_kwargs, llm_call, **kwargs):
    """Invoke pipeline with durable replay check (supports streaming replay)."""
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
    )


def _make_stream_chunks(texts: list[str]) -> list[SimpleNamespace]:
    """Build mock OpenAI-style stream chunks."""
    chunks = []
    for i, text in enumerate(texts):
        chunk = SimpleNamespace(
            id="chatcmpl-test",
            choices=[
                SimpleNamespace(
                    delta=SimpleNamespace(content=text),
                    index=0,
                    finish_reason=None if i < len(texts) - 1 else "stop",
                )
            ],
            usage=None,
        )
        # Add model_dump for serialization
        chunk.model_dump = lambda c=chunk: {
            "id": c.id,
            "choices": [
                {
                    "delta": {"content": c.choices[0].delta.content},
                    "index": 0,
                    "finish_reason": c.choices[0].finish_reason,
                }
            ],
        }
        chunks.append(chunk)
    return chunks


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_durable_streaming_records_chunks(e2e_gate, api_client):
    """Durable session with streaming → chunks serialized in cached_response_json."""
    gate = e2e_gate(cache=False)

    stream_chunks = _make_stream_chunks(["Hello", " world", "!"])

    with gate.session(session_id="durable-stream-record-1", durable=True) as session:
        collected = _invoke_streaming_pipeline(
            gate,
            session,
            {"messages": [{"role": "user", "content": "Hi"}]},
            stream_chunks,
        )

    # Chunks were yielded to the caller
    assert len(collected) == 3

    # Check that cached_response_json has the stream marker
    events = gate.store.get_session_events("durable-stream-record-1", event_type="llm_call")
    assert len(events) >= 1
    llm_event = events[0]
    assert llm_event.cached_response_json is not None
    parsed = json.loads(llm_event.cached_response_json)
    assert parsed["_type"] == "stream"
    assert parsed["provider"] == "openai"
    assert len(parsed["chunks"]) == 3


def test_durable_streaming_replay_returns_cached_stream_chunks(e2e_gate, api_client):
    """Resume durable session → streaming step returns CachedStreamChunks."""
    gate = e2e_gate(cache=False)

    stream_chunks = _make_stream_chunks(["One", " Two"])

    # First run — record streaming chunks
    with gate.session(session_id="durable-stream-replay-1", durable=True) as session:
        _invoke_streaming_pipeline(
            gate,
            session,
            {"messages": [{"role": "user", "content": "Count"}]},
            stream_chunks,
        )

    # Resume — should get CachedStreamChunks from replay engine
    with gate.session(session_id="durable-stream-replay-1", durable=True) as session:
        engine = get_current_replay_engine()
        assert engine is not None
        assert engine.is_active

        step = session.next_step()
        cached = engine.get_cached_response(step)
        assert isinstance(cached, CachedStreamChunks)

        # Iterate — should get back the original chunk data
        replayed = list(cached)
        assert len(replayed) == 2


def test_durable_streaming_non_stream_still_works(e2e_gate, api_client):
    """Non-streaming calls in durable mode still record and replay correctly."""
    gate = e2e_gate(cache=False)
    response = make_openai_response("Non-streaming reply")

    # First run
    with gate.session(session_id="durable-nonstream-1", durable=True) as session:
        from tests.test_production.helpers import invoke_pipeline

        invoke_pipeline(
            gate,
            session,
            {"messages": [{"role": "user", "content": "Hi"}]},
            llm_call=lambda: response,
        )

    # Resume
    with gate.session(session_id="durable-nonstream-1", durable=True) as session:
        result = _invoke_with_replay(
            gate,
            session,
            {"messages": [{"role": "user", "content": "Hi"}]},
            llm_call=lambda: make_openai_response("Should not see this"),
        )

    # Non-stream response replayed (not CachedStreamChunks)
    assert not isinstance(result, CachedStreamChunks)


def test_durable_stream_delay_ms_config(e2e_gate, api_client):
    """durable_stream_delay_ms is propagated to session metadata."""
    gate = e2e_gate(cache=False, durable_stream_delay_ms=50)

    with gate.session(session_id="durable-delay-1", durable=True) as session:
        assert session.metadata.get("durable_stream_delay_ms") == 50


def test_durable_stream_delay_ms_zero_not_stored(e2e_gate, api_client):
    """delay_ms=0 (default) does not pollute session metadata."""
    gate = e2e_gate(cache=False)

    with gate.session(session_id="durable-nodelay-1", durable=True) as session:
        assert "durable_stream_delay_ms" not in session.metadata


def test_durable_streaming_mixed_calls(e2e_gate, api_client):
    """Mix of streaming and non-streaming calls in same durable session."""
    gate = e2e_gate(cache=False)
    response = make_openai_response("Non-stream response")
    stream_chunks = _make_stream_chunks(["Stream", " response"])

    with gate.session(session_id="durable-mixed-1", durable=True) as session:
        # Non-streaming call (step 1)
        from tests.test_production.helpers import invoke_pipeline

        invoke_pipeline(
            gate,
            session,
            {"messages": [{"role": "user", "content": "Q1"}]},
            llm_call=lambda: response,
        )

        # Streaming call (step 2)
        collected = _invoke_streaming_pipeline(
            gate,
            session,
            {"messages": [{"role": "user", "content": "Q2"}]},
            stream_chunks,
        )

    assert len(collected) == 2

    # Verify both events are persisted
    events = gate.store.get_session_events("durable-mixed-1", event_type="llm_call")
    assert len(events) >= 2

    # First event: non-stream (no stream marker)
    e1_json = json.loads(events[0].cached_response_json)
    assert e1_json.get("_type") != "stream"

    # Second event: stream marker
    e2_json = json.loads(events[1].cached_response_json)
    assert e2_json["_type"] == "stream"
    assert len(e2_json["chunks"]) == 2


def test_durable_streaming_empty_stream(e2e_gate, api_client):
    """Empty stream (0 chunks) is recorded and replayed as empty."""
    gate = e2e_gate(cache=False)

    with gate.session(session_id="durable-empty-stream-1", durable=True) as session:
        collected = _invoke_streaming_pipeline(
            gate,
            session,
            {"messages": [{"role": "user", "content": "Hi"}]},
            [],  # empty stream
        )

    assert collected == []

    events = gate.store.get_session_events("durable-empty-stream-1", event_type="llm_call")
    assert len(events) >= 1
    parsed = json.loads(events[0].cached_response_json)
    assert parsed["_type"] == "stream"
    assert parsed["chunks"] == []


def test_durable_streaming_resume_mixed(e2e_gate, api_client):
    """Resume with mixed calls: non-stream replayed as object, stream replayed as CachedStreamChunks."""
    gate = e2e_gate(cache=False)
    response = make_openai_response("Non-stream")
    stream_chunks = _make_stream_chunks(["Streamed"])

    # First run
    with gate.session(session_id="durable-resume-mixed-1", durable=True) as session:
        from tests.test_production.helpers import invoke_pipeline

        invoke_pipeline(
            gate,
            session,
            {"messages": [{"role": "user", "content": "Q1"}]},
            llm_call=lambda: response,
        )
        _invoke_streaming_pipeline(
            gate,
            session,
            {"messages": [{"role": "user", "content": "Q2"}]},
            stream_chunks,
        )

    # Resume — check both replay types
    with gate.session(session_id="durable-resume-mixed-1", durable=True) as session:
        engine = get_current_replay_engine()
        assert engine is not None

        # Step 1: non-stream replay
        step1 = session.next_step()
        cached1 = engine.get_cached_response(step1)
        assert not isinstance(cached1, CachedStreamChunks)

        # Step 2: stream replay
        step2 = session.next_step()
        cached2 = engine.get_cached_response(step2)
        assert isinstance(cached2, CachedStreamChunks)
        replayed = list(cached2)
        assert len(replayed) == 1
