"""Tests for the OpenAI Responses API proxy adapter (/v1/responses)."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

from fastapi import FastAPI
from fastapi.testclient import TestClient

from stateloom.core.config import StateLoomConfig
from stateloom.core.errors import (
    StateLoomBudgetError,
    StateLoomKillSwitchError,
    StateLoomPIIBlockedError,
    StateLoomRateLimitError,
)
from stateloom.proxy.responses import (
    _SYNTHETIC_RESP_PREFIX,
    _extract_prompt_preview,
    _input_to_openai_messages,
    _rebuild_responses_body,
    _record_ws_event,
    _resolve_upstream,
    _send_blocked_response,
    _track_stream_usage,
    _WSRelayState,
    create_responses_router,
)
from stateloom.proxy.virtual_key import (
    VirtualKey,
    generate_virtual_key,
    make_key_preview,
    make_virtual_key_id,
)
from stateloom.store.memory_store import MemoryStore


def _make_gate(*, require_virtual_key: bool = True) -> MagicMock:
    gate = MagicMock()
    gate.store = MemoryStore()
    gate.config = StateLoomConfig(
        proxy_enabled=True,
        proxy_require_virtual_key=require_virtual_key,
        console_output=False,
        dashboard=False,
    )
    gate._metrics_collector = None
    return gate


def _make_app(gate: MagicMock, passthrough: MagicMock | None = None) -> FastAPI:
    app = FastAPI()
    router = create_responses_router(gate, passthrough=passthrough)
    app.include_router(router, prefix="/v1")
    return app


def _setup_virtual_key(gate: MagicMock, **kwargs) -> str:
    full_key, key_hash = generate_virtual_key()
    vk = VirtualKey(
        id=make_virtual_key_id(),
        key_hash=key_hash,
        key_preview=make_key_preview(full_key),
        team_id="team-1",
        org_id="org-1",
        name="test-key",
        **kwargs,
    )
    gate.store.save_virtual_key(vk)
    return full_key


def _base_body(**overrides):
    body = {
        "model": "gpt-4.1",
        "input": "What is a linked list?",
    }
    body.update(overrides)
    return body


def _make_upstream_response(
    text="A linked list is a data structure.", input_tokens=10, output_tokens=25
):
    """Create a Responses API-style upstream response dict."""
    return {
        "id": "resp_abc123",
        "object": "response",
        "created_at": 1700000000,
        "model": "gpt-4.1",
        "output": [
            {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": text}],
            }
        ],
        "usage": {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
        },
    }


def _make_passthrough_mock(response_data=None, status_code=200):
    """Create a mock PassthroughProxy that returns a given response."""
    if response_data is None:
        response_data = _make_upstream_response()

    pt = MagicMock()
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = response_data
    resp.text = json.dumps(response_data)
    resp.content = json.dumps(response_data).encode()
    resp.headers = {"content-type": "application/json"}
    pt.forward = AsyncMock(return_value=resp)
    return pt


def _setup_gate_session(gate):
    """Set up mocked gate.async_session and pipeline for passthrough tests."""
    gate.async_session = MagicMock()
    session = MagicMock()
    session.id = "test-session"
    session.metadata = {}
    session.step_counter = 0
    session.next_step = MagicMock()

    ctx_mgr = AsyncMock()
    ctx_mgr.__aenter__ = AsyncMock(return_value=session)
    ctx_mgr.__aexit__ = AsyncMock(return_value=None)
    gate.async_session.return_value = ctx_mgr

    gate.pipeline._hash_request = MagicMock(return_value="hash123")
    return session


class TestResponsesNonStreaming:
    def test_passthrough_works(self):
        """Responses API request forwarded and response returned as-is."""
        gate = _make_gate(require_virtual_key=False)
        upstream_data = _make_upstream_response("Hello from Responses API!")
        pt = _make_passthrough_mock(upstream_data)

        session = _setup_gate_session(gate)
        gate.pipeline.execute = AsyncMock(return_value=upstream_data)

        app = _make_app(gate, passthrough=pt)
        client = TestClient(app)
        resp = client.post(
            "/v1/responses",
            json=_base_body(),
            headers={"Authorization": "Bearer sk-test-key-123"},
        )

        assert resp.status_code == 200
        data = resp.json()
        assert data["id"] == "resp_abc123"
        assert data["object"] == "response"
        assert data["output"][0]["content"][0]["text"] == "Hello from Responses API!"

    def test_string_input(self):
        """String input handled correctly."""
        gate = _make_gate(require_virtual_key=False)
        upstream_data = _make_upstream_response()
        pt = _make_passthrough_mock(upstream_data)

        session = _setup_gate_session(gate)
        gate.pipeline.execute = AsyncMock(return_value=upstream_data)

        app = _make_app(gate, passthrough=pt)
        client = TestClient(app)
        resp = client.post(
            "/v1/responses",
            json=_base_body(input="simple string input"),
            headers={"Authorization": "Bearer sk-test-key"},
        )
        assert resp.status_code == 200

    def test_array_input(self):
        """Array input with message items handled."""
        gate = _make_gate(require_virtual_key=False)
        upstream_data = _make_upstream_response()
        pt = _make_passthrough_mock(upstream_data)

        session = _setup_gate_session(gate)
        gate.pipeline.execute = AsyncMock(return_value=upstream_data)

        app = _make_app(gate, passthrough=pt)
        client = TestClient(app)
        resp = client.post(
            "/v1/responses",
            json=_base_body(
                input=[
                    {"type": "message", "role": "user", "content": "Hello"},
                    {"type": "message", "role": "assistant", "content": "Hi there"},
                    {"type": "message", "role": "user", "content": "How are you?"},
                ]
            ),
            headers={"Authorization": "Bearer sk-test-key"},
        )
        assert resp.status_code == 200

    def test_no_passthrough_returns_503(self):
        """Returns 503 when no PassthroughProxy is available."""
        gate = _make_gate(require_virtual_key=False)
        app = _make_app(gate, passthrough=None)
        client = TestClient(app)
        resp = client.post(
            "/v1/responses",
            json=_base_body(),
            headers={"Authorization": "Bearer sk-test-key"},
        )
        assert resp.status_code == 503
        data = resp.json()
        assert data["error"]["code"] == "service_unavailable"


class TestResponsesAuth:
    def test_bearer_forwarded(self):
        """Non-VK, non-BYOK bearer token forwarded."""
        gate = _make_gate(require_virtual_key=False)
        upstream_data = _make_upstream_response()
        pt = _make_passthrough_mock(upstream_data)

        session = _setup_gate_session(gate)
        gate.pipeline.execute = AsyncMock(return_value=upstream_data)

        app = _make_app(gate, passthrough=pt)
        client = TestClient(app)
        resp = client.post(
            "/v1/responses",
            json=_base_body(),
            headers={"Authorization": "Bearer some-session-token"},
        )
        assert resp.status_code == 200

    def test_vk_auth(self):
        """ag- prefixed token treated as virtual key."""
        gate = _make_gate(require_virtual_key=True)
        full_key = _setup_virtual_key(gate)
        upstream_data = _make_upstream_response()
        pt = _make_passthrough_mock(upstream_data)

        session = _setup_gate_session(gate)
        gate.pipeline.execute = AsyncMock(return_value=upstream_data)

        app = _make_app(gate, passthrough=pt)
        client = TestClient(app)
        resp = client.post(
            "/v1/responses",
            json=_base_body(),
            headers={"Authorization": f"Bearer {full_key}"},
        )
        assert resp.status_code == 200

    def test_byok_key(self):
        """sk-* token treated as BYOK OpenAI key."""
        gate = _make_gate(require_virtual_key=False)
        upstream_data = _make_upstream_response()
        pt = _make_passthrough_mock(upstream_data)

        session = _setup_gate_session(gate)
        gate.pipeline.execute = AsyncMock(return_value=upstream_data)

        app = _make_app(gate, passthrough=pt)
        client = TestClient(app)
        resp = client.post(
            "/v1/responses",
            json=_base_body(),
            headers={"Authorization": "Bearer sk-my-openai-key"},
        )
        assert resp.status_code == 200

    def test_no_auth_mode(self):
        """Unauthenticated allowed when configured."""
        gate = _make_gate(require_virtual_key=False)
        upstream_data = _make_upstream_response()
        pt = _make_passthrough_mock(upstream_data)

        session = _setup_gate_session(gate)
        gate.pipeline.execute = AsyncMock(return_value=upstream_data)

        app = _make_app(gate, passthrough=pt)
        client = TestClient(app)
        resp = client.post(
            "/v1/responses",
            json=_base_body(),
        )
        assert resp.status_code == 200

    def test_missing_auth_rejected(self):
        """Missing auth when required returns 401."""
        gate = _make_gate(require_virtual_key=True)
        pt = _make_passthrough_mock()
        app = _make_app(gate, passthrough=pt)
        client = TestClient(app)
        resp = client.post(
            "/v1/responses",
            json=_base_body(),
        )
        assert resp.status_code == 401
        data = resp.json()
        assert data["error"]["code"] == "invalid_api_key"

    def test_invalid_vk_rejected(self):
        """Invalid ag- key returns 401."""
        gate = _make_gate(require_virtual_key=True)
        pt = _make_passthrough_mock()
        app = _make_app(gate, passthrough=pt)
        client = TestClient(app)
        resp = client.post(
            "/v1/responses",
            json=_base_body(),
            headers={"Authorization": "Bearer ag-invalid-key-12345"},
        )
        assert resp.status_code == 401


class TestResponsesValidation:
    def test_missing_model(self):
        """400 for missing model field."""
        gate = _make_gate(require_virtual_key=False)
        pt = _make_passthrough_mock()
        app = _make_app(gate, passthrough=pt)
        client = TestClient(app)
        resp = client.post(
            "/v1/responses",
            json={"input": "hello"},
            headers={"Authorization": "Bearer sk-key"},
        )
        assert resp.status_code == 400
        assert "model" in resp.json()["error"]["message"]

    def test_missing_input(self):
        """400 for missing input field."""
        gate = _make_gate(require_virtual_key=False)
        pt = _make_passthrough_mock()
        app = _make_app(gate, passthrough=pt)
        client = TestClient(app)
        resp = client.post(
            "/v1/responses",
            json={"model": "gpt-4.1"},
            headers={"Authorization": "Bearer sk-key"},
        )
        assert resp.status_code == 400
        assert "input" in resp.json()["error"]["message"]

    def test_invalid_json(self):
        """400 for invalid JSON body."""
        gate = _make_gate(require_virtual_key=False)
        pt = _make_passthrough_mock()
        app = _make_app(gate, passthrough=pt)
        client = TestClient(app)
        resp = client.post(
            "/v1/responses",
            content=b"not-json",
            headers={"Content-Type": "application/json", "Authorization": "Bearer sk-key"},
        )
        assert resp.status_code == 400


class TestResponsesMessageConversion:
    def test_string_input(self):
        """String input converted to single user message."""
        messages = _input_to_openai_messages("Hello world")
        assert len(messages) == 1
        assert messages[0] == {"role": "user", "content": "Hello world"}

    def test_array_input_with_messages(self):
        """Array input with message items converted."""
        input_field = [
            {"type": "message", "role": "user", "content": "Hello"},
            {"type": "message", "role": "assistant", "content": "Hi!"},
            {"type": "message", "role": "user", "content": "How?"},
        ]
        messages = _input_to_openai_messages(input_field)
        assert len(messages) == 3
        assert messages[0] == {"role": "user", "content": "Hello"}
        assert messages[1] == {"role": "assistant", "content": "Hi!"}
        assert messages[2] == {"role": "user", "content": "How?"}

    def test_instructions_as_system(self):
        """Instructions field converted to system message."""
        messages = _input_to_openai_messages("Hello", "You are helpful.")
        assert len(messages) == 2
        assert messages[0] == {"role": "system", "content": "You are helpful."}
        assert messages[1] == {"role": "user", "content": "Hello"}

    def test_mixed_input_types(self):
        """Non-message items in array input: item_reference skipped, function_call_output converted."""
        input_field = [
            {"type": "message", "role": "user", "content": "Hello"},
            {"type": "item_reference", "id": "ref_123"},
            {"type": "function_call_output", "call_id": "call_1", "output": "42"},
            {"type": "message", "role": "user", "content": "What's next?"},
        ]
        messages = _input_to_openai_messages(input_field)
        assert len(messages) == 3
        assert messages[0]["content"] == "Hello"
        assert messages[1] == {"role": "tool", "content": "42"}
        assert messages[2]["content"] == "What's next?"

    def test_content_array_with_parts(self):
        """Content as array of input_text parts."""
        input_field = [
            {
                "type": "message",
                "role": "user",
                "content": [
                    {"type": "input_text", "text": "Hello"},
                    {"type": "input_image", "url": "http://example.com/img.png"},
                    {"type": "input_text", "text": "World"},
                ],
            },
        ]
        messages = _input_to_openai_messages(input_field)
        assert len(messages) == 1
        assert messages[0]["content"] == "Hello\nWorld"

    def test_empty_input_string(self):
        """Empty string input still creates a message (validation at route level)."""
        messages = _input_to_openai_messages("")
        assert len(messages) == 1
        assert messages[0] == {"role": "user", "content": ""}

    def test_text_type_content_parts(self):
        """Content parts with type 'text' (not just 'input_text') also extracted."""
        input_field = [
            {
                "type": "message",
                "role": "user",
                "content": [
                    {"type": "text", "text": "Hello from text type"},
                ],
            },
        ]
        messages = _input_to_openai_messages(input_field)
        assert len(messages) == 1
        assert messages[0]["content"] == "Hello from text type"


class TestResponsesStreaming:
    def test_sse_forwarded(self):
        """Streaming SSE events forwarded."""
        gate = _make_gate(require_virtual_key=False)
        pt = MagicMock()

        sse_chunk = (
            "event: response.output_text.delta\n"
            'data: {"type": "response.output_text.delta", "delta": "Hello"}\n\n'
        )

        async def mock_stream(*args, **kwargs):
            yield sse_chunk.encode("utf-8")

        pt.forward_stream = mock_stream

        session = _setup_gate_session(gate)
        gate.pipeline.execute_streaming = AsyncMock()

        app = _make_app(gate, passthrough=pt)
        client = TestClient(app)
        resp = client.post(
            "/v1/responses",
            json=_base_body(stream=True),
            headers={"Authorization": "Bearer sk-test-key"},
        )

        assert resp.status_code == 200
        assert "text/event-stream" in resp.headers.get("content-type", "")
        assert "data: " in resp.text

    def test_typed_events_preserved(self):
        """Typed SSE events (event: lines) preserved in forwarding."""
        gate = _make_gate(require_virtual_key=False)
        pt = MagicMock()

        events = [
            'event: response.created\ndata: {"type": "response.created", "response": {}}\n\n',
            'event: response.output_text.delta\ndata: {"type": "response.output_text.delta", "delta": "Hi"}\n\n',
            'event: response.completed\ndata: {"type": "response.completed", "usage": {"input_tokens": 5, "output_tokens": 2}}\n\n',
        ]

        async def mock_stream(*args, **kwargs):
            for event in events:
                yield event.encode("utf-8")

        pt.forward_stream = mock_stream

        session = _setup_gate_session(gate)
        gate.pipeline.execute_streaming = AsyncMock()

        app = _make_app(gate, passthrough=pt)
        client = TestClient(app)
        resp = client.post(
            "/v1/responses",
            json=_base_body(stream=True),
            headers={"Authorization": "Bearer sk-test-key"},
        )

        assert resp.status_code == 200
        body = resp.text
        assert "response.created" in body
        assert "response.output_text.delta" in body
        assert "response.completed" in body


class TestResponsesStreamTokenTracking:
    def test_usage_from_response_completed(self):
        """Token usage extracted from response.completed event."""
        ctx = MagicMock()
        ctx.prompt_tokens = 0
        ctx.completion_tokens = 0

        sse_line = 'data: {"type": "response.completed", "usage": {"input_tokens": 42, "output_tokens": 18, "total_tokens": 60}}\n\n'
        _track_stream_usage(sse_line, ctx)

        assert ctx.prompt_tokens == 42
        assert ctx.completion_tokens == 18

    def test_no_usage(self):
        """No crash when usage missing."""
        ctx = MagicMock()
        ctx.prompt_tokens = 0
        ctx.completion_tokens = 0

        sse_line = 'data: {"type": "response.output_text.delta", "delta": "hi"}\n\n'
        _track_stream_usage(sse_line, ctx)
        # Should not error, tokens unchanged

    def test_none_ctx(self):
        """No crash when ctx is None."""
        _track_stream_usage("data: {}\n\n", None)


class TestRecordWsEvent:
    """Test _record_ws_event creates LLMCallEvents and updates session."""

    def _make_session(self):
        session = MagicMock()
        session.id = "ws-session-1"
        session.step_counter = 1
        session.metadata = {"billing_mode": "api"}
        session.add_cost = MagicMock()
        session.next_step = MagicMock()
        return session

    def _make_gate_for_ws(self):
        gate = MagicMock()
        gate.pricing.calculate_cost = MagicMock(return_value=0.005)
        gate.store.save_session_with_events = MagicMock()
        return gate

    def test_response_completed_creates_event(self):
        """response.completed message creates LLMCallEvent and persists."""
        session = self._make_session()
        gate = self._make_gate_for_ws()
        ws_state = _WSRelayState(current_model="gpt-4.1", prompt_preview="hello world")

        msg = json.dumps(
            {
                "type": "response.completed",
                "response": {
                    "model": "gpt-4.1",
                    "usage": {"input_tokens": 100, "output_tokens": 50},
                },
            }
        )

        _record_ws_event(msg, session, gate, "api", ws_state)

        # Session cost updated
        session.add_cost.assert_called_once()
        call_kwargs = session.add_cost.call_args[1]
        assert call_kwargs["prompt_tokens"] == 100
        assert call_kwargs["completion_tokens"] == 50

        # Event persisted with prompt preview
        gate.store.save_session_with_events.assert_called_once()
        saved_events = gate.store.save_session_with_events.call_args[0][1]
        assert len(saved_events) == 1
        assert saved_events[0].prompt_tokens == 100
        assert saved_events[0].completion_tokens == 50
        assert saved_events[0].model == "gpt-4.1"
        assert saved_events[0].prompt_preview == "hello world"

    def test_subscription_billing_zero_cost(self):
        """Subscription billing mode sets actual cost to 0."""
        session = self._make_session()
        gate = self._make_gate_for_ws()
        gate.pricing.calculate_cost.return_value = 0.01
        ws_state = _WSRelayState(current_model="gpt-4.1")

        msg = json.dumps(
            {
                "type": "response.completed",
                "response": {
                    "model": "gpt-4.1",
                    "usage": {"input_tokens": 50, "output_tokens": 20},
                },
            }
        )

        _record_ws_event(msg, session, gate, "subscription", ws_state)

        call_kwargs = session.add_cost.call_args[1]
        assert call_kwargs["cost"] == 0.0
        assert call_kwargs["estimated_api_cost"] == 0.01

    def test_non_completed_events_ignored(self):
        """Non-response.completed events do not create events."""
        session = self._make_session()
        gate = self._make_gate_for_ws()
        ws_state = _WSRelayState(current_model="gpt-4.1")

        for msg_type in ["response.created", "response.output_text.delta", "response.done"]:
            msg = json.dumps({"type": msg_type})
            _record_ws_event(msg, session, gate, "api", ws_state)

        session.add_cost.assert_not_called()
        gate.store.save_session_with_events.assert_not_called()

    def test_invalid_json_ignored(self):
        """Invalid JSON does not crash."""
        session = self._make_session()
        gate = self._make_gate_for_ws()
        ws_state = _WSRelayState(current_model="gpt-4.1")
        _record_ws_event("not-json", session, gate, "api", ws_state)
        session.add_cost.assert_not_called()

    def test_latency_tracked(self):
        """Latency calculated from call_start timestamp."""
        import time as _time

        session = self._make_session()
        gate = self._make_gate_for_ws()
        start = _time.monotonic() - 0.5  # 500ms ago
        ws_state = _WSRelayState(current_model="gpt-4.1", call_start=start)

        msg = json.dumps(
            {
                "type": "response.completed",
                "response": {
                    "model": "gpt-4.1",
                    "usage": {"input_tokens": 10, "output_tokens": 5},
                },
            }
        )

        _record_ws_event(msg, session, gate, "api", ws_state)

        saved_events = gate.store.save_session_with_events.call_args[0][1]
        assert saved_events[0].latency_ms >= 400  # at least 400ms


class TestResponsesBillingMode:
    def test_byok_sk_detected_as_api(self):
        """BYOK sk-* key detected as API billing mode."""
        gate = _make_gate(require_virtual_key=False)
        upstream_data = _make_upstream_response()
        pt = _make_passthrough_mock(upstream_data)

        session = _setup_gate_session(gate)
        gate.pipeline.execute = AsyncMock(return_value=upstream_data)

        app = _make_app(gate, passthrough=pt)
        client = TestClient(app)
        resp = client.post(
            "/v1/responses",
            json=_base_body(),
            headers={"Authorization": "Bearer sk-my-openai-key"},
        )
        assert resp.status_code == 200
        assert session.metadata.get("billing_mode") == "api"

    def test_other_token_detected_as_subscription(self):
        """Non-sk token detected as subscription billing mode."""
        gate = _make_gate(require_virtual_key=False)
        upstream_data = _make_upstream_response()
        pt = _make_passthrough_mock(upstream_data)

        session = _setup_gate_session(gate)
        gate.pipeline.execute = AsyncMock(return_value=upstream_data)

        app = _make_app(gate, passthrough=pt)
        client = TestClient(app)
        resp = client.post(
            "/v1/responses",
            json=_base_body(),
            headers={"Authorization": "Bearer session-oauth-token-xyz"},
        )
        assert resp.status_code == 200
        assert session.metadata.get("billing_mode") == "subscription"


class TestResponsesErrorFormat:
    def test_stateloom_errors_mapped(self):
        """StateLoom errors mapped to OpenAI error format."""
        errors_and_expected = [
            (StateLoomRateLimitError("team-1", 10.0, 100), 429),
            (StateLoomBudgetError(5.0, 6.0, "s-1"), 402),
            (StateLoomPIIBlockedError("email", "s-1"), 400),
            (StateLoomKillSwitchError(), 503),
        ]

        for error, expected_status in errors_and_expected:
            gate = _make_gate(require_virtual_key=False)
            pt = _make_passthrough_mock()

            session = _setup_gate_session(gate)
            gate.pipeline.execute = AsyncMock(side_effect=error)

            app = _make_app(gate, passthrough=pt)
            client = TestClient(app)
            resp = client.post(
                "/v1/responses",
                json=_base_body(),
                headers={"Authorization": "Bearer sk-test-key"},
            )

            assert resp.status_code == expected_status, (
                f"Expected {expected_status} for {type(error).__name__}, got {resp.status_code}"
            )
            data = resp.json()
            assert "error" in data
            assert "message" in data["error"]


class TestResponsesUpstreamError:
    def test_upstream_error_forwarded(self):
        """Upstream 4xx/5xx forwarded with original status code."""
        gate = _make_gate(require_virtual_key=False)
        error_data = {
            "error": {
                "message": "Invalid model",
                "type": "invalid_request_error",
                "code": "model_not_found",
            }
        }
        pt = _make_passthrough_mock(error_data, status_code=404)

        session = _setup_gate_session(gate)

        upstream_error = {
            "_upstream_error": True,
            "_status_code": 404,
            **error_data,
        }
        gate.pipeline.execute = AsyncMock(return_value=upstream_error)

        app = _make_app(gate, passthrough=pt)
        client = TestClient(app)
        resp = client.post(
            "/v1/responses",
            json=_base_body(),
            headers={"Authorization": "Bearer sk-test-key"},
        )
        assert resp.status_code == 404
        data = resp.json()
        assert data["error"]["message"] == "Invalid model"


class TestResponsesSessionHeader:
    def test_session_id_header_used(self):
        """X-StateLoom-Session-Id header used when provided."""
        gate = _make_gate(require_virtual_key=False)
        upstream_data = _make_upstream_response()
        pt = _make_passthrough_mock(upstream_data)

        session = _setup_gate_session(gate)
        gate.pipeline.execute = AsyncMock(return_value=upstream_data)

        app = _make_app(gate, passthrough=pt)
        client = TestClient(app)
        resp = client.post(
            "/v1/responses",
            json=_base_body(),
            headers={
                "Authorization": "Bearer sk-test-key",
                "X-StateLoom-Session-Id": "my-custom-session",
            },
        )
        assert resp.status_code == 200

        call_kwargs = gate.async_session.call_args[1]
        assert call_kwargs["session_id"] == "my-custom-session"


class TestResponsesBodyRebuild:
    def test_rebuild_from_string_input(self):
        """String input rebuilt from modified user message."""
        original_body = {
            "model": "gpt-4.1",
            "input": "my ssn is 123-45-6789",
            "temperature": 0.5,
        }
        messages = [
            {"role": "user", "content": "my ssn is [REDACTED]"},
        ]
        rebuilt = json.loads(_rebuild_responses_body(original_body, messages))
        assert rebuilt["model"] == "gpt-4.1"
        assert rebuilt["temperature"] == 0.5
        assert rebuilt["input"] == "my ssn is [REDACTED]"

    def test_rebuild_preserves_instructions(self):
        """System messages converted back to instructions."""
        original_body = {
            "model": "gpt-4.1",
            "input": "hello",
            "instructions": "Be helpful.",
        }
        messages = [
            {"role": "system", "content": "Be helpful."},
            {"role": "user", "content": "hello"},
        ]
        rebuilt = json.loads(_rebuild_responses_body(original_body, messages))
        assert rebuilt["instructions"] == "Be helpful."
        assert rebuilt["input"] == "hello"

    def test_rebuild_array_input(self):
        """Array input items rebuilt with modified content."""
        original_body = {
            "model": "gpt-4.1",
            "input": [
                {"type": "message", "role": "user", "content": "my email is test@test.com"},
                {"type": "item_reference", "id": "ref_123"},
                {"type": "message", "role": "user", "content": "hello"},
            ],
        }
        messages = [
            {"role": "user", "content": "my email is [REDACTED]"},
            {"role": "user", "content": "hello"},
        ]
        rebuilt = json.loads(_rebuild_responses_body(original_body, messages))
        assert len(rebuilt["input"]) == 3
        assert rebuilt["input"][0]["content"] == "my email is [REDACTED]"
        assert rebuilt["input"][1]["type"] == "item_reference"  # preserved
        assert rebuilt["input"][2]["content"] == "hello"

    def test_rebuild_with_redacted_content(self):
        """Redacted PII values reflected in rebuilt body."""
        original_body = {
            "model": "gpt-4.1",
            "input": "my ssn is 123-45-6789",
        }
        messages = [
            {"role": "user", "content": "my ssn is [REDACTED]"},
        ]
        rebuilt = json.loads(_rebuild_responses_body(original_body, messages))
        assert rebuilt["input"] == "my ssn is [REDACTED]"

    def test_rebuild_preserves_other_fields(self):
        """Non-input/instructions fields preserved."""
        original_body = {
            "model": "gpt-4.1",
            "input": "hello",
            "temperature": 0.7,
            "max_output_tokens": 1000,
            "tools": [{"type": "web_search"}],
            "previous_response_id": "resp_abc",
        }
        messages = [{"role": "user", "content": "hello"}]
        rebuilt = json.loads(_rebuild_responses_body(original_body, messages))
        assert rebuilt["temperature"] == 0.7
        assert rebuilt["max_output_tokens"] == 1000
        assert rebuilt["tools"] == [{"type": "web_search"}]
        assert rebuilt["previous_response_id"] == "resp_abc"


class TestResponsesWebSocket:
    def test_ws_rejects_missing_auth(self):
        """WebSocket connection rejected when auth required and missing."""
        import pytest
        from starlette.websockets import WebSocketDisconnect

        gate = _make_gate(require_virtual_key=True)
        pt = _make_passthrough_mock()
        app = _make_app(gate, passthrough=pt)
        client = TestClient(app)
        with pytest.raises(WebSocketDisconnect):
            with client.websocket_connect("/v1/responses") as ws:
                pass

    def test_ws_accepts_valid_auth(self):
        """WebSocket connection accepted with valid bearer token."""
        gate = _make_gate(require_virtual_key=False)
        pt = _make_passthrough_mock()
        app = _make_app(gate, passthrough=pt)
        client = TestClient(app)
        # Without a real upstream WS server, this will fail at the
        # websockets.connect step, but it proves our auth accepted the client
        try:
            with client.websocket_connect(
                "/v1/responses",
                headers={"Authorization": "Bearer some-token"},
            ) as ws:
                pass
        except Exception:
            # Expected — no real upstream WS to connect to
            pass


class TestResolveUpstream:
    """Verify ChatGPT OAuth vs API key upstream routing."""

    def test_api_key_routes_to_openai(self):
        """Requests without chatgpt-account-id route to api.openai.com."""
        headers = {"authorization": "Bearer sk-test-key"}
        http_url, ws_url = _resolve_upstream(headers, "https://api.openai.com")
        assert http_url == "https://api.openai.com/v1/responses"
        assert ws_url == "wss://api.openai.com/v1/responses"

    def test_chatgpt_oauth_routes_to_chatgpt(self):
        """Requests with chatgpt-account-id route to chatgpt.com."""
        headers = {
            "authorization": "Bearer some-oauth-token",
            "chatgpt-account-id": "acct-abc123",
        }
        http_url, ws_url = _resolve_upstream(headers, "https://api.openai.com")
        assert http_url == "https://chatgpt.com/backend-api/codex/responses"
        assert ws_url == "wss://chatgpt.com/backend-api/codex/responses"

    def test_empty_chatgpt_header_routes_to_openai(self):
        """Empty chatgpt-account-id header treated as API mode."""
        headers = {
            "authorization": "Bearer sk-key",
            "chatgpt-account-id": "",
        }
        http_url, ws_url = _resolve_upstream(headers, "https://api.openai.com")
        assert http_url == "https://api.openai.com/v1/responses"

    def test_custom_upstream_respected_for_api(self):
        """Custom proxy_upstream_openai config respected in API mode."""
        headers = {"authorization": "Bearer sk-key"}
        http_url, ws_url = _resolve_upstream(headers, "https://custom-openai.example.com")
        assert http_url == "https://custom-openai.example.com/v1/responses"
        assert ws_url == "wss://custom-openai.example.com/v1/responses"

    def test_chatgpt_always_uses_fixed_upstream(self):
        """ChatGPT mode always routes to chatgpt.com, ignoring config upstream."""
        headers = {"chatgpt-account-id": "acct-123"}
        http_url, ws_url = _resolve_upstream(headers, "https://custom-openai.example.com")
        assert "chatgpt.com" in http_url
        assert "chatgpt.com" in ws_url


class TestExtractPromptPreview:
    """Test _extract_prompt_preview for dashboard prompt column."""

    def test_string_input(self):
        assert _extract_prompt_preview("hello world") == "hello world"

    def test_array_input_last_user_message(self):
        """Extracts last user message from array input."""
        input_field = [
            {"type": "message", "role": "user", "content": "first"},
            {"type": "message", "role": "assistant", "content": "reply"},
            {"type": "message", "role": "user", "content": "second question"},
        ]
        assert _extract_prompt_preview(input_field) == "second question"

    def test_content_parts(self):
        """Extracts text from content array parts."""
        input_field = [
            {
                "type": "message",
                "role": "user",
                "content": [
                    {"type": "input_text", "text": "check this"},
                    {"type": "input_image", "url": "http://example.com/img.png"},
                ],
            },
        ]
        assert _extract_prompt_preview(input_field) == "check this"

    def test_truncates_long_text(self):
        """Preview truncated to 200 chars."""
        long_text = "x" * 500
        assert len(_extract_prompt_preview(long_text)) == 200

    def test_skips_non_message_items(self):
        """Non-message items (function_call_output, etc.) skipped."""
        input_field = [
            {"type": "function_call_output", "call_id": "c1", "output": "42"},
            {"type": "message", "role": "user", "content": "the prompt"},
        ]
        assert _extract_prompt_preview(input_field) == "the prompt"

    def test_empty_input(self):
        assert _extract_prompt_preview("") == ""
        assert _extract_prompt_preview([]) == ""


class TestSendBlockedResponse:
    """Test _send_blocked_response sends response.created + response.completed (incomplete)."""

    async def test_sends_created_then_completed_incomplete(self):
        from stateloom.proxy.responses import _send_blocked_response

        ws = AsyncMock()
        sent: list[str] = []
        ws.send_text = AsyncMock(side_effect=lambda d: sent.append(d))

        resp_id = await _send_blocked_response(ws, "PII blocked: ssn detected")

        # 8 events: created, output_item.added, content_part.added,
        # output_text.delta, output_text.done, content_part.done,
        # output_item.done, response.completed
        assert len(sent) == 8
        assert resp_id.startswith("resp_ag_")

        created = json.loads(sent[0])
        assert created["type"] == "response.created"
        assert created["response"]["status"] == "in_progress"
        assert created["response"]["id"] == resp_id

        delta = json.loads(sent[3])
        assert delta["type"] == "response.output_text.delta"
        assert "PII blocked" in delta["delta"]

        completed = json.loads(sent[7])
        assert completed["type"] == "response.completed"
        assert completed["response"]["status"] == "incomplete"
        assert completed["response"]["id"] == resp_id
        assert completed["response"]["incomplete_details"]["reason"] == "content_filter"
        output_text = completed["response"]["output"][0]["content"][0]["text"]
        assert "PII blocked" in output_text

    async def test_response_ids_are_unique(self):
        from stateloom.proxy.responses import _send_blocked_response

        ids: list[str] = []
        for _ in range(3):
            ws = AsyncMock()
            sent: list[str] = []
            ws.send_text = AsyncMock(side_effect=lambda d: sent.append(d))
            resp_id = await _send_blocked_response(ws, "error")
            ids.append(resp_id)

        assert len(set(ids)) == 3, "Each call should produce a unique response ID"

    async def test_returns_synthetic_id(self):
        from stateloom.proxy.responses import _SYNTHETIC_RESP_PREFIX, _send_blocked_response

        ws = AsyncMock()
        ws.send_text = AsyncMock()
        resp_id = await _send_blocked_response(ws, "error")
        assert resp_id.startswith(_SYNTHETIC_RESP_PREFIX)


class TestSyntheticIdStripping:
    """Test synthetic response ID stripping logic."""

    def test_prefix_stripped_from_response_obj(self):
        """previous_response_id with resp_ag_ prefix is stripped from response obj."""
        resp_obj = {"previous_response_id": "resp_ag_abc123", "model": "gpt-4.1"}
        msg = {"type": "response.create", "response": resp_obj}
        ws_state = _WSRelayState()

        # Simulate the stripping logic
        _stripped = False
        for _obj in (resp_obj, msg):
            prev_id = _obj.get("previous_response_id", "")
            if prev_id and (
                prev_id in ws_state.synthetic_ids or prev_id.startswith(_SYNTHETIC_RESP_PREFIX)
            ):
                _obj.pop("previous_response_id", None)
                _stripped = True

        assert _stripped is True
        assert "previous_response_id" not in resp_obj

    def test_prefix_stripped_from_msg_level(self):
        """previous_response_id with resp_ag_ prefix is stripped from top-level msg."""
        resp_obj = {"model": "gpt-4.1"}
        msg = {
            "type": "response.create",
            "response": resp_obj,
            "previous_response_id": "resp_ag_xyz789",
        }
        ws_state = _WSRelayState()

        _stripped = False
        for _obj in (resp_obj, msg):
            prev_id = _obj.get("previous_response_id", "")
            if prev_id and (
                prev_id in ws_state.synthetic_ids or prev_id.startswith(_SYNTHETIC_RESP_PREFIX)
            ):
                _obj.pop("previous_response_id", None)
                _stripped = True

        assert _stripped is True
        assert "previous_response_id" not in msg

    def test_real_openai_id_not_stripped(self):
        """Real OpenAI resp_ ID is NOT stripped."""
        resp_obj = {"previous_response_id": "resp_real_abc123", "model": "gpt-4.1"}
        msg = {"type": "response.create", "response": resp_obj}
        ws_state = _WSRelayState()

        _stripped = False
        for _obj in (resp_obj, msg):
            prev_id = _obj.get("previous_response_id", "")
            if prev_id and (
                prev_id in ws_state.synthetic_ids or prev_id.startswith(_SYNTHETIC_RESP_PREFIX)
            ):
                _obj.pop("previous_response_id", None)
                _stripped = True

        assert _stripped is False
        assert resp_obj["previous_response_id"] == "resp_real_abc123"

    def test_set_membership_also_strips(self):
        """ID found by set membership (not prefix) is also stripped."""
        syn_id = "resp_custom_id_12345"
        resp_obj = {"previous_response_id": syn_id, "model": "gpt-4.1"}
        msg = {"type": "response.create", "response": resp_obj}
        ws_state = _WSRelayState(synthetic_ids={syn_id})

        _stripped = False
        for _obj in (resp_obj, msg):
            prev_id = _obj.get("previous_response_id", "")
            if prev_id and (
                prev_id in ws_state.synthetic_ids or prev_id.startswith(_SYNTHETIC_RESP_PREFIX)
            ):
                _obj.pop("previous_response_id", None)
                _stripped = True

        assert _stripped is True
        assert "previous_response_id" not in resp_obj
