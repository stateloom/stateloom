"""Tests for the Code Assist proxy adapter (/code-assist)."""

from __future__ import annotations

import json
import types
from unittest.mock import AsyncMock, MagicMock, patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from stateloom.core.config import StateLoomConfig
from stateloom.core.errors import (
    StateLoomBudgetError,
    StateLoomKillSwitchError,
    StateLoomPIIBlockedError,
    StateLoomRateLimitError,
)
from stateloom.proxy.code_assist import (
    _code_assist_error,
    _contents_to_openai_messages,
    _track_stream_usage,
    create_code_assist_router,
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
    router = create_code_assist_router(gate, passthrough=passthrough)
    app.include_router(router, prefix="/code-assist")
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
        "model": "models/gemini-2.0-flash",
        "request": {
            "contents": [
                {"role": "user", "parts": [{"text": "Hello"}]},
            ],
        },
    }
    body.update(overrides)
    return body


def _make_upstream_response(text="Hello!", prompt_tokens=10, completion_tokens=20):
    """Create a Code Assist-style upstream response dict."""
    return {
        "response": {
            "candidates": [
                {
                    "content": {
                        "parts": [{"text": text}],
                        "role": "model",
                    },
                    "finishReason": "STOP",
                }
            ],
            "usageMetadata": {
                "promptTokenCount": prompt_tokens,
                "candidatesTokenCount": completion_tokens,
                "totalTokenCount": prompt_tokens + completion_tokens,
            },
        },
        "traceId": "abc123",
        "consumedCredits": 1,
        "remainingCredits": 99,
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


class TestCodeAssistNonStreaming:
    def test_non_streaming_passthrough(self):
        """Code Assist request forwarded and response returned as-is."""
        gate = _make_gate(require_virtual_key=False)
        upstream_data = _make_upstream_response("Hello from Code Assist!")
        pt = _make_passthrough_mock(upstream_data)

        # Mock gate.async_session and pipeline
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
        gate.pipeline.execute = AsyncMock(return_value=upstream_data)

        app = _make_app(gate, passthrough=pt)
        client = TestClient(app)
        resp = client.post(
            "/code-assist/v1internal:generateContent",
            json=_base_body(),
            headers={"Authorization": "Bearer some-oauth-token"},
        )

        assert resp.status_code == 200
        data = resp.json()
        # Response forwarded as-is — Code Assist format preserved
        assert "response" in data
        assert "traceId" in data
        assert "consumedCredits" in data
        assert (
            data["response"]["candidates"][0]["content"]["parts"][0]["text"]
            == "Hello from Code Assist!"
        )

    def test_no_passthrough_returns_503(self):
        """Returns 503 when no PassthroughProxy is available."""
        gate = _make_gate(require_virtual_key=False)
        app = _make_app(gate, passthrough=None)
        client = TestClient(app)
        resp = client.post(
            "/code-assist/v1internal:generateContent",
            json=_base_body(),
            headers={"Authorization": "Bearer some-oauth-token"},
        )
        assert resp.status_code == 503
        data = resp.json()
        assert data["error"]["status"] == "UNAVAILABLE"


class TestCodeAssistAuth:
    def test_oauth_bearer_forwarded(self):
        """OAuth bearer token forwarded to upstream."""
        gate = _make_gate(require_virtual_key=False)
        upstream_data = _make_upstream_response()
        pt = _make_passthrough_mock(upstream_data)

        # Mock gate.async_session and pipeline
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
        gate.pipeline.execute = AsyncMock(return_value=upstream_data)

        app = _make_app(gate, passthrough=pt)
        client = TestClient(app)
        resp = client.post(
            "/code-assist/v1internal:generateContent",
            json=_base_body(),
            headers={"Authorization": "Bearer ya29.oauth-token-here"},
        )
        assert resp.status_code == 200

    def test_virtual_key_auth(self):
        """ag- prefixed token treated as virtual key."""
        gate = _make_gate(require_virtual_key=True)
        full_key = _setup_virtual_key(gate)
        upstream_data = _make_upstream_response()
        pt = _make_passthrough_mock(upstream_data)

        # Mock gate.async_session and pipeline
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
        gate.pipeline.execute = AsyncMock(return_value=upstream_data)

        app = _make_app(gate, passthrough=pt)
        client = TestClient(app)
        resp = client.post(
            "/code-assist/v1internal:generateContent",
            json=_base_body(),
            headers={"Authorization": f"Bearer {full_key}"},
        )
        assert resp.status_code == 200

    def test_no_auth_mode(self):
        """Unauthenticated allowed when configured."""
        gate = _make_gate(require_virtual_key=False)
        upstream_data = _make_upstream_response()
        pt = _make_passthrough_mock(upstream_data)

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
        gate.pipeline.execute = AsyncMock(return_value=upstream_data)

        app = _make_app(gate, passthrough=pt)
        client = TestClient(app)
        resp = client.post(
            "/code-assist/v1internal:generateContent",
            json=_base_body(),
        )
        assert resp.status_code == 200

    def test_rejects_missing_auth(self):
        """Missing auth when required returns 401."""
        gate = _make_gate(require_virtual_key=True)
        pt = _make_passthrough_mock()
        app = _make_app(gate, passthrough=pt)
        client = TestClient(app)
        resp = client.post(
            "/code-assist/v1internal:generateContent",
            json=_base_body(),
        )
        assert resp.status_code == 401
        data = resp.json()
        assert data["error"]["status"] == "UNAUTHENTICATED"

    def test_rejects_invalid_vk(self):
        """Invalid ag- key returns 401."""
        gate = _make_gate(require_virtual_key=True)
        pt = _make_passthrough_mock()
        app = _make_app(gate, passthrough=pt)
        client = TestClient(app)
        resp = client.post(
            "/code-assist/v1internal:generateContent",
            json=_base_body(),
            headers={"Authorization": "Bearer ag-invalid-key-12345"},
        )
        assert resp.status_code == 401


class TestCodeAssistValidation:
    def test_missing_model(self):
        """400 for missing model field."""
        gate = _make_gate(require_virtual_key=False)
        pt = _make_passthrough_mock()
        app = _make_app(gate, passthrough=pt)
        client = TestClient(app)
        resp = client.post(
            "/code-assist/v1internal:generateContent",
            json={"request": {"contents": [{"role": "user", "parts": [{"text": "Hi"}]}]}},
            headers={"Authorization": "Bearer token"},
        )
        assert resp.status_code == 400
        assert "model" in resp.json()["error"]["message"]

    def test_missing_contents(self):
        """400 for missing request.contents."""
        gate = _make_gate(require_virtual_key=False)
        pt = _make_passthrough_mock()
        app = _make_app(gate, passthrough=pt)
        client = TestClient(app)
        resp = client.post(
            "/code-assist/v1internal:generateContent",
            json={"model": "models/gemini-2.0-flash", "request": {}},
            headers={"Authorization": "Bearer token"},
        )
        assert resp.status_code == 400
        assert "contents" in resp.json()["error"]["message"]

    def test_invalid_json(self):
        """400 for invalid JSON body."""
        gate = _make_gate(require_virtual_key=False)
        pt = _make_passthrough_mock()
        app = _make_app(gate, passthrough=pt)
        client = TestClient(app)
        resp = client.post(
            "/code-assist/v1internal:generateContent",
            content=b"not-json",
            headers={"Content-Type": "application/json", "Authorization": "Bearer token"},
        )
        assert resp.status_code == 400


class TestCodeAssistMessageConversion:
    def test_contents_to_openai_messages(self):
        """Nested contents converted to OpenAI messages format."""
        contents = [
            {"role": "user", "parts": [{"text": "Hello"}]},
            {"role": "model", "parts": [{"text": "Hi there!"}]},
            {"role": "user", "parts": [{"text": "How are you?"}]},
        ]
        messages = _contents_to_openai_messages(contents)
        assert len(messages) == 3
        assert messages[0]["role"] == "user"
        assert messages[0]["content"] == "Hello"
        assert messages[0]["_content_idx"] == 0
        assert messages[1]["role"] == "assistant"
        assert messages[1]["content"] == "Hi there!"
        assert messages[1]["_content_idx"] == 1
        assert messages[2]["role"] == "user"
        assert messages[2]["content"] == "How are you?"
        assert messages[2]["_content_idx"] == 2

    def test_system_instruction_prepended(self):
        """systemInstruction converted to system message."""
        contents = [{"role": "user", "parts": [{"text": "Hello"}]}]
        system_instruction = {"parts": [{"text": "You are helpful."}]}
        messages = _contents_to_openai_messages(contents, system_instruction)
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == "You are helpful."
        assert messages[0]["_content_idx"] == -1
        assert messages[1]["role"] == "user"
        assert messages[1]["content"] == "Hello"

    def test_string_system_instruction(self):
        """String systemInstruction handled."""
        contents = [{"role": "user", "parts": [{"text": "Hello"}]}]
        messages = _contents_to_openai_messages(contents, "Be concise.")
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == "Be concise."

    def test_empty_contents_returns_empty(self):
        """Empty contents returns empty list."""
        assert _contents_to_openai_messages([]) == []

    def test_multi_part_text(self):
        """Multiple text parts joined."""
        contents = [{"role": "user", "parts": [{"text": "Hello"}, {"text": "World"}]}]
        messages = _contents_to_openai_messages(contents)
        assert messages[0]["content"] == "Hello\nWorld"


class TestCodeAssistStreamTokenTracking:
    def test_track_nested_usage_metadata(self):
        """Token usage extracted from Code Assist nested response.usageMetadata."""
        ctx = MagicMock()
        ctx.prompt_tokens = 0
        ctx.completion_tokens = 0

        sse_line = 'data: {"response": {"usageMetadata": {"promptTokenCount": 15, "candidatesTokenCount": 25, "totalTokenCount": 40}}}\n\n'
        _track_stream_usage(sse_line, ctx)

        assert ctx.prompt_tokens == 15
        assert ctx.completion_tokens == 25

    def test_track_no_usage_metadata(self):
        """No crash when usage metadata missing."""
        ctx = MagicMock()
        ctx.prompt_tokens = 0
        ctx.completion_tokens = 0

        sse_line = 'data: {"response": {"candidates": []}}\n\n'
        _track_stream_usage(sse_line, ctx)
        # Should not error

    def test_track_none_ctx(self):
        """No crash when ctx is None."""
        _track_stream_usage("data: {}\n\n", None)


class TestCodeAssistBillingMode:
    def test_billing_always_subscription(self):
        """Code Assist billing mode defaults to subscription."""
        gate = _make_gate(require_virtual_key=False)
        upstream_data = _make_upstream_response()
        pt = _make_passthrough_mock(upstream_data)

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
        gate.pipeline.execute = AsyncMock(return_value=upstream_data)

        app = _make_app(gate, passthrough=pt)
        client = TestClient(app)
        resp = client.post(
            "/code-assist/v1internal:generateContent",
            json=_base_body(),
            headers={"Authorization": "Bearer ya29.oauth-token"},
        )
        assert resp.status_code == 200
        # Verify billing_mode was set to subscription on session metadata
        assert session.metadata.get("billing_mode") == "subscription"


class TestCodeAssistErrorFormat:
    def test_error_format(self):
        """StateLoom errors mapped to Google error format."""
        errors_and_expected = [
            (StateLoomRateLimitError("team-1", 10.0, 100), 429, "RESOURCE_EXHAUSTED"),
            (StateLoomBudgetError(5.0, 6.0, "s-1"), 400, "INVALID_ARGUMENT"),
            (StateLoomPIIBlockedError("email", "s-1"), 400, "INVALID_ARGUMENT"),
            (StateLoomKillSwitchError(), 503, "UNAVAILABLE"),
        ]

        for error, expected_status, expected_status_str in errors_and_expected:
            gate = _make_gate(require_virtual_key=False)
            pt = _make_passthrough_mock()

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
            gate.pipeline.execute = AsyncMock(side_effect=error)

            app = _make_app(gate, passthrough=pt)
            client = TestClient(app)
            resp = client.post(
                "/code-assist/v1internal:generateContent",
                json=_base_body(),
                headers={"Authorization": "Bearer oauth-token"},
            )

            assert resp.status_code == expected_status, (
                f"Expected {expected_status} for {type(error).__name__}, got {resp.status_code}"
            )
            data = resp.json()
            assert "error" in data
            assert data["error"]["status"] == expected_status_str


class TestCodeAssistUtilityPassthrough:
    def test_load_code_assist_forwarded(self):
        """loadCodeAssist forwarded as-is, no middleware."""
        gate = _make_gate(require_virtual_key=False)
        pt = MagicMock()

        utility_resp = {"status": "ok", "features": ["chat", "code"]}
        resp_mock = MagicMock()
        resp_mock.status_code = 200
        resp_mock.content = json.dumps(utility_resp).encode()
        resp_mock.headers = {"content-type": "application/json"}
        pt.forward = AsyncMock(return_value=resp_mock)

        app = _make_app(gate, passthrough=pt)
        client = TestClient(app)
        resp = client.post(
            "/code-assist/v1internal:loadCodeAssist",
            json={"key": "value"},
            headers={"Authorization": "Bearer oauth-token"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"

        # Verify passthrough was called with correct URL
        call_args = pt.forward.call_args
        url = call_args[0][0]
        assert "v1internal:loadCodeAssist" in url

    def test_operations_polling_forwarded(self):
        """Path-based operations forwarded as-is."""
        gate = _make_gate(require_virtual_key=False)
        pt = MagicMock()

        op_resp = {"name": "operations/123", "done": False}
        resp_mock = MagicMock()
        resp_mock.status_code = 200
        resp_mock.content = json.dumps(op_resp).encode()
        resp_mock.headers = {"content-type": "application/json"}
        pt.forward = AsyncMock(return_value=resp_mock)

        app = _make_app(gate, passthrough=pt)
        client = TestClient(app)
        resp = client.get(
            "/code-assist/v1internal/operations/123",
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["name"] == "operations/123"

    def test_utility_no_passthrough_returns_503(self):
        """Utility endpoints return 503 without passthrough."""
        gate = _make_gate(require_virtual_key=False)
        app = _make_app(gate, passthrough=None)
        client = TestClient(app)
        resp = client.post(
            "/code-assist/v1internal:loadCodeAssist",
            json={},
        )
        assert resp.status_code == 503


class TestCodeAssistStreaming:
    def test_streaming_returns_sse(self):
        """streamGenerateContent returns SSE."""
        gate = _make_gate(require_virtual_key=False)
        pt = MagicMock()

        # Mock streaming response
        sse_chunk = 'data: {"response": {"candidates": [{"content": {"parts": [{"text": "Hi"}], "role": "model"}}], "usageMetadata": {"promptTokenCount": 5, "candidatesTokenCount": 10}}}\n\n'

        async def mock_stream(*args, **kwargs):
            yield sse_chunk.encode("utf-8")

        pt.forward_stream = mock_stream

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
        gate.pipeline.execute_streaming = AsyncMock()

        app = _make_app(gate, passthrough=pt)
        client = TestClient(app)
        resp = client.post(
            "/code-assist/v1internal:streamGenerateContent",
            json=_base_body(),
            headers={"Authorization": "Bearer oauth-token"},
        )

        assert resp.status_code == 200
        assert "text/event-stream" in resp.headers.get("content-type", "")
        body = resp.text
        assert "data: " in body


class TestCodeAssistSessionHeader:
    def test_session_id_header_used(self):
        """X-StateLoom-Session-Id header used when provided."""
        gate = _make_gate(require_virtual_key=False)
        upstream_data = _make_upstream_response()
        pt = _make_passthrough_mock(upstream_data)

        gate.async_session = MagicMock()
        session = MagicMock()
        session.id = "my-custom-session"
        session.metadata = {}
        session.step_counter = 0
        session.next_step = MagicMock()

        ctx_mgr = AsyncMock()
        ctx_mgr.__aenter__ = AsyncMock(return_value=session)
        ctx_mgr.__aexit__ = AsyncMock(return_value=None)
        gate.async_session.return_value = ctx_mgr

        gate.pipeline._hash_request = MagicMock(return_value="hash123")
        gate.pipeline.execute = AsyncMock(return_value=upstream_data)

        app = _make_app(gate, passthrough=pt)
        client = TestClient(app)
        resp = client.post(
            "/code-assist/v1internal:generateContent",
            json=_base_body(),
            headers={
                "Authorization": "Bearer oauth-token",
                "X-StateLoom-Session-Id": "my-custom-session",
            },
        )
        assert resp.status_code == 200

        # Verify session_id was passed to async_session
        call_kwargs = gate.async_session.call_args[1]
        assert call_kwargs["session_id"] == "my-custom-session"


class TestCodeAssistUpstreamError:
    def test_upstream_error_forwarded(self):
        """Upstream 4xx/5xx forwarded with original status code."""
        gate = _make_gate(require_virtual_key=False)
        error_data = {"error": {"code": 403, "message": "Forbidden", "status": "PERMISSION_DENIED"}}
        pt = _make_passthrough_mock(error_data, status_code=403)

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

        # The llm_call inside _handle_passthrough marks upstream errors
        upstream_error = {
            "_upstream_error": True,
            "_status_code": 403,
            **error_data,
        }
        gate.pipeline.execute = AsyncMock(return_value=upstream_error)

        app = _make_app(gate, passthrough=pt)
        client = TestClient(app)
        resp = client.post(
            "/code-assist/v1internal:generateContent",
            json=_base_body(),
            headers={"Authorization": "Bearer oauth-token"},
        )
        assert resp.status_code == 403
        data = resp.json()
        assert data["error"]["status"] == "PERMISSION_DENIED"


class TestCodeAssistCostTracker:
    def test_extract_tokens_from_code_assist_format(self):
        """CostTracker extracts tokens from Code Assist nested format."""
        from stateloom.middleware.cost_tracker import CostTracker

        data = {
            "response": {
                "usageMetadata": {
                    "promptTokenCount": 42,
                    "candidatesTokenCount": 18,
                    "totalTokenCount": 60,
                }
            },
            "traceId": "abc",
        }
        prompt, completion = CostTracker._extract_tokens_from_dict(data)
        assert prompt == 42
        assert completion == 18

    def test_extract_tokens_standard_gemini_unaffected(self):
        """Standard Gemini format still works."""
        from stateloom.middleware.cost_tracker import CostTracker

        data = {
            "usageMetadata": {
                "promptTokenCount": 10,
                "candidatesTokenCount": 20,
            }
        }
        prompt, completion = CostTracker._extract_tokens_from_dict(data)
        assert prompt == 10
        assert completion == 20


class TestCodeAssistBodyPatch:
    def test_patch_strips_pii_messages(self):
        """Body is patched when middleware strips messages (e.g. PII removal)."""
        from stateloom.proxy.code_assist import _patch_code_assist_body

        original_body = {
            "model": "models/gemini-2.0-flash",
            "request": {
                "contents": [
                    {"role": "user", "parts": [{"text": "my ssn is 123-12-1234"}]},
                    {"role": "model", "parts": [{"text": "I see your SSN"}]},
                    {"role": "user", "parts": [{"text": "hello"}]},
                ],
            },
            "traceId": "abc",
        }
        original_messages = [
            {"role": "user", "content": "my ssn is 123-12-1234", "_content_idx": 0},
            {"role": "assistant", "content": "I see your SSN", "_content_idx": 1},
            {"role": "user", "content": "hello", "_content_idx": 2},
        ]
        # After PII stripping, the SSN message and its response are removed
        current_messages = [
            {"role": "user", "content": "hello", "_content_idx": 2},
        ]
        rebuilt = json.loads(
            _patch_code_assist_body(original_body, original_messages, current_messages)
        )

        assert rebuilt["model"] == "models/gemini-2.0-flash"
        assert rebuilt["traceId"] == "abc"  # preserved
        assert len(rebuilt["request"]["contents"]) == 1
        assert rebuilt["request"]["contents"][0]["parts"][0]["text"] == "hello"

    def test_patch_removes_system_instruction_when_stripped(self):
        """systemInstruction removed if system message was stripped."""
        from stateloom.proxy.code_assist import _patch_code_assist_body

        original_body = {
            "model": "m",
            "request": {
                "contents": [{"role": "user", "parts": [{"text": "hi"}]}],
                "systemInstruction": {"parts": [{"text": "old system"}]},
            },
        }
        original_messages = [
            {"role": "system", "content": "old system", "_content_idx": -1},
            {"role": "user", "content": "hi", "_content_idx": 0},
        ]
        current_messages = [
            {"role": "user", "content": "hi", "_content_idx": 0},
        ]
        rebuilt = json.loads(
            _patch_code_assist_body(original_body, original_messages, current_messages)
        )

        assert "systemInstruction" not in rebuilt["request"]

    def test_patch_redacts_text(self):
        """Redacted PII values are applied to the original body's text parts."""
        from stateloom.proxy.code_assist import _patch_code_assist_body

        original_body = {
            "model": "m",
            "request": {
                "contents": [
                    {"role": "user", "parts": [{"text": "my ssn is 123-12-1234"}]},
                ],
            },
        }
        original_messages = [
            {"role": "user", "content": "my ssn is 123-12-1234", "_content_idx": 0},
        ]
        current_messages = [
            {"role": "user", "content": "my ssn is [REDACTED]", "_content_idx": 0},
        ]
        rebuilt = json.loads(
            _patch_code_assist_body(original_body, original_messages, current_messages)
        )

        assert rebuilt["request"]["contents"][0]["parts"][0]["text"] == "my ssn is [REDACTED]"

    def test_patch_preserves_thought_signature(self):
        """Gemini-specific fields like thought_signature are preserved."""
        from stateloom.proxy.code_assist import _patch_code_assist_body

        original_body = {
            "model": "m",
            "request": {
                "contents": [
                    {"role": "user", "parts": [{"text": "my ssn is 123-12-1234"}]},
                    {
                        "role": "model",
                        "parts": [
                            {"text": "I'll search for that."},
                            {
                                "functionCall": {
                                    "name": "google_web_search",
                                    "args": {"query": "test"},
                                },
                                "thought_signature": "abc123signature",
                            },
                        ],
                    },
                    {
                        "role": "user",
                        "parts": [
                            {
                                "functionResponse": {
                                    "name": "google_web_search",
                                    "response": {"result": "ok"},
                                }
                            },
                        ],
                    },
                    {"role": "user", "parts": [{"text": "what about now?"}]},
                ],
            },
        }
        original_messages = [
            {"role": "user", "content": "my ssn is 123-12-1234", "_content_idx": 0},
            {
                "role": "assistant",
                "content": "I'll search for that.",
                "tool_calls": [...],
                "_content_idx": 1,
            },
            {
                "role": "tool",
                "content": '{"result": "ok"}',
                "tool_call_id": "google_web_search",
                "_content_idx": 2,
            },
            {"role": "user", "content": "what about now?", "_content_idx": 3},
        ]
        # Strip the SSN message (index 0)
        current_messages = [
            {
                "role": "assistant",
                "content": "I'll search for that.",
                "tool_calls": [...],
                "_content_idx": 1,
            },
            {
                "role": "tool",
                "content": '{"result": "ok"}',
                "tool_call_id": "google_web_search",
                "_content_idx": 2,
            },
            {"role": "user", "content": "what about now?", "_content_idx": 3},
        ]
        rebuilt = json.loads(
            _patch_code_assist_body(original_body, original_messages, current_messages)
        )

        contents = rebuilt["request"]["contents"]
        assert len(contents) == 3  # SSN message stripped
        # functionCall entry should still have thought_signature
        model_entry = contents[0]
        assert model_entry["role"] == "model"
        fc_part = model_entry["parts"][1]
        assert "thought_signature" in fc_part
        assert fc_part["thought_signature"] == "abc123signature"

    def test_patch_no_changes_returns_original(self):
        """When messages are identical, body is unchanged."""
        from stateloom.proxy.code_assist import _patch_code_assist_body

        original_body = {
            "model": "m",
            "request": {
                "contents": [{"role": "user", "parts": [{"text": "hi"}]}],
            },
        }
        messages = [{"role": "user", "content": "hi", "_content_idx": 0}]
        rebuilt = json.loads(_patch_code_assist_body(original_body, messages, messages))

        assert rebuilt["request"]["contents"][0]["parts"][0]["text"] == "hi"


class TestCodeAssistErrorHelper:
    def test_code_assist_error_format(self):
        """Error helper produces correct Google-style error."""
        result = _code_assist_error(400, "Bad request", "INVALID_ARGUMENT")
        assert result == {
            "error": {
                "code": 400,
                "message": "Bad request",
                "status": "INVALID_ARGUMENT",
            },
        }


class TestCodeAssistGeminiSessionId:
    """Verify that Gemini CLI's session_id is used instead of sticky sessions."""

    def test_gemini_session_id_used(self):
        """request.session_id from Gemini CLI becomes the StateLoom session ID."""
        gate = _make_gate(require_virtual_key=False)
        upstream_data = _make_upstream_response()
        pt = _make_passthrough_mock(upstream_data)

        gate.async_session = MagicMock()
        session = MagicMock()
        session.id = "abc123"
        session.metadata = {}
        session.step_counter = 0
        session.next_step = MagicMock()

        ctx_mgr = AsyncMock()
        ctx_mgr.__aenter__ = AsyncMock(return_value=session)
        ctx_mgr.__aexit__ = AsyncMock(return_value=None)
        gate.async_session.return_value = ctx_mgr

        gate.pipeline._hash_request = MagicMock(return_value="hash123")
        gate.pipeline.execute = AsyncMock(return_value=upstream_data)

        app = _make_app(gate, passthrough=pt)
        client = TestClient(app)
        body = _base_body()
        body["request"]["session_id"] = "abc123"
        resp = client.post(
            "/code-assist/v1internal:generateContent",
            json=body,
            headers={"Authorization": "Bearer oauth-token"},
        )
        assert resp.status_code == 200

        # Verify Gemini's session_id used directly
        call_kwargs = gate.async_session.call_args[1]
        assert call_kwargs["session_id"] == "abc123"

    def test_explicit_header_takes_priority(self):
        """X-StateLoom-Session-Id header overrides Gemini session_id."""
        gate = _make_gate(require_virtual_key=False)
        upstream_data = _make_upstream_response()
        pt = _make_passthrough_mock(upstream_data)

        gate.async_session = MagicMock()
        session = MagicMock()
        session.id = "explicit-session"
        session.metadata = {}
        session.step_counter = 0
        session.next_step = MagicMock()

        ctx_mgr = AsyncMock()
        ctx_mgr.__aenter__ = AsyncMock(return_value=session)
        ctx_mgr.__aexit__ = AsyncMock(return_value=None)
        gate.async_session.return_value = ctx_mgr

        gate.pipeline._hash_request = MagicMock(return_value="hash123")
        gate.pipeline.execute = AsyncMock(return_value=upstream_data)

        app = _make_app(gate, passthrough=pt)
        client = TestClient(app)
        body = _base_body()
        body["request"]["session_id"] = "gemini-id"
        resp = client.post(
            "/code-assist/v1internal:generateContent",
            json=body,
            headers={
                "Authorization": "Bearer oauth-token",
                "X-StateLoom-Session-Id": "explicit-session",
            },
        )
        assert resp.status_code == 200

        call_kwargs = gate.async_session.call_args[1]
        assert call_kwargs["session_id"] == "explicit-session"


class TestCodeAssistCliInternal:
    """Verify CLI-internal detection via user_prompt_id."""

    def test_session_summary_marked_cli_internal(self):
        """user_prompt_id='session-summary-generation' sets _cli_internal flag."""
        gate = _make_gate(require_virtual_key=False)
        upstream_data = _make_upstream_response()
        pt = _make_passthrough_mock(upstream_data)

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
        gate.pipeline.execute = AsyncMock(return_value=upstream_data)

        app = _make_app(gate, passthrough=pt)
        client = TestClient(app)
        body = _base_body()
        body["user_prompt_id"] = "session-summary-generation"
        resp = client.post(
            "/code-assist/v1internal:generateContent",
            json=body,
            headers={"Authorization": "Bearer oauth-token"},
        )
        assert resp.status_code == 200

        # Verify _cli_internal was set in request_kwargs
        execute_call = gate.pipeline.execute.call_args
        ctx = execute_call[0][0]  # first positional arg = MiddlewareContext
        assert ctx.request_kwargs.get("_cli_internal") is True

    def test_normal_prompt_not_cli_internal(self):
        """Regular user_prompt_id does NOT set _cli_internal flag."""
        gate = _make_gate(require_virtual_key=False)
        upstream_data = _make_upstream_response()
        pt = _make_passthrough_mock(upstream_data)

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
        gate.pipeline.execute = AsyncMock(return_value=upstream_data)

        app = _make_app(gate, passthrough=pt)
        client = TestClient(app)
        body = _base_body()
        body["user_prompt_id"] = "abc-123########0"
        resp = client.post(
            "/code-assist/v1internal:generateContent",
            json=body,
            headers={"Authorization": "Bearer oauth-token"},
        )
        assert resp.status_code == 200

        execute_call = gate.pipeline.execute.call_args
        ctx = execute_call[0][0]
        assert "_cli_internal" not in ctx.request_kwargs


class TestCodeAssistToolContinuation:
    """Verify functionCall/functionResponse translation enables tool continuation detection."""

    def test_function_response_detected_as_tool_continuation(self):
        """functionResponse in last content item → _is_tool_continuation returns True."""
        from stateloom.middleware.cost_tracker import _is_tool_continuation

        contents = [
            {"role": "user", "parts": [{"text": "check the code"}]},
            {
                "role": "model",
                "parts": [
                    {"text": "I'll look at the files."},
                    {"functionCall": {"name": "read_file", "args": {"path": "main.py"}}},
                ],
            },
            {
                "role": "user",
                "parts": [
                    {
                        "functionResponse": {
                            "name": "read_file",
                            "response": {"content": "print('hi')"},
                        }
                    },
                ],
            },
        ]
        messages = _contents_to_openai_messages(contents)
        assert _is_tool_continuation(messages) is True

    def test_text_only_not_continuation(self):
        """Plain text contents → NOT a tool continuation."""
        from stateloom.middleware.cost_tracker import _is_tool_continuation

        contents = [
            {"role": "user", "parts": [{"text": "check the code"}]},
        ]
        messages = _contents_to_openai_messages(contents)
        assert _is_tool_continuation(messages) is False

    def test_function_call_translated_to_tool_calls(self):
        """functionCall parts become assistant tool_calls in OpenAI format."""
        contents = [
            {
                "role": "model",
                "parts": [
                    {"text": "Let me check."},
                    {"functionCall": {"name": "search", "args": {"q": "test"}}},
                ],
            },
        ]
        messages = _contents_to_openai_messages(contents)
        assert len(messages) == 1
        msg = messages[0]
        assert msg["role"] == "assistant"
        assert msg["content"] == "Let me check."
        assert len(msg["tool_calls"]) == 1
        assert msg["tool_calls"][0]["function"]["name"] == "search"

    def test_multi_turn_new_prompt_after_tools_not_continuation(self):
        """New user text after prior functionResponse history → NOT a continuation."""
        from stateloom.middleware.cost_tracker import _is_tool_continuation

        contents = [
            {"role": "user", "parts": [{"text": "check the code"}]},
            {
                "role": "model",
                "parts": [
                    {"functionCall": {"name": "read_file", "args": {}}},
                ],
            },
            {
                "role": "user",
                "parts": [
                    {"functionResponse": {"name": "read_file", "response": {"content": "..."}}},
                ],
            },
            {"role": "model", "parts": [{"text": "I found the file."}]},
            {"role": "user", "parts": [{"text": "now summarize it"}]},
        ]
        messages = _contents_to_openai_messages(contents)
        assert _is_tool_continuation(messages) is False
