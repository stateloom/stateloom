"""Tests for the Gemini-native /v1beta/models/{model}:generateContent proxy endpoint."""

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
from stateloom.proxy.gemini_native import create_gemini_router
from stateloom.proxy.virtual_key import (
    VirtualKey,
    generate_virtual_key,
    make_key_preview,
    make_virtual_key_id,
)
from stateloom.store.memory_store import MemoryStore

# Mock target: Client is now imported lazily inside the function body
_CLIENT_MOCK_TARGET = "stateloom.chat.Client"


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


def _make_app(gate: MagicMock) -> FastAPI:
    app = FastAPI()
    router = create_gemini_router(gate)
    app.include_router(router, prefix="/v1beta")
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


def _mock_gemini_response(text="Hello!", model="gemini-2.0-flash"):
    """Create a mock response with text content (not a Gemini SDK object)."""
    # Return a dict that looks like a pipeline response with extracted text
    return types.SimpleNamespace(
        choices=[
            types.SimpleNamespace(
                message=types.SimpleNamespace(content=text, role="assistant"),
                finish_reason="stop",
            )
        ],
        usage=types.SimpleNamespace(
            prompt_tokens=10,
            completion_tokens=20,
            total_tokens=30,
        ),
    )


def _base_body(**overrides):
    body = {
        "contents": [
            {"role": "user", "parts": [{"text": "Hello"}]},
        ],
    }
    body.update(overrides)
    return body


class TestGeminiNonStreaming:
    def test_non_streaming(self):
        """Gemini request → Gemini response format."""
        gate = _make_gate(require_virtual_key=False)
        app = _make_app(gate)

        mock_resp = MagicMock()
        mock_resp.raw = {
            "choices": [{"message": {"role": "assistant", "content": "Hello from Gemini!"}}]
        }

        with patch(_CLIENT_MOCK_TARGET) as mock_client:
            instance = MagicMock()
            mock_client.return_value = instance
            instance.__aenter__ = AsyncMock(return_value=instance)
            instance.__aexit__ = AsyncMock(return_value=None)
            instance.achat = AsyncMock(return_value=mock_resp)

            client = TestClient(app)
            resp = client.post(
                "/v1beta/models/gemini-2.0-flash:generateContent",
                json=_base_body(),
            )

            assert resp.status_code == 200
            data = resp.json()
            assert "candidates" in data
            assert data["candidates"][0]["content"]["role"] == "model"
            assert data["candidates"][0]["content"]["parts"][0]["text"] == "Hello from Gemini!"
            assert data["candidates"][0]["finishReason"] == "STOP"
            assert "usageMetadata" in data

    def test_dict_response_fallback(self):
        """Kill switch dicts wrapped in Gemini format."""
        gate = _make_gate(require_virtual_key=False)
        app = _make_app(gate)

        mock_resp = MagicMock()
        mock_resp.raw = {
            "choices": [{"message": {"role": "assistant", "content": "Service paused"}}]
        }

        with patch(_CLIENT_MOCK_TARGET) as mock_client:
            instance = MagicMock()
            mock_client.return_value = instance
            instance.__aenter__ = AsyncMock(return_value=instance)
            instance.__aexit__ = AsyncMock(return_value=None)
            instance.achat = AsyncMock(return_value=mock_resp)

            client = TestClient(app)
            resp = client.post(
                "/v1beta/models/gemini-2.0-flash:generateContent",
                json=_base_body(),
            )

            assert resp.status_code == 200
            data = resp.json()
            assert "candidates" in data
            assert data["candidates"][0]["content"]["parts"][0]["text"] == "Service paused"


class TestGeminiAuth:
    def test_auth_x_goog_api_key_byok(self):
        """x-goog-api-key treated as BYOK."""
        gate = _make_gate(require_virtual_key=True)
        app = _make_app(gate)

        mock_resp = MagicMock()
        mock_resp.raw = {"choices": [{"message": {"role": "assistant", "content": "ok"}}]}

        with patch(_CLIENT_MOCK_TARGET) as mock_client:
            instance = MagicMock()
            mock_client.return_value = instance
            instance.__aenter__ = AsyncMock(return_value=instance)
            instance.__aexit__ = AsyncMock(return_value=None)
            instance.achat = AsyncMock(return_value=mock_resp)

            client = TestClient(app)
            resp = client.post(
                "/v1beta/models/gemini-2.0-flash:generateContent",
                json=_base_body(),
                headers={"x-goog-api-key": "AIzaSyRealGoogleKey123"},
            )
            assert resp.status_code == 200

            # Verify provider_keys has the Google key
            call_kwargs = mock_client.call_args[1]
            assert call_kwargs["provider_keys"]["google"] == "AIzaSyRealGoogleKey123"

    def test_auth_x_goog_api_key_virtual_key(self):
        """x-goog-api-key: ag-xxx validated as VK."""
        gate = _make_gate(require_virtual_key=True)
        full_key = _setup_virtual_key(gate)
        app = _make_app(gate)

        mock_resp = MagicMock()
        mock_resp.raw = {"choices": [{"message": {"role": "assistant", "content": "ok"}}]}

        with patch(_CLIENT_MOCK_TARGET) as mock_client:
            instance = MagicMock()
            mock_client.return_value = instance
            instance.__aenter__ = AsyncMock(return_value=instance)
            instance.__aexit__ = AsyncMock(return_value=None)
            instance.achat = AsyncMock(return_value=mock_resp)

            client = TestClient(app)
            resp = client.post(
                "/v1beta/models/gemini-2.0-flash:generateContent",
                json=_base_body(),
                headers={"x-goog-api-key": full_key},
            )
            assert resp.status_code == 200

    def test_no_auth_mode(self):
        """Unauthenticated allowed when configured."""
        gate = _make_gate(require_virtual_key=False)
        app = _make_app(gate)

        mock_resp = MagicMock()
        mock_resp.raw = {"choices": [{"message": {"role": "assistant", "content": "ok"}}]}

        with patch(_CLIENT_MOCK_TARGET) as mock_client:
            instance = MagicMock()
            mock_client.return_value = instance
            instance.__aenter__ = AsyncMock(return_value=instance)
            instance.__aexit__ = AsyncMock(return_value=None)
            instance.achat = AsyncMock(return_value=mock_resp)

            client = TestClient(app)
            resp = client.post(
                "/v1beta/models/gemini-2.0-flash:generateContent",
                json=_base_body(),
            )
            assert resp.status_code == 200

    def test_rejects_missing_auth(self):
        """Missing auth when required returns 401 in Gemini format."""
        gate = _make_gate(require_virtual_key=True)
        client = TestClient(_make_app(gate))
        resp = client.post(
            "/v1beta/models/gemini-2.0-flash:generateContent",
            json=_base_body(),
        )
        assert resp.status_code == 401
        data = resp.json()
        assert "error" in data
        assert data["error"]["status"] == "UNAUTHENTICATED"

    def test_rejects_invalid_vk(self):
        """Invalid ag- key returns 401."""
        gate = _make_gate(require_virtual_key=True)
        client = TestClient(_make_app(gate))
        resp = client.post(
            "/v1beta/models/gemini-2.0-flash:generateContent",
            json=_base_body(),
            headers={"x-goog-api-key": "ag-invalid-key-12345"},
        )
        assert resp.status_code == 401


class TestGeminiValidation:
    def test_missing_contents(self):
        """400 in Gemini error format for missing contents."""
        gate = _make_gate(require_virtual_key=False)
        client = TestClient(_make_app(gate))
        resp = client.post(
            "/v1beta/models/gemini-2.0-flash:generateContent",
            json={},
        )
        assert resp.status_code == 400
        data = resp.json()
        assert data["error"]["status"] == "INVALID_ARGUMENT"
        assert "contents" in data["error"]["message"]

    def test_invalid_json(self):
        gate = _make_gate(require_virtual_key=False)
        client = TestClient(_make_app(gate))
        resp = client.post(
            "/v1beta/models/gemini-2.0-flash:generateContent",
            content=b"not-json",
            headers={"Content-Type": "application/json"},
        )
        assert resp.status_code == 400

    def test_model_from_url(self):
        """Model extracted from URL path."""
        gate = _make_gate(require_virtual_key=False)
        app = _make_app(gate)

        mock_resp = MagicMock()
        mock_resp.raw = {"choices": [{"message": {"role": "assistant", "content": "ok"}}]}

        with patch(_CLIENT_MOCK_TARGET) as mock_client:
            instance = MagicMock()
            mock_client.return_value = instance
            instance.__aenter__ = AsyncMock(return_value=instance)
            instance.__aexit__ = AsyncMock(return_value=None)
            instance.achat = AsyncMock(return_value=mock_resp)

            client = TestClient(app)
            resp = client.post(
                "/v1beta/models/gemini-1.5-pro:generateContent",
                json=_base_body(),
            )
            assert resp.status_code == 200

            # Verify the model from URL was passed to achat
            call_kwargs = instance.achat.call_args[1]
            assert call_kwargs["model"] == "gemini-1.5-pro"


class TestGeminiSystemInstruction:
    def test_system_instruction(self):
        """systemInstruction converted correctly to system message."""
        gate = _make_gate(require_virtual_key=False)
        app = _make_app(gate)

        mock_resp = MagicMock()
        mock_resp.raw = {"choices": [{"message": {"role": "assistant", "content": "ok"}}]}

        with patch(_CLIENT_MOCK_TARGET) as mock_client:
            instance = MagicMock()
            mock_client.return_value = instance
            instance.__aenter__ = AsyncMock(return_value=instance)
            instance.__aexit__ = AsyncMock(return_value=None)
            instance.achat = AsyncMock(return_value=mock_resp)

            client = TestClient(app)
            resp = client.post(
                "/v1beta/models/gemini-2.0-flash:generateContent",
                json={
                    "contents": [{"role": "user", "parts": [{"text": "Hello"}]}],
                    "systemInstruction": {"parts": [{"text": "You are a helpful assistant."}]},
                },
            )
            assert resp.status_code == 200

            call_kwargs = instance.achat.call_args[1]
            messages = call_kwargs["messages"]
            assert messages[0]["role"] == "system"
            assert messages[0]["content"] == "You are a helpful assistant."
            assert messages[1]["role"] == "user"


class TestGeminiGenerationConfig:
    def test_generation_config(self):
        """maxOutputTokens/temperature forwarded."""
        gate = _make_gate(require_virtual_key=False)
        app = _make_app(gate)

        mock_resp = MagicMock()
        mock_resp.raw = {"choices": [{"message": {"role": "assistant", "content": "ok"}}]}

        with patch(_CLIENT_MOCK_TARGET) as mock_client:
            instance = MagicMock()
            mock_client.return_value = instance
            instance.__aenter__ = AsyncMock(return_value=instance)
            instance.__aexit__ = AsyncMock(return_value=None)
            instance.achat = AsyncMock(return_value=mock_resp)

            client = TestClient(app)
            resp = client.post(
                "/v1beta/models/gemini-2.0-flash:generateContent",
                json={
                    "contents": [{"role": "user", "parts": [{"text": "Hello"}]}],
                    "generationConfig": {
                        "maxOutputTokens": 2048,
                        "temperature": 0.5,
                        "topP": 0.9,
                        "topK": 40,
                    },
                },
            )
            assert resp.status_code == 200

            call_kwargs = instance.achat.call_args[1]
            assert call_kwargs["max_tokens"] == 2048
            assert call_kwargs["temperature"] == 0.5
            assert call_kwargs["top_p"] == 0.9
            assert call_kwargs["top_k"] == 40


class TestGeminiErrorFormat:
    def test_error_format(self):
        """Errors in Gemini error format."""
        gate = _make_gate(require_virtual_key=False)
        app = _make_app(gate)

        errors_and_expected = [
            (StateLoomRateLimitError("team-1", 10.0, 100), 429, "RESOURCE_EXHAUSTED"),
            (StateLoomBudgetError(5.0, 6.0, "s-1"), 400, "INVALID_ARGUMENT"),
            (StateLoomPIIBlockedError("email", "s-1"), 400, "INVALID_ARGUMENT"),
            (StateLoomKillSwitchError(), 503, "UNAVAILABLE"),
        ]

        for error, expected_status, expected_status_str in errors_and_expected:
            with patch(_CLIENT_MOCK_TARGET) as mock_client:
                instance = MagicMock()
                mock_client.return_value = instance
                instance.__aenter__ = AsyncMock(return_value=instance)
                instance.__aexit__ = AsyncMock(return_value=None)
                instance.achat = AsyncMock(side_effect=error)

                client = TestClient(app)
                resp = client.post(
                    "/v1beta/models/gemini-2.0-flash:generateContent",
                    json=_base_body(),
                )

                assert resp.status_code == expected_status, (
                    f"Expected {expected_status} for {type(error).__name__}, got {resp.status_code}"
                )
                data = resp.json()
                assert "error" in data
                assert data["error"]["status"] == expected_status_str


class TestGeminiStreaming:
    def test_streaming_returns_sse(self):
        """streamGenerateContent returns SSE events."""
        gate = _make_gate(require_virtual_key=False)
        app = _make_app(gate)

        mock_resp = MagicMock()
        mock_resp.raw = {"choices": [{"message": {"role": "assistant", "content": "Streamed!"}}]}

        with patch(_CLIENT_MOCK_TARGET) as mock_client:
            instance = MagicMock()
            mock_client.return_value = instance
            instance.__aenter__ = AsyncMock(return_value=instance)
            instance.__aexit__ = AsyncMock(return_value=None)
            instance.achat = AsyncMock(return_value=mock_resp)

            client = TestClient(app)
            resp = client.post(
                "/v1beta/models/gemini-2.0-flash:streamGenerateContent",
                json=_base_body(),
            )

            assert resp.status_code == 200
            assert "text/event-stream" in resp.headers.get("content-type", "")
            body = resp.text
            assert "data: " in body
            # Verify it's a valid Gemini response in the SSE
            data_line = body.strip().split("data: ")[1].split("\n")[0]
            parsed = json.loads(data_line)
            assert "candidates" in parsed


class TestGeminiSessionHeader:
    def test_session_id_header(self):
        """X-StateLoom-Session-Id header used when provided."""
        gate = _make_gate(require_virtual_key=False)
        app = _make_app(gate)

        mock_resp = MagicMock()
        mock_resp.raw = {"choices": [{"message": {"role": "assistant", "content": "ok"}}]}

        with patch(_CLIENT_MOCK_TARGET) as mock_client:
            instance = MagicMock()
            mock_client.return_value = instance
            instance.__aenter__ = AsyncMock(return_value=instance)
            instance.__aexit__ = AsyncMock(return_value=None)
            instance.achat = AsyncMock(return_value=mock_resp)

            client = TestClient(app)
            resp = client.post(
                "/v1beta/models/gemini-2.0-flash:generateContent",
                json=_base_body(),
                headers={"X-StateLoom-Session-Id": "my-custom-session"},
            )
            assert resp.status_code == 200

            # Verify session_id was passed to Client
            call_kwargs = mock_client.call_args[1]
            assert call_kwargs["session_id"] == "my-custom-session"


class TestGeminiRoleConversion:
    def test_model_role_converted_to_assistant(self):
        """Gemini 'model' role converted to 'assistant' for pipeline."""
        gate = _make_gate(require_virtual_key=False)
        app = _make_app(gate)

        mock_resp = MagicMock()
        mock_resp.raw = {"choices": [{"message": {"role": "assistant", "content": "ok"}}]}

        with patch(_CLIENT_MOCK_TARGET) as mock_client:
            instance = MagicMock()
            mock_client.return_value = instance
            instance.__aenter__ = AsyncMock(return_value=instance)
            instance.__aexit__ = AsyncMock(return_value=None)
            instance.achat = AsyncMock(return_value=mock_resp)

            client = TestClient(app)
            resp = client.post(
                "/v1beta/models/gemini-2.0-flash:generateContent",
                json={
                    "contents": [
                        {"role": "user", "parts": [{"text": "Hello"}]},
                        {"role": "model", "parts": [{"text": "Hi there!"}]},
                        {"role": "user", "parts": [{"text": "How are you?"}]},
                    ],
                },
            )
            assert resp.status_code == 200

            call_kwargs = instance.achat.call_args[1]
            messages = call_kwargs["messages"]
            assert messages[0]["role"] == "user"
            assert messages[1]["role"] == "assistant"
            assert messages[2]["role"] == "user"

    def test_function_call_translated_to_tool_calls(self):
        """Gemini functionCall parts translate to assistant message with tool_calls."""
        gate = _make_gate(require_virtual_key=False)
        app = _make_app(gate)

        mock_resp = MagicMock()
        mock_resp.raw = {"choices": [{"message": {"role": "assistant", "content": "ok"}}]}

        with patch(_CLIENT_MOCK_TARGET) as mock_client:
            instance = MagicMock()
            mock_client.return_value = instance
            instance.__aenter__ = AsyncMock(return_value=instance)
            instance.__aexit__ = AsyncMock(return_value=None)
            instance.achat = AsyncMock(return_value=mock_resp)

            client = TestClient(app)
            resp = client.post(
                "/v1beta/models/gemini-2.0-flash:generateContent",
                json={
                    "contents": [
                        {"role": "user", "parts": [{"text": "Search for errors"}]},
                        {
                            "role": "model",
                            "parts": [
                                {"functionCall": {"name": "search", "args": {"q": "errors"}}},
                            ],
                        },
                        {
                            "role": "user",
                            "parts": [
                                {
                                    "functionResponse": {
                                        "name": "search",
                                        "response": {"results": ["err1"]},
                                    }
                                },
                            ],
                        },
                    ],
                },
            )
            assert resp.status_code == 200

            call_kwargs = instance.achat.call_args[1]
            messages = call_kwargs["messages"]
            # user prompt
            assert messages[0]["role"] == "user"
            assert messages[0]["content"] == "Search for errors"
            # model functionCall → assistant with tool_calls
            assert messages[1]["role"] == "assistant"
            assert "tool_calls" in messages[1]
            assert messages[1]["tool_calls"][0]["function"]["name"] == "search"
            # user functionResponse → role="tool"
            assert messages[2]["role"] == "tool"
            assert messages[2]["tool_call_id"] == "search"

    def test_function_response_enables_tool_continuation(self):
        """Gemini functionResponse is last message → _is_tool_continuation detects it."""
        from stateloom.middleware.cost_tracker import _is_tool_continuation

        gate = _make_gate(require_virtual_key=False)
        app = _make_app(gate)

        mock_resp = MagicMock()
        mock_resp.raw = {"choices": [{"message": {"role": "assistant", "content": "ok"}}]}

        with patch(_CLIENT_MOCK_TARGET) as mock_client:
            instance = MagicMock()
            mock_client.return_value = instance
            instance.__aenter__ = AsyncMock(return_value=instance)
            instance.__aexit__ = AsyncMock(return_value=None)
            instance.achat = AsyncMock(return_value=mock_resp)

            client = TestClient(app)
            resp = client.post(
                "/v1beta/models/gemini-2.0-flash:generateContent",
                json={
                    "contents": [
                        {"role": "user", "parts": [{"text": "Check codebase"}]},
                        {
                            "role": "model",
                            "parts": [
                                {"functionCall": {"name": "investigate", "args": {}}},
                            ],
                        },
                        {
                            "role": "user",
                            "parts": [
                                {
                                    "functionResponse": {
                                        "name": "investigate",
                                        "response": {"output": "done"},
                                    }
                                },
                            ],
                        },
                    ],
                },
            )
            assert resp.status_code == 200

            # Verify the translated messages trigger tool continuation detection
            call_kwargs = instance.achat.call_args[1]
            messages = call_kwargs["messages"]
            assert _is_tool_continuation(messages) is True

    def test_function_call_with_text_preserves_content(self):
        """Gemini functionCall with accompanying text preserves content."""
        gate = _make_gate(require_virtual_key=False)
        app = _make_app(gate)

        mock_resp = MagicMock()
        mock_resp.raw = {"choices": [{"message": {"role": "assistant", "content": "ok"}}]}

        with patch(_CLIENT_MOCK_TARGET) as mock_client:
            instance = MagicMock()
            mock_client.return_value = instance
            instance.__aenter__ = AsyncMock(return_value=instance)
            instance.__aexit__ = AsyncMock(return_value=None)
            instance.achat = AsyncMock(return_value=mock_resp)

            client = TestClient(app)
            resp = client.post(
                "/v1beta/models/gemini-2.0-flash:generateContent",
                json={
                    "contents": [
                        {"role": "user", "parts": [{"text": "Help me"}]},
                        {
                            "role": "model",
                            "parts": [
                                {"text": "I'll search for that."},
                                {"functionCall": {"name": "search", "args": {}}},
                            ],
                        },
                    ],
                },
            )
            assert resp.status_code == 200

            call_kwargs = instance.achat.call_args[1]
            messages = call_kwargs["messages"]
            assert messages[1]["role"] == "assistant"
            assert messages[1]["content"] == "I'll search for that."
            assert "tool_calls" in messages[1]

    def test_text_only_parts_unchanged(self):
        """Text-only contents still work exactly as before."""
        gate = _make_gate(require_virtual_key=False)
        app = _make_app(gate)

        mock_resp = MagicMock()
        mock_resp.raw = {"choices": [{"message": {"role": "assistant", "content": "ok"}}]}

        with patch(_CLIENT_MOCK_TARGET) as mock_client:
            instance = MagicMock()
            mock_client.return_value = instance
            instance.__aenter__ = AsyncMock(return_value=instance)
            instance.__aexit__ = AsyncMock(return_value=None)
            instance.achat = AsyncMock(return_value=mock_resp)

            client = TestClient(app)
            resp = client.post(
                "/v1beta/models/gemini-2.0-flash:generateContent",
                json={
                    "contents": [
                        {"role": "user", "parts": [{"text": "Hello"}]},
                        {"role": "model", "parts": [{"text": "Hi!"}]},
                    ],
                },
            )
            assert resp.status_code == 200

            call_kwargs = instance.achat.call_args[1]
            messages = call_kwargs["messages"]
            assert len(messages) == 2
            assert messages[0] == {"role": "user", "content": "Hello"}
            assert messages[1] == {"role": "assistant", "content": "Hi!"}
