"""Tests for the Anthropic-native /v1/messages proxy endpoint."""

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
from stateloom.proxy.anthropic_native import create_anthropic_router
from stateloom.proxy.auth import ProxyAuth
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


def _make_app(gate: MagicMock) -> FastAPI:
    """Create app without passthrough (uses legacy Client fallback)."""
    app = FastAPI()
    router = create_anthropic_router(gate)
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


def _mock_anthropic_response(content="Hello!", model="claude-3-5-sonnet-20241022"):
    """Create a mock that looks like anthropic.types.Message."""
    return types.SimpleNamespace(
        id="msg_abc123",
        type="message",
        role="assistant",
        content=[types.SimpleNamespace(type="text", text=content)],
        model=model,
        stop_reason="end_turn",
        stop_sequence=None,
        usage=types.SimpleNamespace(input_tokens=10, output_tokens=20),
        model_dump=lambda: {
            "id": "msg_abc123",
            "type": "message",
            "role": "assistant",
            "content": [{"type": "text", "text": content}],
            "model": model,
            "stop_reason": "end_turn",
            "stop_sequence": None,
            "usage": {"input_tokens": 10, "output_tokens": 20},
        },
    )


def _base_body(**overrides):
    body = {
        "model": "claude-3-5-sonnet-20241022",
        "max_tokens": 1024,
        "messages": [{"role": "user", "content": "Hello"}],
    }
    body.update(overrides)
    return body


# Mock target: Client is now imported lazily inside the function body
_CLIENT_MOCK_TARGET = "stateloom.chat.Client"


class TestAnthropicNonStreaming:
    def test_non_streaming(self):
        """Native request -> native response format."""
        gate = _make_gate(require_virtual_key=False)
        app = _make_app(gate)

        mock_resp = MagicMock()
        mock_resp.raw = _mock_anthropic_response()

        with patch(_CLIENT_MOCK_TARGET) as mock_client:
            instance = MagicMock()
            mock_client.return_value = instance
            instance.__aenter__ = AsyncMock(return_value=instance)
            instance.__aexit__ = AsyncMock(return_value=None)
            instance.achat = AsyncMock(return_value=mock_resp)

            client = TestClient(app)
            resp = client.post("/v1/messages", json=_base_body())

            assert resp.status_code == 200
            data = resp.json()
            assert data["type"] == "message"
            assert data["role"] == "assistant"
            assert data["content"][0]["type"] == "text"
            assert data["content"][0]["text"] == "Hello!"
            assert data["model"] == "claude-3-5-sonnet-20241022"
            assert "usage" in data

    def test_dict_response_fallback(self):
        """Kill switch dict responses wrapped in Anthropic format."""
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
            resp = client.post("/v1/messages", json=_base_body())

            assert resp.status_code == 200
            data = resp.json()
            assert data["type"] == "message"
            assert data["content"][0]["text"] == "Service paused"


class TestAnthropicStreaming:
    def test_streaming_sse_format(self):
        """SSE events in Anthropic format (event: lines)."""
        gate = _make_gate(require_virtual_key=False)
        app = _make_app(gate)

        mock_resp = MagicMock()
        mock_resp.raw = _mock_anthropic_response("Streamed response")

        with patch(_CLIENT_MOCK_TARGET) as mock_client:
            instance = MagicMock()
            mock_client.return_value = instance
            instance.__aenter__ = AsyncMock(return_value=instance)
            instance.__aexit__ = AsyncMock(return_value=None)
            instance.achat = AsyncMock(return_value=mock_resp)

            client = TestClient(app)
            resp = client.post("/v1/messages", json=_base_body(stream=True))

            assert resp.status_code == 200
            assert "text/event-stream" in resp.headers.get("content-type", "")
            body = resp.text

            # Verify Anthropic SSE format: event: <type>\ndata: <json>\n\n
            assert "event: message_start" in body
            assert "event: content_block_start" in body
            assert "event: content_block_delta" in body
            assert "event: content_block_stop" in body
            assert "event: message_delta" in body
            assert "event: message_stop" in body
            assert "Streamed response" in body


class TestAnthropicAuth:
    def test_auth_x_api_key_byok(self):
        """x-api-key treated as BYOK when not ag- prefix."""
        gate = _make_gate(require_virtual_key=True)
        app = _make_app(gate)

        mock_resp = MagicMock()
        mock_resp.raw = _mock_anthropic_response()

        with patch(_CLIENT_MOCK_TARGET) as mock_client:
            instance = MagicMock()
            mock_client.return_value = instance
            instance.__aenter__ = AsyncMock(return_value=instance)
            instance.__aexit__ = AsyncMock(return_value=None)
            instance.achat = AsyncMock(return_value=mock_resp)

            client = TestClient(app)
            resp = client.post(
                "/v1/messages",
                json=_base_body(),
                headers={"x-api-key": "sk-ant-api03-real-key"},
            )
            # BYOK should pass auth (not 401)
            assert resp.status_code == 200

            # Verify provider_keys was passed with the BYOK key
            call_kwargs = mock_client.call_args[1]
            assert call_kwargs["provider_keys"]["anthropic"] == "sk-ant-api03-real-key"

    def test_auth_x_api_key_virtual_key(self):
        """x-api-key: ag-xxx validated as VK."""
        gate = _make_gate(require_virtual_key=True)
        full_key = _setup_virtual_key(gate)
        app = _make_app(gate)

        mock_resp = MagicMock()
        mock_resp.raw = _mock_anthropic_response()

        with patch(_CLIENT_MOCK_TARGET) as mock_client:
            instance = MagicMock()
            mock_client.return_value = instance
            instance.__aenter__ = AsyncMock(return_value=instance)
            instance.__aexit__ = AsyncMock(return_value=None)
            instance.achat = AsyncMock(return_value=mock_resp)

            client = TestClient(app)
            resp = client.post(
                "/v1/messages",
                json=_base_body(),
                headers={"x-api-key": full_key},
            )
            assert resp.status_code == 200

    def test_auth_bearer(self):
        """Authorization: Bearer also works."""
        gate = _make_gate(require_virtual_key=True)
        app = _make_app(gate)

        mock_resp = MagicMock()
        mock_resp.raw = _mock_anthropic_response()

        with patch(_CLIENT_MOCK_TARGET) as mock_client:
            instance = MagicMock()
            mock_client.return_value = instance
            instance.__aenter__ = AsyncMock(return_value=instance)
            instance.__aexit__ = AsyncMock(return_value=None)
            instance.achat = AsyncMock(return_value=mock_resp)

            client = TestClient(app)
            resp = client.post(
                "/v1/messages",
                json=_base_body(),
                headers={"Authorization": "Bearer sk-ant-real-key"},
            )
            # BYOK via Bearer should pass auth
            assert resp.status_code == 200

    def test_no_auth_mode(self):
        """proxy_require_virtual_key=False allows unauthenticated."""
        gate = _make_gate(require_virtual_key=False)
        app = _make_app(gate)

        mock_resp = MagicMock()
        mock_resp.raw = _mock_anthropic_response()

        with patch(_CLIENT_MOCK_TARGET) as mock_client:
            instance = MagicMock()
            mock_client.return_value = instance
            instance.__aenter__ = AsyncMock(return_value=instance)
            instance.__aexit__ = AsyncMock(return_value=None)
            instance.achat = AsyncMock(return_value=mock_resp)

            client = TestClient(app)
            resp = client.post("/v1/messages", json=_base_body())
            assert resp.status_code == 200

    def test_no_auth_mode_ignores_cli_token(self):
        """In no-auth mode, CLI session tokens are ignored (not used as BYOK)."""
        gate = _make_gate(require_virtual_key=False)
        app = _make_app(gate)

        mock_resp = MagicMock()
        mock_resp.raw = _mock_anthropic_response()

        with patch(_CLIENT_MOCK_TARGET) as mock_client:
            instance = MagicMock()
            mock_client.return_value = instance
            instance.__aenter__ = AsyncMock(return_value=instance)
            instance.__aexit__ = AsyncMock(return_value=None)
            instance.achat = AsyncMock(return_value=mock_resp)

            client = TestClient(app)
            resp = client.post(
                "/v1/messages",
                json=_base_body(),
                headers={"x-api-key": "session-token-from-claude-max"},
            )
            assert resp.status_code == 200

            # Verify the CLI token was NOT passed as BYOK
            call_kwargs = mock_client.call_args[1]
            provider_keys = call_kwargs.get("provider_keys")
            assert provider_keys is None or "anthropic" not in (provider_keys or {})

    def test_subscription_billing_detected_for_cli_token(self):
        """CLI session tokens (non-sk-ant-*) detected as subscription billing."""
        gate = _make_gate(require_virtual_key=False)
        app = _make_app(gate)

        mock_resp = MagicMock()
        mock_resp.raw = _mock_anthropic_response()

        with patch(_CLIENT_MOCK_TARGET) as mock_client:
            instance = MagicMock()
            mock_client.return_value = instance
            instance.__aenter__ = AsyncMock(return_value=instance)
            instance.__aexit__ = AsyncMock(return_value=None)
            instance.achat = AsyncMock(return_value=mock_resp)

            client = TestClient(app)
            resp = client.post(
                "/v1/messages",
                json=_base_body(),
                headers={"x-api-key": "session-token-from-claude-max"},
            )
            assert resp.status_code == 200

            # Subscription billing mode should be detected
            call_kwargs = mock_client.call_args[1]
            assert call_kwargs["billing_mode"] == "subscription"

    def test_api_billing_for_api_key(self):
        """Real API keys (sk-ant-*) detected as API billing."""
        gate = _make_gate(require_virtual_key=True)
        app = _make_app(gate)

        mock_resp = MagicMock()
        mock_resp.raw = _mock_anthropic_response()

        with patch(_CLIENT_MOCK_TARGET) as mock_client:
            instance = MagicMock()
            mock_client.return_value = instance
            instance.__aenter__ = AsyncMock(return_value=instance)
            instance.__aexit__ = AsyncMock(return_value=None)
            instance.achat = AsyncMock(return_value=mock_resp)

            client = TestClient(app)
            resp = client.post(
                "/v1/messages",
                json=_base_body(),
                headers={"x-api-key": "sk-ant-api03-real-key"},
            )
            assert resp.status_code == 200

            call_kwargs = mock_client.call_args[1]
            assert call_kwargs["billing_mode"] == "api"

    def test_vk_billing_mode_override(self):
        """VK with explicit billing_mode=subscription overrides auto-detection."""
        gate = _make_gate(require_virtual_key=True)

        # Create a VK with billing_mode=subscription
        vk = VirtualKey(
            id=make_virtual_key_id(),
            key_hash="test",
            key_preview="ag-test",
            org_id="",
            team_id="",
            name="sub-key",
            billing_mode="subscription",
        )
        gate.store.save_virtual_key(vk)
        raw_key = "ag-test-sub-key-12345"
        # Patch authenticate to return this VK
        app = _make_app(gate)

        mock_resp = MagicMock()
        mock_resp.raw = _mock_anthropic_response()

        with (
            patch(_CLIENT_MOCK_TARGET) as mock_client,
            patch.object(ProxyAuth, "authenticate", return_value=vk),
        ):
            instance = MagicMock()
            mock_client.return_value = instance
            instance.__aenter__ = AsyncMock(return_value=instance)
            instance.__aexit__ = AsyncMock(return_value=None)
            instance.achat = AsyncMock(return_value=mock_resp)

            client = TestClient(app)
            resp = client.post(
                "/v1/messages",
                json=_base_body(),
                headers={"x-api-key": raw_key},
            )
            assert resp.status_code == 200

            call_kwargs = mock_client.call_args[1]
            assert call_kwargs["billing_mode"] == "subscription"

    def test_rejects_missing_auth(self):
        """Missing auth when required returns 401 in Anthropic format."""
        gate = _make_gate(require_virtual_key=True)
        client = TestClient(_make_app(gate))
        resp = client.post("/v1/messages", json=_base_body())
        assert resp.status_code == 401
        data = resp.json()
        assert data["type"] == "error"
        assert data["error"]["type"] == "authentication_error"

    def test_rejects_invalid_vk(self):
        """Invalid ag- key returns 401."""
        gate = _make_gate(require_virtual_key=True)
        client = TestClient(_make_app(gate))
        resp = client.post(
            "/v1/messages",
            json=_base_body(),
            headers={"x-api-key": "ag-invalid-key-12345"},
        )
        assert resp.status_code == 401


class TestAnthropicValidation:
    def test_missing_model(self):
        """400 in Anthropic error format for missing model."""
        gate = _make_gate(require_virtual_key=False)
        client = TestClient(_make_app(gate))
        resp = client.post(
            "/v1/messages",
            json={
                "max_tokens": 1024,
                "messages": [{"role": "user", "content": "hi"}],
            },
        )
        assert resp.status_code == 400
        data = resp.json()
        assert data["type"] == "error"
        assert data["error"]["type"] == "invalid_request_error"
        assert "model" in data["error"]["message"]

    def test_missing_max_tokens(self):
        """400 in Anthropic error format for missing max_tokens."""
        gate = _make_gate(require_virtual_key=False)
        client = TestClient(_make_app(gate))
        resp = client.post(
            "/v1/messages",
            json={
                "model": "claude-3-5-sonnet-20241022",
                "messages": [{"role": "user", "content": "hi"}],
            },
        )
        assert resp.status_code == 400
        data = resp.json()
        assert data["type"] == "error"
        assert "max_tokens" in data["error"]["message"]

    def test_missing_messages(self):
        gate = _make_gate(require_virtual_key=False)
        client = TestClient(_make_app(gate))
        resp = client.post(
            "/v1/messages",
            json={
                "model": "claude-3-5-sonnet-20241022",
                "max_tokens": 1024,
            },
        )
        assert resp.status_code == 400

    def test_invalid_json(self):
        gate = _make_gate(require_virtual_key=False)
        client = TestClient(_make_app(gate))
        resp = client.post(
            "/v1/messages",
            content=b"not-json",
            headers={"Content-Type": "application/json"},
        )
        assert resp.status_code == 400
        assert resp.json()["type"] == "error"


class TestAnthropicSystemPrompt:
    def test_system_prompt_passthrough(self):
        """System field converted to system message and round-trips correctly."""
        gate = _make_gate(require_virtual_key=False)
        app = _make_app(gate)

        mock_resp = MagicMock()
        mock_resp.raw = _mock_anthropic_response()

        with patch(_CLIENT_MOCK_TARGET) as mock_client:
            instance = MagicMock()
            mock_client.return_value = instance
            instance.__aenter__ = AsyncMock(return_value=instance)
            instance.__aexit__ = AsyncMock(return_value=None)
            instance.achat = AsyncMock(return_value=mock_resp)

            client = TestClient(app)
            resp = client.post(
                "/v1/messages",
                json={
                    "model": "claude-3-5-sonnet-20241022",
                    "max_tokens": 1024,
                    "system": "You are a helpful assistant.",
                    "messages": [{"role": "user", "content": "Hello"}],
                },
            )
            assert resp.status_code == 200

            # Verify system was injected as first message
            call_args = instance.achat.call_args
            messages = call_args[1]["messages"]
            assert messages[0]["role"] == "system"
            assert messages[0]["content"] == "You are a helpful assistant."
            assert messages[1]["role"] == "user"

    def test_system_prompt_as_blocks(self):
        """System as list of content blocks converted correctly."""
        gate = _make_gate(require_virtual_key=False)
        app = _make_app(gate)

        mock_resp = MagicMock()
        mock_resp.raw = _mock_anthropic_response()

        with patch(_CLIENT_MOCK_TARGET) as mock_client:
            instance = MagicMock()
            mock_client.return_value = instance
            instance.__aenter__ = AsyncMock(return_value=instance)
            instance.__aexit__ = AsyncMock(return_value=None)
            instance.achat = AsyncMock(return_value=mock_resp)

            client = TestClient(app)
            resp = client.post(
                "/v1/messages",
                json={
                    "model": "claude-3-5-sonnet-20241022",
                    "max_tokens": 1024,
                    "system": [
                        {"type": "text", "text": "First instruction."},
                        {"type": "text", "text": "Second instruction."},
                    ],
                    "messages": [{"role": "user", "content": "Hello"}],
                },
            )
            assert resp.status_code == 200

            call_args = instance.achat.call_args
            messages = call_args[1]["messages"]
            assert messages[0]["role"] == "system"
            assert "First instruction." in messages[0]["content"]
            assert "Second instruction." in messages[0]["content"]


class TestAnthropicErrorFormat:
    def test_error_format(self):
        """StateLoom errors returned in Anthropic error format."""
        gate = _make_gate(require_virtual_key=False)
        app = _make_app(gate)

        errors_and_expected = [
            (StateLoomRateLimitError("team-1", 10.0, 100), 429, "rate_limit_error"),
            (StateLoomBudgetError(5.0, 6.0, "s-1"), 400, "invalid_request_error"),
            (StateLoomPIIBlockedError("email", "s-1"), 400, "invalid_request_error"),
            (StateLoomKillSwitchError(), 503, "api_error"),
        ]

        for error, expected_status, expected_type in errors_and_expected:
            with patch(_CLIENT_MOCK_TARGET) as mock_client:
                instance = MagicMock()
                mock_client.return_value = instance
                instance.__aenter__ = AsyncMock(return_value=instance)
                instance.__aexit__ = AsyncMock(return_value=None)
                instance.achat = AsyncMock(side_effect=error)

                client = TestClient(app)
                resp = client.post("/v1/messages", json=_base_body())

                assert resp.status_code == expected_status, (
                    f"Expected {expected_status} for {type(error).__name__}, got {resp.status_code}"
                )
                data = resp.json()
                assert data["type"] == "error"
                assert data["error"]["type"] == expected_type


class TestAnthropicVKScopes:
    def test_vk_model_scope(self):
        """Virtual key model restrictions enforced."""
        gate = _make_gate(require_virtual_key=True)
        full_key = _setup_virtual_key(gate, allowed_models=["gpt-*"])
        app = _make_app(gate)

        client = TestClient(app)
        resp = client.post(
            "/v1/messages",
            json=_base_body(),
            headers={"x-api-key": full_key},
        )
        assert resp.status_code == 403
        assert "not allowed" in resp.json()["error"]["message"]

    def test_vk_budget_scope(self):
        """Budget limits enforced."""
        gate = _make_gate(require_virtual_key=True)
        full_key = _setup_virtual_key(gate, budget_limit=1.0, budget_spent=2.0)
        app = _make_app(gate)

        client = TestClient(app)
        resp = client.post(
            "/v1/messages",
            json=_base_body(),
            headers={"x-api-key": full_key},
        )
        assert resp.status_code == 403
        assert "budget" in resp.json()["error"]["message"].lower()


class TestAnthropicExtraParams:
    def test_extra_params_forwarded(self):
        """temperature, top_p, etc. forwarded to achat."""
        gate = _make_gate(require_virtual_key=False)
        app = _make_app(gate)

        mock_resp = MagicMock()
        mock_resp.raw = _mock_anthropic_response()

        with patch(_CLIENT_MOCK_TARGET) as mock_client:
            instance = MagicMock()
            mock_client.return_value = instance
            instance.__aenter__ = AsyncMock(return_value=instance)
            instance.__aexit__ = AsyncMock(return_value=None)
            instance.achat = AsyncMock(return_value=mock_resp)

            client = TestClient(app)
            resp = client.post(
                "/v1/messages",
                json={
                    **_base_body(),
                    "temperature": 0.7,
                    "top_p": 0.9,
                    "top_k": 40,
                    "stop_sequences": ["\n\nHuman:"],
                },
            )
            assert resp.status_code == 200

            call_kwargs = instance.achat.call_args[1]
            assert call_kwargs["temperature"] == 0.7
            assert call_kwargs["top_p"] == 0.9
            assert call_kwargs["top_k"] == 40
            assert call_kwargs["stop_sequences"] == ["\n\nHuman:"]
