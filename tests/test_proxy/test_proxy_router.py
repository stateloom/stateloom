"""Tests for the proxy router endpoints."""

from __future__ import annotations

from dataclasses import dataclass, field
from unittest.mock import AsyncMock, MagicMock, patch

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient

from stateloom.core.config import StateLoomConfig
from stateloom.core.errors import (
    StateLoomBudgetError,
    StateLoomKillSwitchError,
    StateLoomPIIBlockedError,
    StateLoomRateLimitError,
)
from stateloom.proxy.router import create_proxy_router
from stateloom.proxy.virtual_key import (
    VirtualKey,
    generate_virtual_key,
    make_key_preview,
    make_virtual_key_id,
)
from stateloom.store.memory_store import MemoryStore


@dataclass
class MockPricing:
    _prices: dict = field(
        default_factory=lambda: {
            "gpt-4": MagicMock(input_per_token=0.00003, output_per_token=0.00006),
            "claude-3-opus": MagicMock(input_per_token=0.000015, output_per_token=0.000075),
            "gemini-pro": MagicMock(input_per_token=0.000001, output_per_token=0.000002),
        }
    )


def _make_gate(*, require_virtual_key: bool = True) -> MagicMock:
    """Create a mock Gate."""
    gate = MagicMock()
    gate.store = MemoryStore()
    gate.config = StateLoomConfig(
        proxy_enabled=True,
        proxy_require_virtual_key=require_virtual_key,
        console_output=False,
        dashboard=False,
    )
    gate.pricing = MockPricing()
    return gate


def _make_app(gate: MagicMock) -> FastAPI:
    """Create a FastAPI app with the proxy router mounted."""
    app = FastAPI()
    router = create_proxy_router(gate)
    app.include_router(router, prefix="/v1")
    return app


def _setup_virtual_key(gate: MagicMock) -> str:
    """Create a virtual key and return the full key string."""
    full_key, key_hash = generate_virtual_key()
    vk = VirtualKey(
        id=make_virtual_key_id(),
        key_hash=key_hash,
        key_preview=make_key_preview(full_key),
        team_id="team-1",
        org_id="org-1",
        name="test-key",
    )
    gate.store.save_virtual_key(vk)
    return full_key


_MOCK_JSON_RESPONSE = JSONResponse(
    content={
        "id": "chatcmpl-test",
        "choices": [{"message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
        "model": "gpt-4",
    }
)


class TestModelsEndpoint:
    def test_list_models_no_auth_when_not_required(self):
        gate = _make_gate(require_virtual_key=False)
        client = TestClient(_make_app(gate))
        resp = client.get("/v1/models")
        assert resp.status_code == 200
        data = resp.json()
        assert data["object"] == "list"
        assert len(data["data"]) == 3
        model_ids = {m["id"] for m in data["data"]}
        assert "gpt-4" in model_ids

    def test_list_models_requires_auth(self):
        gate = _make_gate(require_virtual_key=True)
        client = TestClient(_make_app(gate))
        resp = client.get("/v1/models")
        assert resp.status_code == 401

    def test_list_models_with_auth(self):
        gate = _make_gate(require_virtual_key=True)
        full_key = _setup_virtual_key(gate)
        client = TestClient(_make_app(gate))
        resp = client.get("/v1/models", headers={"Authorization": f"Bearer {full_key}"})
        assert resp.status_code == 200
        assert len(resp.json()["data"]) == 3

    def test_model_owner_mapping(self):
        gate = _make_gate(require_virtual_key=False)
        client = TestClient(_make_app(gate))
        resp = client.get("/v1/models")
        data = resp.json()
        owners = {m["id"]: m["owned_by"] for m in data["data"]}
        assert owners["gpt-4"] == "openai"
        assert owners["claude-3-opus"] == "anthropic"
        assert owners["gemini-pro"] == "google"


class TestChatCompletionsAuth:
    def test_rejects_missing_auth(self):
        gate = _make_gate(require_virtual_key=True)
        client = TestClient(_make_app(gate))
        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": "gpt-4",
                "messages": [{"role": "user", "content": "hi"}],
            },
        )
        assert resp.status_code == 401
        assert "invalid_api_key" in resp.json()["error"]["code"]

    def test_rejects_invalid_key(self):
        gate = _make_gate(require_virtual_key=True)
        client = TestClient(_make_app(gate))
        resp = client.post(
            "/v1/chat/completions",
            json={"model": "gpt-4", "messages": [{"role": "user", "content": "hi"}]},
            headers={"Authorization": "Bearer ag-fake-key"},
        )
        assert resp.status_code == 401

    def test_accepts_revoked_key_rejected(self):
        gate = _make_gate(require_virtual_key=True)
        full_key = _setup_virtual_key(gate)
        # Revoke
        keys = gate.store.list_virtual_keys()
        gate.store.revoke_virtual_key(keys[0].id)

        client = TestClient(_make_app(gate))
        resp = client.post(
            "/v1/chat/completions",
            json={"model": "gpt-4", "messages": [{"role": "user", "content": "hi"}]},
            headers={"Authorization": f"Bearer {full_key}"},
        )
        assert resp.status_code == 401

    def test_allows_no_auth_when_not_required(self):
        gate = _make_gate(require_virtual_key=False)
        app = _make_app(gate)

        with patch(
            "stateloom.proxy.router._handle_provider_sdk",
            new_callable=AsyncMock,
            return_value=_MOCK_JSON_RESPONSE,
        ):
            client = TestClient(app)
            resp = client.post(
                "/v1/chat/completions",
                json={"model": "gpt-4", "messages": [{"role": "user", "content": "hi"}]},
            )
        # Should NOT be 401
        assert resp.status_code != 401


class TestChatCompletionsValidation:
    def test_missing_model(self):
        gate = _make_gate(require_virtual_key=False)
        client = TestClient(_make_app(gate))
        resp = client.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "hi"}]},
        )
        assert resp.status_code == 400
        assert "missing_model" in resp.json()["error"]["code"]

    def test_missing_messages(self):
        gate = _make_gate(require_virtual_key=False)
        client = TestClient(_make_app(gate))
        resp = client.post(
            "/v1/chat/completions",
            json={"model": "gpt-4"},
        )
        assert resp.status_code == 400
        assert "missing_messages" in resp.json()["error"]["code"]

    def test_empty_messages(self):
        gate = _make_gate(require_virtual_key=False)
        client = TestClient(_make_app(gate))
        resp = client.post(
            "/v1/chat/completions",
            json={"model": "gpt-4", "messages": []},
        )
        assert resp.status_code == 400

    def test_invalid_json(self):
        gate = _make_gate(require_virtual_key=False)
        client = TestClient(_make_app(gate))
        resp = client.post(
            "/v1/chat/completions",
            content=b"not-json",
            headers={"Content-Type": "application/json"},
        )
        assert resp.status_code == 400
        assert "invalid_json" in resp.json()["error"]["code"]


class TestChatCompletionsErrorMapping:
    """Test that StateLoom errors are mapped to OpenAI error format.

    _handle_provider_sdk catches errors internally and returns mapped
    JSONResponses.  These tests mock the handler to return the mapped
    error response directly, verifying the endpoint returns it correctly.
    """

    @staticmethod
    def _error_response(error):
        """Build the JSONResponse that _handle_provider_sdk would return for the error."""
        from stateloom.proxy.errors import error_status_code, to_openai_error_dict

        return JSONResponse(
            status_code=error_status_code(error),
            content=to_openai_error_dict(error),
        )

    def test_rate_limit_error(self):
        error = StateLoomRateLimitError("team-1", 10.0, 100)
        gate = _make_gate(require_virtual_key=False)
        app = _make_app(gate)

        with patch(
            "stateloom.proxy.router._handle_provider_sdk",
            new_callable=AsyncMock,
            return_value=self._error_response(error),
        ):
            client = TestClient(app)
            resp = client.post(
                "/v1/chat/completions",
                json={
                    "model": "gpt-4",
                    "messages": [{"role": "user", "content": "hi"}],
                },
            )
            assert resp.status_code == 429
            assert resp.json()["error"]["type"] == "rate_limit_error"

    def test_budget_error(self):
        error = StateLoomBudgetError(5.0, 6.0, "session-1")
        gate = _make_gate(require_virtual_key=False)
        app = _make_app(gate)

        with patch(
            "stateloom.proxy.router._handle_provider_sdk",
            new_callable=AsyncMock,
            return_value=self._error_response(error),
        ):
            client = TestClient(app)
            resp = client.post(
                "/v1/chat/completions",
                json={
                    "model": "gpt-4",
                    "messages": [{"role": "user", "content": "hi"}],
                },
            )
            assert resp.status_code == 402

    def test_pii_blocked_error(self):
        error = StateLoomPIIBlockedError("email", "session-1")
        gate = _make_gate(require_virtual_key=False)
        app = _make_app(gate)

        with patch(
            "stateloom.proxy.router._handle_provider_sdk",
            new_callable=AsyncMock,
            return_value=self._error_response(error),
        ):
            client = TestClient(app)
            resp = client.post(
                "/v1/chat/completions",
                json={
                    "model": "gpt-4",
                    "messages": [{"role": "user", "content": "hi"}],
                },
            )
            assert resp.status_code == 400

    def test_kill_switch_error(self):
        error = StateLoomKillSwitchError()
        gate = _make_gate(require_virtual_key=False)
        app = _make_app(gate)

        with patch(
            "stateloom.proxy.router._handle_provider_sdk",
            new_callable=AsyncMock,
            return_value=self._error_response(error),
        ):
            client = TestClient(app)
            resp = client.post(
                "/v1/chat/completions",
                json={
                    "model": "gpt-4",
                    "messages": [{"role": "user", "content": "hi"}],
                },
            )
            assert resp.status_code == 503


class TestChatCompletionsStreaming:
    def test_streaming_returns_sse(self):
        from starlette.responses import StreamingResponse

        gate = _make_gate(require_virtual_key=False)
        app = _make_app(gate)

        async def _mock_sse_handler(**kwargs):
            async def generate():
                yield 'data: {"choices":[{"delta":{"content":"streamed"}}]}\n\n'
                yield "data: [DONE]\n\n"

            return StreamingResponse(generate(), media_type="text/event-stream")

        with patch(
            "stateloom.proxy.router._handle_provider_sdk",
            new_callable=AsyncMock,
            side_effect=_mock_sse_handler,
        ):
            client = TestClient(app)
            resp = client.post(
                "/v1/chat/completions",
                json={
                    "model": "gpt-4",
                    "messages": [{"role": "user", "content": "hi"}],
                    "stream": True,
                },
            )
            assert resp.status_code == 200
            assert "text/event-stream" in resp.headers.get("content-type", "")
            body = resp.text
            assert "data: " in body
            assert "[DONE]" in body


class TestBYOKHeaders:
    """Test Bring Your Own Key header support."""

    def _post_with_headers(self, gate, app, headers=None):
        """Helper: POST to chat/completions with mock handler, return mock and its call args."""
        with patch(
            "stateloom.proxy.router._handle_provider_sdk",
            new_callable=AsyncMock,
            return_value=_MOCK_JSON_RESPONSE,
        ) as mock_handler:
            client = TestClient(app)
            resp = client.post(
                "/v1/chat/completions",
                json={
                    "model": "gpt-4",
                    "messages": [{"role": "user", "content": "hi"}],
                },
                headers=headers or {},
            )
            return resp, mock_handler

    def test_byok_openai_key_header(self):
        gate = _make_gate(require_virtual_key=False)
        app = _make_app(gate)
        resp, mock_handler = self._post_with_headers(
            gate, app, headers={"X-StateLoom-OpenAI-Key": "sk-test-openai"}
        )
        assert resp.status_code == 200
        call_kwargs = mock_handler.call_args.kwargs
        assert call_kwargs["provider_keys"]["openai"] == "sk-test-openai"

    def test_byok_anthropic_key_header(self):
        gate = _make_gate(require_virtual_key=False)
        app = _make_app(gate)
        resp, mock_handler = self._post_with_headers(
            gate, app, headers={"X-StateLoom-Anthropic-Key": "sk-ant-test"}
        )
        assert resp.status_code == 200
        call_kwargs = mock_handler.call_args.kwargs
        assert call_kwargs["provider_keys"]["anthropic"] == "sk-ant-test"

    def test_byok_google_key_header(self):
        gate = _make_gate(require_virtual_key=False)
        app = _make_app(gate)
        resp, mock_handler = self._post_with_headers(
            gate, app, headers={"X-StateLoom-Google-Key": "AIza-test"}
        )
        assert resp.status_code == 200
        call_kwargs = mock_handler.call_args.kwargs
        assert call_kwargs["provider_keys"]["google"] == "AIza-test"

    def test_byok_headers_override_org_keys(self):
        gate = _make_gate(require_virtual_key=True)
        full_key = _setup_virtual_key(gate)
        app = _make_app(gate)

        # Set org-level secret so get_provider_keys returns it
        gate.store.save_secret("org:org-1:provider_key_openai", "sk-org-openai")

        resp, mock_handler = self._post_with_headers(
            gate,
            app,
            headers={
                "Authorization": f"Bearer {full_key}",
                "X-StateLoom-OpenAI-Key": "sk-byok-override",
            },
        )
        assert resp.status_code == 200
        call_kwargs = mock_handler.call_args.kwargs
        # BYOK header should win over org secret
        assert call_kwargs["provider_keys"]["openai"] == "sk-byok-override"

    def test_byok_no_headers_uses_org_keys(self):
        gate = _make_gate(require_virtual_key=True)
        full_key = _setup_virtual_key(gate)
        app = _make_app(gate)

        # Set org-level secret
        gate.store.save_secret("org:org-1:provider_key_openai", "sk-org-key")

        resp, mock_handler = self._post_with_headers(
            gate,
            app,
            headers={"Authorization": f"Bearer {full_key}"},
        )
        assert resp.status_code == 200
        call_kwargs = mock_handler.call_args.kwargs
        # Should use org key since no BYOK header
        assert call_kwargs["provider_keys"].get("openai") == "sk-org-key"
