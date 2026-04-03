"""Tests for the universal catch-all passthrough proxy."""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import httpx
from fastapi import APIRouter, FastAPI
from httpx import ASGITransport, AsyncClient

from stateloom.proxy.catch_all import (
    _detect_provider,
    _error_for_provider,
    _is_streaming_request,
    create_catch_all_routers,
)
from stateloom.proxy.passthrough import PassthroughProxy, UpstreamStreamError

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_request(headers: dict[str, str] | None = None) -> MagicMock:
    """Build a mock Starlette Request with headers."""
    req = MagicMock()
    req.headers = headers or {}
    req.query_params = {}
    return req


def _make_gate(**overrides):
    """Build a minimal mock Gate for the catch-all router."""
    proxy_config = SimpleNamespace(
        upstream_openai="https://api.openai.com",
        upstream_anthropic="https://api.anthropic.com",
        upstream_gemini="https://generativelanguage.googleapis.com",
        timeout=30.0,
    )
    config = SimpleNamespace(proxy=proxy_config)
    gate = MagicMock()
    gate.config = config
    for k, v in overrides.items():
        setattr(gate, k, v)
    return gate


def _build_app(gate=None, passthrough=None) -> FastAPI:
    """Build a FastAPI app with only the catch-all routers mounted."""
    gate = gate or _make_gate()
    v1_router, v1beta_router = create_catch_all_routers(gate, passthrough=passthrough)
    app = FastAPI()
    app.include_router(v1_router, prefix="/v1")
    app.include_router(v1beta_router, prefix="/v1beta")
    return app


# ---------------------------------------------------------------------------
# Unit tests: provider detection
# ---------------------------------------------------------------------------


class TestDetectProvider:
    def test_default_is_openai(self):
        req = _make_request({"content-type": "application/json"})
        assert _detect_provider(req) == "openai"

    def test_x_api_key_means_anthropic(self):
        req = _make_request({"x-api-key": "sk-ant-test123"})
        assert _detect_provider(req) == "anthropic"

    def test_x_goog_api_key_means_gemini(self):
        req = _make_request({"x-goog-api-key": "AIzaSyFoo"})
        assert _detect_provider(req) == "gemini"

    def test_anthropic_takes_priority_over_openai(self):
        """x-api-key is checked before defaulting to OpenAI."""
        req = _make_request(
            {
                "x-api-key": "sk-ant-test",
                "authorization": "Bearer sk-openai",
            }
        )
        assert _detect_provider(req) == "anthropic"

    def test_both_headers_anthropic_wins(self):
        """x-api-key is checked first."""
        req = _make_request(
            {
                "x-api-key": "sk-ant-test",
                "x-goog-api-key": "AIzaSy",
            }
        )
        assert _detect_provider(req) == "anthropic"


# ---------------------------------------------------------------------------
# Unit tests: streaming detection
# ---------------------------------------------------------------------------


class TestIsStreamingRequest:
    def test_stream_true_in_json_body(self):
        body = json.dumps({"model": "gpt-4o", "stream": True}).encode()
        req = _make_request()
        assert _is_streaming_request(body, req) is True

    def test_stream_false_in_json_body(self):
        body = json.dumps({"model": "gpt-4o", "stream": False}).encode()
        req = _make_request()
        assert _is_streaming_request(body, req) is False

    def test_no_stream_key(self):
        body = json.dumps({"model": "gpt-4o"}).encode()
        req = _make_request()
        assert _is_streaming_request(body, req) is False

    def test_alt_sse_query_param(self):
        req = MagicMock()
        req.query_params = {"alt": "sse"}
        assert _is_streaming_request(b"", req) is True

    def test_empty_body(self):
        req = _make_request()
        assert _is_streaming_request(b"", req) is False

    def test_invalid_json_body(self):
        req = _make_request()
        assert _is_streaming_request(b"not-json", req) is False


# ---------------------------------------------------------------------------
# Unit tests: error format
# ---------------------------------------------------------------------------


class TestErrorForProvider:
    def test_openai_format(self):
        err = _error_for_provider("openai", 502, "Upstream failed")
        assert "error" in err
        assert err["error"]["message"] == "Upstream failed"
        assert err["error"]["type"] == "server_error"

    def test_anthropic_format(self):
        err = _error_for_provider("anthropic", 503, "Unavailable")
        assert err["type"] == "error"
        assert err["error"]["message"] == "Unavailable"

    def test_gemini_format(self):
        err = _error_for_provider("gemini", 502, "Bad gateway")
        assert err["error"]["code"] == 502
        assert err["error"]["message"] == "Bad gateway"
        assert err["error"]["status"] == "INTERNAL"

    def test_gemini_503_status(self):
        err = _error_for_provider("gemini", 503, "Down")
        assert err["error"]["status"] == "UNAVAILABLE"


# ---------------------------------------------------------------------------
# Integration tests: ASGI app
# ---------------------------------------------------------------------------


class TestCatchAllNoPassthrough:
    """When no PassthroughProxy is provided, all requests return 503."""

    async def test_v1_returns_503(self):
        app = _build_app(passthrough=None)
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.get("/v1/models")
            assert resp.status_code == 503

    async def test_v1beta_returns_503(self):
        app = _build_app(passthrough=None)
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.post(
                "/v1beta/models/gemini-pro:countTokens",
                json={"contents": [{"parts": [{"text": "hello"}]}]},
            )
            assert resp.status_code == 503


class TestCatchAllForwarding:
    """Test actual HTTP forwarding via mocked PassthroughProxy."""

    async def test_get_forwarded_to_openai(self):
        """GET /v1/models with no special headers goes to OpenAI."""
        mock_proxy = AsyncMock(spec=PassthroughProxy)
        mock_proxy.forward_any.return_value = httpx.Response(
            200,
            json={"data": [{"id": "gpt-4o"}]},
            headers={"content-type": "application/json"},
        )

        app = _build_app(passthrough=mock_proxy)
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.get("/v1/models")

        assert resp.status_code == 200
        assert resp.json()["data"][0]["id"] == "gpt-4o"
        mock_proxy.forward_any.assert_called_once()
        call_args = mock_proxy.forward_any.call_args
        assert call_args[0][0] == "GET"  # method
        assert "api.openai.com/v1/models" in call_args[0][1]  # upstream_url

    async def test_post_forwarded_to_anthropic(self):
        """POST with x-api-key goes to Anthropic upstream."""
        mock_proxy = AsyncMock(spec=PassthroughProxy)
        mock_proxy.forward_any.return_value = httpx.Response(
            200,
            json={"count": 42},
            headers={"content-type": "application/json"},
        )

        app = _build_app(passthrough=mock_proxy)
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.post(
                "/v1/messages/batches",
                json={"requests": []},
                headers={"x-api-key": "sk-ant-test"},
            )

        assert resp.status_code == 200
        call_args = mock_proxy.forward_any.call_args
        assert "api.anthropic.com/v1/messages/batches" in call_args[0][1]

    async def test_v1beta_always_gemini(self):
        """All /v1beta paths route to Gemini regardless of headers."""
        mock_proxy = AsyncMock(spec=PassthroughProxy)
        mock_proxy.forward_any.return_value = httpx.Response(
            200,
            json={"totalTokens": 5},
            headers={"content-type": "application/json"},
        )

        app = _build_app(passthrough=mock_proxy)
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.post(
                "/v1beta/models/gemini-pro:countTokens",
                json={"contents": [{"parts": [{"text": "hello"}]}]},
            )

        assert resp.status_code == 200
        call_args = mock_proxy.forward_any.call_args
        assert "generativelanguage.googleapis.com/v1beta/" in call_args[0][1]

    async def test_query_params_preserved(self):
        """Query parameters are forwarded to upstream."""
        mock_proxy = AsyncMock(spec=PassthroughProxy)
        mock_proxy.forward_any.return_value = httpx.Response(
            200,
            json={},
            headers={"content-type": "application/json"},
        )

        app = _build_app(passthrough=mock_proxy)
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.get("/v1/models?page=2&limit=10")

        assert resp.status_code == 200
        upstream_url = mock_proxy.forward_any.call_args[0][1]
        assert "page=2" in upstream_url
        assert "limit=10" in upstream_url

    async def test_put_method_forwarded(self):
        """PUT requests are forwarded correctly."""
        mock_proxy = AsyncMock(spec=PassthroughProxy)
        mock_proxy.forward_any.return_value = httpx.Response(
            200,
            json={"ok": True},
            headers={"content-type": "application/json"},
        )

        app = _build_app(passthrough=mock_proxy)
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.put("/v1/fine_tuning/jobs/ft-123/cancel")

        assert resp.status_code == 200
        assert mock_proxy.forward_any.call_args[0][0] == "PUT"

    async def test_delete_method_forwarded(self):
        """DELETE requests are forwarded correctly."""
        mock_proxy = AsyncMock(spec=PassthroughProxy)
        mock_proxy.forward_any.return_value = httpx.Response(
            200,
            json={"deleted": True},
            headers={"content-type": "application/json"},
        )

        app = _build_app(passthrough=mock_proxy)
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.delete("/v1/files/file-abc")

        assert resp.status_code == 200
        assert mock_proxy.forward_any.call_args[0][0] == "DELETE"

    async def test_google_key_routes_to_gemini(self):
        """x-goog-api-key header routes /v1 paths to Gemini."""
        mock_proxy = AsyncMock(spec=PassthroughProxy)
        mock_proxy.forward_any.return_value = httpx.Response(
            200,
            json={"models": []},
            headers={"content-type": "application/json"},
        )

        app = _build_app(passthrough=mock_proxy)
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.get(
                "/v1/models",
                headers={"x-goog-api-key": "AIzaSyTest"},
            )

        assert resp.status_code == 200
        upstream_url = mock_proxy.forward_any.call_args[0][1]
        assert "generativelanguage.googleapis.com" in upstream_url


class TestCatchAllResponseHeaders:
    """Verify hop-by-hop headers are stripped from upstream responses."""

    async def test_hop_by_hop_stripped_from_response(self):
        # Build a mock httpx.Response with hop-by-hop headers.
        # We can't set content-encoding: gzip with non-gzip body on httpx.Response,
        # so we use a SimpleNamespace to simulate the upstream response instead.
        upstream_resp = SimpleNamespace(
            status_code=200,
            content=b'{"ok":true}',
            headers={
                "content-type": "application/json",
                "content-encoding": "gzip",
                "transfer-encoding": "chunked",
                "connection": "keep-alive",
                "x-request-id": "abc-123",
            },
        )
        mock_proxy = AsyncMock(spec=PassthroughProxy)
        mock_proxy.forward_any.return_value = upstream_resp

        app = _build_app(passthrough=mock_proxy)
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.get("/v1/models")

        assert resp.status_code == 200
        # Hop-by-hop should be stripped
        assert "content-encoding" not in resp.headers
        assert "transfer-encoding" not in resp.headers
        assert "connection" not in resp.headers
        # Non-hop-by-hop preserved
        assert resp.headers.get("x-request-id") == "abc-123"


class TestCatchAllAuthPassthrough:
    """Auth headers should be forwarded as-is (no replacement)."""

    async def test_authorization_header_forwarded(self):
        mock_proxy = AsyncMock(spec=PassthroughProxy)
        mock_proxy.forward_any.return_value = httpx.Response(
            200,
            json={},
            headers={"content-type": "application/json"},
        )

        app = _build_app(passthrough=mock_proxy)
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            await client.get(
                "/v1/models",
                headers={"authorization": "Bearer sk-test-key"},
            )

        # Check the headers passed to forward_any
        upstream_headers = mock_proxy.forward_any.call_args[0][3]
        assert upstream_headers.get("authorization") == "Bearer sk-test-key"

    async def test_x_api_key_forwarded(self):
        mock_proxy = AsyncMock(spec=PassthroughProxy)
        mock_proxy.forward_any.return_value = httpx.Response(
            200,
            json={},
            headers={"content-type": "application/json"},
        )

        app = _build_app(passthrough=mock_proxy)
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            await client.get(
                "/v1/models",
                headers={"x-api-key": "sk-ant-my-key"},
            )

        upstream_headers = mock_proxy.forward_any.call_args[0][3]
        assert upstream_headers.get("x-api-key") == "sk-ant-my-key"


class TestCatchAllStreaming:
    """Streaming requests should be relayed via forward_stream."""

    async def test_streaming_post_uses_forward_stream(self):
        mock_proxy = AsyncMock(spec=PassthroughProxy)

        async def mock_stream(*args, **kwargs):
            yield b'data: {"id": "chunk1"}\n\n'
            yield b"data: [DONE]\n\n"

        mock_proxy.forward_stream = mock_stream

        app = _build_app(passthrough=mock_proxy)
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.post(
                "/v1/embeddings",
                json={"input": "hello", "stream": True},
            )

        assert resp.status_code == 200
        assert "text/event-stream" in resp.headers.get("content-type", "")

    async def test_streaming_alt_sse_param(self):
        """Gemini-style ?alt=sse triggers streaming."""
        mock_proxy = AsyncMock(spec=PassthroughProxy)

        async def mock_stream(*args, **kwargs):
            yield b'data: {"candidates": []}\n\n'

        mock_proxy.forward_stream = mock_stream

        app = _build_app(passthrough=mock_proxy)
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.post(
                "/v1beta/models/gemini-pro:generateContent?alt=sse",
                json={"contents": []},
            )

        assert resp.status_code == 200
        assert "text/event-stream" in resp.headers.get("content-type", "")


class TestCatchAllUpstreamErrors:
    """Upstream errors are propagated with correct status codes."""

    async def test_upstream_4xx_propagated(self):
        mock_proxy = AsyncMock(spec=PassthroughProxy)
        mock_proxy.forward_any.return_value = httpx.Response(
            401,
            json={"error": {"message": "Invalid key"}},
            headers={"content-type": "application/json"},
        )

        app = _build_app(passthrough=mock_proxy)
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.get("/v1/models")

        assert resp.status_code == 401

    async def test_upstream_5xx_propagated(self):
        mock_proxy = AsyncMock(spec=PassthroughProxy)
        mock_proxy.forward_any.return_value = httpx.Response(
            500,
            json={"error": {"message": "Internal error"}},
            headers={"content-type": "application/json"},
        )

        app = _build_app(passthrough=mock_proxy)
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.get("/v1/models")

        assert resp.status_code == 500

    async def test_connection_error_returns_502(self):
        mock_proxy = AsyncMock(spec=PassthroughProxy)
        mock_proxy.forward_any.side_effect = httpx.ConnectError("Connection refused")

        app = _build_app(passthrough=mock_proxy)
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.get("/v1/models")

        assert resp.status_code == 502

    async def test_stream_upstream_error_detected(self):
        """Streaming upstream errors return proper HTTP error, not 200 SSE."""
        mock_proxy = AsyncMock(spec=PassthroughProxy)

        async def mock_stream(*args, **kwargs):
            raise UpstreamStreamError(429, b'{"error":"rate limited"}', "application/json")
            yield  # make it a generator  # pragma: no cover

        mock_proxy.forward_stream = mock_stream

        app = _build_app(passthrough=mock_proxy)
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.post(
                "/v1/completions",
                json={"stream": True},
            )

        assert resp.status_code == 429


class TestCatchAllStateloomHeadersStripped:
    """StateLoom-internal headers should not be forwarded."""

    async def test_stateloom_headers_stripped(self):
        mock_proxy = AsyncMock(spec=PassthroughProxy)
        mock_proxy.forward_any.return_value = httpx.Response(
            200,
            json={},
            headers={"content-type": "application/json"},
        )

        app = _build_app(passthrough=mock_proxy)
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            await client.get(
                "/v1/models",
                headers={
                    "x-stateloom-session-id": "sess-123",
                    "x-stateloom-openai-key": "sk-secret",
                    "authorization": "Bearer sk-real",
                },
            )

        upstream_headers = mock_proxy.forward_any.call_args[0][3]
        assert "x-stateloom-session-id" not in upstream_headers
        assert "x-stateloom-openai-key" not in upstream_headers
        # Real auth is preserved
        assert upstream_headers.get("authorization") == "Bearer sk-real"


class TestCatchAllRouterPriority:
    """Specific routes should take priority over catch-all when mounted first."""

    async def test_specific_route_takes_priority(self):
        """When a specific /v1/chat/completions route exists, it wins."""
        app = FastAPI()

        # Mount a specific route first (like server.py does)
        specific_router = APIRouter()

        @specific_router.post("/chat/completions")
        async def specific_chat():
            return {"source": "specific"}

        app.include_router(specific_router, prefix="/v1")

        # Then mount catch-all
        gate = _make_gate()
        mock_proxy = AsyncMock(spec=PassthroughProxy)
        mock_proxy.forward_any.return_value = httpx.Response(
            200,
            json={"source": "catch-all"},
            headers={"content-type": "application/json"},
        )
        v1_catch, _ = create_catch_all_routers(gate, passthrough=mock_proxy)
        app.include_router(v1_catch, prefix="/v1")

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.post("/v1/chat/completions")

        assert resp.status_code == 200
        assert resp.json()["source"] == "specific"
        # catch-all should NOT have been called
        mock_proxy.forward_any.assert_not_called()

    async def test_unmatched_path_hits_catch_all(self):
        """Paths not matched by specific routes fall through to catch-all."""
        app = FastAPI()

        specific_router = APIRouter()

        @specific_router.post("/chat/completions")
        async def specific_chat():
            return {"source": "specific"}

        app.include_router(specific_router, prefix="/v1")

        gate = _make_gate()
        mock_proxy = AsyncMock(spec=PassthroughProxy)
        mock_proxy.forward_any.return_value = httpx.Response(
            200,
            json={"source": "catch-all"},
            headers={"content-type": "application/json"},
        )
        v1_catch, _ = create_catch_all_routers(gate, passthrough=mock_proxy)
        app.include_router(v1_catch, prefix="/v1")

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.get("/v1/models")

        assert resp.status_code == 200
        assert resp.json()["source"] == "catch-all"
        mock_proxy.forward_any.assert_called_once()
