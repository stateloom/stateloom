"""Tests for the HTTP reverse proxy passthrough engine."""

from __future__ import annotations

import httpx
import pytest
import respx

from stateloom.proxy.passthrough import (
    DEFAULT_UPSTREAM_URLS,
    REQUEST_HOP_BY_HOP_HEADERS,
    RESPONSE_HOP_BY_HOP_HEADERS,
    PassthroughProxy,
    filter_headers,
)


class TestFilterHeaders:
    def test_strips_hop_by_hop_headers(self):
        headers = {
            "host": "localhost:4782",
            "connection": "keep-alive",
            "transfer-encoding": "chunked",
            "content-type": "application/json",
            "x-api-key": "sk-ant-test",
        }
        result = filter_headers(headers)
        assert "host" not in result
        assert "connection" not in result
        assert "transfer-encoding" not in result
        assert result["content-type"] == "application/json"
        assert result["x-api-key"] == "sk-ant-test"

    def test_strips_stateloom_internal_headers(self):
        headers = {
            "content-type": "application/json",
            "x-stateloom-session-id": "sess-123",
            "x-stateloom-openai-key": "sk-openai",
            "x-stateloom-anthropic-key": "sk-ant",
            "x-stateloom-google-key": "AIza",
            "x-api-key": "sk-ant-test",
        }
        result = filter_headers(headers)
        assert "x-stateloom-session-id" not in result
        assert "x-stateloom-openai-key" not in result
        assert "x-stateloom-anthropic-key" not in result
        assert "x-stateloom-google-key" not in result
        assert result["x-api-key"] == "sk-ant-test"

    def test_strips_content_length(self):
        headers = {
            "content-type": "application/json",
            "content-length": "123",
        }
        result = filter_headers(headers)
        assert "content-length" not in result
        assert result["content-type"] == "application/json"

    def test_replaces_auth_header(self):
        headers = {
            "x-api-key": "original-token",
            "content-type": "application/json",
        }
        result = filter_headers(
            headers,
            auth_header_name="x-api-key",
            auth_header_value="resolved-provider-key",
        )
        assert result["x-api-key"] == "resolved-provider-key"
        assert result["content-type"] == "application/json"

    def test_no_auth_replacement_passes_original(self):
        headers = {
            "x-api-key": "cli-session-token",
            "content-type": "application/json",
        }
        result = filter_headers(headers)
        assert result["x-api-key"] == "cli-session-token"

    def test_empty_auth_header_name_skips_injection(self):
        headers = {"content-type": "application/json"}
        result = filter_headers(headers, auth_header_name="", auth_header_value="some-value")
        assert "x-api-key" not in result

    def test_case_insensitive_hop_by_hop(self):
        headers = {
            "Host": "example.com",
            "Connection": "keep-alive",
            "Content-Type": "application/json",
        }
        result = filter_headers(headers)
        assert "Host" not in result
        assert "Connection" not in result
        assert result["Content-Type"] == "application/json"

    def test_starlette_headers_object(self):
        """Works with Starlette/FastAPI Headers objects (dict-like)."""
        # Starlette Headers are dict-like with .items()
        from starlette.datastructures import Headers

        raw = [(b"content-type", b"application/json"), (b"x-api-key", b"test")]
        headers = Headers(raw=raw)
        result = filter_headers(headers)
        assert result["content-type"] == "application/json"
        assert result["x-api-key"] == "test"


class TestDefaultUpstreamUrls:
    def test_anthropic_url(self):
        assert DEFAULT_UPSTREAM_URLS["anthropic"] == "https://api.anthropic.com"

    def test_openai_url(self):
        assert DEFAULT_UPSTREAM_URLS["openai"] == "https://api.openai.com"

    def test_gemini_url(self):
        assert DEFAULT_UPSTREAM_URLS["gemini"] == "https://generativelanguage.googleapis.com"


class TestPassthroughProxyForward:
    @respx.mock
    async def test_forward_success(self):
        """Forward a request and get successful response."""
        respx.post("https://api.anthropic.com/v1/messages").mock(
            return_value=httpx.Response(
                200,
                json={
                    "id": "msg_123",
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "text", "text": "Hello!"}],
                    "model": "claude-3-5-sonnet-20241022",
                    "usage": {"input_tokens": 10, "output_tokens": 5},
                },
            )
        )

        proxy = PassthroughProxy(timeout=10.0)
        try:
            resp = await proxy.forward(
                "https://api.anthropic.com/v1/messages",
                b'{"model":"claude-3-5-sonnet-20241022","messages":[{"role":"user","content":"hi"}]}',
                {"content-type": "application/json", "x-api-key": "sk-ant-test"},
            )
            assert resp.status_code == 200
            data = resp.json()
            assert data["type"] == "message"
            assert data["content"][0]["text"] == "Hello!"
        finally:
            await proxy.close()

    @respx.mock
    async def test_forward_upstream_error(self):
        """Forward propagates upstream error status."""
        respx.post("https://api.anthropic.com/v1/messages").mock(
            return_value=httpx.Response(
                401,
                json={"error": {"type": "authentication_error", "message": "Bad key"}},
            )
        )

        proxy = PassthroughProxy(timeout=10.0)
        try:
            resp = await proxy.forward(
                "https://api.anthropic.com/v1/messages",
                b'{"model":"claude","messages":[]}',
                {"x-api-key": "bad-key"},
            )
            assert resp.status_code == 401
            assert resp.json()["error"]["type"] == "authentication_error"
        finally:
            await proxy.close()

    @respx.mock
    async def test_forward_headers_passed(self):
        """Verify that headers are forwarded to upstream."""
        route = respx.post("https://api.openai.com/v1/chat/completions").mock(
            return_value=httpx.Response(200, json={"choices": []})
        )

        proxy = PassthroughProxy(timeout=10.0)
        try:
            await proxy.forward(
                "https://api.openai.com/v1/chat/completions",
                b"{}",
                {
                    "authorization": "Bearer sk-test",
                    "content-type": "application/json",
                    "anthropic-version": "2023-06-01",
                },
            )
            # Verify the headers were sent
            request = route.calls[0].request
            assert request.headers["authorization"] == "Bearer sk-test"
            assert request.headers["content-type"] == "application/json"
            assert request.headers["anthropic-version"] == "2023-06-01"
        finally:
            await proxy.close()

    @respx.mock
    async def test_forward_body_passed(self):
        """Verify that the request body is forwarded correctly."""
        route = respx.post("https://api.anthropic.com/v1/messages").mock(
            return_value=httpx.Response(200, json={"type": "message"})
        )

        body = b'{"model":"claude","messages":[{"role":"user","content":"test"}]}'
        proxy = PassthroughProxy(timeout=10.0)
        try:
            await proxy.forward(
                "https://api.anthropic.com/v1/messages",
                body,
                {"content-type": "application/json"},
            )
            request = route.calls[0].request
            assert request.content == body
        finally:
            await proxy.close()


class TestPassthroughProxyStream:
    @respx.mock
    async def test_forward_stream_success(self):
        """Stream response forwards SSE events."""
        sse_content = (
            b"event: message_start\n"
            b'data: {"type":"message_start"}\n\n'
            b"event: content_block_delta\n"
            b'data: {"type":"content_block_delta","delta":{"text":"Hello"}}\n\n'
            b"event: message_stop\n"
            b'data: {"type":"message_stop"}\n\n'
        )
        respx.post("https://api.anthropic.com/v1/messages").mock(
            return_value=httpx.Response(200, content=sse_content)
        )

        proxy = PassthroughProxy(timeout=10.0)
        try:
            chunks = []
            async for chunk in proxy.forward_stream(
                "https://api.anthropic.com/v1/messages",
                b'{"stream":true}',
                {"content-type": "application/json"},
            ):
                chunks.append(chunk)
            assert len(chunks) > 0
            combined = b"".join(chunks)
            assert b"message_start" in combined
        finally:
            await proxy.close()

    @respx.mock
    async def test_forward_stream_error(self):
        """Stream raises UpstreamStreamError on upstream error."""
        from stateloom.proxy.passthrough import UpstreamStreamError

        respx.post("https://api.anthropic.com/v1/messages").mock(
            return_value=httpx.Response(
                500,
                content=b'{"error": "internal"}',
            )
        )

        proxy = PassthroughProxy(timeout=10.0)
        try:
            with pytest.raises(UpstreamStreamError) as exc_info:
                async for _chunk in proxy.forward_stream(
                    "https://api.anthropic.com/v1/messages",
                    b'{"stream":true}',
                    {},
                ):
                    pass  # pragma: no cover
            assert exc_info.value.status_code == 500
            assert b"error" in exc_info.value.content
        finally:
            await proxy.close()


class TestPassthroughProxyLifecycle:
    async def test_close(self):
        """Proxy can be created and closed without error."""
        proxy = PassthroughProxy(timeout=5.0)
        await proxy.close()

    def test_default_timeout(self):
        """Default timeout is 600 seconds."""
        proxy = PassthroughProxy()
        assert proxy._client.timeout.read == 600.0
        assert proxy._client.timeout.connect == 10.0

    def test_custom_timeout(self):
        """Custom timeout is applied."""
        proxy = PassthroughProxy(timeout=30.0)
        assert proxy._client.timeout.read == 30.0


class TestHeaderConstants:
    def test_request_headers_contain_expected_members(self):
        for h in ("host", "connection", "transfer-encoding", "keep-alive", "upgrade"):
            assert h in REQUEST_HOP_BY_HOP_HEADERS

    def test_response_headers_contain_expected_members(self):
        for h in ("content-length", "content-encoding", "connection", "keep-alive"):
            assert h in RESPONSE_HOP_BY_HOP_HEADERS

    def test_content_encoding_not_in_request_set(self):
        assert "content-encoding" not in REQUEST_HOP_BY_HOP_HEADERS

    def test_host_not_in_response_set(self):
        assert "host" not in RESPONSE_HOP_BY_HOP_HEADERS
