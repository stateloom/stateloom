"""Universal catch-all passthrough for unmatched /v1/* and /v1beta/* paths.

Specific proxy routes (chat/completions, messages, generateContent, responses)
are registered first in ``server.py``.  These catch-all routers are mounted
**last** so that any path not matched by a specific handler is forwarded to the
correct upstream provider as pure HTTP passthrough (no middleware pipeline).

This enables SDK utility calls (``countTokens``, model listing, embeddings,
batch APIs, etc.) to work transparently through the gateway.
"""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Any

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, StreamingResponse
from starlette.responses import Response

from stateloom.proxy.passthrough import (
    DEFAULT_UPSTREAM_URLS,
    RESPONSE_HOP_BY_HOP_HEADERS,
    PassthroughProxy,
    UpstreamStreamError,
    filter_headers,
)
from stateloom.proxy.stream_helpers import SSE_HEADERS

if TYPE_CHECKING:
    from stateloom.gate import Gate

logger = logging.getLogger("stateloom.proxy.catch_all")


def _detect_provider(request: Request) -> str:
    """Detect the target provider from request headers.

    Priority:
      - ``x-api-key`` header present -> Anthropic
      - ``x-goog-api-key`` header present -> Gemini
      - Otherwise -> OpenAI (default)
    """
    if request.headers.get("x-api-key"):
        return "anthropic"
    if request.headers.get("x-goog-api-key"):
        return "gemini"
    return "openai"


def _upstream_base(provider: str, gate: Gate) -> str:
    """Resolve the upstream base URL for a provider."""
    if provider == "anthropic":
        return gate.config.proxy.upstream_anthropic or DEFAULT_UPSTREAM_URLS["anthropic"]
    if provider == "gemini":
        return gate.config.proxy.upstream_gemini or DEFAULT_UPSTREAM_URLS["gemini"]
    return gate.config.proxy.upstream_openai or DEFAULT_UPSTREAM_URLS["openai"]


def _is_streaming_request(body: bytes, request: Request) -> bool:
    """Detect whether a request expects a streaming response.

    Checks for ``"stream": true`` in a JSON body (OpenAI/Anthropic convention)
    and ``?alt=sse`` query parameter (Gemini convention).
    """
    if request.query_params.get("alt") == "sse":
        return True
    if body:
        try:
            data = json.loads(body)
            if isinstance(data, dict) and data.get("stream") is True:
                return True
        except Exception:
            pass
    return False


def _error_for_provider(provider: str, status: int, message: str) -> dict[str, Any]:
    """Build a provider-matched error response body."""
    if provider == "anthropic":
        return {
            "type": "error",
            "error": {"type": "api_error", "message": message},
        }
    if provider == "gemini":
        return {
            "error": {
                "code": status,
                "message": message,
                "status": "UNAVAILABLE" if status == 503 else "INTERNAL",
            },
        }
    # OpenAI format (default)
    return {
        "error": {
            "message": message,
            "type": "server_error",
            "code": None,
        },
    }


def create_catch_all_routers(
    gate: Gate,
    passthrough: PassthroughProxy | None = None,
) -> tuple[APIRouter, APIRouter]:
    """Create catch-all routers for ``/v1`` and ``/v1beta`` prefixes.

    Returns:
        A ``(v1_router, v1beta_router)`` tuple.  Mount **after** all specific
        proxy routers so FastAPI's registration-order matching gives them
        priority.
    """
    v1_router = APIRouter()
    v1beta_router = APIRouter()

    async def _forward_and_respond(
        request: Request,
        prefix: str,
        path: str,
        provider: str,
    ) -> Response:
        """Core forwarding logic shared by both routers."""
        if passthrough is None:
            return JSONResponse(
                status_code=503,
                content=_error_for_provider(provider, 503, "Proxy passthrough is not available"),
            )

        upstream_base = _upstream_base(provider, gate)
        upstream_url = f"{upstream_base}/{prefix}/{path}"

        # Preserve query parameters
        query = str(request.url.query)
        if query:
            upstream_url += f"?{query}"

        # Read body (may be empty for GET/DELETE)
        raw_body = await request.body()
        body = raw_body if raw_body else None

        # Filter headers — no auth replacement, forward as-is
        upstream_headers = filter_headers(request.headers)

        method = request.method

        logger.debug(
            "Catch-all %s /%s/%s -> %s (%s)",
            method,
            prefix,
            path,
            upstream_url[:120],
            provider,
        )

        # Detect streaming
        is_stream = method == "POST" and _is_streaming_request(raw_body, request)

        try:
            if is_stream:
                return await _handle_stream(
                    passthrough, upstream_url, raw_body, upstream_headers, provider
                )

            resp = await passthrough.forward_any(method, upstream_url, body, upstream_headers)

            # Filter response hop-by-hop headers
            resp_headers = {
                k: v
                for k, v in resp.headers.items()
                if k.lower() not in RESPONSE_HOP_BY_HOP_HEADERS
            }

            return Response(
                content=resp.content,
                status_code=resp.status_code,
                headers=resp_headers,
            )
        except Exception:
            logger.exception("Catch-all passthrough error: %s /%s/%s", method, prefix, path)
            return JSONResponse(
                status_code=502,
                content=_error_for_provider(provider, 502, "Upstream request failed"),
            )

    async def _handle_stream(
        passthrough: PassthroughProxy,
        upstream_url: str,
        body: bytes,
        headers: dict[str, str],
        provider: str,
    ) -> Response:
        """Handle streaming passthrough with eager error detection."""
        gen = passthrough.forward_stream(upstream_url, body, headers)

        try:
            first_chunk = await gen.__anext__()
        except UpstreamStreamError as e:
            return Response(
                content=e.content,
                status_code=e.status_code,
                media_type=e.content_type or "application/json",
            )
        except StopAsyncIteration:
            first_chunk = b""

        async def generate() -> AsyncIterator[bytes]:
            try:
                if first_chunk:
                    yield first_chunk
                async for chunk in gen:
                    yield chunk
            except Exception:
                logger.exception("Catch-all streaming error")

        return StreamingResponse(
            generate(),
            media_type="text/event-stream",
            headers=SSE_HEADERS,
        )

    # --- /v1 catch-all: provider detected from headers ---

    @v1_router.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH"])
    async def v1_catch_all(path: str, request: Request) -> Response:
        provider = _detect_provider(request)
        return await _forward_and_respond(request, "v1", path, provider)

    # --- /v1beta catch-all: always Gemini ---

    @v1beta_router.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH"])
    async def v1beta_catch_all(path: str, request: Request) -> Response:
        return await _forward_and_respond(request, "v1beta", path, "gemini")

    return v1_router, v1beta_router
