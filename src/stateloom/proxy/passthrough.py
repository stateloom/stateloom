"""HTTP reverse proxy engine for forwarding requests to upstream LLM APIs.

Instead of instantiating SDK clients (which require API keys), this module
forwards raw HTTP requests to upstream providers using httpx. The CLI's own
authentication headers pass through transparently, enabling subscription users
(Claude Max, Gemini Ultra) whose CLIs use OAuth/session tokens.
"""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncGenerator
from typing import Any, cast

import httpx

logger = logging.getLogger("stateloom.proxy.passthrough")

# Shared base hop-by-hop headers (common to both request and response filtering)
_BASE_HOP_BY_HOP = frozenset({"connection", "transfer-encoding", "keep-alive"})

# Request-direction: strip before forwarding to upstream
REQUEST_HOP_BY_HOP_HEADERS = _BASE_HOP_BY_HOP | frozenset(
    {
        "host",
        "upgrade",
        "proxy-connection",
        "te",
        "trailer",
    }
)

# Response-direction: strip before returning to client
RESPONSE_HOP_BY_HOP_HEADERS = _BASE_HOP_BY_HOP | frozenset(
    {
        "content-length",  # httpx/ASGI recalculates
        "content-encoding",  # httpx auto-decompresses; forwarding causes double-decompress
    }
)

# StateLoom-internal headers that should be stripped
_STATELOOM_HEADERS = frozenset(
    {
        "x-stateloom-session-id",
        "x-stateloom-openai-key",
        "x-stateloom-anthropic-key",
        "x-stateloom-google-key",
        "x-stateloom-end-user",
    }
)

# Default upstream URLs
DEFAULT_UPSTREAM_URLS = {
    "anthropic": "https://api.anthropic.com",
    "openai": "https://api.openai.com",
    "gemini": "https://generativelanguage.googleapis.com",
    "code_assist": "https://cloudcode-pa.googleapis.com",
}


class UpstreamStreamError(Exception):
    """Raised by :meth:`forward_stream` when the upstream returns 4xx/5xx.

    Callers catch this *before* creating a ``StreamingResponse`` so they
    can return a proper HTTP error response instead of wrapping the error
    body inside a 200 SSE stream (which CLIs can't parse).
    """

    def __init__(self, status_code: int, content: bytes, content_type: str = "") -> None:
        self.status_code = status_code
        self.content = content
        self.content_type = content_type
        super().__init__(f"Upstream returned {status_code}")

    def error_json(self) -> dict[str, Any]:
        """Parse the error body as JSON, with a fallback."""
        try:
            data = json.loads(self.content)
            # Gemini wraps errors in a JSON array
            if isinstance(data, list) and data:
                return cast(dict[str, Any], data[0])
            return cast(dict[str, Any], data)
        except Exception:
            return {
                "error": {
                    "code": self.status_code,
                    "message": self.content.decode("utf-8", errors="replace")[:500],
                }
            }


class PassthroughProxy:
    """HTTP reverse proxy for forwarding requests to upstream LLM APIs."""

    def __init__(self, timeout: float = 600.0) -> None:
        """Initialize the reverse proxy with a pooled httpx client.

        Args:
            timeout: Total request timeout in seconds (default 600).
                The connect timeout is always 10 s.  Connection pool
                allows up to 100 concurrent connections with 20 keepalive.
        """
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(timeout, connect=10.0),
            follow_redirects=True,
            limits=httpx.Limits(max_connections=100, max_keepalive_connections=20),
        )

    async def forward(
        self,
        upstream_url: str,
        body: bytes,
        headers: dict[str, str],
    ) -> httpx.Response:
        """Forward a request and return the complete response.

        Args:
            upstream_url: Full URL to POST to (e.g.
                ``https://api.openai.com/v1/chat/completions``).
            body: Raw request body bytes.
            headers: Pre-filtered headers (see ``filter_headers()``).

        Returns:
            The upstream ``httpx.Response`` (caller checks status code).
        """
        logger.debug("Forward request: POST %s (%d bytes)", upstream_url[:120], len(body))
        response = await self._client.post(
            upstream_url,
            content=body,
            headers=headers,
        )
        logger.debug(
            "Forward response: %d %s (%s)",
            response.status_code,
            upstream_url[:80],
            response.headers.get("content-type", ""),
        )
        return response

    async def forward_any(
        self,
        method: str,
        upstream_url: str,
        body: bytes | None,
        headers: dict[str, str],
    ) -> httpx.Response:
        """Forward a request with any HTTP method.

        Unlike :meth:`forward` (POST-only), this supports GET, PUT, DELETE, etc.
        for forwarding SDK utility calls (token counting, model listing, etc.).

        Args:
            method: HTTP method (``GET``, ``POST``, ``PUT``, ``DELETE``, etc.).
            upstream_url: Full URL to send the request to.
            body: Raw request body bytes, or ``None`` for body-less methods.
            headers: Pre-filtered headers (see ``filter_headers()``).

        Returns:
            The upstream ``httpx.Response``.
        """
        logger.debug("Forward request: %s %s", method, upstream_url[:120])
        response = await self._client.request(
            method,
            upstream_url,
            content=body,
            headers=headers,
        )
        logger.debug(
            "Forward response: %d %s (%s)",
            response.status_code,
            upstream_url[:80],
            response.headers.get("content-type", ""),
        )
        return response

    async def forward_stream(
        self,
        upstream_url: str,
        body: bytes,
        headers: dict[str, str],
    ) -> AsyncGenerator[bytes, None]:
        """Forward a request and stream the response line-by-line.

        Args:
            upstream_url: Full URL to POST to.
            body: Raw request body bytes.
            headers: Pre-filtered headers.

        Yields:
            UTF-8 encoded SSE lines (each ending with ``\\n\\n``).

        Raises:
            UpstreamStreamError: If the upstream returns 4xx/5xx.  The
                caller should catch this *before* creating a
                ``StreamingResponse`` to return a proper HTTP error.
        """
        async with self._client.stream(
            "POST",
            upstream_url,
            content=body,
            headers=headers,
        ) as response:
            logger.debug(
                "Upstream stream response: %d %s (url=%s)",
                response.status_code,
                response.headers.get("content-type", ""),
                upstream_url[:120],
            )
            if response.status_code >= 400:
                await response.aread()
                logger.debug("Upstream error body: %s", response.content[:500])
                raise UpstreamStreamError(
                    response.status_code,
                    response.content,
                    response.headers.get("content-type", ""),
                )
            line_count = 0
            async for line in response.aiter_lines():
                if line:
                    line_count += 1
                    yield (line + "\n\n").encode("utf-8")
            logger.debug("Upstream stream completed: %d lines yielded", line_count)

    async def close(self) -> None:
        """Close the underlying HTTP client."""
        await self._client.aclose()


def filter_headers(
    request_headers: dict[str, str] | Any,
    *,
    auth_header_name: str = "",
    auth_header_value: str = "",
) -> dict[str, str]:
    """Filter request headers for upstream forwarding.

    Strips hop-by-hop and StateLoom-internal headers. Optionally replaces
    the auth header (for VK mode where we inject the resolved provider key).

    Args:
        request_headers: Original request headers (may be a Starlette Headers object).
        auth_header_name: If set, the auth header to replace (e.g. "x-api-key").
        auth_header_value: The value to set for the auth header.
    """
    filtered: dict[str, str] = {}
    for key, value in request_headers.items():
        lower_key = key.lower()
        if lower_key in REQUEST_HOP_BY_HOP_HEADERS:
            continue
        if lower_key in _STATELOOM_HEADERS:
            continue
        # If we're replacing auth, skip the original
        if auth_header_name and lower_key == auth_header_name.lower():
            continue
        # Skip content-length as httpx will recalculate
        if lower_key == "content-length":
            continue
        filtered[key] = value

    # Inject the resolved auth header
    if auth_header_name and auth_header_value:
        filtered[auth_header_name] = auth_header_value

    return filtered
