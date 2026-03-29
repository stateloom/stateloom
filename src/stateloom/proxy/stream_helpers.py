"""Shared helpers for passthrough streaming relay and SSE responses."""

from __future__ import annotations

import logging
from collections.abc import AsyncGenerator, Callable
from typing import Any

from fastapi.responses import StreamingResponse
from starlette.responses import Response

from stateloom.proxy.passthrough import PassthroughProxy, UpstreamStreamError
from stateloom.proxy.rate_limiter import ProxyRateLimiter

logger = logging.getLogger("stateloom.proxy.stream_helpers")

SSE_HEADERS = {
    "Cache-Control": "no-cache",
    "Connection": "keep-alive",
    "X-Accel-Buffering": "no",
}


async def passthrough_stream_relay(
    passthrough: PassthroughProxy,
    upstream_url: str,
    body: bytes,
    headers: dict[str, str],
    *,
    ctx: Any = None,
    track_usage: Callable[[str, Any], None] | None = None,
    format_error: Callable[[Exception], bytes] | None = None,
    proxy_rate_limiter: ProxyRateLimiter | None = None,
    vk_id: str | None = None,
) -> Response:
    """Shared async streaming helper with upstream error detection.

    Eagerly peeks the first chunk from the upstream to detect errors
    (4xx/5xx) *before* committing to a 200 ``StreamingResponse``.
    Upstream errors are returned with their original status code so CLIs
    can display them properly.

    Args:
        passthrough: HTTP reverse proxy engine.
        upstream_url: Full URL to POST to.
        body: Raw request body bytes.
        headers: Pre-filtered headers.
        ctx: Optional MiddlewareContext for stream-complete callbacks.
        track_usage: Optional per-chunk usage tracker ``(chunk_str, ctx) -> None``.
        format_error: Optional error formatter ``(exc) -> bytes``. If None,
            errors are silently logged.
        proxy_rate_limiter: Optional per-VK rate limiter to release on completion.
        vk_id: Virtual key ID for rate limiter release.
    """
    gen = passthrough.forward_stream(upstream_url, body, headers)
    logger.debug("Stream relay started: %s (%d bytes)", upstream_url[:120], len(body))

    # Peek the first chunk — if upstream errored, forward_stream raises
    # UpstreamStreamError instead of yielding data.
    try:
        first_chunk = await gen.__anext__()
    except UpstreamStreamError as e:
        logger.debug("Upstream error %d for %s", e.status_code, upstream_url[:120])
        # Clean up: run stream-complete callbacks and release rate limiter slot
        if ctx is not None:
            for cb in ctx._on_stream_complete:
                try:
                    cb()
                except Exception:
                    pass
        if proxy_rate_limiter and vk_id:
            proxy_rate_limiter.on_request_complete(vk_id)
        return Response(
            content=e.content,
            status_code=e.status_code,
            media_type=e.content_type or "application/json",
        )
    except StopAsyncIteration:
        first_chunk = b""

    async def generate() -> AsyncGenerator[bytes, None]:
        try:
            if first_chunk:
                if track_usage is not None and ctx is not None:
                    s = first_chunk.decode() if isinstance(first_chunk, bytes) else first_chunk
                    track_usage(s, ctx)
                yield first_chunk
            async for chunk in gen:
                if track_usage is not None and ctx is not None:
                    s = chunk.decode("utf-8") if isinstance(chunk, bytes) else chunk
                    track_usage(s, ctx)
                yield chunk
        except Exception as exc:
            logger.exception("Passthrough streaming error")
            if format_error is not None:
                yield format_error(exc)
        finally:
            if ctx is not None:
                for cb in ctx._on_stream_complete:
                    try:
                        cb()
                    except Exception:
                        pass
            if proxy_rate_limiter and vk_id:
                proxy_rate_limiter.on_request_complete(vk_id)

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers=SSE_HEADERS,
    )
