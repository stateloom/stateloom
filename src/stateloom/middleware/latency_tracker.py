"""Latency tracking middleware — per-call timing."""

from __future__ import annotations

import time
from collections.abc import Awaitable, Callable
from typing import Any

from stateloom.middleware.base import MiddlewareContext


class LatencyTracker:
    """Records latency for each LLM call."""

    async def process(
        self,
        ctx: MiddlewareContext,
        call_next: Callable[[MiddlewareContext], Awaitable[Any]],
    ) -> Any:
        start = time.perf_counter()
        result = await call_next(ctx)

        if ctx.is_streaming:

            def _on_complete() -> None:
                ctx.latency_ms = (time.perf_counter() - start) * 1000

            ctx._on_stream_complete.append(_on_complete)
        else:
            ctx.latency_ms = (time.perf_counter() - start) * 1000

        return result
