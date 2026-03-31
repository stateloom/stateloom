"""Parallel pre-computation middleware for expensive semantic operations.

Launches semantic complexity scoring in a background executor so it runs
concurrently with PII → Budget → Cache → Loop middleware. By the time
AutoRouter needs the score, it's already computed (or nearly so).
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from typing import Any

from stateloom.middleware.base import MiddlewareContext

logger = logging.getLogger("stateloom.middleware.precompute")


def _extract_last_user_text(request_kwargs: dict[str, Any]) -> str:
    """Extract the last user message text from request kwargs."""
    messages = request_kwargs.get("messages", [])
    for msg in reversed(messages):
        if isinstance(msg, dict) and msg.get("role") == "user":
            content = msg.get("content", "")
            if isinstance(content, list):
                content = " ".join(
                    b.get("text", "")
                    for b in content
                    if isinstance(b, dict) and b.get("type") == "text"
                )
            return str(content)
    return ""


class PrecomputeMiddleware:
    """Launches expensive semantic scoring in a background executor.

    Sits after ExperimentMiddleware (which may change model/kwargs)
    and before PII/Cache/AutoRouter. The scoring runs concurrently
    with PII → Budget → Cache → Loop middleware. AutoRouter awaits the
    result instead of computing inline.
    """

    def __init__(
        self,
        semantic_classifier: Any,
        compliance_fn: Any = None,
    ) -> None:
        self._classifier = semantic_classifier
        self._compliance_fn = compliance_fn

    async def process(
        self,
        ctx: MiddlewareContext,
        call_next: Callable[[MiddlewareContext], Awaitable[Any]],
    ) -> Any:
        if self._classifier is None:
            return await call_next(ctx)

        # Skip for CLI internal overhead calls — no point scoring
        # complexity for haiku/flash quota checks.
        if ctx.request_kwargs.get("_cli_internal"):
            return await call_next(ctx)

        # Skip when compliance profile blocks local routing — the
        # AutoRouter will bail out anyway, so don't waste a background
        # thread loading the CrossEncoder model.
        if self._compliance_fn:
            try:
                profile = self._compliance_fn(ctx.session.org_id, ctx.session.team_id)
                if profile and profile.block_local_routing:
                    return await call_next(ctx)
            except Exception:
                logger.debug("Compliance check in precompute failed", exc_info=True)

        last_text = _extract_last_user_text(ctx.request_kwargs)
        if not last_text:
            return await call_next(ctx)

        loop = asyncio.get_event_loop()
        ctx._precomputed_complexity_future = loop.run_in_executor(
            None, self._classifier.classify, last_text
        )

        return await call_next(ctx)
