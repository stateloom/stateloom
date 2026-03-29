"""Loop detection middleware — detect repeated LLM call patterns."""

from __future__ import annotations

import logging
import threading
from collections import defaultdict
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any

from stateloom.core.config import StateLoomConfig
from stateloom.core.event import LoopDetectionEvent
from stateloom.middleware.base import MiddlewareContext

if TYPE_CHECKING:
    from stateloom.store.base import Store

logger = logging.getLogger("stateloom.middleware.loop")


class LoopDetector:
    """Detects repeated LLM call patterns within a session.

    Exact-match detection: tracks request hashes per session.
    If the same request hash appears N times (configurable threshold),
    triggers the configured action.

    Note: Tool-call loop detection is handled separately by the
    Gate.tool() decorator via Gate._check_tool_loop().
    """

    def __init__(self, config: StateLoomConfig, store: Store | None = None) -> None:
        self._threshold = config.loop_exact_threshold
        self._store = store
        # session_id -> {request_hash -> count}
        self._counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        # Lock: single _lock guards _counts dict
        self._lock = threading.Lock()

    async def process(
        self,
        ctx: MiddlewareContext,
        call_next: Callable[[MiddlewareContext], Awaitable[Any]],
    ) -> Any:
        request_hash = ctx.request_hash
        session_id = ctx.session.id

        if not request_hash:
            return await call_next(ctx)

        # Increment count
        with self._lock:
            self._counts[session_id][request_hash] += 1
            count = self._counts[session_id][request_hash]

        if count >= self._threshold:
            event = LoopDetectionEvent(
                session_id=session_id,
                step=ctx.session.step_counter,
                pattern_hash=request_hash,
                repeat_count=count,
                action="circuit_break",
            )
            ctx.events.append(event)

            logger.warning(
                "Loop detected in session '%s': request repeated %d times",
                session_id,
                count,
            )

            # Block the request — skip LLM call and return a static response
            ctx.skip_call = True
            ctx.cached_response = {
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": (
                                f"[StateLoom] Request blocked: loop detected "
                                f"(pattern repeated {count} times in session '{session_id}')"
                            ),
                        },
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            }
            return await call_next(ctx)

        return await call_next(ctx)

    def on_session_end(self, session_id: str) -> None:
        """Clean up per-session loop tracking data when a session ends."""
        with self._lock:
            self._counts.pop(session_id, None)
