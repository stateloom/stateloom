"""Priority-aware rate limiting middleware with request queueing."""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from collections import deque
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from stateloom.core.errors import StateLoomRateLimitError
from stateloom.core.event import RateLimitEvent
from stateloom.core.organization import Team
from stateloom.middleware.base import MiddlewareContext

if TYPE_CHECKING:
    from stateloom.observability.collector import MetricsCollector

logger = logging.getLogger("stateloom.middleware.rate_limiter")


@dataclass
class _QueuedRequest:
    """A request waiting for a token bucket slot."""

    event: threading.Event
    priority: int
    enqueued_at: float
    released: bool = False
    timed_out: bool = False


class _TokenBucket:
    """Per-team token bucket with priority queue."""

    def __init__(
        self,
        tps: float,
        priority: int = 0,
        max_queue_size: int = 100,
        queue_timeout: float = 30.0,
    ) -> None:
        self.tps = tps
        self.tokens = tps  # Start full
        self.last_refill = time.monotonic()
        self.priority = priority
        self.max_queue_size = max_queue_size
        self.queue_timeout = queue_timeout
        # priority -> FIFO deque of queued requests
        self.queue: dict[int, deque[_QueuedRequest]] = {}
        self.queue_size = 0
        self.active_count = 0
        # Lock: per-bucket lock (no cross-bucket locking)
        self.lock = threading.Lock()

    def refill(self) -> None:
        """Lazily refill tokens based on elapsed time. Must hold self.lock."""
        now = time.monotonic()
        elapsed = now - self.last_refill
        if elapsed > 0:
            self.tokens = min(self.tps, self.tokens + elapsed * self.tps)
            self.last_refill = now

    def try_acquire(self) -> bool:
        """Try to acquire a token. Must hold self.lock. Returns True if acquired."""
        self.refill()
        if self.tokens >= 1.0:
            self.tokens -= 1.0
            self.active_count += 1
            return True
        return False

    def release_next_waiter(self) -> None:
        """Release the highest-priority waiter if a token is available. Must hold self.lock."""
        self.refill()
        if self.tokens < 1.0 or self.queue_size == 0:
            return
        # Find highest priority with waiters
        priorities = sorted(self.queue.keys(), reverse=True)
        for prio in priorities:
            q = self.queue[prio]
            while q:
                req = q[0]
                if req.timed_out:
                    q.popleft()
                    self.queue_size -= 1
                    continue
                # Release this request
                self.tokens -= 1.0
                self.active_count += 1
                req.released = True
                q.popleft()
                self.queue_size -= 1
                # Signal the waiting thread (Event.set is thread-safe, no lock needed)
                req.event.set()
                return
            # Clean up empty priority level
            if not q:
                del self.queue[prio]


class RateLimiterMiddleware:
    """Per-team TPS limiting with priority-aware request queueing.

    Sits after BlastRadius and before Experiment in the middleware chain.
    Teams without rate_limit_tps configured pass through unthrottled.
    """

    def __init__(
        self,
        team_lookup: Callable[[str], Team | None],
        store: Any = None,
        metrics: MetricsCollector | None = None,
    ) -> None:
        self._team_lookup = team_lookup
        self._store = store
        self._metrics = metrics
        self._buckets: dict[str, _TokenBucket] = {}
        self._buckets_lock = threading.Lock()

    async def process(
        self,
        ctx: MiddlewareContext,
        call_next: Callable[[MiddlewareContext], Awaitable[Any]],
    ) -> Any:
        if not ctx.config.rate_limiting_enabled:
            return await call_next(ctx)

        team_id = ctx.session.team_id
        if not team_id:
            return await call_next(ctx)

        team = self._team_lookup(team_id)
        if team is None or team.rate_limit_tps is None:
            return await call_next(ctx)

        bucket = self._get_or_create_bucket(team)
        start_time = time.monotonic()
        acquired = False
        req: _QueuedRequest | None = None

        with bucket.lock:
            # Hot-reload config from team object
            bucket.tps = team.rate_limit_tps
            bucket.priority = team.rate_limit_priority
            bucket.max_queue_size = team.rate_limit_max_queue
            bucket.queue_timeout = team.rate_limit_queue_timeout

            if bucket.try_acquire():
                acquired = True
                logger.debug("Rate limiter: token acquired for team=%s", team_id)
            elif bucket.queue_size >= bucket.max_queue_size:
                # Queue full — reject immediately
                logger.warning(
                    "Rate limiter: queue full, rejecting request for team=%s "
                    "(queue_size=%d, max=%d, tps=%.1f)",
                    team_id, bucket.queue_size, bucket.max_queue_size, bucket.tps,
                )
                event = RateLimitEvent(
                    session_id=ctx.session.id,
                    step=ctx.session.step_counter,
                    team_id=team_id,
                    queued=False,
                    rejected=True,
                )
                ctx.events.append(event)
                self._save_events_directly(ctx)
                if self._metrics:
                    self._metrics.record_rate_limit(
                        team_id=team_id,
                        outcome="rejected",
                    )
                raise StateLoomRateLimitError(
                    team_id=team_id,
                    tps=bucket.tps,
                    queue_size=bucket.queue_size,
                    message="Rate limit exceeded: queue full",
                )
            else:
                # Enqueue
                logger.debug(
                    "Rate limiter: queuing request for team=%s (queue_size=%d, priority=%d)",
                    team_id, bucket.queue_size, bucket.priority,
                )
                req = _QueuedRequest(
                    event=threading.Event(),
                    priority=bucket.priority,
                    enqueued_at=time.monotonic(),
                )
                prio = bucket.priority
                if prio not in bucket.queue:
                    bucket.queue[prio] = deque()
                bucket.queue[prio].append(req)
                bucket.queue_size += 1

        # Lock is released here

        if acquired:
            # Token was available — proceed immediately
            event = RateLimitEvent(
                session_id=ctx.session.id,
                step=ctx.session.step_counter,
                team_id=team_id,
                queued=False,
                wait_ms=0.0,
            )
            ctx.events.append(event)
            if self._metrics:
                self._metrics.record_rate_limit(
                    team_id=team_id,
                    outcome="passed",
                )
            if ctx.is_streaming:
                result = await call_next(ctx)
                ctx._on_stream_complete.append(lambda: self._on_request_complete(bucket))
                return result
            try:
                result = await call_next(ctx)
            finally:
                self._on_request_complete(bucket)
            return result

        # Wait for slot (outside bucket.lock, using run_in_executor to not block event loop)
        assert req is not None
        loop = asyncio.get_running_loop()
        slot_acquired = await loop.run_in_executor(None, self._wait_for_slot, req, bucket)

        wait_ms = (time.monotonic() - start_time) * 1000

        if not slot_acquired:
            # Timeout
            logger.warning(
                "Rate limiter: queue timeout for team=%s after %.0fms (timeout=%.1fs)",
                team_id, wait_ms, bucket.queue_timeout,
            )
            event = RateLimitEvent(
                session_id=ctx.session.id,
                step=ctx.session.step_counter,
                team_id=team_id,
                queued=True,
                wait_ms=wait_ms,
                timed_out=True,
            )
            ctx.events.append(event)
            self._save_events_directly(ctx)
            if self._metrics:
                self._metrics.record_rate_limit(
                    team_id=team_id,
                    outcome="timed_out",
                    wait_ms=wait_ms,
                )
            raise StateLoomRateLimitError(
                team_id=team_id,
                tps=bucket.tps,
                queue_size=bucket.queue_size,
                message="Rate limit exceeded: queue timeout",
            )

        # Released from queue — proceed
        logger.debug("Rate limiter: released from queue for team=%s after %.0fms", team_id, wait_ms)
        event = RateLimitEvent(
            session_id=ctx.session.id,
            step=ctx.session.step_counter,
            team_id=team_id,
            queued=True,
            wait_ms=wait_ms,
        )
        ctx.events.append(event)
        if self._metrics:
            self._metrics.record_rate_limit(
                team_id=team_id,
                outcome="queued",
                wait_ms=wait_ms,
            )
        if ctx.is_streaming:
            result = await call_next(ctx)
            ctx._on_stream_complete.append(lambda: self._on_request_complete(bucket))
            return result
        try:
            result = await call_next(ctx)
        finally:
            self._on_request_complete(bucket)
        return result

    def _wait_for_slot(self, req: _QueuedRequest, bucket: _TokenBucket) -> bool:
        """Block the current thread until a slot opens or timeout.

        Returns True if slot acquired, False on timeout.
        """
        deadline = req.enqueued_at + bucket.queue_timeout

        while not req.released:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                req.timed_out = True
                # Clean up from queue
                with bucket.lock:
                    prio_q = bucket.queue.get(req.priority)
                    if prio_q:
                        try:
                            prio_q.remove(req)
                            bucket.queue_size -= 1
                        except ValueError:
                            pass
                return False

            # Try to self-acquire via refill
            with bucket.lock:
                if req.released:
                    return True
                bucket.refill()
                if bucket.tokens >= 1.0:
                    bucket.tokens -= 1.0
                    bucket.active_count += 1
                    req.released = True
                    # Remove from queue
                    prio_q = bucket.queue.get(req.priority)
                    if prio_q:
                        try:
                            prio_q.remove(req)
                            bucket.queue_size -= 1
                        except ValueError:
                            pass
                    return True

            # Wait for signal or periodic self-check
            wait_time = min(0.5, remaining)
            req.event.wait(timeout=wait_time)

        return True

    def _on_request_complete(self, bucket: _TokenBucket) -> None:
        """Called when a request finishes — release next waiter if possible."""
        with bucket.lock:
            bucket.active_count = max(0, bucket.active_count - 1)
            bucket.release_next_waiter()

    def _save_events_directly(self, ctx: MiddlewareContext) -> None:
        """Persist events directly to the store (bypass EventRecorder).

        Exception after this call prevents EventRecorder from running — no
        duplicate risk.
        """
        if not self._store:
            return
        for event in ctx.events:
            try:
                self._store.save_event(event)
            except Exception:
                logger.debug("Failed to persist rate limit event", exc_info=True)
        try:
            self._store.save_session(ctx.session)
        except Exception:
            logger.debug("Failed to persist session after rate limit", exc_info=True)

    def _get_or_create_bucket(self, team: Team) -> _TokenBucket:
        """Get or create a token bucket for a team."""
        team_id = team.id
        with self._buckets_lock:
            bucket = self._buckets.get(team_id)
            if bucket is None:
                bucket = _TokenBucket(
                    tps=team.rate_limit_tps or 1.0,
                    priority=team.rate_limit_priority,
                    max_queue_size=team.rate_limit_max_queue,
                    queue_timeout=team.rate_limit_queue_timeout,
                )
                self._buckets[team_id] = bucket
            return bucket

    def remove_bucket(self, team_id: str) -> None:
        """Remove a team's bucket (when TPS is set to None)."""
        with self._buckets_lock:
            self._buckets.pop(team_id, None)

    def get_status(self) -> dict:
        """Return current rate limiter state for all teams."""
        teams: dict[str, dict] = {}
        with self._buckets_lock:
            for team_id, bucket in self._buckets.items():
                with bucket.lock:
                    bucket.refill()
                    teams[team_id] = {
                        "tps": bucket.tps,
                        "tokens_available": round(bucket.tokens, 2),
                        "queue_size": bucket.queue_size,
                        "active_requests": bucket.active_count,
                        "priority": bucket.priority,
                        "max_queue_size": bucket.max_queue_size,
                        "queue_timeout": bucket.queue_timeout,
                    }
        return {"teams": teams}
