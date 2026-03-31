"""Per-virtual-key TPS rate limiting for the proxy gateway."""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from typing import TYPE_CHECKING, Any

from stateloom.core.errors import StateLoomRateLimitError
from stateloom.middleware.rate_limiter import _QueuedRequest, _TokenBucket

if TYPE_CHECKING:
    from stateloom.observability.collector import MetricsCollector
    from stateloom.proxy.virtual_key import VirtualKey

logger = logging.getLogger("stateloom.proxy.rate_limiter")


class ProxyRateLimiter:
    """Per-virtual-key TPS rate limiting for the proxy gateway.

    Reuses _TokenBucket and _QueuedRequest from the middleware layer.
    Keyed by virtual key ID. Keys without rate_limit_tps pass through.
    """

    def __init__(self, metrics: MetricsCollector | None = None, enabled: bool = True) -> None:
        self._buckets: dict[str, _TokenBucket] = {}
        self._buckets_lock = threading.Lock()
        self._metrics = metrics
        self._enabled = enabled

        # Guardrail 3: Global TPS cap for restricted dev mode
        self._global_dev_bucket: _TokenBucket | None = None
        try:
            from stateloom.ee import DEV_MODE_LIMITS, is_restricted_dev_mode

            if is_restricted_dev_mode():
                self._global_dev_bucket = _TokenBucket(
                    tps=DEV_MODE_LIMITS["max_tps"],
                    priority=0,
                    max_queue_size=10,
                    queue_timeout=5.0,
                )
        except ImportError:
            pass

    async def check(self, vk: VirtualKey) -> None:
        """Acquire a rate limit slot for this virtual key.

        Raises StateLoomRateLimitError if the queue is full or times out.
        Does nothing if vk.rate_limit_tps is None or rate limiting is disabled.
        """
        if not self._enabled:
            return

        # Guardrail 3: Global dev mode TPS cap (checked before per-VK)
        if self._global_dev_bucket is not None:
            with self._global_dev_bucket.lock:
                self._global_dev_bucket.refill()
                if self._global_dev_bucket.tokens < 1.0:
                    raise StateLoomRateLimitError(
                        team_id="__dev_mode__",
                        tps=self._global_dev_bucket.tps,
                        queue_size=0,
                        message=(
                            "Dev mode global rate limit: max 5 TPS. Purchase a license to remove."
                        ),
                    )
                self._global_dev_bucket.tokens -= 1.0

        tps = vk.rate_limit_tps
        if tps is None:
            return

        start_time = time.monotonic()
        bucket = self._get_or_create_bucket(vk)
        req: _QueuedRequest | None = None

        with bucket.lock:
            # Hot-reload config from VK
            bucket.tps = tps
            bucket.max_queue_size = vk.rate_limit_max_queue
            bucket.queue_timeout = vk.rate_limit_queue_timeout

            if bucket.try_acquire():
                self._record(vk.id, outcome="passed")
                return

            if bucket.queue_size >= bucket.max_queue_size:
                self._record(vk.id, outcome="rejected")
                raise StateLoomRateLimitError(
                    team_id=vk.id,
                    tps=bucket.tps,
                    queue_size=bucket.queue_size,
                    message=f"Rate limit exceeded for key {vk.key_preview}: queue full",
                )

            # Enqueue — all per-key requests use priority 0 (FIFO)
            req = _QueuedRequest(
                event=threading.Event(),
                priority=0,
                enqueued_at=time.monotonic(),
            )
            if 0 not in bucket.queue:
                from collections import deque

                bucket.queue[0] = deque()
            bucket.queue[0].append(req)
            bucket.queue_size += 1

        # Lock released — wait for slot
        assert req is not None
        loop = asyncio.get_running_loop()
        slot_acquired = await loop.run_in_executor(None, self._wait_for_slot, req, bucket)

        wait_ms = (time.monotonic() - start_time) * 1000.0

        if not slot_acquired:
            self._record(vk.id, outcome="timed_out", wait_ms=wait_ms)
            raise StateLoomRateLimitError(
                team_id=vk.id,
                tps=bucket.tps,
                queue_size=bucket.queue_size,
                message=f"Rate limit exceeded for key {vk.key_preview}: queue timeout",
            )

        self._record(vk.id, outcome="queued", wait_ms=wait_ms)

    def _record(
        self,
        virtual_key_id: str,
        *,
        outcome: str,
        wait_ms: float = 0.0,
    ) -> None:
        """Record a rate limit metric if metrics are enabled."""
        if self._metrics:
            self._metrics.record_rate_limit(
                team_id="",
                virtual_key_id=virtual_key_id,
                outcome=outcome,
                wait_ms=wait_ms,
            )

    def on_request_complete(self, vk_id: str) -> None:
        """Release a slot after request finishes — triggers next waiter."""
        with self._buckets_lock:
            bucket = self._buckets.get(vk_id)
        if bucket is None:
            return
        with bucket.lock:
            bucket.active_count = max(0, bucket.active_count - 1)
            bucket.release_next_waiter()

    def remove_bucket(self, vk_id: str) -> None:
        """Remove a key's bucket (when rate limit is removed)."""
        with self._buckets_lock:
            self._buckets.pop(vk_id, None)

    def get_status(self) -> dict[str, Any]:
        """Return current state for all keys (dashboard use)."""
        keys: dict[str, dict[str, Any]] = {}
        with self._buckets_lock:
            for vk_id, bucket in self._buckets.items():
                with bucket.lock:
                    bucket.refill()
                    keys[vk_id] = {
                        "tps": bucket.tps,
                        "tokens_available": round(bucket.tokens, 2),
                        "queue_size": bucket.queue_size,
                        "active_requests": bucket.active_count,
                        "max_queue_size": bucket.max_queue_size,
                        "queue_timeout": bucket.queue_timeout,
                    }
        return {"keys": keys}

    def _get_or_create_bucket(self, vk: VirtualKey) -> _TokenBucket:
        """Get or create a token bucket for a virtual key."""
        with self._buckets_lock:
            bucket = self._buckets.get(vk.id)
            if bucket is None:
                bucket = _TokenBucket(
                    tps=vk.rate_limit_tps or 1.0,
                    priority=0,
                    max_queue_size=vk.rate_limit_max_queue,
                    queue_timeout=vk.rate_limit_queue_timeout,
                )
                self._buckets[vk.id] = bucket
            return bucket

    @staticmethod
    def _wait_for_slot(req: _QueuedRequest, bucket: _TokenBucket) -> bool:
        """Block the current thread until a slot opens or timeout.

        Returns True if slot acquired, False on timeout.
        """
        deadline = req.enqueued_at + bucket.queue_timeout

        while not req.released:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                req.timed_out = True
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
