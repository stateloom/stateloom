"""Session model and SessionManager for StateLoom."""

from __future__ import annotations

import threading
import uuid
from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr

from stateloom.core.types import SessionStatus


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _new_session_id() -> str:
    return uuid.uuid4().hex[:12]


class Session(BaseModel):
    """A unit of agent work — tracks cost, tokens, events, and PII detections.

    All accumulator fields (``total_cost``, ``total_tokens``, etc.) are
    updated through thread-safe helper methods (``add_cost``,
    ``add_cache_hit``, ``add_pii_detection``) guarded by ``_lock``.
    Multiple middleware or concurrent requests can safely update the same
    session in parallel.

    Timeout and cancellation state is checked by
    ``TimeoutCheckerMiddleware`` before every LLM call.
    """

    model_config = ConfigDict(extra="ignore", arbitrary_types_allowed=True)

    id: str = Field(default_factory=_new_session_id)
    name: str | None = None
    org_id: str = ""
    team_id: str = ""
    started_at: datetime = Field(default_factory=_utcnow)
    ended_at: datetime | None = None
    status: SessionStatus = SessionStatus.ACTIVE

    # Accumulators (updated in real-time)
    total_cost: float = Field(default=0.0, ge=0)
    # Always per-token cost regardless of billing mode (for analytics).
    estimated_api_cost: float = Field(default=0.0, ge=0)
    # Per-model cost and token breakdowns within a session.
    cost_by_model: dict[str, float] = Field(default_factory=dict)
    tokens_by_model: dict[str, dict[str, int]] = Field(default_factory=dict)
    total_tokens: int = Field(default=0, ge=0)
    total_prompt_tokens: int = Field(default=0, ge=0)
    total_completion_tokens: int = Field(default=0, ge=0)
    call_count: int = Field(default=0, ge=0)
    cache_hits: int = Field(default=0, ge=0)
    cache_savings: float = Field(default=0.0, ge=0)
    pii_detections: int = Field(default=0, ge=0)
    guardrail_detections: int = Field(default=0, ge=0)

    # Budget
    budget: float | None = None

    # Monotonically increasing counter used by next_step() to assign a
    # unique step number to each LLM call and tool invocation.  Also used
    # by durable replay to match cached responses to their original steps.
    step_counter: int = Field(default=0, ge=0)

    # Arbitrary metadata (experiment assignment, etc.)
    metadata: dict[str, Any] = Field(default_factory=dict)

    # End-user attribution (optional, set by proxy X-StateLoom-End-User header)
    end_user: str = ""

    # Typed metadata fields — promoted from metadata dict for type safety,
    # IDE support, and direct-column persistence in SQLite.  These are also
    # dual-written to metadata["<key>"] for backward compatibility.
    billing_mode: str = ""  # "api" | "subscription" | ""
    durable: bool = False  # Durable replay enabled
    agent_id: str = ""  # agt-* UUID
    agent_slug: str = ""  # URL slug
    agent_version_id: str = ""  # agv-* UUID
    agent_version_number: int = 0  # Semantic version
    agent_name: str = ""  # Display name (used by blast_radius)
    transport: str = ""  # "websocket" for WS sessions

    # Parent-child hierarchy
    parent_session_id: str | None = None

    # Timeouts
    timeout: float | None = None
    idle_timeout: float | None = None
    last_heartbeat: datetime | None = None

    # Private: Cancellation
    _cancelled: bool = PrivateAttr(default=False)

    # Private: Suspension (human-in-the-loop).
    # _suspend_event is a threading.Event used by wait_for_signal() to
    # block the calling thread until signal() is invoked from another
    # thread (e.g. the dashboard API handler).
    _suspended: bool = PrivateAttr(default=False)
    _suspend_event: threading.Event = PrivateAttr(default_factory=threading.Event)
    _signal_payload: Any = PrivateAttr(default=None)

    # Private: Thread safety for accumulators
    _lock: threading.Lock = PrivateAttr(default_factory=threading.Lock)

    # Private: Durable concurrency guard — tracks in-flight LLM calls to
    # detect concurrent calls that would make step ordinals non-deterministic.
    _durable_in_flight: int = PrivateAttr(default=0)

    def next_step(self) -> int:
        """Increment and return the next step number (thread-safe)."""
        with self._lock:
            self.step_counter += 1
            return self.step_counter

    def acquire_durable_step(self) -> int:
        """Increment and return the next step number, with concurrency guard for durable sessions.

        For durable sessions, raises ``StateLoomError`` if another LLM call is
        already in-flight. For non-durable sessions, behaves identically to
        ``next_step()``.
        """
        with self._lock:
            if self.durable and self._durable_in_flight > 0:
                from stateloom.core.errors import StateLoomError

                raise StateLoomError(
                    "Concurrent LLM calls detected in durable session "
                    f"'{self.id}'. Durable sessions require sequential calls "
                    "to ensure deterministic step ordering. Use asyncio.gather() "
                    "only with non-durable sessions.",
                )
            self.step_counter += 1
            if self.durable:
                self._durable_in_flight += 1
            return self.step_counter

    def release_durable_step(self) -> None:
        """Release the durable in-flight guard after an LLM call completes."""
        with self._lock:
            if self._durable_in_flight > 0:
                self._durable_in_flight -= 1

    def add_cost(
        self,
        cost: float,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        *,
        estimated_api_cost: float = 0.0,
        model: str = "",
    ) -> None:
        """Add cost and token counts to session accumulators (thread-safe).

        Args:
            cost: Actual cost in USD.  For subscription billing this is 0.0;
                for API billing it equals ``estimated_api_cost``.
            prompt_tokens: Number of input tokens for this call.
            completion_tokens: Number of output tokens for this call.
            estimated_api_cost: Per-token cost regardless of billing mode,
                used for analytics comparisons.
            model: Model name for per-model breakdown tracking.

        Also increments ``call_count`` by 1.
        """
        with self._lock:
            self.total_cost += cost
            self.estimated_api_cost += estimated_api_cost
            self.total_tokens += prompt_tokens + completion_tokens
            self.total_prompt_tokens += prompt_tokens
            self.total_completion_tokens += completion_tokens
            self.call_count += 1
            if model:
                self.cost_by_model[model] = self.cost_by_model.get(model, 0.0) + cost
                existing = self.tokens_by_model.get(
                    model, {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
                )
                self.tokens_by_model[model] = {
                    "prompt_tokens": existing["prompt_tokens"] + prompt_tokens,
                    "completion_tokens": existing["completion_tokens"] + completion_tokens,
                    "total_tokens": existing["total_tokens"] + prompt_tokens + completion_tokens,
                }

    def add_cache_hit(self, saved_cost: float) -> None:
        """Record a cache hit and its estimated cost savings (thread-safe).

        Args:
            saved_cost: Estimated USD that would have been spent without cache.
        """
        with self._lock:
            self.cache_hits += 1
            self.cache_savings += saved_cost

    def add_pii_detection(self) -> None:
        """Increment the PII detection counter by 1 (thread-safe)."""
        with self._lock:
            self.pii_detections += 1

    def add_guardrail_detection(self) -> None:
        """Increment the guardrail detection counter by 1 (thread-safe)."""
        with self._lock:
            self.guardrail_detections += 1

    def heartbeat(self) -> None:
        """Update the last heartbeat timestamp (thread-safe)."""
        with self._lock:
            self.last_heartbeat = _utcnow()

    def is_timed_out(self) -> tuple[bool, str, float, float]:
        """Check if the session has exceeded its timeout or idle timeout.

        Returns (timed_out, timeout_type, elapsed, limit).
        """
        now = _utcnow()
        if self.timeout is not None:
            elapsed = (now - self.started_at).total_seconds()
            if elapsed > self.timeout:
                return True, "session_timeout", elapsed, self.timeout
        if self.idle_timeout is not None and self.last_heartbeat is not None:
            idle = (now - self.last_heartbeat).total_seconds()
            if idle > self.idle_timeout:
                return True, "idle_timeout", idle, self.idle_timeout
        return False, "", 0.0, 0.0

    def cancel(self) -> None:
        """Mark the session as cancelled.

        Sets both the private ``_cancelled`` flag (for fast in-memory checks)
        and ``metadata["_cancelled"]`` (persisted to the store for
        cross-process visibility).
        """
        with self._lock:
            self._cancelled = True
            self.metadata["_cancelled"] = True

    @property
    def is_cancelled(self) -> bool:
        """Check if the session has been cancelled."""
        return self._cancelled or self.metadata.get("_cancelled", False)

    def suspend(self, reason: str = "", data: dict[str, Any] | None = None) -> None:
        """Suspend the session, awaiting an external signal to resume.

        Sets the session status to SUSPENDED and blocks further LLM calls
        until ``signal()`` is called (human-in-the-loop approval pattern).

        Args:
            reason: Why the session is being suspended (shown in dashboard).
            data: Arbitrary context data for the human reviewer.
        """
        with self._lock:
            self._suspended = True
            self._signal_payload = None
            self._suspend_event.clear()
            self.status = SessionStatus.SUSPENDED
            if reason:
                self.metadata["_suspend_reason"] = reason
            if data:
                self.metadata["_suspend_data"] = data

    def signal(self, payload: Any = None) -> None:
        """Resume a suspended session with an optional payload.

        Args:
            payload: Arbitrary data (approval decision, human feedback, etc.)
                accessible via ``session.signal_payload`` after resumption.
        """
        with self._lock:
            self._signal_payload = payload
            self._suspended = False
            self.status = SessionStatus.ACTIVE
            self._suspend_event.set()

    def wait_for_signal(self, timeout: float | None = None) -> Any:
        """Block until ``signal()`` is called. Returns the signal payload.

        Args:
            timeout: Max seconds to wait. None = wait indefinitely.

        Returns:
            The payload passed to ``signal()``, or None on timeout.
        """
        signaled = self._suspend_event.wait(timeout=timeout)
        if not signaled:
            return None
        return self._signal_payload

    @property
    def is_suspended(self) -> bool:
        """Check if the session is suspended."""
        return self._suspended

    @property
    def signal_payload(self) -> Any:
        """Get the most recent signal payload."""
        return self._signal_payload

    def end(self, status: SessionStatus = SessionStatus.COMPLETED) -> None:
        """Finalize the session by recording ``ended_at`` and setting status.

        Args:
            status: Terminal status (default ``COMPLETED``).  May also be
                ``TIMED_OUT``, ``CANCELLED``, etc.
        """
        with self._lock:
            self.ended_at = _utcnow()
            self.status = status


class SessionManager:
    """Manages active sessions. Thread-safe."""

    def __init__(self) -> None:
        """Initialize the session manager with an empty registry."""
        # TODO(memory-leak): _sessions grows indefinitely — completed/errored
        # sessions are never evicted.  Each entry is small (~1-2KB) so this is
        # only a concern for long-running servers with high session throughput.
        # Fix: add TTL-based eviction (e.g. remove terminal sessions after 30
        # min).  Note: simple eviction-on-end would break cancel_session() /
        # suspend_session() / signal_session() which look up in-memory sessions
        # for _cancelled / _suspend_event state.
        self._sessions: dict[str, Session] = {}
        self._lock = threading.Lock()
        self._default_budget: float | None = None

    def set_default_budget(self, budget: float | None) -> None:
        """Set the default per-session budget applied when no explicit budget is given.

        Args:
            budget: Budget in USD, or None to disable default budgets.
        """
        self._default_budget = budget

    def create(
        self,
        session_id: str | None = None,
        name: str | None = None,
        budget: float | None = None,
        org_id: str = "",
        team_id: str = "",
        parent_session_id: str | None = None,
        timeout: float | None = None,
        idle_timeout: float | None = None,
    ) -> Session:
        """Create and register a new session.

        Args:
            session_id: Explicit ID, or None to auto-generate.
            name: Human-readable label (shown in dashboard).
            budget: Per-session budget in USD; falls back to default.
            org_id: Organization scope.
            team_id: Team scope.
            parent_session_id: Parent session for hierarchical grouping.
            timeout: Max session duration in seconds.
            idle_timeout: Max idle time between calls in seconds.

        Returns:
            The newly created ``Session``, already registered in the manager.
        """
        session = Session(
            id=session_id or _new_session_id(),
            name=name,
            org_id=org_id,
            team_id=team_id,
            budget=budget or self._default_budget,
            parent_session_id=parent_session_id,
            timeout=timeout,
            idle_timeout=idle_timeout,
            # Heartbeat is only initialized when a timeout is active;
            # without a timeout there is nothing to check idle against.
            last_heartbeat=_utcnow() if (timeout or idle_timeout) else None,
        )
        with self._lock:
            self._sessions[session.id] = session
        return session

    def get(self, session_id: str) -> Session | None:
        """Get a session by ID."""
        with self._lock:
            return self._sessions.get(session_id)

    def get_or_create(self, session_id: str | None = None) -> Session:
        """Get an existing session or create a new one.

        Args:
            session_id: If provided and a session with this ID exists, return
                it.  Otherwise create a new session with this ID.

        Returns:
            An existing or freshly created ``Session``.
        """
        if session_id:
            session = self.get(session_id)
            if session:
                return session
        return self.create(session_id=session_id)

    def end(self, session_id: str, status: SessionStatus = SessionStatus.COMPLETED) -> None:
        """End a session by ID.

        Args:
            session_id: The session to end.
            status: Terminal status (default ``COMPLETED``).
        """
        session = self.get(session_id)
        if session:
            session.end(status)

    def list_active(self) -> list[Session]:
        """Return all sessions with ``ACTIVE`` status.

        Returns:
            List of currently active sessions (snapshot; thread-safe).
        """
        with self._lock:
            return [s for s in self._sessions.values() if s.status == SessionStatus.ACTIVE]

    def list_all(self) -> list[Session]:
        """Return all tracked sessions regardless of status.

        Returns:
            List of all sessions (snapshot; thread-safe).
        """
        with self._lock:
            return list(self._sessions.values())
