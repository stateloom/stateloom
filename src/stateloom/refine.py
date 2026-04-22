"""Evaluator-aware refinement loops — scored retry with feedback history.

Generalises "retry on failure" to an evaluator-driven loop: the user supplies
a scorer ``V(y) -> (score, feedback)``.  The loop keeps a history of
``(candidate, score, feedback)`` nodes and lets the generator condition on
that history to refine its next attempt.

Provides two interfaces:

- :func:`refine_loop` — one-shot function.  Usable inside or outside a
  :func:`stateloom.session`.
- :func:`durable_refine` — decorator combining a fresh durable session with
  the refinement loop.

This module is intentionally **isolated** from :mod:`stateloom.retry` — the
two share only the :data:`_NON_RETRYABLE` error tuple (re-imported, not
re-exported).  The "single durable session spans all attempts, step counter
advances across attempts" pattern is re-implemented here by reading
``retry.py`` as a reference; no code is shared.

Caveat: scorer non-determinism (e.g. scoring via a random LLM judge with
``temperature > 0``) will cause divergence on durable resume.  Recommended
mitigation: route scorer LLM calls through :mod:`stateloom.client` so those
responses are also cached for replay.
"""

from __future__ import annotations

import functools
import inspect
import logging
import math
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal

from stateloom.retry import _NON_RETRYABLE  # shared error tuple only

logger = logging.getLogger("stateloom")


# ---------------------------------------------------------------------------
# Public return / input types
# ---------------------------------------------------------------------------


@dataclass
class ScoreResult:
    """Return value for a scorer function.

    Scorers may return either a plain ``float`` (score only) or a
    ``ScoreResult`` with free-form feedback surfaced in the next
    generator's history plus arbitrary metadata.
    """

    score: float
    feedback: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class RefineNode:
    """One attempt in a refinement trajectory."""

    attempt: int  # 1-based
    result: Any
    score: float
    feedback: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class RefineResult:
    """Final outcome of a :func:`refine_loop` run."""

    best: RefineNode
    history: list[RefineNode]
    threshold_met: bool
    attempts_used: int


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _normalize_score(value: float | ScoreResult) -> ScoreResult:
    """Coerce a scorer return value into a :class:`ScoreResult`."""
    if isinstance(value, ScoreResult):
        return value
    return ScoreResult(score=float(value))


def _sentinel_score(direction: Literal["max", "min"]) -> float:
    """The worst-possible score in the requested direction.

    Used when the scorer raises a generic exception — the attempt is recorded
    but will never be selected as "best".
    """
    return -math.inf if direction == "max" else math.inf


def _is_better(new: float, old: float, direction: Literal["max", "min"]) -> bool:
    """Return True if ``new`` is strictly better than ``old``."""
    return new > old if direction == "max" else new < old


def _threshold_met(score: float, threshold: float, direction: Literal["max", "min"]) -> bool:
    """Return True if ``score`` crosses the threshold in the configured direction."""
    return score >= threshold if direction == "max" else score <= threshold


def _record_refine_event(
    session_id: str,
    *,
    step: int,
    attempt: int,
    max_attempts: int,
    score: float,
    best_score_so_far: float,
    feedback: str,
    threshold: float | None,
    direction: str,
    threshold_met: bool,
) -> None:
    """Persist a :class:`RefineAttemptEvent` to the store (fail-open)."""
    try:
        import stateloom
        from stateloom.core.event import RefineAttemptEvent

        event = RefineAttemptEvent(
            session_id=session_id,
            step=step,
            attempt=attempt,
            max_attempts=max_attempts,
            score=score,
            best_score_so_far=best_score_so_far,
            feedback_summary=feedback[:500],
            threshold=threshold,
            direction=direction,
            threshold_met=threshold_met,
        )
        stateloom.get_gate().store.save_event(event)
    except Exception:
        logger.warning(
            "Failed to record refine event for attempt %d (fail-open)",
            attempt,
            exc_info=True,
        )


def format_history_as_messages(
    history: list[RefineNode],
    *,
    style: Literal["openai", "anthropic"] = "openai",
) -> list[dict[str, Any]]:
    """Format refinement history as a chat transcript.

    Emits a minimal ``(assistant, user-feedback)`` exchange per attempt, so
    generators that want the paper's "four-part prompt" shape don't have to
    build it themselves.  Optional — generators are free to format history
    in any way they want.

    Args:
        history: The running list of :class:`RefineNode` objects.
        style: Either ``"openai"`` (role+content dicts) or ``"anthropic"``
            (role+content-list dicts).

    Returns:
        A list of chat message dicts in the requested style.
    """
    messages: list[dict[str, Any]] = []
    for node in history:
        result_text = node.result if isinstance(node.result, str) else repr(node.result)
        feedback_text = (
            f"Score: {node.score:.3f}. Feedback: {node.feedback}"
            if node.feedback
            else f"Score: {node.score:.3f}."
        )
        if style == "anthropic":
            messages.append(
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": result_text}],
                }
            )
            messages.append(
                {
                    "role": "user",
                    "content": [{"type": "text", "text": feedback_text}],
                }
            )
        else:
            messages.append({"role": "assistant", "content": result_text})
            messages.append({"role": "user", "content": feedback_text})
    return messages


# ---------------------------------------------------------------------------
# Core loop
# ---------------------------------------------------------------------------


def refine_loop(
    generator: Callable[[list[RefineNode]], Any],
    scorer: Callable[[Any], float | ScoreResult],
    *,
    max_attempts: int = 3,
    threshold: float | None = None,
    direction: Literal["max", "min"] = "max",
    on_attempt: Callable[[RefineNode], None] | None = None,
) -> RefineResult:
    """Run an evaluator-driven refinement loop.

    Args:
        generator: ``generator(history) -> candidate``.  Receives the running
            list of :class:`RefineNode`\\ s and returns the next candidate.
        scorer: ``scorer(candidate) -> float | ScoreResult``.  Plain ``float``
            returns are treated as score-only; :class:`ScoreResult` additionally
            carries feedback (appended to history) and metadata.
        max_attempts: Maximum number of generator/scorer cycles (default 3).
        threshold: Optional early-stop threshold.  When met (``>=`` for
            ``direction="max"``, ``<=`` for ``"min"``) the loop returns.
        direction: ``"max"`` (default) or ``"min"`` — whether higher or lower
            scores are better.
        on_attempt: Optional callback invoked with each completed
            :class:`RefineNode` (useful for progress reporting).

    Returns:
        A :class:`RefineResult` with the best node seen, full history,
        whether the threshold was met, and the number of attempts used.

    Raises:
        Any of the StateLoom non-retryable errors (budget, PII, kill switch,
        guardrail, etc.) propagate immediately without being counted as
        attempts.  Other scorer exceptions are logged and the attempt is
        recorded with a sentinel worst-case score; the loop continues.
    """
    if direction not in ("max", "min"):
        raise ValueError(f"direction must be 'max' or 'min', got {direction!r}")
    if max_attempts < 1:
        raise ValueError(f"max_attempts must be >= 1, got {max_attempts}")

    from stateloom.core.context import get_current_session

    history: list[RefineNode] = []
    best: RefineNode | None = None
    threshold_crossed = False
    attempts_used = 0

    for attempt in range(1, max_attempts + 1):
        attempts_used = attempt

        # Generator invocation — non-retryable errors propagate, other errors
        # also propagate (the refinement loop can only refine a scored
        # candidate; if we never got one, there's nothing to salvage).
        try:
            candidate = generator(list(history))
        except _NON_RETRYABLE:
            raise

        # Scorer invocation — non-retryable errors propagate, other errors
        # record a sentinel-score attempt and continue.
        try:
            raw_score = scorer(candidate)
            score_obj = _normalize_score(raw_score)
            score = score_obj.score
            feedback = score_obj.feedback
            metadata = dict(score_obj.metadata)
        except _NON_RETRYABLE:
            raise
        except Exception as e:
            logger.warning(
                "refine_loop scorer raised on attempt %d: %s (recording sentinel score)",
                attempt,
                e,
                exc_info=True,
            )
            score = _sentinel_score(direction)
            feedback = f"scorer raised: {type(e).__name__}: {e}"
            metadata = {"scorer_error": type(e).__name__}

        node = RefineNode(
            attempt=attempt,
            result=candidate,
            score=score,
            feedback=feedback,
            metadata=metadata,
        )
        history.append(node)

        if best is None or _is_better(score, best.score, direction):
            best = node

        if threshold is not None and _threshold_met(score, threshold, direction):
            threshold_crossed = True

        if on_attempt is not None:
            try:
                on_attempt(node)
            except Exception:
                logger.warning(
                    "refine_loop on_attempt callback raised on attempt %d (fail-open)",
                    attempt,
                    exc_info=True,
                )

        # Record an event if we're inside a session (fail-open).
        sess = get_current_session()
        if sess is not None:
            _record_refine_event(
                sess.id,
                step=sess.step_counter,
                attempt=attempt,
                max_attempts=max_attempts,
                score=score,
                best_score_so_far=best.score,
                feedback=feedback,
                threshold=threshold,
                direction=direction,
                threshold_met=threshold_crossed,
            )

        if threshold_crossed:
            break

    # history always has at least one node (max_attempts >= 1 enforced above
    # and the generator either succeeded or raised before reaching here).
    assert best is not None
    return RefineResult(
        best=best,
        history=history,
        threshold_met=threshold_crossed,
        attempts_used=attempts_used,
    )


# ---------------------------------------------------------------------------
# Decorator
# ---------------------------------------------------------------------------


def durable_refine(
    *,
    scorer: Callable[[Any], float | ScoreResult],
    max_attempts: int = 3,
    threshold: float | None = None,
    direction: Literal["max", "min"] = "max",
    session_id: str | None = None,
    name: str | None = None,
    budget: float | None = None,
    on_attempt: Callable[[RefineNode], None] | None = None,
) -> Callable[..., Any]:
    """Decorator: fresh durable session + refinement loop.

    The decorated function must accept a ``history: list[RefineNode]`` keyword
    argument — it is the generator half of :func:`refine_loop`.  A single
    durable session spans all attempts; on resume, cached LLM responses from
    prior attempts replay for free and only the live/new attempts hit the
    network.

    Args:
        scorer: Same as :func:`refine_loop`.
        max_attempts: Max refinement iterations (default 3).
        threshold: Optional early-stop threshold.
        direction: ``"max"`` or ``"min"`` (default ``"max"``).
        session_id: Fixed session ID — enables durable resume across process
            restarts.  If ``None``, one is auto-generated per call.
        name: Session name (defaults to the function name).
        budget: Per-session budget in USD.
        on_attempt: Optional per-attempt progress callback.

    Returns:
        A decorated function that returns a :class:`RefineResult`.  Supports
        both sync and async wrapped functions (dispatch via
        :func:`inspect.iscoroutinefunction`).
    """

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        task_name = name or func.__name__

        if inspect.iscoroutinefunction(func):

            @functools.wraps(func)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                return await _run_refine_async(
                    func,
                    args,
                    kwargs,
                    scorer=scorer,
                    max_attempts=max_attempts,
                    threshold=threshold,
                    direction=direction,
                    session_id=session_id,
                    task_name=task_name,
                    budget=budget,
                    on_attempt=on_attempt,
                )

            return async_wrapper

        @functools.wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            return _run_refine_sync(
                func,
                args,
                kwargs,
                scorer=scorer,
                max_attempts=max_attempts,
                threshold=threshold,
                direction=direction,
                session_id=session_id,
                task_name=task_name,
                budget=budget,
                on_attempt=on_attempt,
            )

        return sync_wrapper

    return decorator


def _run_refine_sync(
    func: Callable[..., Any],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    *,
    scorer: Callable[[Any], float | ScoreResult],
    max_attempts: int,
    threshold: float | None,
    direction: Literal["max", "min"],
    session_id: str | None,
    task_name: str,
    budget: float | None,
    on_attempt: Callable[[RefineNode], None] | None,
) -> RefineResult:
    """Execute a sync refinement loop inside a fresh durable session.

    The single session spans all attempts — the step counter advances
    monotonically across attempts, so on durable resume each attempt's
    cached LLM responses replay in order and no live calls are made until
    the session catches up with the resumed attempt boundary.
    """
    import stateloom

    sid = session_id or f"{task_name}-{uuid.uuid4().hex[:8]}"

    with stateloom.session(
        session_id=sid,
        name=task_name,
        budget=budget,
        durable=True,
    ):

        def generator(history: list[RefineNode]) -> Any:
            return func(*args, history=history, **kwargs)

        return refine_loop(
            generator=generator,
            scorer=scorer,
            max_attempts=max_attempts,
            threshold=threshold,
            direction=direction,
            on_attempt=on_attempt,
        )


async def _run_refine_async(
    func: Callable[..., Any],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    *,
    scorer: Callable[[Any], float | ScoreResult],
    max_attempts: int,
    threshold: float | None,
    direction: Literal["max", "min"],
    session_id: str | None,
    task_name: str,
    budget: float | None,
    on_attempt: Callable[[RefineNode], None] | None,
) -> RefineResult:
    """Execute an async refinement loop inside a fresh durable session.

    Async mirror of :func:`_run_refine_sync`.  Awaits the user's async
    generator for each attempt while scoring and event recording run
    synchronously inline.
    """
    import stateloom

    sid = session_id or f"{task_name}-{uuid.uuid4().hex[:8]}"

    async with stateloom.async_session(
        session_id=sid,
        name=task_name,
        budget=budget,
        durable=True,
    ):
        # We can't pass an async generator directly into refine_loop (which is
        # sync), so inline the loop here.  Logic mirrors refine_loop exactly.
        from stateloom.core.context import get_current_session

        if direction not in ("max", "min"):
            raise ValueError(f"direction must be 'max' or 'min', got {direction!r}")
        if max_attempts < 1:
            raise ValueError(f"max_attempts must be >= 1, got {max_attempts}")

        history: list[RefineNode] = []
        best: RefineNode | None = None
        threshold_crossed = False
        attempts_used = 0

        for attempt in range(1, max_attempts + 1):
            attempts_used = attempt

            try:
                candidate = await func(*args, history=list(history), **kwargs)
            except _NON_RETRYABLE:
                raise

            try:
                raw_score = scorer(candidate)
                score_obj = _normalize_score(raw_score)
                score = score_obj.score
                feedback = score_obj.feedback
                metadata = dict(score_obj.metadata)
            except _NON_RETRYABLE:
                raise
            except Exception as e:
                logger.warning(
                    "durable_refine scorer raised on attempt %d: %s (recording sentinel)",
                    attempt,
                    e,
                    exc_info=True,
                )
                score = _sentinel_score(direction)
                feedback = f"scorer raised: {type(e).__name__}: {e}"
                metadata = {"scorer_error": type(e).__name__}

            node = RefineNode(
                attempt=attempt,
                result=candidate,
                score=score,
                feedback=feedback,
                metadata=metadata,
            )
            history.append(node)

            if best is None or _is_better(score, best.score, direction):
                best = node

            if threshold is not None and _threshold_met(score, threshold, direction):
                threshold_crossed = True

            if on_attempt is not None:
                try:
                    on_attempt(node)
                except Exception:
                    logger.warning(
                        "durable_refine on_attempt callback raised (fail-open)",
                        exc_info=True,
                    )

            sess = get_current_session()
            if sess is not None:
                _record_refine_event(
                    sess.id,
                    step=sess.step_counter,
                    attempt=attempt,
                    max_attempts=max_attempts,
                    score=score,
                    best_score_so_far=best.score,
                    feedback=feedback,
                    threshold=threshold,
                    direction=direction,
                    threshold_met=threshold_crossed,
                )

            if threshold_crossed:
                break

        assert best is not None
        return RefineResult(
            best=best,
            history=history,
            threshold_met=threshold_crossed,
            attempts_used=attempts_used,
        )
