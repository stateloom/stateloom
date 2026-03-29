"""Semantic retry logic — automatic retries for LLM output failures.

Provides two interfaces:
- ``RetryLoop``: iterable retry helper for use inside existing sessions
- ``durable_task``: decorator combining durable session + automatic retry
"""

from __future__ import annotations

import functools
import inspect
import logging
import uuid
from collections.abc import Callable, Iterator
from typing import Any

from stateloom.core.errors import (
    StateLoomBlastRadiusError,
    StateLoomBudgetError,
    StateLoomCancellationError,
    StateLoomComplianceError,
    StateLoomError,
    StateLoomGuardrailError,
    StateLoomKillSwitchError,
    StateLoomPIIBlockedError,
    StateLoomRateLimitError,
    StateLoomRetryError,
    StateLoomSuspendedError,
    StateLoomTimeoutError,
)

logger = logging.getLogger("stateloom")

# Errors that should never be retried.  These represent infrastructure or
# security boundaries (budget exhaustion, PII policy, kill switch, etc.)
# where retrying would either violate policy or waste resources.
_NON_RETRYABLE = (
    StateLoomBudgetError,
    StateLoomPIIBlockedError,
    StateLoomKillSwitchError,
    StateLoomBlastRadiusError,
    StateLoomRateLimitError,
    StateLoomRetryError,
    StateLoomTimeoutError,
    StateLoomCancellationError,
    StateLoomComplianceError,
    StateLoomSuspendedError,
    StateLoomGuardrailError,
)


class RetryAttempt:
    """Context manager for a single retry attempt.

    Used as ``with attempt:`` inside a ``RetryLoop`` iteration.  Exception
    handling follows four branches:

    1. **No exception** — mark success, propagate normally (return False).
    2. **Non-retryable error** — always propagate immediately.
    3. **Last attempt** — propagate (caller sees the exception).
    4. **Retryable + attempts remaining** — suppress (return True), allowing
       the ``RetryLoop`` iterator to yield the next attempt.
    """

    def __init__(self, loop: RetryLoop, number: int) -> None:
        """Initialize a retry attempt.

        Args:
            loop: The owning ``RetryLoop``.
            number: 1-based attempt number.
        """
        self.number = number
        self._loop = loop
        self._success = False

    def __enter__(self) -> RetryAttempt:
        """Enter the attempt context."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> bool:
        """Handle attempt completion or failure.

        Returns:
            True to suppress the exception (retry), False to propagate.
        """
        # Branch 1: no exception — success
        if exc_type is None:
            self._success = True
            self._loop._succeeded = True
            return False

        # Branch 2: non-retryable infrastructure/security errors — always propagate
        if isinstance(exc_val, _NON_RETRYABLE):
            return False

        # Record retry event
        assert isinstance(exc_val, Exception)
        self._loop._record_retry(self.number, exc_val)

        # Branch 3: last attempt exhausted — propagate
        if self.number >= self._loop._max_attempts:
            return False

        # Branch 4: suppress exception, allow next iteration
        return True


class RetryLoop:
    """Iterable retry loop for LLM call blocks.

    Usage::

        for attempt in RetryLoop(retries=3):
            with attempt:
                response = openai.chat.completions.create(...)
                data = json.loads(response.choices[0].message.content)
    """

    def __init__(
        self,
        retries: int = 3,
        validate: Callable[[Any], bool] | None = None,
        on_retry: Callable[[int, Exception], None] | None = None,
    ) -> None:
        """Initialize the retry loop.

        Args:
            retries: Maximum number of attempts.
            validate: Optional validator called with the return value.
                If it returns False, triggers a retry.
            on_retry: Optional callback ``(attempt_number, error)`` invoked
                after each failed attempt.
        """
        self._max_attempts = retries
        self._validate = validate
        self._on_retry = on_retry
        self._succeeded = False
        self._attempts = 0

    def __iter__(self) -> Iterator[RetryAttempt]:
        """Yield ``RetryAttempt`` context managers up to ``retries`` times.

        Iteration stops early if an attempt succeeds (``_succeeded`` flag).
        """
        for i in range(1, self._max_attempts + 1):
            if self._succeeded:
                return
            self._attempts = i
            yield RetryAttempt(self, i)

    def _record_retry(self, attempt: int, error: Exception) -> None:
        """Record a retry event to the current session.

        Args:
            attempt: The 1-based attempt number that just failed.
            error: The exception that triggered the retry.
        """
        if self._on_retry:
            self._on_retry(attempt, error)

        # Persist SemanticRetryEvent to session store (fail-open)
        try:
            from stateloom.core.context import get_current_session
            from stateloom.core.event import SemanticRetryEvent

            session = get_current_session()
            if session is None:
                return

            event = SemanticRetryEvent(
                session_id=session.id,
                step=session.step_counter,
                attempt=attempt,
                max_attempts=self._max_attempts,
                error_type=type(error).__name__,
                error_message=str(error)[:500],
                resolved=False,
            )

            import stateloom

            stateloom.get_gate().store.save_event(event)
        except Exception:
            logger.warning(
                "Failed to record retry event for attempt %d (fail-open)",
                attempt,
                exc_info=True,
            )


def retry_loop(
    retries: int = 3,
    *,
    validate: Callable[[Any], bool] | None = None,
    on_retry: Callable[[int, Exception], None] | None = None,
) -> RetryLoop:
    """Create an iterable retry loop for LLM call blocks.

    Args:
        retries: Maximum number of attempts.
        validate: Optional validator called with the return value.
        on_retry: Optional callback ``(attempt_number, error)`` on failure.

    Returns:
        A ``RetryLoop`` iterable that yields ``RetryAttempt`` context managers.

    Usage::

        for attempt in stateloom.retry_loop(retries=3):
            with attempt:
                response = openai.chat.completions.create(...)
                result = json.loads(response.choices[0].message.content)
    """
    return RetryLoop(retries=retries, validate=validate, on_retry=on_retry)


def _record_retry_event(
    session_id: str,
    attempt: int,
    max_attempts: int,
    error: Exception,
) -> None:
    """Record a SemanticRetryEvent to the store (fail-open).

    Args:
        session_id: The session this retry belongs to.
        attempt: 1-based attempt number that just failed.
        max_attempts: Total allowed attempts.
        error: The exception that triggered the retry.
    """
    try:
        import stateloom
        from stateloom.core.event import SemanticRetryEvent

        event = SemanticRetryEvent(
            session_id=session_id,
            attempt=attempt,
            max_attempts=max_attempts,
            error_type=type(error).__name__,
            error_message=str(error)[:500],
            resolved=False,
        )
        stateloom.get_gate().store.save_event(event)
    except Exception:
        logger.warning(
            "Failed to record retry event for attempt %d (fail-open)",
            attempt,
            exc_info=True,
        )


def _record_success_event(
    session: Any,
    attempt: int,
    max_attempts: int,
) -> None:
    """Record a resolved SemanticRetryEvent on success (fail-open).

    Args:
        session: The active session.
        attempt: The attempt number that succeeded.
        max_attempts: Total allowed attempts.

    Only records an event when ``attempt > 1`` — first-try successes
    don't need a "resolved" marker since no retries occurred.
    """
    if attempt <= 1:
        return
    try:
        import stateloom
        from stateloom.core.event import SemanticRetryEvent

        event = SemanticRetryEvent(
            session_id=session.id,
            step=session.step_counter,
            attempt=attempt,
            max_attempts=max_attempts,
            resolved=True,
        )
        stateloom.get_gate().store.save_event(event)
    except Exception:
        logger.warning(
            "Failed to record retry success event for attempt %d (fail-open)",
            attempt,
            exc_info=True,
        )


def durable_task(
    retries: int = 3,
    *,
    validate: Callable[[Any], bool] | None = None,
    session_id: str | None = None,
    name: str | None = None,
    budget: float | None = None,
    on_retry: Callable[[int, Exception], None] | None = None,
) -> Callable:
    """Decorator: durable session + automatic retry on exception.

    Creates a single durable session that spans all retry attempts.
    If the function raises, retries up to N times. On crash recovery,
    cached LLM responses are replayed automatically.

    Args:
        retries: Max retry attempts (default 3).
        validate: Optional validator — called with the return value.
            If it returns False, triggers a retry.
        session_id: Fixed session ID (enables durable resume across restarts).
            If None, auto-generated per call.
        name: Session name (defaults to function name).
        budget: Per-session budget in USD.
        on_retry: Optional callback(attempt_number, error) on each retry.
    """

    def decorator(func: Callable) -> Callable:
        task_name = name or func.__name__

        if inspect.iscoroutinefunction(func):

            @functools.wraps(func)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                return await _run_durable_task_async(
                    func,
                    args,
                    kwargs,
                    retries=retries,
                    validate=validate,
                    session_id=session_id,
                    task_name=task_name,
                    budget=budget,
                    on_retry=on_retry,
                )

            return async_wrapper

        @functools.wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            return _run_durable_task_sync(
                func,
                args,
                kwargs,
                retries=retries,
                validate=validate,
                session_id=session_id,
                task_name=task_name,
                budget=budget,
                on_retry=on_retry,
            )

        return sync_wrapper

    return decorator


def _run_durable_task_sync(
    func: Callable,
    args: tuple,
    kwargs: dict,
    *,
    retries: int,
    validate: Callable[[Any], bool] | None,
    session_id: str | None,
    task_name: str,
    budget: float | None,
    on_retry: Callable[[int, Exception], None] | None,
) -> Any:
    """Execute a sync durable task with retry logic.

    Args:
        func: The user's sync function to execute.
        args: Positional arguments for ``func``.
        kwargs: Keyword arguments for ``func``.
        retries: Max retry attempts.
        validate: Optional validator; False return triggers retry.
        session_id: Fixed session ID for durable resume, or None to
            auto-generate.
        task_name: Session name (defaults to function name).
        budget: Per-session budget in USD.
        on_retry: Optional callback on each failed attempt.

    Returns:
        The return value of ``func`` on success.

    Raises:
        StateLoomRetryError: When all attempts are exhausted.
    """
    import stateloom

    sid = session_id or f"{task_name}-{uuid.uuid4().hex[:8]}"
    last_error: Exception | None = None

    # A single durable session spans all attempts.  The step counter
    # advances across retries (attempt 1 uses steps 1..N, attempt 2 uses
    # N+1..M), so on durable resume the cached bad responses replay,
    # the same validation failure fires, retry kicks in, and the cached
    # good response from the successful attempt is returned — zero
    # wasted API calls.
    with stateloom.session(
        session_id=sid,
        name=task_name,
        budget=budget,
        durable=True,
    ) as sess:
        for attempt in range(1, retries + 1):
            try:
                result = func(*args, **kwargs)
                if validate and not validate(result):
                    raise ValueError("durable_task validation failed")
                _record_success_event(sess, attempt, retries)
                return result
            except _NON_RETRYABLE:
                raise
            except Exception as e:
                last_error = e
                _record_retry_event(sid, attempt, retries, e)
                if on_retry:
                    on_retry(attempt, e)
                if attempt == retries:
                    raise StateLoomRetryError(
                        retries,
                        str(e),
                        session_id=sid,
                    ) from e
                logger.info(
                    "durable_task '%s' attempt %d/%d failed: %s",
                    task_name,
                    attempt,
                    retries,
                    e,
                )

    # Should not reach here, but satisfy type checker
    raise StateLoomRetryError(retries, str(last_error), session_id=sid)


async def _run_durable_task_async(
    func: Callable,
    args: tuple,
    kwargs: dict,
    *,
    retries: int,
    validate: Callable[[Any], bool] | None,
    session_id: str | None,
    task_name: str,
    budget: float | None,
    on_retry: Callable[[int, Exception], None] | None,
) -> Any:
    """Execute an async durable task with retry logic.

    Async mirror of ``_run_durable_task_sync``.  See that function for
    full Args/Returns/Raises documentation.
    """
    import stateloom

    sid = session_id or f"{task_name}-{uuid.uuid4().hex[:8]}"
    last_error: Exception | None = None

    # Single session spans all attempts — step counter advances across retries
    async with stateloom.async_session(
        session_id=sid,
        name=task_name,
        budget=budget,
        durable=True,
    ) as sess:
        for attempt in range(1, retries + 1):
            try:
                result = await func(*args, **kwargs)
                if validate and not validate(result):
                    raise ValueError("durable_task validation failed")
                _record_success_event(sess, attempt, retries)
                return result
            except _NON_RETRYABLE:
                raise
            except Exception as e:
                last_error = e
                _record_retry_event(sid, attempt, retries, e)
                if on_retry:
                    on_retry(attempt, e)
                if attempt == retries:
                    raise StateLoomRetryError(
                        retries,
                        str(e),
                        session_id=sid,
                    ) from e
                logger.info(
                    "durable_task '%s' attempt %d/%d failed: %s",
                    task_name,
                    attempt,
                    retries,
                    e,
                )

    # Should not reach here, but satisfy type checker
    raise StateLoomRetryError(retries, str(last_error), session_id=sid)
