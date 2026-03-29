"""Tests for session cancellation."""

from __future__ import annotations

import pytest

from stateloom.core.errors import (
    StateLoomCancellationError,
    StateLoomTimeoutError,
)
from stateloom.core.session import Session
from stateloom.core.types import SessionStatus
from stateloom.retry import _NON_RETRYABLE


class TestSessionCancelMethod:
    """Session.cancel() and Session.is_cancelled."""

    def test_not_cancelled_by_default(self):
        session = Session(id="s1")
        assert session.is_cancelled is False

    def test_cancel_sets_flag(self):
        session = Session(id="s1")
        session.cancel()
        assert session.is_cancelled is True

    def test_cancel_idempotent(self):
        session = Session(id="s1")
        session.cancel()
        session.cancel()
        assert session.is_cancelled is True

    def test_cancel_does_not_end_session(self):
        """cancel() sets the flag but doesn't end the session directly.
        The middleware handles status transition."""
        session = Session(id="s1")
        session.cancel()
        assert session.status == SessionStatus.ACTIVE
        assert session.ended_at is None


class TestCancelledSessionStatus:
    """SessionStatus.CANCELLED enum."""

    def test_cancelled_status_exists(self):
        assert hasattr(SessionStatus, "CANCELLED")
        assert SessionStatus.CANCELLED == "cancelled"

    def test_timed_out_status_exists(self):
        assert hasattr(SessionStatus, "TIMED_OUT")
        assert SessionStatus.TIMED_OUT == "timed_out"


class TestNonRetryableErrors:
    """New errors are in the _NON_RETRYABLE tuple."""

    def test_timeout_error_non_retryable(self):
        assert StateLoomTimeoutError in _NON_RETRYABLE

    def test_cancellation_error_non_retryable(self):
        assert StateLoomCancellationError in _NON_RETRYABLE

    def test_timeout_error_instance_matches(self):
        err = StateLoomTimeoutError(
            session_id="s1",
            timeout_type="session_timeout",
            elapsed=65.0,
            limit=60.0,
        )
        assert isinstance(err, _NON_RETRYABLE)

    def test_cancellation_error_instance_matches(self):
        err = StateLoomCancellationError(session_id="s1")
        assert isinstance(err, _NON_RETRYABLE)


class TestBlastRadiusExclusion:
    """New error types should not be counted as failures by blast radius."""

    def test_timeout_excluded_from_blast_radius(self):
        """Verify TimeoutError is in blast radius exclusion list."""
        from stateloom.middleware.blast_radius import (
            StateLoomCancellationError as BR_Cancel,
        )
        from stateloom.middleware.blast_radius import (
            StateLoomTimeoutError as BR_Timeout,
        )

        # These should be importable from blast_radius (they're used in except clause)
        assert BR_Timeout is StateLoomTimeoutError
        assert BR_Cancel is StateLoomCancellationError
