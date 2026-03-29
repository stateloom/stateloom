"""Map StateLoom errors to OpenAI-format error responses."""

from __future__ import annotations

from typing import Any

from stateloom.core.errors import (
    StateLoomBlastRadiusError,
    StateLoomBudgetError,
    StateLoomCancellationError,
    StateLoomError,
    StateLoomKillSwitchError,
    StateLoomPIIBlockedError,
    StateLoomRateLimitError,
    StateLoomTimeoutError,
)


def to_openai_error_dict(exc: Exception) -> dict[str, Any]:
    """Convert an exception to an OpenAI-format error dict.

    Returns:
        {"error": {"message": ..., "type": ..., "code": ...}}
    """
    if isinstance(exc, StateLoomRateLimitError):
        return _make_error(str(exc), "rate_limit_error", "rate_limit_exceeded")
    if isinstance(exc, StateLoomBudgetError):
        return _make_error(str(exc), "insufficient_funds", "budget_exceeded")
    if isinstance(exc, StateLoomPIIBlockedError):
        return _make_error(str(exc), "invalid_request_error", "pii_blocked")
    if isinstance(exc, StateLoomKillSwitchError):
        return _make_error(str(exc), "server_error", "service_unavailable")
    if isinstance(exc, StateLoomBlastRadiusError):
        return _make_error(str(exc), "server_error", "blast_radius_paused")
    if isinstance(exc, StateLoomTimeoutError):
        return _make_error(str(exc), "server_error", "session_timed_out")
    if isinstance(exc, StateLoomCancellationError):
        return _make_error(str(exc), "server_error", "session_cancelled")
    if isinstance(exc, StateLoomError):
        return _make_error(str(exc), "server_error", getattr(exc, "error_code", "stateloom_error"))
    return _make_error(str(exc), "server_error", "internal_error")


def error_status_code(exc: Exception) -> int:
    """Map an exception to an HTTP status code."""
    if isinstance(exc, StateLoomRateLimitError):
        return 429
    if isinstance(exc, StateLoomBudgetError):
        return 402
    if isinstance(exc, StateLoomPIIBlockedError):
        return 400
    if isinstance(exc, StateLoomKillSwitchError):
        return 503
    if isinstance(exc, StateLoomBlastRadiusError):
        return 503
    if isinstance(exc, StateLoomTimeoutError):
        return 504
    if isinstance(exc, StateLoomCancellationError):
        return 499
    if isinstance(exc, StateLoomError):
        return 500
    return 500


def _make_error(message: str, error_type: str, code: str) -> dict[str, Any]:
    return {
        "error": {
            "message": message,
            "type": error_type,
            "code": code,
        }
    }
