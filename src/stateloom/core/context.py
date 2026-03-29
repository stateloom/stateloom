"""ContextVar-based async-safe session and replay tracking."""

from __future__ import annotations

from contextvars import ContextVar
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from stateloom.core.session import Session

# The current session for this async context / thread
_current_session: ContextVar[Session | None] = ContextVar("stateloom_session", default=None)

# The current session ID (lighter weight, for header injection)
_current_session_id: ContextVar[str | None] = ContextVar("stateloom_session_id", default=None)

# The active replay engine for this context (avoids global state on Gate)
_current_replay_engine: ContextVar[Any] = ContextVar("stateloom_replay_engine", default=None)


def get_current_session() -> Session | None:
    """Get the current session for this context."""
    return _current_session.get()


def set_current_session(session: Session | None) -> None:
    """Set the current session for this context."""
    _current_session.set(session)
    if session:
        _current_session_id.set(session.id)
    else:
        _current_session_id.set(None)


def get_current_session_id() -> str | None:
    """Get the current session ID for this context."""
    return _current_session_id.get()


def set_current_session_id(session_id: str | None) -> None:
    """Set the current session ID (for distributed context propagation)."""
    _current_session_id.set(session_id)


def get_current_replay_engine() -> Any:
    """Get the active replay engine for this context."""
    return _current_replay_engine.get()


def set_current_replay_engine(engine: Any) -> None:
    """Set the active replay engine for this context."""
    _current_replay_engine.set(engine)


# Framework-specific metadata for the current context (e.g., LangChain run_id, tags).
# Keyed by framework name: {"langchain": {"run_id": "...", "tags": [...]}}
_framework_context: ContextVar[dict[str, Any]] = ContextVar(
    "stateloom_framework_context", default={}
)


def get_framework_context() -> dict[str, Any]:
    """Get framework metadata for the current context (e.g., LangChain run_id, tags)."""
    return _framework_context.get()


def set_framework_context(context: dict[str, Any] | None) -> None:
    """Set framework metadata. Keys are framework names, values are metadata dicts."""
    _framework_context.set(context or {})


def clear_framework_context() -> None:
    """Clear framework metadata after the SDK call completes."""
    _framework_context.set({})
