"""Core types, models, and errors for StateLoom."""

from stateloom.core.config import PIIRule, StateLoomConfig
from stateloom.core.context import (
    get_current_session,
    get_current_session_id,
    set_current_session,
    set_current_session_id,
)
from stateloom.core.errors import (
    StateLoomBudgetError,
    StateLoomError,
    StateLoomLoopError,
    StateLoomPIIBlockedError,
    StateLoomReplayError,
    StateLoomSideEffectError,
)
from stateloom.core.event import (
    BudgetEnforcementEvent,
    CacheHitEvent,
    Event,
    LLMCallEvent,
    LoopDetectionEvent,
    PIIDetectionEvent,
    ToolCallEvent,
)
from stateloom.core.session import Session, SessionManager
from stateloom.core.types import (
    BudgetAction,
    EventType,
    FailureAction,
    LoopAction,
    PIIMode,
    Provider,
    SessionStatus,
)

__all__ = [
    "StateLoomBudgetError",
    "StateLoomConfig",
    "StateLoomError",
    "StateLoomLoopError",
    "StateLoomPIIBlockedError",
    "StateLoomReplayError",
    "StateLoomSideEffectError",
    "BudgetAction",
    "BudgetEnforcementEvent",
    "CacheHitEvent",
    "Event",
    "EventType",
    "FailureAction",
    "LLMCallEvent",
    "LoopAction",
    "LoopDetectionEvent",
    "PIIDetectionEvent",
    "PIIMode",
    "PIIRule",
    "Provider",
    "Session",
    "SessionManager",
    "SessionStatus",
    "ToolCallEvent",
    "get_current_session",
    "get_current_session_id",
    "set_current_session",
    "set_current_session_id",
]
