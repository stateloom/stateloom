"""Step tracking for replay — step numbering and StepRecord model."""

from __future__ import annotations

from dataclasses import dataclass

from stateloom.core.types import EventType


@dataclass
class StepRecord:
    """A recorded step in a session, used for replay."""

    step: int
    event_type: EventType
    request_hash: str = ""
    cached_response: object | None = None
    cached_response_json: str | None = None
    tool_name: str | None = None
    mutates_state: bool = False
    provider: str | None = None
    model: str | None = None
