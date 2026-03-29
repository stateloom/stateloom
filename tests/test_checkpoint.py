"""Tests for checkpoint events."""

from __future__ import annotations

import pytest

from stateloom.core.event import CheckpointEvent
from stateloom.core.session import Session
from stateloom.core.types import EventType
from stateloom.store.memory_store import MemoryStore


class TestCheckpointEvent:
    """CheckpointEvent dataclass tests."""

    def test_default_fields(self):
        event = CheckpointEvent(session_id="s1")
        assert event.event_type == EventType.CHECKPOINT
        assert event.label == ""
        assert event.description == ""
        assert event.session_id == "s1"

    def test_custom_fields(self):
        event = CheckpointEvent(
            session_id="s1",
            label="step-3",
            description="After user validation",
        )
        assert event.label == "step-3"
        assert event.description == "After user validation"
        assert event.event_type == EventType.CHECKPOINT

    def test_event_type_is_checkpoint(self):
        event = CheckpointEvent(session_id="s1", label="x")
        assert event.event_type == EventType.CHECKPOINT
        assert event.event_type.value == "checkpoint"


class TestCheckpointEventType:
    """EventType.CHECKPOINT enum value."""

    def test_checkpoint_enum_exists(self):
        assert hasattr(EventType, "CHECKPOINT")
        assert EventType.CHECKPOINT == "checkpoint"


class TestCheckpointStore:
    """Checkpoint events are persisted and retrieved via MemoryStore."""

    def test_save_and_retrieve_checkpoint_event(self):
        store = MemoryStore()
        session = Session(id="s1")
        store.save_session(session)

        event = CheckpointEvent(
            session_id="s1",
            step=3,
            label="mid-flow",
            description="After validation",
        )
        store.save_event(event)

        events = store.get_session_events("s1")
        assert len(events) == 1
        assert events[0].event_type == EventType.CHECKPOINT
        assert events[0].label == "mid-flow"
        assert events[0].description == "After validation"

    def test_multiple_checkpoints(self):
        store = MemoryStore()
        session = Session(id="s1")
        store.save_session(session)

        for i in range(3):
            store.save_event(CheckpointEvent(session_id="s1", step=i + 1, label=f"cp-{i + 1}"))

        events = store.get_session_events("s1")
        assert len(events) == 3
        labels = [e.label for e in events]
        assert labels == ["cp-1", "cp-2", "cp-3"]
