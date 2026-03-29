"""Replay engine — time-travel debugging for agent sessions."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from stateloom.core.context import set_current_replay_engine, set_current_session
from stateloom.core.errors import StateLoomReplayError
from stateloom.replay.safety import analyze_replay_safety, display_safety_warnings
from stateloom.replay.schema import deserialize_response, serialize_response
from stateloom.replay.step import StepRecord

if TYPE_CHECKING:
    from stateloom.gate import Gate

logger = logging.getLogger("stateloom.replay")


class ReplayEngine:
    """Orchestrates time-travel debugging for a recorded session.

    Usage:
        engine = ReplayEngine(gate, session_id="ticket-123", mock_until_step=13)
        engine.start()

    Steps 1-13 return cached responses instantly.
    Steps 14+ execute live.
    """

    def __init__(
        self,
        gate: Gate,
        session_id: str,
        mock_until_step: int,
        strict: bool = True,
        allow_hosts: list[str] | None = None,
    ) -> None:
        self.gate = gate
        self.session_id = session_id
        self.mock_until_step = mock_until_step
        self.strict = strict
        self.allow_hosts = set(allow_hosts or [])
        self._steps: list[StepRecord] = []
        self._step_index: dict[int, StepRecord] = {}
        self._active = False
        self._network_blocker: Any = None

    def start(self) -> None:
        """Start a replay session."""
        # Load recorded steps
        self._steps = self._load_steps()
        self._step_index = {s.step: s for s in self._steps}

        if not self._steps:
            raise StateLoomReplayError(
                f"No recorded steps found for session '{self.session_id}'",
                session_id=self.session_id,
            )

        max_step = max(s.step for s in self._steps)
        if self.mock_until_step > max_step:
            raise StateLoomReplayError(
                f"mock_until_step ({self.mock_until_step}) exceeds recorded steps ({max_step})",
                session_id=self.session_id,
            )

        # Safety check
        warnings = analyze_replay_safety(self._steps, self.mock_until_step)
        if not display_safety_warnings(warnings):
            return

        # Activate network blocker in strict mode
        if self.strict:
            try:
                from stateloom.replay.network_blocker import NetworkBlocker

                self._network_blocker = NetworkBlocker(
                    session_id=self.session_id,
                )
                self._network_blocker.activate(self.allow_hosts)
            except ImportError:
                logger.warning("Network blocker not available")

        # Store reference in ContextVar (async/thread-safe, no global mutation)
        set_current_replay_engine(self)

        # Set up the replay session
        session = self.gate.session_manager.create(
            session_id=f"replay-{self.session_id}",
            name=f"Replay of {self.session_id}",
        )
        set_current_session(session)
        self._active = True

        mode = ", strict mode" if self.strict else ""
        logger.info(
            "Replay started: session '%s', mocking steps 1-%d, live from step %d%s",
            self.session_id,
            self.mock_until_step,
            self.mock_until_step + 1,
            mode,
        )

    @property
    def is_active(self) -> bool:
        """Whether the replay engine is currently active."""
        return self._active

    def should_mock(self, current_step: int) -> bool:
        """Check if a step should return cached response."""
        return self._active and current_step <= self.mock_until_step

    def should_mock_tool(self, current_step: int) -> bool:
        """Check if a tool call at this step should return cached result.

        Used by the @gate.tool() wrapper to skip execution of mutates_state
        tools during replay (dry run mode).
        """
        return self.should_mock(current_step)

    def get_cached_response(self, step: int) -> object | None:
        """Get the cached response for a mocked step. O(1) lookup."""
        record = self._step_index.get(step)
        if record is None:
            return None
        if record.cached_response_json is not None:
            return deserialize_response(record.cached_response_json, record.provider)
        return record.cached_response

    def stop(self) -> None:
        """Stop the replay and deactivate network blocker."""
        self._active = False
        set_current_replay_engine(None)
        if self._network_blocker is not None:
            self._network_blocker.deactivate()
            self._network_blocker = None

    def __enter__(self) -> ReplayEngine:
        """Context manager entry — starts the replay engine."""
        self.start()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Context manager exit — always stops the engine, even on exception."""
        self.stop()

    def _load_steps(self) -> list[StepRecord]:
        """Load recorded steps from the store."""
        events = self.gate.store.get_session_events(self.session_id)

        steps: list[StepRecord] = []
        for event in events:
            cached_result = getattr(event, "cached_result", None)
            cached_json = None
            if cached_result is not None:
                cached_json = serialize_response(cached_result)

            # Also check for cached_response_json stored in event metadata
            if hasattr(event, "metadata") and event.metadata:
                cached_json = event.metadata.get("cached_response_json", cached_json)

            # Check for cached_response_json stored directly on LLMCallEvent
            event_cached_json = getattr(event, "cached_response_json", None)
            if event_cached_json is not None:
                cached_json = event_cached_json

            step = StepRecord(
                step=event.step,
                event_type=event.event_type,
                provider=getattr(event, "provider", None),
                model=getattr(event, "model", None),
                tool_name=getattr(event, "tool_name", None),
                mutates_state=getattr(event, "mutates_state", False),
                cached_response=cached_result,
                cached_response_json=cached_json,
            )
            steps.append(step)

        steps.sort(key=lambda s: s.step)
        return steps


class DurableReplayEngine:
    """Lightweight replay engine for durable session resumption.

    Unlike ReplayEngine (time-travel debugging), this:
    - Uses the SAME session ID (no "replay-" prefix)
    - Auto-determines which steps to mock from persisted events
    - No network blocker, no safety warnings
    - Tools re-execute by default; set ``cache_tools=True`` to replay
      cached tool results (prevents re-sending emails, re-writing to DBs, etc.)
    """

    def __init__(
        self,
        steps: list[StepRecord],
        *,
        cache_tools: bool = False,
        stream_delay_ms: float = 0,
    ) -> None:
        self._step_index: dict[int, StepRecord] = {
            s.step: s for s in steps if s.cached_response_json is not None
        }
        self._tool_step_index: dict[int, StepRecord] = {
            s.step: s
            for s in steps
            if s.event_type == "tool_call" and s.cached_response_json is not None
        }
        self._active = bool(self._step_index) or bool(self._tool_step_index)
        self._cache_tools = cache_tools
        self._stream_delay_ms = stream_delay_ms

    @property
    def is_active(self) -> bool:
        return self._active

    def should_mock(self, current_step: int) -> bool:
        """Mock only steps that have a cached LLM response."""
        return self._active and current_step in self._step_index

    def should_mock_tool(self, current_step: int) -> bool:
        """Mock tool calls if cache_tools is enabled and a cached result exists."""
        if not self._cache_tools:
            return False
        return self._active and current_step in self._tool_step_index

    def get_cached_response(self, step: int) -> object | None:
        record = self._step_index.get(step)
        if record is None or record.cached_response_json is None:
            return None
        return deserialize_response(
            record.cached_response_json,
            record.provider,
            delay_ms=self._stream_delay_ms,
        )

    def stop(self) -> None:
        self._active = False
        set_current_replay_engine(None)


def _load_durable_steps(gate: Gate, session_id: str) -> list[StepRecord]:
    """Load recorded LLM call steps that have cached responses."""
    events = gate.store.get_session_events(session_id)
    steps: list[StepRecord] = []
    for event in events:
        cached_json = getattr(event, "cached_response_json", None)
        if cached_json is None:
            continue
        steps.append(
            StepRecord(
                step=event.step,
                event_type=event.event_type,
                provider=getattr(event, "provider", None),
                model=getattr(event, "model", None),
                cached_response_json=cached_json,
            )
        )
    return steps
