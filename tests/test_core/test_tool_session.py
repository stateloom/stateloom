"""Tests for @gate.tool() — session_root, dry_run replay, async support, loop detection."""

import asyncio

import pytest

import stateloom
from stateloom.core.context import get_current_session, set_current_session
from stateloom.core.errors import StateLoomLoopError
from stateloom.core.session import Session
from stateloom.core.types import SessionStatus


def _init_gate(**kwargs):
    return stateloom.init(
        auto_patch=False,
        dashboard=False,
        console_output=False,
        store_backend="memory",
        **kwargs,
    )


# ───────────────────────────────────────────────────────────
# session_root=True tests
# ───────────────────────────────────────────────────────────


def test_session_root_creates_session():
    """session_root=True should create a new session for the call."""
    gate = _init_gate()

    @gate.tool(session_root=True)
    def my_agent():
        session = get_current_session()
        assert session is not None
        assert session.name == "my_agent"
        assert session.id.startswith("my_agent-")
        return "done"

    result = my_agent()
    assert result == "done"


def test_session_root_unique_per_call():
    """Each call to a session_root tool gets a unique session ID."""
    gate = _init_gate()
    session_ids = []

    @gate.tool(session_root=True)
    def my_agent():
        session = get_current_session()
        session_ids.append(session.id)

    my_agent()
    my_agent()
    assert len(session_ids) == 2
    assert session_ids[0] != session_ids[1]


def test_session_root_nested_inherits():
    """An inner @gate.tool() (without session_root) uses the outer session."""
    gate = _init_gate()
    inner_session_ids = []

    @gate.tool()
    def inner_tool():
        session = get_current_session()
        inner_session_ids.append(session.id)

    @gate.tool(session_root=True)
    def outer_agent():
        session = get_current_session()
        inner_tool()
        return session.id

    outer_id = outer_agent()
    assert len(inner_session_ids) == 1
    assert inner_session_ids[0] == outer_id


def test_session_root_restores_previous():
    """After a session_root call, the previous session is restored in ContextVar."""
    gate = _init_gate()
    previous = Session(id="previous-session", name="Previous")
    set_current_session(previous)

    @gate.tool(session_root=True)
    def my_agent():
        session = get_current_session()
        assert session.id != "previous-session"

    my_agent()
    restored = get_current_session()
    assert restored is not None
    assert restored.id == "previous-session"


def test_session_root_false_default():
    """Default behavior (session_root=False) doesn't create a new session."""
    gate = _init_gate()

    with gate.session(session_id="existing-session") as s:

        @gate.tool()
        def my_tool():
            session = get_current_session()
            return session.id

        result = my_tool()
        assert result == "existing-session"


def test_session_root_completes_on_return():
    """session_root session is marked COMPLETED after the function returns."""
    gate = _init_gate()
    created_session_id = []

    @gate.tool(session_root=True)
    def my_agent():
        session = get_current_session()
        created_session_id.append(session.id)
        assert session.status == SessionStatus.ACTIVE

    my_agent()
    # The session should now be completed
    completed_session = gate.session_manager.get(created_session_id[0])
    assert completed_session is not None
    assert completed_session.status == SessionStatus.COMPLETED


# ───────────────────────────────────────────────────────────
# Async tool tests
# ───────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_async_tool_basic():
    """Async tool functions should be properly wrapped."""
    gate = _init_gate()
    executed = []

    @gate.tool()
    async def async_fetch(url):
        executed.append(url)
        return f"fetched {url}"

    with gate.session("test-async") as s:
        result = await async_fetch("https://example.com")

    assert result == "fetched https://example.com"
    assert len(executed) == 1


@pytest.mark.asyncio
async def test_async_tool_session_root():
    """Async tool with session_root=True creates a scoped session."""
    gate = _init_gate()

    @gate.tool(session_root=True)
    async def async_agent():
        session = get_current_session()
        assert session is not None
        assert session.name == "async_agent"
        return "done"

    result = await async_agent()
    assert result == "done"


@pytest.mark.asyncio
async def test_async_tool_restores_session():
    """Async tool with session_root restores previous session after completion."""
    gate = _init_gate()
    previous = Session(id="prev-async", name="Previous")
    set_current_session(previous)

    @gate.tool(session_root=True)
    async def async_agent():
        session = get_current_session()
        assert session.id != "prev-async"
        return session.id

    await async_agent()
    restored = get_current_session()
    assert restored is not None
    assert restored.id == "prev-async"


# ───────────────────────────────────────────────────────────
# Async session context manager tests
# ───────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_async_session_context_manager():
    """async_session() creates and cleans up a session."""
    gate = _init_gate()

    async with gate.async_session(session_id="async-session-1", budget=5.0) as s:
        assert s.id == "async-session-1"
        assert s.budget == 5.0
        assert s.status == SessionStatus.ACTIVE

    assert s.status == SessionStatus.COMPLETED


@pytest.mark.asyncio
async def test_async_session_public_api():
    """stateloom.async_session() works through the public API."""
    _init_gate()

    async with stateloom.async_session("pub-async") as s:
        assert s.id == "pub-async"

    assert s.status == SessionStatus.COMPLETED


# ───────────────────────────────────────────────────────────
# dry_run during replay tests
# ───────────────────────────────────────────────────────────


class FakeReplayEngine:
    """Minimal fake replay engine for testing dry_run behavior."""

    def __init__(self, active=True, mock_until_step=10, cached_results=None):
        self._active = active
        self.mock_until_step = mock_until_step
        self._cached_results = cached_results or {}

    @property
    def is_active(self):
        return self._active

    def should_mock(self, step):
        return self._active and step <= self.mock_until_step

    def should_mock_tool(self, step):
        return self.should_mock(step)

    def get_cached_response(self, step):
        return self._cached_results.get(step)


def test_tool_dry_run_during_replay():
    """mutates_state tool returns cached result during replay instead of executing."""
    from stateloom.core.context import set_current_replay_engine

    gate = _init_gate()
    executed = []

    @gate.tool(mutates_state=True)
    def write_file(path):
        executed.append(path)
        return f"wrote {path}"

    # Simulate active replay engine via ContextVar
    engine = FakeReplayEngine(
        active=True,
        mock_until_step=10,
        cached_results={1: "cached-result"},
    )
    set_current_replay_engine(engine)

    try:
        with gate.session("test-replay") as s:
            result = write_file("/tmp/test.txt")

        assert result == "cached-result"
        assert len(executed) == 0  # Function body should NOT have executed
    finally:
        set_current_replay_engine(None)


def test_tool_no_dry_run_outside_replay():
    """Normal execution when no replay engine is active."""
    gate = _init_gate()
    executed = []

    @gate.tool(mutates_state=True)
    def write_file(path):
        executed.append(path)
        return f"wrote {path}"

    with gate.session("test-normal") as s:
        result = write_file("/tmp/test.txt")

    assert result == "wrote /tmp/test.txt"
    assert len(executed) == 1


def test_tool_dry_run_only_for_mutates_state():
    """Non-mutating tools execute normally even during replay."""
    from stateloom.core.context import set_current_replay_engine

    gate = _init_gate()
    executed = []

    @gate.tool(mutates_state=False)
    def read_data():
        executed.append(True)
        return "data"

    set_current_replay_engine(FakeReplayEngine(active=True, mock_until_step=10))

    try:
        with gate.session("test-replay-read") as s:
            result = read_data()

        assert result == "data"
        assert len(executed) == 1  # Should execute — not mutates_state
    finally:
        set_current_replay_engine(None)


def test_tool_dry_run_past_mock_step():
    """Tools beyond mock_until_step execute live even during replay."""
    from stateloom.core.context import set_current_replay_engine

    gate = _init_gate()
    executed = []

    @gate.tool(mutates_state=True)
    def write_file(path):
        executed.append(path)
        return f"wrote {path}"

    # mock_until_step=0 means no steps are mocked
    set_current_replay_engine(FakeReplayEngine(active=True, mock_until_step=0))

    try:
        with gate.session("test-replay-live") as s:
            result = write_file("/tmp/test.txt")

        assert result == "wrote /tmp/test.txt"
        assert len(executed) == 1
    finally:
        set_current_replay_engine(None)


# ───────────────────────────────────────────────────────────
# Replay LLM call mocking tests
# ───────────────────────────────────────────────────────────


def test_replay_check_in_openai_patch():
    """Verify _check_replay returns cached response when replay is active."""
    from stateloom.core.context import set_current_replay_engine
    from stateloom.intercept.generic_interceptor import _check_replay

    gate = _init_gate()
    set_current_replay_engine(
        FakeReplayEngine(
            active=True,
            mock_until_step=5,
            cached_results={3: {"mocked": True}},
        )
    )

    try:
        # Step within mock range
        result = _check_replay(gate, 3)
        assert result == {"mocked": True}

        # Step beyond mock range
        result = _check_replay(gate, 6)
        assert result is None
    finally:
        set_current_replay_engine(None)


def test_replay_check_no_engine():
    """_check_replay returns None when no replay engine is set."""
    from stateloom.core.context import set_current_replay_engine
    from stateloom.intercept.generic_interceptor import _check_replay

    gate = _init_gate()
    set_current_replay_engine(None)
    assert _check_replay(gate, 1) is None


def test_replay_check_in_anthropic_patch():
    """Verify _check_replay in anthropic patch matches openai behavior."""
    from stateloom.core.context import set_current_replay_engine
    from stateloom.intercept.generic_interceptor import _check_replay

    gate = _init_gate()
    set_current_replay_engine(
        FakeReplayEngine(
            active=True,
            mock_until_step=3,
            cached_results={2: "anthropic-cached"},
        )
    )

    try:
        assert _check_replay(gate, 2) == "anthropic-cached"
        assert _check_replay(gate, 4) is None
    finally:
        set_current_replay_engine(None)


# ───────────────────────────────────────────────────────────
# Tool loop detection tests
# ───────────────────────────────────────────────────────────


def test_tool_loop_detection_triggers():
    """Tool called >= threshold times in a session raises StateLoomLoopError."""
    gate = _init_gate(loop_exact_threshold=3)

    @gate.tool()
    def my_tool():
        return "ok"

    with gate.session("loop-session") as s:
        my_tool()  # call 1
        my_tool()  # call 2
        with pytest.raises(StateLoomLoopError):
            my_tool()  # call 3 = threshold


def test_tool_loop_detection_different_tools():
    """Different tools have independent loop counters."""
    gate = _init_gate(loop_exact_threshold=3)

    @gate.tool()
    def tool_a():
        return "a"

    @gate.tool()
    def tool_b():
        return "b"

    with gate.session("loop-multi") as s:
        # Interleave calls — each tool has its own counter
        tool_a()  # tool_a: 1
        tool_b()  # tool_b: 1
        tool_a()  # tool_a: 2
        tool_b()  # tool_b: 2

        # tool_b hits threshold at call 3
        with pytest.raises(StateLoomLoopError) as exc_info:
            tool_b()  # tool_b: 3 = threshold -> raises

        # Verify it was tool_b that triggered, not tool_a
        assert "tool_b" in str(exc_info.value)


def test_tool_loop_detection_disabled():
    """Loop detection disabled when threshold=0."""
    gate = _init_gate(loop_exact_threshold=0)

    @gate.tool()
    def my_tool():
        return "ok"

    with gate.session("no-loop") as s:
        for _ in range(10):
            my_tool()  # Should not raise


@pytest.mark.asyncio
async def test_async_tool_loop_detection():
    """Async tools also trigger loop detection."""
    gate = _init_gate(loop_exact_threshold=2)

    @gate.tool()
    async def async_tool():
        return "ok"

    with gate.session("async-loop") as s:
        await async_tool()  # call 1
        with pytest.raises(StateLoomLoopError):
            await async_tool()  # call 2 = threshold
