"""VCR-cassette mocking for tests — record once, replay forever.

Usage::

    import stateloom

    # As decorator (session ID auto-derived from function identity)
    @stateloom.mock()
    def test_my_agent():
        response = openai.chat.completions.create(model="gpt-4o", messages=[...])
        assert "hello" in response.choices[0].message.content.lower()

    # As context manager
    def test_inline():
        with stateloom.mock("my-cassette") as m:
            response = openai.chat.completions.create(...)
            print(m.is_replay)  # True on subsequent runs

    # Force re-record
    @stateloom.mock(force_record=True)
    def test_rerecord():
        ...

    # Pytest fixture (auto-registered when pytest is available)
    def test_with_fixture(stateloom_mock):
        response = openai.chat.completions.create(...)
        print(stateloom_mock.is_replay)
"""

from __future__ import annotations

import functools
import hashlib
import inspect
import logging
from typing import Any

logger = logging.getLogger("stateloom.mock")


def _derive_session_id(func: Any = None, explicit_id: str | None = None) -> str:
    """Derive a deterministic session ID for mock recording/replay.

    Uses the function's module + qualname hashed for uniqueness, or
    the explicit ID provided by the caller.
    """
    if explicit_id is not None:
        return f"mock-{explicit_id}" if not explicit_id.startswith("mock-") else explicit_id
    if func is not None:
        module = getattr(func, "__module__", "")
        qualname = getattr(func, "__qualname__", func.__name__)
        raw = f"{module}.{qualname}"
        digest = hashlib.sha256(raw.encode()).hexdigest()[:12]
        return f"mock-{qualname}-{digest}"
    raise ValueError("mock() requires either session_id or a function")


def _ensure_gate() -> Any:
    """Get the existing Gate or auto-init one suitable for mock recording."""
    import stateloom

    if stateloom._gate is not None:
        return stateloom._gate

    return stateloom.init(
        auto_patch=True,
        dashboard=False,
        console_output=False,
        store_backend="sqlite",
        store_path=".stateloom/mock.db",
        cache=False,
    )


class MockSession:
    """VCR-cassette mock session — record on first run, replay on subsequent.

    Delegates to ``gate.session(durable=True)`` for all heavy lifting.
    Usable as a decorator, sync context manager, or async context manager.

    Args:
        session_id: Explicit cassette ID. If None, auto-derived from function identity.
        force_record: Purge existing cassette and re-record.
        network_block: Block outbound HTTP in replay mode (default True).
        allow_hosts: Hosts to allow through the network blocker.
    """

    def __init__(
        self,
        session_id: str | None = None,
        *,
        force_record: bool = False,
        network_block: bool = True,
        allow_hosts: list[str] | None = None,
    ) -> None:
        self._explicit_id = session_id
        self._force_record = force_record
        self._network_block = network_block
        self._allow_hosts = allow_hosts or []
        # Runtime state
        self._resolved_id: str | None = None
        self._gate: Any = None
        self._session_cm: Any = None
        self._session: Any = None
        self._network_blocker: Any = None
        self._is_replay = False

    @property
    def is_replay(self) -> bool:
        """True if the current run is replaying cached responses."""
        return self._is_replay

    @property
    def session_id(self) -> str | None:
        """The resolved mock session ID."""
        return self._resolved_id

    def _enter(self, func: Any = None) -> MockSession:
        """Core entry logic shared by decorator and context manager paths."""
        self._gate = _ensure_gate()
        self._resolved_id = _derive_session_id(func, self._explicit_id)

        # Force re-record: purge existing cassette
        if self._force_record:
            try:
                self._gate.store.purge_session(self._resolved_id)
            except Exception:
                pass  # No-op if session doesn't exist

        # Detect replay: existing session with cached steps
        existing = self._gate.store.get_session(self._resolved_id)
        if existing is not None and existing.step_counter > 0 and not self._force_record:
            from stateloom.replay.engine import _load_durable_steps

            durable_steps = _load_durable_steps(self._gate, self._resolved_id)
            if durable_steps:
                self._is_replay = True

        # Enter gate.session(durable=True) — handles all replay/record logic
        self._session_cm = self._gate.session(
            session_id=self._resolved_id,
            durable=True,
        )
        self._session = self._session_cm.__enter__()

        # Tag session metadata
        self._session.metadata["mock"] = True
        self._session.metadata["mock_mode"] = "replay" if self._is_replay else "record"

        # Network blocker in replay mode
        if self._is_replay and self._network_block:
            try:
                from stateloom.replay.network_blocker import NetworkBlocker

                self._network_blocker = NetworkBlocker(session_id=self._resolved_id)
                self._network_blocker.activate(
                    set(self._allow_hosts) if self._allow_hosts else None
                )
            except ImportError:
                logger.debug("Network blocker not available")

        return self

    def _exit(self, exc_type: Any = None, exc_val: Any = None, exc_tb: Any = None) -> bool:
        """Core exit logic shared by decorator and context manager paths."""
        # Deactivate network blocker
        if self._network_blocker is not None:
            self._network_blocker.deactivate()
            self._network_blocker = None

        # Exit gate.session()
        result = False
        if self._session_cm is not None:
            result = self._session_cm.__exit__(exc_type, exc_val, exc_tb) or False
            self._session_cm = None

        return result

    # --- Sync context manager ---

    def __enter__(self) -> MockSession:
        return self._enter(func=None)

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> bool:
        return self._exit(exc_type, exc_val, exc_tb)

    # --- Async context manager ---

    async def __aenter__(self) -> MockSession:
        return self._enter(func=None)

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> bool:
        return self._exit(exc_type, exc_val, exc_tb)

    # --- Decorator ---

    def __call__(self, func: Any) -> Any:
        """Use MockSession as a decorator on sync or async test functions."""
        if inspect.iscoroutinefunction(func):

            @functools.wraps(func)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                self._enter(func=func)
                try:
                    return await func(*args, **kwargs)
                finally:
                    self._exit()

            return async_wrapper

        @functools.wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            self._enter(func=func)
            try:
                return func(*args, **kwargs)
            finally:
                self._exit()

        return sync_wrapper


def mock(
    session_id: str | None = None,
    *,
    force_record: bool = False,
    network_block: bool = True,
    allow_hosts: list[str] | None = None,
) -> MockSession:
    """Create a VCR-cassette mock: record once, replay forever.

    Returns a ``MockSession`` usable as a decorator or context manager.

    Args:
        session_id: Explicit cassette ID. If None, auto-derived from function identity.
        force_record: Purge existing cassette and re-record.
        network_block: Block outbound HTTP in replay mode (default True).
        allow_hosts: Hosts to allow through the network blocker.

    Examples::

        @stateloom.mock()
        def test_my_agent():
            response = openai.chat.completions.create(...)
            assert response.choices[0].message.content

        with stateloom.mock("my-cassette") as m:
            response = openai.chat.completions.create(...)
            print(m.is_replay)
    """
    return MockSession(
        session_id=session_id,
        force_record=force_record,
        network_block=network_block,
        allow_hosts=allow_hosts,
    )


# --- Pytest fixture (auto-registered when pytest is available) ---

try:
    import pytest

    @pytest.fixture
    def stateloom_mock(request: Any) -> Any:
        """Pytest fixture providing a VCR-cassette mock session.

        Auto-derives a unique session ID from the test's node ID.
        Use the ``stateloom_force_record`` marker to force re-recording.

        Usage::

            def test_something(stateloom_mock):
                response = openai.chat.completions.create(...)
                assert stateloom_mock.is_replay or response.choices
        """
        func = request.function
        node_id = request.node.nodeid
        digest = hashlib.sha256(node_id.encode()).hexdigest()[:12]
        sid = f"mock-{func.__qualname__}-{digest}"

        marker = request.node.get_closest_marker("stateloom_force_record")
        force_record = marker is not None

        m = MockSession(session_id=sid, force_record=force_record)
        m._enter(func=func)
        try:
            yield m
        finally:
            m._exit()

    def pytest_configure(config: Any) -> None:
        config.addinivalue_line(
            "markers",
            "stateloom_force_record: Force re-record the mock cassette",
        )

except ImportError:
    pass
