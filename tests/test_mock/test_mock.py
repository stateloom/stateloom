"""Tests for stateloom.mock() — VCR-cassette mocking."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

import stateloom
from stateloom.core.event import LLMCallEvent
from stateloom.core.session import Session
from stateloom.mock import MockSession, _derive_session_id, _ensure_gate, mock


def _init_gate():
    """Init gate with test-friendly defaults. Returns the gate."""
    return stateloom.init(
        auto_patch=False,
        dashboard=False,
        console_output=False,
        store_backend="memory",
    )


# ---------------------------------------------------------------------------
# _derive_session_id
# ---------------------------------------------------------------------------


class TestDeriveSessionId:
    def test_explicit_id(self):
        assert _derive_session_id(explicit_id="my-cassette") == "mock-my-cassette"

    def test_explicit_id_already_prefixed(self):
        assert _derive_session_id(explicit_id="mock-already") == "mock-already"

    def test_from_function(self):
        def some_func():
            pass

        sid = _derive_session_id(func=some_func)
        assert sid.startswith("mock-")
        assert "some_func" in sid
        # Should end with a 12-char hex digest
        assert len(sid.rsplit("-", 1)[-1]) == 12

    def test_from_function_deterministic(self):
        def my_test():
            pass

        sid1 = _derive_session_id(func=my_test)
        sid2 = _derive_session_id(func=my_test)
        assert sid1 == sid2

    def test_explicit_takes_precedence(self):
        def some_func():
            pass

        sid = _derive_session_id(func=some_func, explicit_id="override")
        assert sid == "mock-override"

    def test_no_func_no_id_raises(self):
        with pytest.raises(ValueError, match="requires either"):
            _derive_session_id()

    def test_class_method_includes_class(self):
        class MyClass:
            def test_method(self):
                pass

        sid = _derive_session_id(func=MyClass.test_method)
        assert "MyClass.test_method" in sid

    def test_different_functions_different_ids(self):
        def func_a():
            pass

        def func_b():
            pass

        assert _derive_session_id(func=func_a) != _derive_session_id(func=func_b)


# ---------------------------------------------------------------------------
# _ensure_gate
# ---------------------------------------------------------------------------


class TestEnsureGate:
    def test_returns_existing_gate(self):
        gate = _init_gate()
        result = _ensure_gate()
        assert result is gate

    def test_auto_init_when_no_gate(self):
        stateloom.shutdown()
        assert stateloom._gate is None

        gate = _ensure_gate()
        assert gate is not None
        assert stateloom._gate is gate
        assert gate.config.store_backend == "sqlite"
        assert gate.config.store_path == ".stateloom/mock.db"
        assert gate.config.dashboard is False
        assert gate.config.console_output is False
        assert gate.config.cache_enabled is False


# ---------------------------------------------------------------------------
# MockSession — constructor / properties
# ---------------------------------------------------------------------------


class TestMockSessionInit:
    def test_defaults(self):
        m = MockSession()
        assert m._explicit_id is None
        assert m._force_record is False
        assert m._network_block is True
        assert m._allow_hosts == []
        assert m.is_replay is False
        assert m.session_id is None

    def test_custom_args(self):
        m = MockSession(
            session_id="my-id",
            force_record=True,
            network_block=False,
            allow_hosts=["example.com"],
        )
        assert m._explicit_id == "my-id"
        assert m._force_record is True
        assert m._network_block is False
        assert m._allow_hosts == ["example.com"]


# ---------------------------------------------------------------------------
# MockSession — context manager (record mode)
# ---------------------------------------------------------------------------


class TestMockSessionContextManager:
    def test_sync_context_manager_record_mode(self):
        _init_gate()
        with mock("test-cm-record") as m:
            assert m.is_replay is False
            assert m.session_id == "mock-test-cm-record"
            assert m._session is not None
            assert m._session.metadata.get("mock") is True
            assert m._session.metadata.get("mock_mode") == "record"

    def test_sync_context_manager_cleanup(self):
        _init_gate()
        m = mock("test-cleanup")
        with m:
            assert m._session_cm is not None
        assert m._session_cm is None

    @pytest.mark.asyncio
    async def test_async_context_manager(self):
        _init_gate()
        async with mock("test-async-cm") as m:
            assert m.is_replay is False
            assert m.session_id == "mock-test-async-cm"
            assert m._session.metadata.get("mock") is True


# ---------------------------------------------------------------------------
# MockSession — decorator mode
# ---------------------------------------------------------------------------


class TestMockSessionDecorator:
    def test_sync_decorator(self):
        _init_gate()

        @mock()
        def my_test_func():
            return "result"

        assert my_test_func() == "result"

    def test_sync_decorator_with_explicit_id(self):
        _init_gate()

        @mock("explicit-deco")
        def my_test():
            return 42

        assert my_test() == 42

    @pytest.mark.asyncio
    async def test_async_decorator(self):
        _init_gate()

        @mock()
        async def my_async_test():
            return "async-result"

        assert await my_async_test() == "async-result"

    def test_decorator_preserves_function_name(self):
        @mock()
        def important_test():
            pass

        assert important_test.__name__ == "important_test"

    @pytest.mark.asyncio
    async def test_async_decorator_preserves_function_name(self):
        @mock()
        async def async_important():
            pass

        assert async_important.__name__ == "async_important"


# ---------------------------------------------------------------------------
# Helpers to set up replay state
# ---------------------------------------------------------------------------


def _setup_replay(gate, cassette_id, num_steps=1):
    """Save a session + LLMCallEvents so mock() detects replay mode."""
    prev = Session(id=cassette_id, name="prev")
    prev.step_counter = num_steps
    gate.store.save_session(prev)
    for step in range(1, num_steps + 1):
        gate.store.save_event(
            LLMCallEvent(
                session_id=cassette_id,
                step=step,
                provider="openai",
                model="gpt-4o",
                prompt_tokens=10,
                completion_tokens=20,
                cost=0.001,
                cached_response_json='{"_type": "raw", "data": {}}',
            )
        )


# ---------------------------------------------------------------------------
# MockSession — record → replay cycle
# ---------------------------------------------------------------------------


class TestRecordReplayCycle:
    def test_first_run_is_record_mode(self):
        _init_gate()
        with mock("record-test") as m:
            assert m.is_replay is False
            assert m._session.metadata["mock_mode"] == "record"

    def test_replay_detected_with_cached_steps(self):
        gate = _init_gate()
        _setup_replay(gate, "mock-replay-detect")

        with mock("replay-detect") as m:
            assert m.is_replay is True
            assert m._session.metadata["mock_mode"] == "replay"

    def test_no_replay_with_zero_steps(self):
        gate = _init_gate()
        prev = Session(id="mock-no-replay", name="prev")
        prev.step_counter = 0
        gate.store.save_session(prev)

        with mock("no-replay") as m:
            assert m.is_replay is False


# ---------------------------------------------------------------------------
# MockSession — force_record
# ---------------------------------------------------------------------------


class TestForceRecord:
    def test_force_record_purges_existing(self):
        gate = _init_gate()
        _setup_replay(gate, "mock-force-rec", num_steps=2)

        with mock("force-rec", force_record=True) as m:
            assert m.is_replay is False
            assert m._session.metadata["mock_mode"] == "record"

    def test_force_record_on_nonexistent_session(self):
        _init_gate()
        with mock("nonexistent", force_record=True) as m:
            assert m.is_replay is False


# ---------------------------------------------------------------------------
# MockSession — network blocking
# ---------------------------------------------------------------------------


class TestNetworkBlocking:
    def test_network_blocker_active_in_replay(self):
        gate = _init_gate()
        _setup_replay(gate, "mock-net-block")

        with mock("net-block") as m:
            assert m.is_replay is True
            assert m._network_blocker is not None

    def test_network_blocker_not_active_in_record(self):
        _init_gate()
        with mock("net-no-block") as m:
            assert m.is_replay is False
            assert m._network_blocker is None

    def test_network_blocker_disabled(self):
        gate = _init_gate()
        _setup_replay(gate, "mock-net-disabled")

        with mock("net-disabled", network_block=False) as m:
            assert m.is_replay is True
            assert m._network_blocker is None

    def test_network_blocker_deactivated_on_exit(self):
        gate = _init_gate()
        _setup_replay(gate, "mock-net-deact")

        m = mock("net-deact")
        with m:
            assert m._network_blocker is not None
        assert m._network_blocker is None

    def test_allow_hosts_passed_to_blocker(self):
        gate = _init_gate()
        _setup_replay(gate, "mock-allow-hosts")

        with mock("allow-hosts", allow_hosts=["example.com"]) as m:
            assert m.is_replay is True
            assert m._network_blocker is not None
            assert "example.com" in m._network_blocker._allowed_hosts


# ---------------------------------------------------------------------------
# MockSession — session metadata
# ---------------------------------------------------------------------------


class TestSessionMetadata:
    def test_mock_metadata_set_in_record(self):
        _init_gate()
        with mock("meta-record") as m:
            assert m._session.metadata["mock"] is True
            assert m._session.metadata["mock_mode"] == "record"

    def test_durable_metadata_set(self):
        _init_gate()
        with mock("meta-durable") as m:
            assert m._session.metadata.get("durable") is True


# ---------------------------------------------------------------------------
# mock() factory function
# ---------------------------------------------------------------------------


class TestMockFactory:
    def test_returns_mock_session(self):
        m = mock()
        assert isinstance(m, MockSession)

    def test_passes_args(self):
        m = mock("my-id", force_record=True, network_block=False, allow_hosts=["a.com"])
        assert m._explicit_id == "my-id"
        assert m._force_record is True
        assert m._network_block is False
        assert m._allow_hosts == ["a.com"]


# ---------------------------------------------------------------------------
# stateloom.mock() (public API)
# ---------------------------------------------------------------------------


class TestPublicAPI:
    def test_stateloom_mock_available(self):
        assert hasattr(stateloom, "mock")
        assert callable(stateloom.mock)

    def test_mock_session_in_all(self):
        assert "mock" in stateloom.__all__
        assert "MockSession" in stateloom.__all__

    def test_stateloom_mock_returns_mock_session(self):
        m = stateloom.mock("test-public")
        assert isinstance(m, MockSession)

    def test_public_api_as_context_manager(self):
        _init_gate()
        with stateloom.mock("pub-cm") as m:
            assert m.is_replay is False
            assert m.session_id == "mock-pub-cm"

    def test_public_api_as_decorator(self):
        _init_gate()

        @stateloom.mock()
        def my_test():
            return "works"

        assert my_test() == "works"


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_fewer_calls_than_cached(self):
        """Replay mode with fewer calls than cached steps should not error."""
        gate = _init_gate()
        _setup_replay(gate, "mock-fewer-calls", num_steps=3)

        with mock("fewer-calls") as m:
            assert m.is_replay is True

    def test_exception_in_body_still_exits_cleanly(self):
        _init_gate()
        m_ref = None
        with pytest.raises(ValueError, match="boom"):
            with mock("exc-body") as m:
                m_ref = m
                raise ValueError("boom")

        assert m_ref._session_cm is None
        assert m_ref._network_blocker is None

    def test_decorator_exception_still_exits(self):
        _init_gate()

        @mock("exc-deco")
        def failing_test():
            raise RuntimeError("test failure")

        with pytest.raises(RuntimeError, match="test failure"):
            failing_test()

    def test_multiple_mock_sessions(self):
        """Multiple mock sessions with different IDs should not interfere."""
        _init_gate()
        with mock("session-a") as a:
            assert a.session_id == "mock-session-a"

        with mock("session-b") as b:
            assert b.session_id == "mock-session-b"

    def test_decorator_with_args_and_return(self):
        _init_gate()

        @mock("args-test")
        def compute(x, y):
            return x + y

        assert compute(2, 3) == 5

    @pytest.mark.asyncio
    async def test_async_decorator_with_args(self):
        _init_gate()

        @mock("async-args-test")
        async def async_compute(x, y):
            return x * y

        assert await async_compute(3, 4) == 12


# ---------------------------------------------------------------------------
# Pytest fixture
# ---------------------------------------------------------------------------


class TestPytestFixture:
    def test_fixture_exists(self):
        """The stateloom_mock fixture should be importable."""
        from stateloom.mock import stateloom_mock  # noqa: F401

    def test_fixture_is_pytest_fixture(self):
        from stateloom.mock import stateloom_mock

        # pytest wraps fixtures — check it's callable and has fixture marker
        assert callable(stateloom_mock)
        # pytest >= 8 uses pytest.fixture which sets this attribute
        assert hasattr(stateloom_mock, "pytestmark") or "fixture" in str(type(stateloom_mock))

    def test_marker_registration(self):
        from stateloom.mock import pytest_configure

        mock_config = MagicMock()
        pytest_configure(mock_config)
        mock_config.addinivalue_line.assert_called_once()
        args = mock_config.addinivalue_line.call_args
        assert "stateloom_force_record" in args[0][1]
