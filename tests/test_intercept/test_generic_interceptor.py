"""Tests for the generic interceptor (shared intercept logic)."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from stateloom.core.context import set_current_replay_engine, set_current_session
from stateloom.intercept.generic_interceptor import (
    _check_replay,
    _intercept_sync,
    _wrap_stream_sync,
    patch_provider,
)
from stateloom.intercept.provider_adapter import BaseProviderAdapter, PatchTarget
from stateloom.intercept.provider_registry import clear_adapters


class _MockAdapter(BaseProviderAdapter):
    """A minimal adapter for testing."""

    def __init__(self, name: str = "mock-provider"):
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    @property
    def method_label(self) -> str:
        return "mock.call"


@pytest.fixture(autouse=True)
def _clean_state():
    clear_adapters()
    set_current_session(None)
    set_current_replay_engine(None)
    yield
    clear_adapters()
    set_current_session(None)
    set_current_replay_engine(None)


def _make_gate(fail_open: bool = True) -> MagicMock:
    gate = MagicMock()
    gate.config.fail_open = fail_open
    gate.config.durable_stream_buffer = False
    gate.pricing.calculate_cost.return_value = 0.001
    session = MagicMock()
    session.id = "test-session"
    session.durable = False
    session.next_step.return_value = 1
    gate.get_or_create_session.return_value = session
    return gate


class TestCheckReplay:
    def test_no_engine(self):
        gate = MagicMock()
        assert _check_replay(gate, 1) is None

    def test_inactive_engine(self):
        engine = MagicMock()
        engine.is_active = False
        set_current_replay_engine(engine)
        gate = MagicMock()
        assert _check_replay(gate, 1) is None

    def test_active_engine_should_mock(self):
        cached = SimpleNamespace(text="cached")
        engine = MagicMock()
        engine.is_active = True
        engine.should_mock.return_value = True
        engine.get_cached_response.return_value = cached
        set_current_replay_engine(engine)
        gate = MagicMock()
        result = _check_replay(gate, 5)
        assert result is cached
        engine.should_mock.assert_called_once_with(5)

    def test_active_engine_should_not_mock(self):
        engine = MagicMock()
        engine.is_active = True
        engine.should_mock.return_value = False
        set_current_replay_engine(engine)
        gate = MagicMock()
        assert _check_replay(gate, 1) is None


class TestInterceptSync:
    def test_passes_provider_name_and_model(self):
        gate = _make_gate()
        adapter = _MockAdapter("my-provider")
        response = SimpleNamespace(text="hello")
        original = MagicMock(return_value=response)
        gate.pipeline.execute_sync.return_value = response

        result = _intercept_sync(gate, adapter, original, None, (), {"model": "my-model"})

        gate.pipeline.execute_sync.assert_called_once()
        call_kwargs = gate.pipeline.execute_sync.call_args
        assert call_kwargs.kwargs["provider"] == "my-provider"
        assert call_kwargs.kwargs["method"] == "mock.call"
        assert call_kwargs.kwargs["model"] == "my-model"

    def test_cost_tracking_delegated_to_pipeline(self):
        """Interceptor should NOT call session.add_cost — CostTracker middleware handles it."""
        gate = _make_gate()
        adapter = _MockAdapter("tok")
        response = SimpleNamespace()
        gate.pipeline.execute_sync.return_value = response

        _intercept_sync(gate, adapter, MagicMock(), None, (), {"model": "m"})

        session = gate.get_or_create_session()
        session.add_cost.assert_not_called()

    def test_fail_open_on_middleware_error(self):
        gate = _make_gate(fail_open=True)
        adapter = _MockAdapter()
        response = SimpleNamespace(text="fallback")
        original = MagicMock(return_value=response)
        gate.pipeline.execute_sync.side_effect = RuntimeError("boom")

        result = _intercept_sync(gate, adapter, original, None, (), {"model": "m"})
        assert result is response
        original.assert_called_once()

    def test_fail_closed_on_middleware_error(self):
        gate = _make_gate(fail_open=False)
        adapter = _MockAdapter()
        original = MagicMock()
        gate.pipeline.execute_sync.side_effect = RuntimeError("boom")

        with pytest.raises(RuntimeError, match="boom"):
            _intercept_sync(gate, adapter, original, None, (), {"model": "m"})

    def test_replay_returns_cached(self):
        cached = SimpleNamespace(text="from replay")
        engine = MagicMock()
        engine.is_active = True
        engine.should_mock.return_value = True
        engine.get_cached_response.return_value = cached
        set_current_replay_engine(engine)

        gate = _make_gate()
        adapter = _MockAdapter()
        original = MagicMock()

        result = _intercept_sync(gate, adapter, original, None, (), {"model": "m"})
        assert result is cached
        gate.pipeline.execute_sync.assert_not_called()


class TestComplianceStreamingOverride:
    def test_block_streaming_forces_stream_false(self):
        """Compliance profile with block_streaming forces stream=False before pipeline."""
        from stateloom.core.config import ComplianceProfile

        gate = _make_gate()
        adapter = _MockAdapter()
        response = SimpleNamespace(text="ok")
        original = MagicMock(return_value=response)
        gate.pipeline.execute_sync.return_value = response

        # Configure compliance profile with block_streaming
        profile = ComplianceProfile(standard="hipaa", block_streaming=True)
        gate._get_compliance_profile.return_value = profile

        # Adapter says it's streaming
        adapter.is_streaming = MagicMock(return_value=True)

        kwargs = {"model": "gpt-4", "stream": True}
        result = _intercept_sync(gate, adapter, original, None, (), kwargs)

        # Should go through pipeline (not streaming bypass)
        gate.pipeline.execute_sync.assert_called_once()
        # stream should be forced to False
        assert kwargs["stream"] is False

    def test_no_block_streaming_allows_stream(self):
        """Without block_streaming, streaming requests bypass the pipeline."""
        from stateloom.core.config import ComplianceProfile

        gate = _make_gate()
        adapter = _MockAdapter()
        response = MagicMock()
        response.__iter__ = MagicMock(return_value=iter([]))
        original = MagicMock(return_value=response)

        profile = ComplianceProfile(standard="hipaa", block_streaming=False)
        gate._get_compliance_profile.return_value = profile

        adapter.is_streaming = MagicMock(return_value=True)

        kwargs = {"model": "gpt-4", "stream": True}
        # Streaming path — returns a generator, pipeline not called
        result = _intercept_sync(gate, adapter, original, None, (), kwargs)
        gate.pipeline.execute_sync.assert_not_called()

    def test_no_compliance_profile_allows_stream(self):
        """No compliance profile = streaming proceeds normally."""
        gate = _make_gate()
        adapter = _MockAdapter()
        response = MagicMock()
        response.__iter__ = MagicMock(return_value=iter([]))
        original = MagicMock(return_value=response)

        gate._get_compliance_profile.return_value = None

        adapter.is_streaming = MagicMock(return_value=True)

        kwargs = {"model": "gpt-4", "stream": True}
        result = _intercept_sync(gate, adapter, original, None, (), kwargs)
        gate.pipeline.execute_sync.assert_not_called()


class TestWrapStreamSync:
    def test_accumulates_tokens(self):
        class StreamAdapter(_MockAdapter):
            def extract_stream_tokens(
                self, chunk: Any, accumulated: dict[str, int]
            ) -> dict[str, int]:
                accumulated["prompt_tokens"] = chunk.pt
                accumulated["completion_tokens"] = chunk.ct
                return accumulated

        gate = _make_gate()
        session = MagicMock()
        session.id = "s1"
        adapter = StreamAdapter()

        chunks = [
            SimpleNamespace(pt=10, ct=5),
            SimpleNamespace(pt=10, ct=15),
        ]

        collected = list(_wrap_stream_sync(gate, adapter, iter(chunks), session, "model", 1, {}))

        assert len(collected) == 2
        session.add_cost.assert_called_once()
        call_kwargs = session.add_cost.call_args
        assert call_kwargs.kwargs["prompt_tokens"] == 10
        assert call_kwargs.kwargs["completion_tokens"] == 15
        gate.store.save_event.assert_called_once()


class TestPatchProvider:
    def test_returns_empty_for_no_targets(self):
        gate = _make_gate()
        adapter = _MockAdapter()
        result = patch_provider(gate, adapter)
        assert result == []

    def test_patches_sync_target(self):
        class Dummy:
            def method(self):
                return "original"

        class PatchableAdapter(_MockAdapter):
            def get_patch_targets(self):
                return [PatchTarget(Dummy, "method", is_async=False, description="dummy.method")]

        gate = _make_gate()
        adapter = PatchableAdapter()
        original = Dummy.method

        try:
            result = patch_provider(gate, adapter)
            assert len(result) == 1
            assert "method" in result[0]
            # Method should be replaced
            assert Dummy.method is not original
        finally:
            # Restore for other tests
            Dummy.method = original
