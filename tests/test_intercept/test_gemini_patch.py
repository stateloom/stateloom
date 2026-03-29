"""Tests for Gemini intercept patch (using mocks — no real Gemini API calls)."""

from __future__ import annotations

import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest


def test_patch_gemini_skips_when_not_installed():
    """patch_gemini returns empty list when google.generativeai is not installed."""
    from stateloom.intercept.gemini_patch import patch_gemini

    gate = MagicMock()

    with patch.dict(sys.modules, {"google": None, "google.generativeai": None}):
        result = patch_gemini(gate)

    assert result == []


def test_extract_model_from_instance():
    """Model name is extracted from instance.model_name."""
    from stateloom.intercept.gemini_patch import _extract_model

    instance = SimpleNamespace(model_name="gemini-2.0-flash")
    assert _extract_model(instance, {}) == "gemini-2.0-flash"


def test_extract_model_fallback():
    """Falls back to 'unknown' when no model_name on instance."""
    from stateloom.intercept.gemini_patch import _extract_model

    instance = SimpleNamespace()  # no model_name
    assert _extract_model(instance, {}) == "unknown"


def test_extract_model_from_kwargs():
    """Falls back to kwargs['model'] when instance has no model_name."""
    from stateloom.intercept.gemini_patch import _extract_model

    instance = SimpleNamespace()
    assert _extract_model(instance, {"model": "gemini-1.5-pro"}) == "gemini-1.5-pro"


def test_extract_tokens_from_response():
    """Parses usage_metadata correctly."""
    from stateloom.intercept.gemini_patch import _extract_tokens_from_response

    usage_metadata = SimpleNamespace(
        prompt_token_count=100,
        candidates_token_count=50,
        total_token_count=150,
    )
    response = SimpleNamespace(usage_metadata=usage_metadata)

    pt, ct, tt = _extract_tokens_from_response(response)
    assert pt == 100
    assert ct == 50
    assert tt == 150


def test_extract_tokens_missing_usage():
    """Returns (0, 0, 0) when usage_metadata is missing."""
    from stateloom.intercept.gemini_patch import _extract_tokens_from_response

    response = SimpleNamespace()  # no usage_metadata
    assert _extract_tokens_from_response(response) == (0, 0, 0)


def test_extract_tokens_none_usage():
    """Returns (0, 0, 0) when usage_metadata is None."""
    from stateloom.intercept.gemini_patch import _extract_tokens_from_response

    response = SimpleNamespace(usage_metadata=None)
    assert _extract_tokens_from_response(response) == (0, 0, 0)


def test_check_replay_with_active_engine():
    """Returns cached result during replay."""
    from stateloom.core.context import set_current_replay_engine
    from stateloom.intercept.gemini_patch import _check_replay

    cached_response = SimpleNamespace(text="cached response")
    engine = MagicMock()
    engine.is_active = True
    engine.should_mock.return_value = True
    engine.get_cached_response.return_value = cached_response

    set_current_replay_engine(engine)

    try:
        gate = MagicMock()
        result = _check_replay(gate, step=3)
        assert result is cached_response
        engine.should_mock.assert_called_once_with(3)
        engine.get_cached_response.assert_called_once_with(3)
    finally:
        set_current_replay_engine(None)


def test_check_replay_without_engine():
    """Returns None when no replay engine."""
    from stateloom.core.context import set_current_replay_engine
    from stateloom.intercept.gemini_patch import _check_replay

    set_current_replay_engine(None)

    gate = MagicMock()
    result = _check_replay(gate, step=1)
    assert result is None


def test_check_replay_inactive_engine():
    """Returns None when engine is not active."""
    from stateloom.core.context import set_current_replay_engine
    from stateloom.intercept.gemini_patch import _check_replay

    engine = MagicMock()
    engine.is_active = False

    set_current_replay_engine(engine)

    try:
        gate = MagicMock()
        result = _check_replay(gate, step=1)
        assert result is None
    finally:
        set_current_replay_engine(None)
