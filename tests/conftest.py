"""Shared test fixtures for StateLoom."""

import pytest

import stateloom
from stateloom.core.config import StateLoomConfig
from stateloom.core.session import Session, SessionManager
from stateloom.store.memory_store import MemoryStore


@pytest.fixture
def config():
    """Default test config with in-memory store and no dashboard."""
    return StateLoomConfig(
        auto_patch=False,
        dashboard=False,
        console_output=False,
        store_backend="memory",
    )


@pytest.fixture
def memory_store():
    """In-memory store for testing."""
    return MemoryStore()


@pytest.fixture
def session_manager():
    """Session manager for testing."""
    return SessionManager()


@pytest.fixture
def session():
    """A test session."""
    return Session(id="test-session-001", name="Test Session")


@pytest.fixture(autouse=True)
def cleanup_gate():
    """Ensure stateloom is shut down between tests."""
    yield
    stateloom.shutdown()
