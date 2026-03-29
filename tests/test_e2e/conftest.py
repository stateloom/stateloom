"""Shared fixtures for E2E integration tests."""

from __future__ import annotations

from typing import Any

import pytest
from fastapi import FastAPI
from starlette.testclient import TestClient

import stateloom
from stateloom.dashboard.api import create_api_router
from stateloom.gate import Gate


@pytest.fixture
def e2e_gate():
    """Factory fixture that initialises a real Gate with a MemoryStore.

    Returns a factory so each test can pass custom kwargs.
    Cleanup is handled by the autouse ``cleanup_gate`` fixture in tests/conftest.py.
    """

    def _factory(**kwargs: Any) -> Gate:
        defaults = {
            "auto_patch": False,
            "dashboard": False,
            "console_output": False,
            "store_backend": "memory",
            "local_model": None,
        }
        defaults.update(kwargs)
        return stateloom.init(**defaults)

    return _factory


@pytest.fixture
def api_client():
    """Factory fixture that creates a TestClient for the dashboard API.

    Usage:
        gate = e2e_gate()
        client = api_client(gate)
        resp = client.get("/stats")
    """

    def _factory(gate: Gate) -> TestClient:
        app = FastAPI()
        app.include_router(create_api_router(gate))
        return TestClient(app)

    return _factory
