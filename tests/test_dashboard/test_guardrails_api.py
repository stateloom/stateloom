"""Tests for guardrails dashboard API endpoints."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from stateloom.core.config import StateLoomConfig
from stateloom.core.types import GuardrailMode
from stateloom.gate import Gate


@pytest.fixture
def gate():
    """Create a Gate instance with guardrails enabled."""
    config = StateLoomConfig(
        auto_patch=False,
        dashboard=False,
        console_output=False,
        store_backend="memory",
        guardrails_enabled=True,
        guardrails_mode=GuardrailMode.AUDIT,
    )
    g = Gate(config)
    g._setup_middleware()
    return g


@pytest.fixture
def client(gate):
    """Create a FastAPI test client for the dashboard API."""
    from stateloom.dashboard.api import create_api_router

    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    app = FastAPI()
    router = create_api_router(gate)
    app.include_router(router, prefix="/api")
    return TestClient(app)


class TestGuardrailsConfigureEndpoint:
    def test_enable_nli(self, client, gate):
        """POST /security/guardrails/configure can enable NLI."""
        assert gate.config.guardrails_nli_enabled is False

        response = client.post(
            "/api/security/guardrails/configure",
            json={"nli_enabled": True},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["config"]["nli_enabled"] is True
        assert gate.config.guardrails_nli_enabled is True

    def test_set_nli_threshold(self, client, gate):
        """POST /security/guardrails/configure can set NLI threshold."""
        response = client.post(
            "/api/security/guardrails/configure",
            json={"nli_threshold": 0.85},
        )
        assert response.status_code == 200
        assert gate.config.guardrails_nli_threshold == pytest.approx(0.85)

    def test_change_mode(self, client, gate):
        """POST /security/guardrails/configure can change guardrail mode."""
        response = client.post(
            "/api/security/guardrails/configure",
            json={"mode": "enforce"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["config"]["mode"] == "enforce"
        assert gate.config.guardrails_mode == GuardrailMode.ENFORCE


class TestGuardrailsStatusEndpoint:
    def test_status_includes_nli_fields(self, client, gate):
        """GET /security/guardrails includes NLI config fields."""
        response = client.get("/api/security/guardrails")
        assert response.status_code == 200
        data = response.json()
        assert "nli_enabled" in data["config"]
        assert "nli_available" in data["config"]
