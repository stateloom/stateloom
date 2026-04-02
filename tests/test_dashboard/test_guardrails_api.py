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
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from stateloom.dashboard.api import create_api_router

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

    def test_toggle_enabled(self, client, gate):
        """POST /security/guardrails/configure can toggle guardrails enabled."""
        assert gate.config.guardrails_enabled is True

        resp = client.post(
            "/api/security/guardrails/configure",
            json={"enabled": False},
        )
        assert resp.status_code == 200
        assert resp.json()["config"]["enabled"] is False
        assert gate.config.guardrails_enabled is False

    def test_toggle_local_model_enabled(self, client, gate):
        """POST /security/guardrails/configure can toggle local model."""
        resp = client.post(
            "/api/security/guardrails/configure",
            json={"local_model_enabled": True},
        )
        assert resp.status_code == 200
        assert resp.json()["config"]["local_model_enabled"] is True
        assert gate.config.guardrails_local_model_enabled is True

    def test_toggle_output_scanning(self, client, gate):
        """POST /security/guardrails/configure can toggle output scanning."""
        resp = client.post(
            "/api/security/guardrails/configure",
            json={"output_scanning_enabled": False},
        )
        assert resp.status_code == 200
        assert resp.json()["config"]["output_scanning_enabled"] is False
        assert gate.config.guardrails_output_scanning_enabled is False


class TestGuardrailsStatusEndpoint:
    def test_status_includes_nli_fields(self, client, gate):
        """GET /security/guardrails includes NLI config fields."""
        response = client.get("/api/security/guardrails")
        assert response.status_code == 200
        data = response.json()
        assert "nli_enabled" in data["config"]
        assert "nli_available" in data["config"]

    def test_nli_toggle_persists_across_config_reset(self, client, gate):
        """NLI enabled via dashboard survives in-memory config reset."""
        # Toggle NLI on via dashboard
        resp = client.post(
            "/api/security/guardrails/configure",
            json={"nli_enabled": True},
        )
        assert resp.status_code == 200
        assert resp.json()["config"]["nli_enabled"] is True

        # Simulate server restart: reset in-memory config to default
        gate.config.guardrails_nli_enabled = False

        # GET should load persisted value from store
        resp = client.get("/api/security/guardrails")
        assert resp.json()["config"]["nli_enabled"] is True

    def test_disable_persists_across_refresh(self, client, gate):
        """NLI disabled via dashboard stays disabled on refresh."""
        # Enable NLI first
        client.post(
            "/api/security/guardrails/configure",
            json={"nli_enabled": True},
        )
        # Disable NLI
        resp = client.post(
            "/api/security/guardrails/configure",
            json={"nli_enabled": False},
        )
        assert resp.json()["config"]["nli_enabled"] is False

        # Simulate page refresh — store has False, should stay False
        resp = client.get("/api/security/guardrails")
        assert resp.json()["config"]["nli_enabled"] is False

    def test_toggle_on_off_cycle(self, client, gate):
        """Full enable→disable→refresh cycle preserves each state."""
        # Enable NLI
        resp = client.post(
            "/api/security/guardrails/configure",
            json={"nli_enabled": True},
        )
        assert resp.json()["config"]["nli_enabled"] is True

        # Simulate restart: reset in-memory config
        gate.config.guardrails_nli_enabled = False

        # GET reads from store — should still be True
        resp = client.get("/api/security/guardrails")
        assert resp.json()["config"]["nli_enabled"] is True

        # Disable NLI
        resp = client.post(
            "/api/security/guardrails/configure",
            json={"nli_enabled": False},
        )
        assert resp.json()["config"]["nli_enabled"] is False

        # Simulate restart again: set in-memory to True (opposite of store)
        gate.config.guardrails_nli_enabled = True

        # GET reads from store — should be False (store wins)
        resp = client.get("/api/security/guardrails")
        assert resp.json()["config"]["nli_enabled"] is False
