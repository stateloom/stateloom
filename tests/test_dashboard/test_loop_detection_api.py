"""Tests for loop detection dashboard config toggle."""

from __future__ import annotations

import pytest

from stateloom.core.config import StateLoomConfig
from stateloom.gate import Gate


@pytest.fixture
def gate():
    """Create a Gate with loop detection disabled (default)."""
    config = StateLoomConfig(
        auto_patch=False,
        dashboard=False,
        console_output=False,
        store_backend="memory",
    )
    g = Gate(config)
    g._setup_middleware()
    return g


@pytest.fixture
def gate_with_loop():
    """Create a Gate with loop detection enabled."""
    config = StateLoomConfig(
        auto_patch=False,
        dashboard=False,
        console_output=False,
        store_backend="memory",
        loop_detection_enabled=True,
        loop_exact_threshold=5,
    )
    g = Gate(config)
    g._setup_middleware()
    return g


@pytest.fixture
def client(gate):
    from stateloom.dashboard.api import create_api_router

    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    app = FastAPI()
    router = create_api_router(gate)
    app.include_router(router, prefix="/api")
    return TestClient(app)


@pytest.fixture
def client_with_loop(gate_with_loop):
    from stateloom.dashboard.api import create_api_router

    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    app = FastAPI()
    router = create_api_router(gate_with_loop)
    app.include_router(router, prefix="/api")
    return TestClient(app)


def _has_loop_detector(gate: Gate) -> bool:
    return any(hasattr(mw, "_threshold") for mw in gate.pipeline.middlewares)


class TestLoopDetectionConfig:
    def test_disabled_by_default(self, client, gate):
        """GET /config shows loop_detection_enabled=False by default."""
        resp = client.get("/api/config").json()
        assert resp["loop_detection_enabled"] is False
        assert resp["loop_exact_threshold"] == 5
        assert not _has_loop_detector(gate)

    def test_enable_via_dashboard(self, client, gate):
        """PATCH /config can enable loop detection and add middleware."""
        assert not _has_loop_detector(gate)

        resp = client.patch(
            "/api/config",
            json={"loop_detection_enabled": True},
        )
        assert resp.status_code == 200

        assert gate.config.loop_detection_enabled is True
        assert _has_loop_detector(gate)

    def test_disable_via_dashboard(self, client_with_loop, gate_with_loop):
        """PATCH /config can disable loop detection and remove middleware."""
        assert _has_loop_detector(gate_with_loop)

        resp = client_with_loop.patch(
            "/api/config",
            json={"loop_detection_enabled": False},
        )
        assert resp.status_code == 200

        assert gate_with_loop.config.loop_detection_enabled is False
        assert not _has_loop_detector(gate_with_loop)

    def test_update_threshold_via_dashboard(self, client_with_loop, gate_with_loop):
        """PATCH /config can update loop threshold on running middleware."""
        resp = client_with_loop.patch(
            "/api/config",
            json={"loop_exact_threshold": 10},
        )
        assert resp.status_code == 200

        assert gate_with_loop.config.loop_exact_threshold == 10
        # Verify the running middleware also updated
        for mw in gate_with_loop.pipeline.middlewares:
            if hasattr(mw, "_threshold"):
                assert mw._threshold == 10

    def test_enable_does_not_duplicate_middleware(self, client_with_loop, gate_with_loop):
        """Enabling when already enabled does not add a second LoopDetector."""
        count_before = sum(1 for mw in gate_with_loop.pipeline.middlewares if hasattr(mw, "_threshold"))
        assert count_before == 1

        resp = client_with_loop.patch(
            "/api/config",
            json={"loop_detection_enabled": True},
        )
        assert resp.status_code == 200

        count_after = sum(1 for mw in gate_with_loop.pipeline.middlewares if hasattr(mw, "_threshold"))
        assert count_after == 1

    def test_toggle_on_off_cycle(self, client, gate):
        """Enable then disable — middleware is added then removed cleanly."""
        # Enable
        client.patch("/api/config", json={"loop_detection_enabled": True})
        assert _has_loop_detector(gate)

        # Disable
        client.patch("/api/config", json={"loop_detection_enabled": False})
        assert not _has_loop_detector(gate)

        # Re-enable
        client.patch("/api/config", json={"loop_detection_enabled": True})
        assert _has_loop_detector(gate)

    def test_enable_and_set_threshold_together(self, client, gate):
        """PATCH /config can enable and set threshold in a single call."""
        resp = client.patch(
            "/api/config",
            json={"loop_detection_enabled": True, "loop_exact_threshold": 3},
        )
        assert resp.status_code == 200

        assert gate.config.loop_detection_enabled is True
        assert gate.config.loop_exact_threshold == 3
        assert _has_loop_detector(gate)
        for mw in gate.pipeline.middlewares:
            if hasattr(mw, "_threshold"):
                assert mw._threshold == 3
