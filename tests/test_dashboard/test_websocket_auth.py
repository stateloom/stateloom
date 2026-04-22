"""Tests for dashboard WebSocket auth enforcement."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

import stateloom
from stateloom.dashboard.server import DashboardServer


@pytest.fixture
def dashboard_client() -> TestClient:
    gate = stateloom.init(
        auto_patch=False,
        dashboard=False,
        proxy=False,
        store_backend="memory",
    )
    gate.config.dashboard_api_key = "test-key-123"
    server = DashboardServer(gate)
    client = TestClient(server.app)
    try:
        yield client
    finally:
        stateloom.shutdown()


class TestDashboardWebSocketAuth:
    def test_api_accepts_query_api_key(self, dashboard_client: TestClient) -> None:
        resp = dashboard_client.get("/api/v1/stats?api_key=test-key-123")
        assert resp.status_code == 200

    def test_ws_rejects_missing_credentials(self, dashboard_client: TestClient) -> None:
        with pytest.raises(WebSocketDisconnect):
            with dashboard_client.websocket_connect("/ws"):
                pass

    def test_ws_accepts_query_api_key(self, dashboard_client: TestClient) -> None:
        with dashboard_client.websocket_connect("/ws?api_key=test-key-123") as websocket:
            assert websocket.receive_json()["type"] == "heartbeat"

    def test_ws_accepts_bearer_api_key(self, dashboard_client: TestClient) -> None:
        with dashboard_client.websocket_connect(
            "/ws",
            headers={"Authorization": "Bearer test-key-123"},
        ) as websocket:
            assert websocket.receive_json()["type"] == "heartbeat"
