"""Tests for observability API endpoints."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from stateloom.core.config import StateLoomConfig
from stateloom.core.event import LLMCallEvent
from stateloom.store.memory_store import MemoryStore


@pytest.fixture
def gate():
    """Create a minimal Gate-like object for the API."""
    store = MemoryStore()

    # Create a mock gate with the pieces the API needs
    mock_gate = MagicMock()
    mock_gate.store = store
    mock_gate.config = StateLoomConfig(
        store_backend="memory",
        dashboard=False,
        auto_patch=False,
    )

    # Set up observability
    from stateloom.observability.aggregator import TimeSeriesAggregator

    mock_gate._observability_aggregator = TimeSeriesAggregator(store)
    mock_gate._metrics_collector = None

    return mock_gate


@pytest.fixture
def client(gate):
    """Create a FastAPI TestClient with observability routes."""
    from fastapi import FastAPI

    from stateloom.dashboard.observability_api import (
        create_metrics_endpoint,
        create_observability_router,
    )

    app = FastAPI()
    router = create_observability_router(gate)
    app.include_router(router, prefix="/api/v1")

    @app.get("/metrics")
    async def metrics():
        return create_metrics_endpoint(gate)

    return TestClient(app)


class TestTimeseriesEndpoint:
    def test_default_window(self, client):
        resp = client.get("/api/v1/observability/timeseries")
        assert resp.status_code == 200
        data = resp.json()
        assert data["window"] == "1h"
        assert data["bucket_seconds"] == 60
        assert isinstance(data["buckets"], list)

    def test_custom_window(self, client):
        resp = client.get("/api/v1/observability/timeseries?window=24h")
        assert resp.status_code == 200
        data = resp.json()
        assert data["window"] == "24h"
        assert data["bucket_seconds"] == 900

    def test_invalid_window(self, client):
        resp = client.get("/api/v1/observability/timeseries?window=2h")
        assert resp.status_code == 422

    def test_with_events(self, client, gate):
        # Add some events
        event = LLMCallEvent(
            session_id="test",
            model="gpt-4",
            provider="openai",
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=150,
            cost=0.01,
            latency_ms=500.0,
            timestamp=datetime.now(timezone.utc) - timedelta(minutes=5),
        )
        gate.store.save_event(event)

        resp = client.get("/api/v1/observability/timeseries")
        assert resp.status_code == 200
        data = resp.json()
        total = sum(b["requests"] for b in data["buckets"])
        assert total == 1


class TestLatencyEndpoint:
    def test_empty(self, client):
        resp = client.get("/api/v1/observability/latency")
        assert resp.status_code == 200
        data = resp.json()
        assert data["p50"] == 0
        assert data["p90"] == 0
        assert data["histogram_buckets"] == []

    def test_with_events(self, client, gate):
        for lat in [100, 200, 500, 1000, 2000]:
            event = LLMCallEvent(
                session_id="test",
                model="gpt-4",
                provider="openai",
                latency_ms=float(lat),
                timestamp=datetime.now(timezone.utc) - timedelta(minutes=5),
            )
            gate.store.save_event(event)

        resp = client.get("/api/v1/observability/latency")
        assert resp.status_code == 200
        data = resp.json()
        assert data["p50"] > 0
        assert data["avg_ms"] > 0
        assert len(data["histogram_buckets"]) > 0


class TestBreakdownEndpoint:
    def test_empty(self, client):
        resp = client.get("/api/v1/observability/breakdown")
        assert resp.status_code == 200
        data = resp.json()
        assert data["by_model"] == {}
        assert data["by_provider"] == {}

    def test_with_events(self, client, gate):
        for model in ["gpt-4", "gpt-4", "claude-3"]:
            event = LLMCallEvent(
                session_id="test",
                model=model,
                provider="openai" if model.startswith("gpt") else "anthropic",
                cost=0.01,
                total_tokens=100,
                timestamp=datetime.now(timezone.utc) - timedelta(minutes=5),
            )
            gate.store.save_event(event)

        resp = client.get("/api/v1/observability/breakdown")
        assert resp.status_code == 200
        data = resp.json()
        assert "gpt-4" in data["by_model"]
        assert data["by_model"]["gpt-4"]["requests"] == 2


class TestMetricsEndpoint:
    def test_metrics_disabled(self, client):
        resp = client.get("/metrics")
        assert resp.status_code == 200
        text = resp.text
        assert "not enabled" in text.lower() or "disabled" in text.lower()

    def test_metrics_enabled(self, gate):
        """Test /metrics with prometheus_client installed."""
        pytest.importorskip("prometheus_client")
        from fastapi import FastAPI

        from stateloom.dashboard.observability_api import (
            create_metrics_endpoint,
            create_observability_router,
        )
        from stateloom.observability.collector import MetricsCollector

        collector = MetricsCollector(enabled=True)
        collector.record_llm_call(
            model="gpt-4",
            provider="openai",
            latency_ms=100,
            prompt_tokens=10,
            completion_tokens=5,
            cost=0.01,
        )
        gate._metrics_collector = collector

        app = FastAPI()
        router = create_observability_router(gate)
        app.include_router(router, prefix="/api/v1")

        @app.get("/metrics")
        async def metrics():
            return create_metrics_endpoint(gate)

        client = TestClient(app)
        resp = client.get("/metrics")
        assert resp.status_code == 200
        assert "stateloom_llm_requests_total" in resp.text
