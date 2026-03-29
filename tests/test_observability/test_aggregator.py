"""Tests for the TimeSeriesAggregator."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from stateloom.core.event import CacheHitEvent, LLMCallEvent
from stateloom.observability.aggregator import TimeSeriesAggregator
from stateloom.store.memory_store import MemoryStore


@pytest.fixture
def store():
    return MemoryStore()


@pytest.fixture
def aggregator(store):
    return TimeSeriesAggregator(store)


def _make_llm_event(
    *,
    model: str = "gpt-4",
    provider: str = "openai",
    latency_ms: float = 500.0,
    prompt_tokens: int = 100,
    completion_tokens: int = 50,
    cost: float = 0.01,
    minutes_ago: float = 0,
) -> LLMCallEvent:
    return LLMCallEvent(
        session_id="test-session",
        model=model,
        provider=provider,
        latency_ms=latency_ms,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=prompt_tokens + completion_tokens,
        cost=cost,
        timestamp=datetime.now(timezone.utc) - timedelta(minutes=minutes_ago),
    )


def _make_cache_event(*, minutes_ago: float = 0) -> CacheHitEvent:
    return CacheHitEvent(
        session_id="test-session",
        original_model="gpt-4",
        saved_cost=0.01,
        match_type="exact",
        timestamp=datetime.now(timezone.utc) - timedelta(minutes=minutes_ago),
    )


class TestGetTimeseries:
    def test_empty_store(self, aggregator):
        result = aggregator.get_timeseries("1h")
        assert result["window"] == "1h"
        assert result["bucket_seconds"] == 60
        assert isinstance(result["buckets"], list)
        # All buckets should have 0 requests
        total = sum(b["requests"] for b in result["buckets"])
        assert total == 0

    def test_events_in_window(self, store, aggregator):
        # Add events within the last hour
        for i in range(5):
            store.save_event(_make_llm_event(minutes_ago=i * 10, cost=0.02))

        result = aggregator.get_timeseries("1h")
        total = sum(b["requests"] for b in result["buckets"])
        assert total == 5
        total_cost = sum(b["cost"] for b in result["buckets"])
        assert abs(total_cost - 0.10) < 0.001

    def test_events_outside_window_excluded(self, store, aggregator):
        # Add an event 2 hours ago (outside 1h window)
        store.save_event(_make_llm_event(minutes_ago=120))
        # Add an event 30 minutes ago (inside 1h window)
        store.save_event(_make_llm_event(minutes_ago=30))

        result = aggregator.get_timeseries("1h")
        total = sum(b["requests"] for b in result["buckets"])
        assert total == 1

    def test_different_windows(self, store, aggregator):
        for window, expected_bucket_secs in [
            ("1h", 60),
            ("6h", 300),
            ("24h", 900),
            ("7d", 3600),
        ]:
            result = aggregator.get_timeseries(window)
            assert result["bucket_seconds"] == expected_bucket_secs


class TestGetLatency:
    def test_empty_store(self, aggregator):
        result = aggregator.get_latency("1h")
        assert result["p50"] == 0
        assert result["p90"] == 0
        assert result["p95"] == 0
        assert result["p99"] == 0
        assert result["avg_ms"] == 0
        assert result["histogram_buckets"] == []

    def test_percentile_calculation(self, store, aggregator):
        # Add 100 events with latencies 1ms to 100ms
        for i in range(1, 101):
            store.save_event(
                _make_llm_event(
                    latency_ms=float(i),
                    minutes_ago=i * 0.5,
                )
            )

        result = aggregator.get_latency("1h")
        assert result["p50"] > 0
        assert result["p90"] > 0
        assert result["p95"] > 0
        assert result["p99"] > 0
        assert result["avg_ms"] > 0
        # p90 should be >= p50
        assert result["p90"] >= result["p50"]
        assert result["p99"] >= result["p95"]

    def test_histogram_buckets(self, store, aggregator):
        store.save_event(_make_llm_event(latency_ms=200.0, minutes_ago=5))
        store.save_event(_make_llm_event(latency_ms=1500.0, minutes_ago=10))
        store.save_event(_make_llm_event(latency_ms=8000.0, minutes_ago=15))

        result = aggregator.get_latency("1h")
        assert len(result["histogram_buckets"]) == 9
        # The 100ms bucket should have 0 events (200ms > 100ms)
        assert result["histogram_buckets"][0]["le"] == 100
        assert result["histogram_buckets"][0]["count"] == 0
        # The 250ms bucket should have 1 event (200ms <= 250ms)
        assert result["histogram_buckets"][1]["le"] == 250
        assert result["histogram_buckets"][1]["count"] == 1


class TestGetBreakdown:
    def test_empty_store(self, aggregator):
        result = aggregator.get_breakdown("1h")
        assert result["by_model"] == {}
        assert result["by_provider"] == {}
        assert result["cache"]["hits"] == 0

    def test_model_breakdown(self, store, aggregator):
        store.save_event(_make_llm_event(model="gpt-4", cost=0.05, minutes_ago=5))
        store.save_event(_make_llm_event(model="gpt-4", cost=0.03, minutes_ago=10))
        store.save_event(
            _make_llm_event(model="claude-3", provider="anthropic", cost=0.02, minutes_ago=15)
        )

        result = aggregator.get_breakdown("1h")
        assert "gpt-4" in result["by_model"]
        assert result["by_model"]["gpt-4"]["requests"] == 2
        assert abs(result["by_model"]["gpt-4"]["cost"] - 0.08) < 0.001
        assert "claude-3" in result["by_model"]
        assert result["by_model"]["claude-3"]["requests"] == 1

    def test_provider_breakdown(self, store, aggregator):
        store.save_event(_make_llm_event(provider="openai", minutes_ago=5))
        store.save_event(_make_llm_event(provider="anthropic", minutes_ago=10))
        store.save_event(_make_llm_event(provider="openai", minutes_ago=15))

        result = aggregator.get_breakdown("1h")
        assert result["by_provider"]["openai"]["requests"] == 2
        assert result["by_provider"]["anthropic"]["requests"] == 1

    def test_cache_stats(self, store, aggregator):
        store.save_event(_make_llm_event(minutes_ago=5))
        store.save_event(_make_llm_event(minutes_ago=10))
        store.save_event(_make_cache_event(minutes_ago=15))

        result = aggregator.get_breakdown("1h")
        assert result["cache"]["hits"] == 1
        assert result["cache"]["misses"] == 2
        # hit_rate = 1/3 = 0.3333
        assert abs(result["cache"]["hit_rate"] - 0.3333) < 0.01


class TestPercentileHelper:
    def test_single_value(self):
        assert TimeSeriesAggregator._percentile([100.0], 50) == 100.0
        assert TimeSeriesAggregator._percentile([100.0], 99) == 100.0

    def test_two_values(self):
        result = TimeSeriesAggregator._percentile([10.0, 20.0], 50)
        assert result == 15.0

    def test_empty(self):
        assert TimeSeriesAggregator._percentile([], 50) == 0.0
