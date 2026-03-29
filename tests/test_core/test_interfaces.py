"""Tests for Protocol compliance — null and real implementations."""

from stateloom.core.interfaces import (
    AlertManagerProtocol,
    MetricsCollectorProtocol,
    TimeSeriesAggregatorProtocol,
)
from stateloom.core.observability_protocol import (
    NullAlertManager,
    NullMetricsCollector,
    NullTimeSeriesAggregator,
)


def test_null_metrics_satisfies_protocol():
    assert isinstance(NullMetricsCollector(), MetricsCollectorProtocol)


def test_null_alert_satisfies_protocol():
    assert isinstance(NullAlertManager(), AlertManagerProtocol)


def test_null_aggregator_satisfies_protocol():
    assert isinstance(NullTimeSeriesAggregator(), TimeSeriesAggregatorProtocol)


def test_real_metrics_satisfies_protocol():
    try:
        from stateloom.observability.collector import MetricsCollector

        assert isinstance(MetricsCollector(enabled=False), MetricsCollectorProtocol)
    except ImportError:
        pass  # prometheus_client not installed


def test_real_alert_satisfies_protocol():
    try:
        from stateloom.observability.alerting import AlertManager

        assert isinstance(AlertManager(), AlertManagerProtocol)
    except ImportError:
        pass


def test_real_aggregator_satisfies_protocol():
    try:
        from stateloom.observability.aggregator import TimeSeriesAggregator
        from stateloom.store.memory_store import MemoryStore

        assert isinstance(TimeSeriesAggregator(MemoryStore()), TimeSeriesAggregatorProtocol)
    except ImportError:
        pass
