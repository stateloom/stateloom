"""Tests for the Prometheus MetricsCollector."""

from __future__ import annotations

import pytest


class TestMetricsCollectorDisabled:
    """Tests for MetricsCollector when disabled or prometheus_client is missing."""

    def test_disabled_collector_is_noop(self):
        from stateloom.observability.collector import MetricsCollector

        mc = MetricsCollector(enabled=False)
        assert not mc.enabled
        assert mc.registry is None

        # All record methods should be no-ops (no exceptions)
        mc.record_llm_call(
            model="gpt-4",
            provider="openai",
            latency_ms=100,
            prompt_tokens=10,
            completion_tokens=20,
            cost=0.01,
            org_id="acme",
            team_id="ml-team",
        )
        mc.record_cache_hit(match_type="exact", org_id="acme", team_id="ml-team")
        mc.record_cache_miss(org_id="acme", team_id="ml-team")
        mc.record_pii_detection(
            pii_type="email",
            action="redacted",
            org_id="acme",
            team_id="ml-team",
        )
        mc.record_budget_violation(org_id="acme", team_id="ml-team")
        mc.record_local_routing(decision="local", org_id="acme", team_id="ml-team")
        mc.record_kill_switch_block(org_id="acme", team_id="ml-team")
        mc.record_blast_radius_pause(
            pause_type="session",
            org_id="acme",
            team_id="ml-team",
        )
        mc.record_rate_limit(team_id="t1", outcome="passed")
        mc.set_active_sessions(5)

    def test_disabled_generate_metrics_returns_comment(self):
        from stateloom.observability.collector import MetricsCollector

        mc = MetricsCollector(enabled=False)
        output = mc.generate_metrics()
        assert output.startswith("#")
        assert "disabled" in output.lower()


class TestMetricsCollectorEnabled:
    """Tests for MetricsCollector with prometheus_client installed."""

    @pytest.fixture
    def collector(self):
        pytest.importorskip("prometheus_client")
        from stateloom.observability.collector import MetricsCollector

        return MetricsCollector(enabled=True)

    def test_enabled_has_registry(self, collector):
        assert collector.enabled
        assert collector.registry is not None

    def test_record_llm_call(self, collector):
        collector.record_llm_call(
            model="gpt-4",
            provider="openai",
            latency_ms=1500,
            prompt_tokens=100,
            completion_tokens=50,
            cost=0.05,
            org_id="acme",
            team_id="ml-team",
        )
        output = collector.generate_metrics()
        assert "stateloom_llm_requests_total" in output
        assert "stateloom_llm_tokens_total" in output
        assert "stateloom_llm_request_cost_usd" in output
        assert "stateloom_llm_request_duration_seconds" in output
        assert 'model="gpt-4"' in output
        assert 'provider="openai"' in output
        assert 'org_id="acme"' in output
        assert 'team_id="ml-team"' in output

    def test_record_cache_hit(self, collector):
        collector.record_cache_hit(
            match_type="semantic",
            org_id="acme",
            team_id="ml-team",
        )
        output = collector.generate_metrics()
        assert "stateloom_cache_hits_total" in output
        assert 'match_type="semantic"' in output
        assert 'org_id="acme"' in output
        assert 'team_id="ml-team"' in output

    def test_record_cache_miss(self, collector):
        collector.record_cache_miss(org_id="acme", team_id="ml-team")
        output = collector.generate_metrics()
        assert "stateloom_cache_misses_total" in output
        assert 'org_id="acme"' in output
        assert 'team_id="ml-team"' in output

    def test_record_pii_detection(self, collector):
        collector.record_pii_detection(
            pii_type="email",
            action="redacted",
            org_id="acme",
            team_id="ml-team",
        )
        output = collector.generate_metrics()
        assert "stateloom_pii_detections_total" in output
        assert 'pii_type="email"' in output
        assert 'action="redacted"' in output
        assert 'org_id="acme"' in output
        assert 'team_id="ml-team"' in output

    def test_record_budget_violation(self, collector):
        collector.record_budget_violation(org_id="acme", team_id="ml-team")
        output = collector.generate_metrics()
        assert "stateloom_budget_violations_total" in output
        assert 'org_id="acme"' in output
        assert 'team_id="ml-team"' in output

    def test_record_local_routing(self, collector):
        collector.record_local_routing(
            decision="local",
            org_id="acme",
            team_id="ml-team",
        )
        output = collector.generate_metrics()
        assert "stateloom_local_routing_total" in output
        assert 'decision="local"' in output
        assert 'org_id="acme"' in output
        assert 'team_id="ml-team"' in output

    def test_record_kill_switch_block(self, collector):
        collector.record_kill_switch_block(org_id="acme", team_id="ml-team")
        output = collector.generate_metrics()
        assert "stateloom_kill_switch_blocks_total" in output
        assert 'org_id="acme"' in output
        assert 'team_id="ml-team"' in output

    def test_record_blast_radius_pause(self, collector):
        collector.record_blast_radius_pause(
            pause_type="agent",
            org_id="acme",
            team_id="ml-team",
        )
        output = collector.generate_metrics()
        assert "stateloom_blast_radius_pauses_total" in output
        assert 'type="agent"' in output
        assert 'org_id="acme"' in output
        assert 'team_id="ml-team"' in output

    def test_active_sessions_gauge(self, collector):
        collector.set_active_sessions(3)
        output = collector.generate_metrics()
        assert "stateloom_active_sessions" in output

    def test_token_labels(self, collector):
        collector.record_llm_call(
            model="claude-3",
            provider="anthropic",
            latency_ms=500,
            prompt_tokens=200,
            completion_tokens=100,
            cost=0.02,
            org_id="acme",
            team_id="ml-team",
        )
        output = collector.generate_metrics()
        assert 'type="prompt"' in output
        assert 'type="completion"' in output

    def test_multiple_calls_accumulate(self, collector):
        for _ in range(3):
            collector.record_llm_call(
                model="gpt-4",
                provider="openai",
                latency_ms=100,
                prompt_tokens=10,
                completion_tokens=5,
                cost=0.01,
                org_id="acme",
                team_id="ml-team",
            )
        output = collector.generate_metrics()
        # Counter should show 3.0
        assert "3.0" in output

    def test_dedicated_registry_no_default_conflict(self, collector):
        """Collector uses its own registry, not the global default."""
        from prometheus_client import REGISTRY

        assert collector.registry is not REGISTRY

    def test_record_rate_limit(self, collector):
        """Rate limit counter records team_id, virtual_key_id, and outcome labels."""
        collector.record_rate_limit(team_id="t1", outcome="passed")
        output = collector.generate_metrics()
        assert "stateloom_rate_limit_requests_total" in output
        assert 'team_id="t1"' in output
        assert 'outcome="passed"' in output
        assert 'virtual_key_id=""' in output

    def test_record_rate_limit_with_virtual_key_id(self, collector):
        """virtual_key_id label appears when provided."""
        collector.record_rate_limit(
            team_id="t1",
            outcome="queued",
            virtual_key_id="vk-abc",
        )
        output = collector.generate_metrics()
        assert 'virtual_key_id="vk-abc"' in output

    def test_record_rate_limit_wait_histogram(self, collector):
        """Histogram records wait time when wait_ms > 0."""
        collector.record_rate_limit(
            team_id="t1",
            outcome="queued",
            wait_ms=250.0,
            virtual_key_id="vk-1",
        )
        output = collector.generate_metrics()
        assert "stateloom_rate_limit_wait_seconds" in output
        assert 'virtual_key_id="vk-1"' in output

    def test_default_empty_org_and_team(self, collector):
        """Calling record_* without org_id/team_id uses empty-string defaults."""
        collector.record_llm_call(
            model="gpt-4",
            provider="openai",
            latency_ms=100,
            prompt_tokens=10,
            completion_tokens=5,
            cost=0.01,
        )
        collector.record_cache_hit(match_type="exact")
        collector.record_cache_miss()
        collector.record_pii_detection(pii_type="ssn", action="blocked")
        collector.record_budget_violation()
        collector.record_local_routing(decision="cloud")
        collector.record_kill_switch_block()
        collector.record_blast_radius_pause(pause_type="session")
        output = collector.generate_metrics()
        assert 'org_id=""' in output
        assert 'team_id=""' in output
