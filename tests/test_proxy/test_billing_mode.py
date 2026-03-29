"""Tests for billing mode (API vs subscription cost tracking)."""

from __future__ import annotations

import types
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from stateloom.core.event import LLMCallEvent
from stateloom.core.session import Session
from stateloom.core.types import BillingMode
from stateloom.middleware.base import MiddlewareContext
from stateloom.middleware.budget_enforcer import BudgetEnforcer
from stateloom.middleware.cost_tracker import CostTracker
from stateloom.pricing.registry import PricingRegistry
from stateloom.proxy.billing import detect_billing_mode
from stateloom.proxy.virtual_key import VirtualKey

# --- detect_billing_mode tests ---


class TestDetectBillingMode:
    def test_detect_anthropic_api_key(self):
        """sk-ant-api* prefix → api."""
        assert detect_billing_mode("sk-ant-api03-abc123xyz", "anthropic") == "api"
        assert detect_billing_mode("sk-ant-api04-xyz", "anthropic") == "api"

    def test_detect_anthropic_subscription(self):
        """OAuth tokens (sk-ant-oat*) and other non-api tokens → subscription."""
        assert detect_billing_mode("sk-ant-oat01-y4abc123", "anthropic") == "subscription"
        assert detect_billing_mode("sess_abc123xyz", "anthropic") == "subscription"
        assert detect_billing_mode("eyJhbGciOiJSUzI...", "anthropic") == "subscription"

    def test_detect_openai_api_key(self):
        """sk-* prefix → api."""
        assert detect_billing_mode("sk-proj-abc123", "openai") == "api"
        assert detect_billing_mode("sk-abc123", "openai") == "api"

    def test_detect_openai_subscription(self):
        """Non sk-* OpenAI token → subscription (e.g. ChatGPT Plus)."""
        assert detect_billing_mode("sess_chatgpt_abc", "openai") == "subscription"

    def test_detect_google_api_key(self):
        """AIzaSy* prefix → api."""
        assert detect_billing_mode("AIzaSyABC123", "google") == "api"

    def test_detect_google_subscription(self):
        """Non-AIzaSy Google token → subscription."""
        assert detect_billing_mode("ya29.a0AfB_byBq", "google") == "subscription"

    def test_detect_unknown_defaults_api(self):
        """Empty token or unknown provider → api."""
        assert detect_billing_mode("", "anthropic") == "api"
        assert detect_billing_mode("", "openai") == "api"
        assert detect_billing_mode("some-token", "unknown_provider") == "api"
        assert detect_billing_mode("", "") == "api"

    def test_detect_empty_token_known_provider(self):
        """Empty token with known provider → api (server key scenario)."""
        assert detect_billing_mode("", "google") == "api"


# --- BillingMode enum tests ---


class TestBillingModeEnum:
    def test_enum_values(self):
        assert BillingMode.API == "api"
        assert BillingMode.SUBSCRIPTION == "subscription"

    def test_str_comparison(self):
        """BillingMode extends str, so direct string comparison works."""
        assert BillingMode.API == "api"
        assert BillingMode.SUBSCRIPTION == "subscription"


# --- VirtualKey billing_mode field ---


class TestVirtualKeyBillingMode:
    def test_default_billing_mode(self):
        vk = VirtualKey(
            id="vk-test",
            key_hash="hash123",
            key_preview="ag-...1234",
            team_id="team-1",
            org_id="org-1",
        )
        assert vk.billing_mode == "api"

    def test_subscription_billing_mode(self):
        vk = VirtualKey(
            id="vk-test",
            key_hash="hash123",
            key_preview="ag-...1234",
            team_id="team-1",
            org_id="org-1",
            billing_mode="subscription",
        )
        assert vk.billing_mode == "subscription"


# --- Session estimated_api_cost accumulator ---


class TestSessionEstimatedApiCost:
    def test_session_estimated_api_cost_accumulator(self):
        """Session accumulates estimated_api_cost across calls."""
        session = Session()
        assert session.estimated_api_cost == 0.0

        session.add_cost(0.0, prompt_tokens=100, completion_tokens=50, estimated_api_cost=0.05)
        assert session.estimated_api_cost == 0.05
        assert session.total_cost == 0.0

        session.add_cost(0.0, prompt_tokens=200, completion_tokens=100, estimated_api_cost=0.10)
        assert session.estimated_api_cost == pytest.approx(0.15)
        assert session.total_cost == 0.0

    def test_api_mode_both_costs_equal(self):
        """For API mode, total_cost and estimated_api_cost accumulate together."""
        session = Session()
        session.add_cost(0.05, prompt_tokens=100, completion_tokens=50, estimated_api_cost=0.05)
        assert session.total_cost == 0.05
        assert session.estimated_api_cost == 0.05

    def test_tokens_always_tracked(self):
        """Tokens are tracked regardless of billing mode."""
        session = Session()
        session.add_cost(0.0, prompt_tokens=500, completion_tokens=200, estimated_api_cost=0.10)
        assert session.total_tokens == 700
        assert session.total_prompt_tokens == 500
        assert session.total_completion_tokens == 200
        assert session.call_count == 1


# --- CostTracker dual cost tracking ---


class TestCostTrackerBillingMode:
    def _make_ctx(self, billing_mode: str = "api") -> MiddlewareContext:
        session = Session(id="test-session")
        if billing_mode != "api":
            session.billing_mode = billing_mode
            session.metadata["billing_mode"] = billing_mode
        config = MagicMock()
        config.budget_action = MagicMock(value="hard_stop")
        return MiddlewareContext(
            session=session,
            config=config,
            provider="openai",
            method="chat",
            model="gpt-4",
            request_kwargs={"model": "gpt-4", "messages": []},
            request_hash="hash123",
        )

    def _make_pricing(self) -> PricingRegistry:
        pricing = PricingRegistry()
        pricing.register("gpt-4", input_per_token=0.03 / 1000, output_per_token=0.06 / 1000)
        return pricing

    @pytest.mark.asyncio
    async def test_subscription_cost_zero(self):
        """CostTracker sets cost=0.0 for subscription sessions."""
        ctx = self._make_ctx(billing_mode="subscription")
        pricing = self._make_pricing()
        tracker = CostTracker(pricing)

        # Mock response with usage
        response = types.SimpleNamespace(
            usage=types.SimpleNamespace(prompt_tokens=100, completion_tokens=50)
        )
        ctx.response = response

        async def call_next(c):
            return response

        await tracker.process(ctx, call_next)

        # Find the LLMCallEvent
        llm_events = [e for e in ctx.events if isinstance(e, LLMCallEvent)]
        assert len(llm_events) == 1
        event = llm_events[0]
        assert event.cost == 0.0
        assert event.prompt_tokens == 100
        assert event.completion_tokens == 50

    @pytest.mark.asyncio
    async def test_subscription_estimated_cost_tracked(self):
        """CostTracker sets estimated_api_cost for subscription sessions."""
        ctx = self._make_ctx(billing_mode="subscription")
        pricing = self._make_pricing()
        tracker = CostTracker(pricing)

        response = types.SimpleNamespace(
            usage=types.SimpleNamespace(prompt_tokens=100, completion_tokens=50)
        )
        ctx.response = response

        async def call_next(c):
            return response

        await tracker.process(ctx, call_next)

        llm_events = [e for e in ctx.events if isinstance(e, LLMCallEvent)]
        event = llm_events[0]
        # estimated_api_cost should be non-zero (per-token price)
        assert event.estimated_api_cost > 0.0
        # Session should also accumulate
        assert ctx.session.estimated_api_cost > 0.0
        assert ctx.session.total_cost == 0.0

    @pytest.mark.asyncio
    async def test_subscription_tokens_tracked(self):
        """Tokens are fully tracked even for subscription sessions."""
        ctx = self._make_ctx(billing_mode="subscription")
        pricing = self._make_pricing()
        tracker = CostTracker(pricing)

        response = types.SimpleNamespace(
            usage=types.SimpleNamespace(prompt_tokens=100, completion_tokens=50)
        )
        ctx.response = response

        async def call_next(c):
            return response

        await tracker.process(ctx, call_next)

        assert ctx.session.total_tokens == 150
        assert ctx.session.total_prompt_tokens == 100
        assert ctx.session.total_completion_tokens == 50

    @pytest.mark.asyncio
    async def test_api_billing_unchanged(self):
        """Default behavior identical for API-billed sessions."""
        ctx = self._make_ctx(billing_mode="api")
        pricing = self._make_pricing()
        tracker = CostTracker(pricing)

        response = types.SimpleNamespace(
            usage=types.SimpleNamespace(prompt_tokens=100, completion_tokens=50)
        )
        ctx.response = response

        async def call_next(c):
            return response

        await tracker.process(ctx, call_next)

        llm_events = [e for e in ctx.events if isinstance(e, LLMCallEvent)]
        event = llm_events[0]
        # Both cost and estimated_api_cost should be equal and non-zero
        assert event.cost > 0.0
        assert event.cost == event.estimated_api_cost
        assert ctx.session.total_cost > 0.0
        assert ctx.session.total_cost == ctx.session.estimated_api_cost


# --- BudgetEnforcer skip for subscription ---


class TestBudgetEnforcerBillingMode:
    @pytest.mark.asyncio
    async def test_subscription_budget_skipped(self):
        """BudgetEnforcer skips enforcement for subscription sessions."""
        session = Session(id="sub-session", budget=0.01)
        session.total_cost = 0.0  # subscription cost is always 0
        session.billing_mode = "subscription"
        session.metadata["billing_mode"] = "subscription"

        config = MagicMock()
        config.budget_action = MagicMock(value="hard_stop")
        from stateloom.core.types import BudgetAction, FailureAction

        config.budget_action = BudgetAction.HARD_STOP
        config.budget_on_middleware_failure = FailureAction.PASS

        ctx = MiddlewareContext(
            session=session,
            config=config,
            provider="openai",
            method="chat",
            model="gpt-4",
            request_kwargs={},
            request_hash="hash",
        )

        enforcer = BudgetEnforcer()
        called = False

        async def call_next(c):
            nonlocal called
            called = True
            return "response"

        result = await enforcer.process(ctx, call_next)
        assert called
        assert result == "response"

    @pytest.mark.asyncio
    async def test_api_budget_still_enforced(self):
        """BudgetEnforcer still enforces for API sessions."""
        from stateloom.core.errors import StateLoomBudgetError
        from stateloom.core.types import BudgetAction, FailureAction

        session = Session(id="api-session", budget=0.01)
        session.total_cost = 0.02  # over budget

        config = MagicMock()
        config.budget_action = BudgetAction.HARD_STOP
        config.budget_on_middleware_failure = FailureAction.PASS

        ctx = MiddlewareContext(
            session=session,
            config=config,
            provider="openai",
            method="chat",
            model="gpt-4",
            request_kwargs={},
            request_hash="hash",
        )

        enforcer = BudgetEnforcer()

        async def call_next(c):
            return "response"

        with pytest.raises(StateLoomBudgetError):
            await enforcer.process(ctx, call_next)


# --- VK billing_mode SQLite round-trip ---


class TestVKBillingModeSQLite:
    def test_vk_billing_mode_persisted(self, tmp_path):
        """VK with billing_mode=subscription round-trips through SQLite."""
        from stateloom.store.sqlite_store import SQLiteStore

        db_path = str(tmp_path / "test.db")
        store = SQLiteStore(db_path)

        vk = VirtualKey(
            id="vk-billing-test",
            key_hash="hash-billing",
            key_preview="ag-...test",
            team_id="team-1",
            org_id="org-1",
            name="billing-test-key",
            billing_mode="subscription",
        )
        store.save_virtual_key(vk)

        loaded = store.get_virtual_key("vk-billing-test")
        assert loaded is not None
        assert loaded.billing_mode == "subscription"

    def test_vk_default_billing_mode(self, tmp_path):
        """VK with default billing_mode round-trips as 'api'."""
        from stateloom.store.sqlite_store import SQLiteStore

        db_path = str(tmp_path / "test.db")
        store = SQLiteStore(db_path)

        vk = VirtualKey(
            id="vk-default",
            key_hash="hash-default",
            key_preview="ag-...dflt",
            team_id="team-1",
            org_id="org-1",
        )
        store.save_virtual_key(vk)

        loaded = store.get_virtual_key("vk-default")
        assert loaded is not None
        assert loaded.billing_mode == "api"


# --- Session estimated_api_cost SQLite round-trip ---


class TestSessionEstimatedCostSQLite:
    def test_session_estimated_api_cost_persisted(self, tmp_path):
        """Session estimated_api_cost persists through SQLite."""
        from stateloom.store.sqlite_store import SQLiteStore

        db_path = str(tmp_path / "test.db")
        store = SQLiteStore(db_path)

        session = Session(id="eac-test")
        session.estimated_api_cost = 1.23
        session.total_cost = 0.0
        store.save_session(session)

        loaded = store.get_session("eac-test")
        assert loaded is not None
        assert loaded.estimated_api_cost == pytest.approx(1.23)
        assert loaded.total_cost == 0.0


# --- LLMCallEvent estimated_api_cost SQLite round-trip ---


class TestLLMCallEventEstimatedCostSQLite:
    def test_event_estimated_api_cost_persisted(self, tmp_path):
        """LLMCallEvent estimated_api_cost round-trips through SQLite."""
        from stateloom.store.sqlite_store import SQLiteStore

        db_path = str(tmp_path / "test.db")
        store = SQLiteStore(db_path)

        session = Session(id="event-eac-test")
        store.save_session(session)

        event = LLMCallEvent(
            session_id="event-eac-test",
            provider="openai",
            model="gpt-4",
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=150,
            cost=0.0,
            estimated_api_cost=0.045,
        )
        store.save_event(event)

        events = store.get_session_events("event-eac-test")
        llm_events = [e for e in events if isinstance(e, LLMCallEvent)]
        assert len(llm_events) == 1
        assert llm_events[0].estimated_api_cost == pytest.approx(0.045)
        assert llm_events[0].cost == 0.0
