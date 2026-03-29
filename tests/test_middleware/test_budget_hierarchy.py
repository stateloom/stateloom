"""Tests for BudgetEnforcer with hierarchy budget checks."""

from __future__ import annotations

import pytest

from stateloom.core.config import StateLoomConfig
from stateloom.core.errors import StateLoomBudgetError
from stateloom.core.session import Session
from stateloom.core.types import BudgetAction, FailureAction
from stateloom.middleware.base import MiddlewareContext
from stateloom.middleware.budget_enforcer import BudgetEnforcer


def _make_ctx(
    session: Session | None = None,
    budget_action: BudgetAction = BudgetAction.HARD_STOP,
) -> MiddlewareContext:
    if session is None:
        session = Session(id="test")
    config = StateLoomConfig(
        budget_action=budget_action,
        budget_per_session=session.budget,
        budget_on_middleware_failure=FailureAction.PASS,
    )
    return MiddlewareContext(
        session=session,
        config=config,
        provider="openai",
        model="gpt-4",
        request_kwargs={"messages": [{"role": "user", "content": "hi"}]},
    )


async def _noop(ctx: MiddlewareContext):
    return {"result": "ok"}


class TestBudgetEnforcerHierarchy:
    @pytest.mark.asyncio
    async def test_no_hierarchy_callback_backward_compat(self):
        """BudgetEnforcer without hierarchy callback works as before."""
        enforcer = BudgetEnforcer()
        session = Session(id="s1", budget=10.0, total_cost=5.0)
        ctx = _make_ctx(session=session)
        result = await enforcer.process(ctx, _noop)
        assert result == {"result": "ok"}

    @pytest.mark.asyncio
    async def test_hierarchy_team_budget_exceeded(self):
        """Team budget exceeded triggers hard stop."""

        def check(org_id, team_id):
            if team_id == "team-1":
                return (50.0, 55.0)  # budget=50, spent=55
            return None

        enforcer = BudgetEnforcer(hierarchy_budget_check=check)
        session = Session(
            id="s1",
            org_id="org-1",
            team_id="team-1",
            budget=100.0,
            total_cost=5.0,
        )
        ctx = _make_ctx(session=session)

        with pytest.raises(StateLoomBudgetError):
            await enforcer.process(ctx, _noop)

    @pytest.mark.asyncio
    async def test_hierarchy_org_budget_exceeded(self):
        """Org budget exceeded triggers hard stop."""

        def check(org_id, team_id):
            if org_id == "org-1":
                return (100.0, 120.0)  # budget=100, spent=120
            return None

        enforcer = BudgetEnforcer(hierarchy_budget_check=check)
        session = Session(
            id="s1",
            org_id="org-1",
            team_id="team-1",
            budget=200.0,
            total_cost=5.0,
        )
        ctx = _make_ctx(session=session)

        with pytest.raises(StateLoomBudgetError):
            await enforcer.process(ctx, _noop)

    @pytest.mark.asyncio
    async def test_hierarchy_no_budget_exceeded(self):
        """When hierarchy budgets are fine, request passes through."""

        def check(org_id, team_id):
            return None

        enforcer = BudgetEnforcer(hierarchy_budget_check=check)
        session = Session(id="s1", org_id="org-1", team_id="team-1")
        ctx = _make_ctx(session=session)
        result = await enforcer.process(ctx, _noop)
        assert result == {"result": "ok"}

    @pytest.mark.asyncio
    async def test_hierarchy_warn_mode(self):
        """Warn mode logs but doesn't raise."""

        def check(org_id, team_id):
            return (10.0, 15.0)

        enforcer = BudgetEnforcer(hierarchy_budget_check=check)
        session = Session(id="s1", org_id="org-1", team_id="team-1")
        ctx = _make_ctx(session=session, budget_action=BudgetAction.WARN)
        result = await enforcer.process(ctx, _noop)
        assert result == {"result": "ok"}
        # Pre-call warn event + post-call budget-crossed event
        assert len(ctx.events) == 2

    @pytest.mark.asyncio
    async def test_post_call_hierarchy_event(self):
        """Post-call check records event when budget is crossed by the call."""
        team_cost = 0.0

        def check(org_id, team_id):
            # Returns current team cost (changes after call_next applies cost)
            if team_cost >= 50.0:
                return (50.0, team_cost)
            return None

        async def call_next_with_cost(ctx):
            nonlocal team_cost
            # Simulate CostTracker applying cost that crosses the budget
            team_cost = 55.0
            return {"result": "ok"}

        enforcer = BudgetEnforcer(hierarchy_budget_check=check)
        session = Session(id="s1", org_id="org-1", team_id="team-1")
        ctx = _make_ctx(session=session)
        result = await enforcer.process(ctx, call_next_with_cost)
        assert result == {"result": "ok"}
        # Pre-call: budget fine (no event). Post-call: budget crossed (1 event).
        assert len(ctx.events) == 1
        assert ctx.events[0].limit == 50.0
        assert ctx.events[0].spent == 55.0

    @pytest.mark.asyncio
    async def test_post_call_session_event(self):
        """Post-call check records event when session budget is crossed."""

        async def call_next_with_cost(ctx):
            # Simulate CostTracker applying cost that crosses session budget
            ctx.session.add_cost(20.0, prompt_tokens=100, completion_tokens=50)
            return {"result": "ok"}

        enforcer = BudgetEnforcer()
        session = Session(id="s1", budget=10.0, total_cost=5.0)
        ctx = _make_ctx(session=session)
        # Pre-call: 5.0 < 10.0, passes. Post-call: 25.0 >= 10.0, event recorded.
        result = await enforcer.process(ctx, call_next_with_cost)
        assert result == {"result": "ok"}
        assert len(ctx.events) == 1
        assert ctx.events[0].limit == 10.0
        assert ctx.events[0].spent == 25.0

    @pytest.mark.asyncio
    async def test_post_call_no_event_when_under_budget(self):
        """Post-call check records no event when budget is still fine."""

        async def call_next_with_small_cost(ctx):
            ctx.session.add_cost(1.0, prompt_tokens=10, completion_tokens=5)
            return {"result": "ok"}

        enforcer = BudgetEnforcer()
        session = Session(id="s1", budget=100.0, total_cost=0.0)
        ctx = _make_ctx(session=session)
        result = await enforcer.process(ctx, call_next_with_small_cost)
        assert result == {"result": "ok"}
        assert len(ctx.events) == 0

    @pytest.mark.asyncio
    async def test_session_budget_checked_before_hierarchy(self):
        """Session budget should be checked first."""

        def check(org_id, team_id):
            return None  # hierarchy is fine

        enforcer = BudgetEnforcer(hierarchy_budget_check=check)
        session = Session(
            id="s1",
            org_id="org-1",
            team_id="team-1",
            budget=10.0,
            total_cost=15.0,
        )
        ctx = _make_ctx(session=session)

        with pytest.raises(StateLoomBudgetError) as exc_info:
            await enforcer.process(ctx, _noop)
        assert exc_info.value.limit == 10.0
