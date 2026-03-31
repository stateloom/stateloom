"""Budget enforcement middleware — per-session, global, and hierarchy budget checks."""

from __future__ import annotations

import json
import logging
import threading
import time
from collections.abc import Awaitable, Callable
from typing import Any

from stateloom.core.errors import StateLoomBudgetError
from stateloom.core.event import BudgetEnforcementEvent
from stateloom.core.types import BudgetAction, FailureAction
from stateloom.middleware.base import MiddlewareContext

logger = logging.getLogger("stateloom.middleware.budget")

# Store key for persisted budget config state
_STORE_KEY_BUDGET_CONFIG = "budget_config_json"

# How often to poll the store for cross-process state (seconds)
_STORE_POLL_INTERVAL = 2.0


class BudgetEnforcer:
    """Enforces per-session, global, and hierarchical (team/org) budget limits.

    Pre-call: blocks if already over budget (raises StateLoomBudgetError
    in hard_stop mode, or logs warning in warn mode).

    Post-call: after CostTracker (downstream) applies cost, re-checks
    session, global, and hierarchy budgets. If a budget was just crossed,
    records a BudgetEnforcementEvent for immediate visibility. The call's
    response is still returned (already paid for); the *next* call
    will be blocked by the pre-call check.
    """

    def __init__(
        self,
        config: Any = None,
        hierarchy_budget_check: (
            Callable[[str, str], tuple[float | None, float, str] | None] | None
        ) = None,
        store: Any = None,
    ) -> None:
        self._config = config
        self._hierarchy_budget_check = hierarchy_budget_check
        self._store = store
        self._last_store_poll: float = 0.0

        # Global spend accumulator — seeded from the store, incremented per call.
        # Lock: single _global_lock guards _global_spend
        self._global_spend: float = 0.0
        self._global_lock = threading.Lock()
        self._global_seeded = False

    def _seed_global_spend(self) -> None:
        """Load cumulative spend from the store once on first call."""
        if self._global_seeded or not self._store:
            return
        self._global_seeded = True
        try:
            stats = self._store.get_global_stats()
            total = float(stats.get("total_cost", 0.0))
            with self._global_lock:
                self._global_spend = total
            logger.debug("Global budget: seeded spend=%.6f from store", total)
        except Exception:
            logger.debug("Failed to seed global spend from store", exc_info=True)

    @property
    def global_spend(self) -> float:
        """Current cumulative global spend (thread-safe read)."""
        with self._global_lock:
            return self._global_spend

    def _sync_from_store(self) -> None:
        """Poll persisted budget config from the store (cross-process sync)."""
        if not self._store or not self._config:
            return
        now = time.monotonic()
        if now - self._last_store_poll < _STORE_POLL_INTERVAL:
            return
        self._last_store_poll = now
        try:
            blob = self._store.get_secret(_STORE_KEY_BUDGET_CONFIG)
            if not blob:
                return
            data = json.loads(blob)
            if "budget_per_session" in data:
                val = data["budget_per_session"]
                self._config.budget_per_session = val
            if "budget_global" in data:
                val = data["budget_global"]
                self._config.budget_global = val
            if "budget_action" in data:
                self._config.budget_action = BudgetAction(data["budget_action"])
        except Exception:
            logger.debug("Failed to sync budget config from store", exc_info=True)

    async def process(
        self,
        ctx: MiddlewareContext,
        call_next: Callable[[MiddlewareContext], Awaitable[Any]],
    ) -> Any:
        self._seed_global_spend()
        self._sync_from_store()
        # Subscription sessions pay a flat fee — skip budget enforcement
        if ctx.session.billing_mode == "subscription":
            return await call_next(ctx)

        try:
            return await self._do_process(ctx, call_next)
        except (StateLoomBudgetError, KeyboardInterrupt, SystemExit):
            raise
        except Exception as exc:
            failure_action = ctx.config.budget_on_middleware_failure
            if failure_action == FailureAction.BLOCK:
                logger.error(f"[StateLoom] Budget enforcer failed, blocking: {exc}")
                raise
            logger.warning(f"[StateLoom] Budget enforcer failed, passing through: {exc}")
            return await call_next(ctx)

    async def _do_process(
        self,
        ctx: MiddlewareContext,
        call_next: Callable[[MiddlewareContext], Awaitable[Any]],
    ) -> Any:
        budget = ctx.session.budget
        if budget is not None:
            spent = ctx.session.total_cost

            if spent >= budget:
                event = BudgetEnforcementEvent(
                    session_id=ctx.session.id,
                    step=ctx.session.step_counter,
                    limit=budget,
                    spent=spent,
                    action=ctx.config.budget_action.value,
                    budget_level="session",
                )
                ctx.events.append(event)

                if ctx.config.budget_action == BudgetAction.HARD_STOP:
                    self._save_events_directly(ctx)
                    raise StateLoomBudgetError(
                        limit=budget,
                        spent=spent,
                        session_id=ctx.session.id,
                    )
                logger.warning(
                    f"[StateLoom] Budget warning: ${spent:.4f} spent, "
                    f"limit is ${budget:.4f} (session '{ctx.session.id}')"
                )

        # Check global budget
        self._pre_call_global_check(ctx)

        # Check team/org budgets if hierarchy callback is available
        self._pre_call_hierarchy_check(ctx)

        cost_before = ctx.session.total_cost
        result = await call_next(ctx)
        cost_after = ctx.session.total_cost

        # Accumulate this call's cost into global spend
        call_cost = cost_after - cost_before
        if call_cost > 0:
            with self._global_lock:
                self._global_spend += call_cost

        # Post-call: CostTracker (downstream) has applied cost to session
        # and team/org accumulators. Re-check so budget crossings are
        # recorded on the call that caused them, not the next attempt.
        self._post_call_session_check(ctx)
        self._post_call_global_check(ctx)
        self._post_call_hierarchy_check(ctx)

        return result

    def _pre_call_global_check(self, ctx: MiddlewareContext) -> None:
        """Pre-call: block or warn if global budget already exceeded."""
        global_limit = getattr(self._config, "budget_global", None)
        if global_limit is None:
            return
        with self._global_lock:
            spent = self._global_spend
        if spent < global_limit:
            return

        event = BudgetEnforcementEvent(
            session_id=ctx.session.id,
            step=ctx.session.step_counter,
            limit=global_limit,
            spent=spent,
            action=ctx.config.budget_action.value,
            budget_level="global",
        )
        ctx.events.append(event)

        if ctx.config.budget_action == BudgetAction.HARD_STOP:
            self._save_events_directly(ctx)
            raise StateLoomBudgetError(
                limit=global_limit,
                spent=spent,
                session_id=ctx.session.id,
            )
        logger.warning(
            "[StateLoom] Global budget warning: $%.4f spent, limit is $%.4f",
            spent,
            global_limit,
        )

    def _post_call_global_check(self, ctx: MiddlewareContext) -> None:
        """Post-call: record event if global budget was just crossed."""
        global_limit = getattr(self._config, "budget_global", None)
        if global_limit is None:
            return
        with self._global_lock:
            spent = self._global_spend
        if spent < global_limit:
            return

        event = BudgetEnforcementEvent(
            session_id=ctx.session.id,
            step=ctx.session.step_counter,
            limit=global_limit,
            spent=spent,
            action=ctx.config.budget_action.value,
            budget_level="global",
        )
        ctx.events.append(event)
        self._save_event_directly(event)
        logger.warning(
            "[StateLoom] Global budget crossed: $%.4f spent, limit $%.4f. "
            "Next call will be blocked.",
            spent,
            global_limit,
        )

    def _call_hierarchy_check(self, ctx: MiddlewareContext) -> tuple[float, float, str] | None:
        """Run the hierarchy callback and extract (limit, spent, level)."""
        if not self._hierarchy_budget_check:
            return None
        result = self._hierarchy_budget_check(ctx.session.org_id, ctx.session.team_id)
        if result is None:
            return None
        budget_limit = result[0]
        hierarchy_spent = result[1]
        level = result[2] if len(result) > 2 else "team"
        if budget_limit is not None and hierarchy_spent >= budget_limit:
            return (budget_limit, hierarchy_spent, level)
        return None

    def _pre_call_hierarchy_check(self, ctx: MiddlewareContext) -> None:
        """Pre-call: block or warn if hierarchy budget already exceeded."""
        check = self._call_hierarchy_check(ctx)
        if check is None:
            return
        budget_limit, hierarchy_spent, level = check
        event = BudgetEnforcementEvent(
            session_id=ctx.session.id,
            step=ctx.session.step_counter,
            limit=budget_limit,
            spent=hierarchy_spent,
            action=ctx.config.budget_action.value,
            budget_level=level,
        )
        ctx.events.append(event)

        if ctx.config.budget_action == BudgetAction.HARD_STOP:
            self._save_events_directly(ctx)
            raise StateLoomBudgetError(
                limit=budget_limit,
                spent=hierarchy_spent,
                session_id=ctx.session.id,
            )
        logger.warning(
            f"[StateLoom] Hierarchy budget warning ({level}): "
            f"${hierarchy_spent:.4f} spent, "
            f"limit is ${budget_limit:.4f} "
            f"(session '{ctx.session.id}')"
        )

    def _post_call_session_check(self, ctx: MiddlewareContext) -> None:
        """Record event if session budget was just crossed by this call."""
        budget = ctx.session.budget
        if budget is None:
            return
        spent = ctx.session.total_cost
        if spent >= budget:
            event = BudgetEnforcementEvent(
                session_id=ctx.session.id,
                step=ctx.session.step_counter,
                limit=budget,
                spent=spent,
                action=ctx.config.budget_action.value,
                budget_level="session",
            )
            ctx.events.append(event)
            self._save_event_directly(event)
            logger.warning(
                f"[StateLoom] Budget crossed (session): ${spent:.4f} spent, "
                f"limit ${budget:.4f} (session '{ctx.session.id}'). "
                f"Next call will be blocked."
            )

    def _post_call_hierarchy_check(self, ctx: MiddlewareContext) -> None:
        """Record event if team/org budget was just crossed by this call."""
        check = self._call_hierarchy_check(ctx)
        if check is None:
            return
        budget_limit, hierarchy_spent, level = check
        event = BudgetEnforcementEvent(
            session_id=ctx.session.id,
            step=ctx.session.step_counter,
            limit=budget_limit,
            spent=hierarchy_spent,
            action=ctx.config.budget_action.value,
            budget_level=level,
        )
        ctx.events.append(event)
        self._save_event_directly(event)
        logger.warning(
            f"[StateLoom] Hierarchy budget crossed ({level}): "
            f"${hierarchy_spent:.4f} spent, "
            f"limit ${budget_limit:.4f} "
            f"(session '{ctx.session.id}'). "
            f"Next call will be blocked."
        )

    def _save_event_directly(self, event: BudgetEnforcementEvent) -> None:
        """Persist a single event directly to the store.

        Post-call events are appended after EventRecorder (downstream) has
        already flushed, so they must be saved here to appear in the dashboard.
        """
        if not self._store:
            return
        try:
            self._store.save_event(event)
        except Exception:
            logger.debug("Failed to persist post-call budget event", exc_info=True)

    def _save_events_directly(self, ctx: MiddlewareContext) -> None:
        """Persist events directly to the store (bypass EventRecorder).

        When BudgetEnforcer raises StateLoomBudgetError, EventRecorder
        (which is downstream in the pipeline) never runs. This method
        ensures budget_enforcement events are persisted before the raise.
        """
        if not self._store:
            return
        for event in ctx.events:
            try:
                self._store.save_event(event)
            except Exception:
                logger.debug("Failed to persist budget event", exc_info=True)
        try:
            self._store.save_session(ctx.session)
        except Exception:
            logger.debug("Failed to persist session after budget block", exc_info=True)
