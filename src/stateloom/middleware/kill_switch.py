"""Kill switch middleware — global emergency stop for all LLM traffic."""

from __future__ import annotations

import json
import logging
import threading
import time
from collections.abc import Awaitable, Callable
from fnmatch import fnmatch
from typing import TYPE_CHECKING, Any

from stateloom.core.config import KillSwitchRule, StateLoomConfig
from stateloom.core.errors import StateLoomKillSwitchError
from stateloom.core.event import KillSwitchEvent
from stateloom.middleware.base import MiddlewareContext

if TYPE_CHECKING:
    from stateloom.observability.collector import MetricsCollector

logger = logging.getLogger("stateloom.middleware.kill_switch")

# Store keys for persisted kill switch state
_STORE_KEY_ACTIVE = "kill_switch_active"
_STORE_KEY_MESSAGE = "kill_switch_message"
_STORE_KEY_RESPONSE_MODE = "kill_switch_response_mode"
_STORE_KEY_RULES = "kill_switch_rules_json"

# How often to poll the store for cross-process state (seconds)
_STORE_POLL_INTERVAL = 2.0


class KillSwitchMiddleware:
    """Global kill switch — blocks ALL LLM traffic when active.

    Sits at position 0 in the middleware chain. When active, it short-circuits
    the entire pipeline without calling call_next, preventing shadow calls,
    PII scanning, budget checks, and any other downstream middleware from running.

    Supports granular rules that match on model (glob), provider, environment,
    and agent_version. A rule must have at least one non-None filter to match.

    Kill switch state is persisted to the store so it propagates across
    processes sharing the same store (e.g. SQLite).
    """

    def __init__(
        self,
        config: StateLoomConfig,
        store: Any = None,
        metrics: MetricsCollector | None = None,
    ) -> None:
        self._config = config
        self._store = store
        self._metrics = metrics
        self._last_store_poll: float = 0.0
        # Snapshot config at init time so _sync_from_store only applies
        # changes made *externally* (dashboard API), never clobbering
        # in-memory state set by the local process via stateloom.kill_switch().
        self._last_known_active: bool = config.kill_switch_active

    def _sync_from_store(self) -> None:
        """Poll persisted kill switch state from the store.

        Called at most once per _STORE_POLL_INTERVAL seconds to avoid
        hitting the store on every request.

        Only applies store state when the in-memory config has NOT been
        changed locally since the last sync.  This prevents the store
        (which may hold stale data from a previous process) from
        clobbering a programmatic ``stateloom.kill_switch(active=True)``.
        """
        if not self._store:
            return
        now = time.monotonic()
        if now - self._last_store_poll < _STORE_POLL_INTERVAL:
            return
        self._last_store_poll = now

        # If the in-memory value was changed since our last sync (by the
        # local process calling stateloom.kill_switch()), respect that —
        # don't let a stale store value overwrite it.
        if self._config.kill_switch_active != self._last_known_active:
            self._last_known_active = self._config.kill_switch_active
            return

        try:
            active = self._store.get_secret(_STORE_KEY_ACTIVE)
            if active:
                self._config.kill_switch_active = active == "1"
                self._last_known_active = self._config.kill_switch_active
            message = self._store.get_secret(_STORE_KEY_MESSAGE)
            if message:
                self._config.kill_switch_message = message
            response_mode = self._store.get_secret(_STORE_KEY_RESPONSE_MODE)
            if response_mode:
                self._config.kill_switch_response_mode = response_mode
            rules_json = self._store.get_secret(_STORE_KEY_RULES)
            if rules_json:
                rules_data = json.loads(rules_json)
                self._config.kill_switch_rules = [KillSwitchRule(**r) for r in rules_data]
        except Exception:
            logger.debug("Failed to sync kill switch state from store", exc_info=True)

    async def process(
        self,
        ctx: MiddlewareContext,
        call_next: Callable[[MiddlewareContext], Awaitable[Any]],
    ) -> Any:
        # Sync state from store (cross-process propagation)
        self._sync_from_store()

        # Check granular rules first
        matched_rule = self._check_rules(ctx)

        if not self._config.kill_switch_active and matched_rule is None:
            return await call_next(ctx)

        if matched_rule is not None:
            logger.warning(
                "Kill switch: rule matched for session=%s model=%s provider=%s rule=%s",
                ctx.session.id,
                ctx.model,
                ctx.provider,
                matched_rule.model_dump(exclude_none=True),
            )
        else:
            logger.warning(
                "Kill switch: global kill switch active, blocking session=%s model=%s",
                ctx.session.id,
                ctx.model,
            )

        # Determine message: rule-specific overrides global
        if matched_rule is not None:
            message = matched_rule.message or self._config.kill_switch_message
            reason = matched_rule.reason or "rule_matched"
            rule_dict = matched_rule.model_dump(exclude_none=True)
        else:
            message = self._config.kill_switch_message
            reason = "kill_switch_active"
            rule_dict = {}

        webhook_url = self._config.kill_switch_webhook_url

        event = KillSwitchEvent(
            session_id=ctx.session.id,
            step=ctx.session.step_counter,
            reason=reason,
            message=message,
            matched_rule=rule_dict,
            blocked_model=ctx.model,
            blocked_provider=ctx.provider,
            webhook_fired=bool(webhook_url),
            webhook_url=webhook_url,
        )
        ctx.events.append(event)

        response_mode = self._config.kill_switch_response_mode

        if response_mode == "response":
            # Return a static response instead of raising
            ctx.skip_call = True
            ctx.cached_response = {"kill_switch": True, "message": message}
            # Fire webhook in daemon thread (non-blocking, fail-open)
            if webhook_url:
                threading.Thread(
                    target=self._fire_webhook,
                    args=(
                        webhook_url,
                        ctx.session.id,
                        reason,
                        message,
                        ctx.model,
                        ctx.provider,
                        rule_dict,
                    ),
                    daemon=True,
                ).start()
            return await call_next(ctx)

        # Default: raise error
        self._save_events_directly(ctx)
        if self._metrics is not None:
            try:
                self._metrics.record_kill_switch_block(
                    org_id=ctx.session.org_id or "",
                    team_id=ctx.session.team_id or "",
                )
            except Exception:
                pass
        # Fire webhook in daemon thread (non-blocking, fail-open)
        if webhook_url:
            threading.Thread(
                target=self._fire_webhook,
                args=(
                    webhook_url,
                    ctx.session.id,
                    reason,
                    message,
                    ctx.model,
                    ctx.provider,
                    rule_dict,
                ),
                daemon=True,
            ).start()
        raise StateLoomKillSwitchError(message, model=ctx.model, provider=ctx.provider)

    def _check_rules(self, ctx: MiddlewareContext) -> KillSwitchRule | None:
        """Check granular rules against the current request context.

        Returns the first matching rule, or None if no rule matches.
        A rule must have at least one non-None filter field to be eligible.
        """
        for rule in self._config.kill_switch_rules:
            if self._rule_matches(rule, ctx):
                return rule
        return None

    def _rule_matches(self, rule: KillSwitchRule, ctx: MiddlewareContext) -> bool:
        """Check if a single rule matches the current context.

        A rule with no filters (all None/empty) matches nothing.
        Each non-None filter must match for the rule to match.
        """
        has_filter = False

        if rule.model is not None:
            has_filter = True
            if not fnmatch(ctx.model, rule.model):
                return False

        if rule.provider is not None:
            has_filter = True
            if not fnmatch(ctx.provider, rule.provider):
                return False

        if rule.environment is not None:
            has_filter = True
            if self._config.kill_switch_environment != rule.environment:
                return False

        if rule.agent_version is not None:
            has_filter = True
            if self._config.kill_switch_agent_version != rule.agent_version:
                return False

        return has_filter

    def _save_events_directly(self, ctx: MiddlewareContext) -> None:
        """Persist events directly to the store (bypass EventRecorder).

        Exception after this call prevents EventRecorder from running — no
        duplicate risk.
        """
        if not self._store:
            return
        for event in ctx.events:
            try:
                self._store.save_event(event)
            except Exception:
                logger.debug("Failed to persist kill switch event", exc_info=True)
        try:
            self._store.save_session(ctx.session)
        except Exception:
            logger.debug("Failed to persist session after kill switch", exc_info=True)

    @staticmethod
    def _fire_webhook(
        url: str,
        session_id: str,
        reason: str,
        message: str,
        blocked_model: str,
        blocked_provider: str,
        matched_rule: dict[str, Any],
    ) -> None:
        """Fire a webhook notification (background thread, fail-open)."""
        try:
            import httpx

            payload = {
                "event": "kill_switch",
                "session_id": session_id,
                "reason": reason,
                "message": message,
                "blocked_model": blocked_model,
                "blocked_provider": blocked_provider,
                "matched_rule": matched_rule,
            }
            with httpx.Client(timeout=10.0) as client:
                client.post(url, json=payload)
        except Exception:
            logger.debug("Kill switch webhook failed", exc_info=True)
