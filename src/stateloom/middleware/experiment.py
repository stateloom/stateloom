"""Experiment middleware — applies variant overrides to LLM calls."""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any

from stateloom.core.types import Provider
from stateloom.middleware.base import MiddlewareContext

if TYPE_CHECKING:
    from stateloom.store.base import Store

logger = logging.getLogger("stateloom.experiment")


class ExperimentMiddleware:
    """Applies experiment variant config overrides to LLM calls.

    Reads variant config from session metadata and overrides model,
    request kwargs, and system prompt in a provider-aware manner.

    When a variant references an agent version (via ``agent_version_id``),
    the agent's model, system prompt, and request overrides are applied as a
    base layer — the variant's explicit overrides take priority on top.

    Fail-open: errors in override application never crash the user's chain.
    """

    def __init__(self, store: Store | None = None) -> None:
        self._store = store

    async def process(
        self,
        ctx: MiddlewareContext,
        call_next: Callable[[MiddlewareContext], Awaitable[Any]],
    ) -> Any:
        try:
            self._apply_overrides(ctx)
        except Exception:
            logger.debug("Experiment middleware override failed", exc_info=True)

        return await call_next(ctx)

    def _apply_overrides(self, ctx: MiddlewareContext) -> None:
        """Apply variant config overrides to the middleware context."""
        variant_config = ctx.session.metadata.get("experiment_variant_config")
        if not variant_config:
            return

        # --- Agent version layer (base) ---
        self._apply_agent_overrides(ctx, variant_config)

        # --- Variant layer (top, overrides agent) ---
        # Override model if variant specifies one
        model = variant_config.get("model")
        if model:
            ctx.model = model
            ctx.request_kwargs["model"] = model

        # Merge request_overrides into request_kwargs
        request_overrides = variant_config.get("request_overrides", {})
        for key, value in request_overrides.items():
            if key == "system_prompt":
                self._apply_system_prompt(ctx, value)
            else:
                ctx.request_kwargs[key] = value

    def _apply_agent_overrides(
        self, ctx: MiddlewareContext, variant_config: dict[str, Any]
    ) -> None:
        """Apply agent version overrides as the base layer."""
        # Prefer snapshot from assignment time (immutable)
        resolved = variant_config.get("_resolved_agent_overrides")
        if resolved:
            self._merge_agent_layer(ctx, variant_config, resolved)
            return

        # Fall back to live resolution for backward compat
        agent_version_id = variant_config.get("agent_version_id")
        if not agent_version_id or not self._store:
            return

        try:
            version = self._store.get_agent_version(agent_version_id)
            if version:
                resolved = {
                    "model": version.model,
                    "system_prompt": version.system_prompt,
                    "request_overrides": version.request_overrides or {},
                }
                self._merge_agent_layer(ctx, variant_config, resolved)
        except Exception:
            logger.debug("Agent version resolution failed for experiment", exc_info=True)

    def _merge_agent_layer(
        self,
        ctx: MiddlewareContext,
        variant_config: dict[str, Any],
        agent_overrides: dict[str, Any],
    ) -> None:
        """Merge agent version overrides as the base layer (variant wins on conflict)."""
        variant_request_overrides = variant_config.get("request_overrides", {})

        # Agent model as base (variant model overrides)
        agent_model = agent_overrides.get("model")
        if agent_model and not variant_config.get("model"):
            ctx.model = agent_model
            ctx.request_kwargs["model"] = agent_model

        # Agent request_overrides as base (variant overrides on top)
        agent_req_overrides = agent_overrides.get("request_overrides", {})
        for key, value in agent_req_overrides.items():
            if key not in variant_request_overrides:
                if key == "system_prompt":
                    # Only apply agent system prompt if variant doesn't override it
                    if "system_prompt" not in variant_request_overrides:
                        self._apply_system_prompt(ctx, value)
                else:
                    ctx.request_kwargs[key] = value

        # Agent system prompt if not overridden by variant
        agent_system_prompt = agent_overrides.get("system_prompt")
        if (
            agent_system_prompt
            and "system_prompt" not in variant_request_overrides
            and "system_prompt" not in agent_req_overrides
        ):
            self._apply_system_prompt(ctx, agent_system_prompt)

    def _apply_system_prompt(self, ctx: MiddlewareContext, prompt: str) -> None:
        """Apply system prompt override in a provider-aware manner."""
        from stateloom.intercept.provider_registry import get_adapter

        adapter = get_adapter(ctx.provider)
        if adapter:
            adapter.apply_system_prompt(ctx.request_kwargs, prompt)
        elif ctx.provider == Provider.ANTHROPIC:
            ctx.request_kwargs["system"] = prompt
        elif ctx.provider == Provider.GEMINI:
            ctx.request_kwargs["system_instruction"] = prompt
        else:
            # OpenAI-style (default): system prompt is a message in the messages list
            messages = ctx.request_kwargs.get("messages", [])
            if messages and messages[0].get("role") == "system":
                messages[0]["content"] = prompt
            else:
                messages.insert(0, {"role": "system", "content": prompt})
            ctx.request_kwargs["messages"] = messages
