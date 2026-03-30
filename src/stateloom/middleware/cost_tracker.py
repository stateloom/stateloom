"""Cost tracking middleware — token counting and cost calculation."""

from __future__ import annotations

import logging
import re
from collections.abc import Awaitable, Callable
from typing import Any

logger = logging.getLogger("stateloom.middleware.cost_tracker")

from stateloom.core.event import CacheHitEvent, LLMCallEvent
from stateloom.middleware.base import MiddlewareContext
from stateloom.pricing.registry import PricingRegistry

_SYSTEM_REMINDER_RE = re.compile(r"<system-reminder>.*?</system-reminder>", re.DOTALL)


def _extract_prompt_preview(messages: list[dict]) -> str:
    """Extract a short preview of the last user message, stripping system-reminder tags."""
    for msg in reversed(messages):
        if msg.get("role") != "user":
            continue
        content = msg.get("content", "")
        if isinstance(content, str):
            text = _SYSTEM_REMINDER_RE.sub("", content).strip()
        elif isinstance(content, list):
            text = ""
            for part in reversed(content):
                if isinstance(part, dict) and part.get("type") == "text":
                    candidate = _SYSTEM_REMINDER_RE.sub("", part.get("text", "")).strip()
                    if candidate:
                        text = candidate
                        break
        else:
            continue
        if text:
            return (text[:50] + "...") if len(text) > 50 else text
    return ""


def _extract_tool_preview(messages: list[dict]) -> str:
    """Extract a preview for tool-continuation calls.

    Shows the last assistant's tool call names and reasoning text instead of
    the original user prompt (which is the same for all tool continuations).
    """
    # Walk backwards to find the last assistant message with tool_calls
    for msg in reversed(messages):
        if msg.get("role") != "assistant":
            continue

        tool_calls = msg.get("tool_calls", [])
        tool_names = []
        for tc in tool_calls:
            fn = tc.get("function", {})
            name = fn.get("name", "") if isinstance(fn, dict) else ""
            if name:
                tool_names.append(name)

        # Also check Anthropic tool_use content blocks
        content = msg.get("content")
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("type") == "tool_use":
                    name = block.get("name", "")
                    if name:
                        tool_names.append(name)

        if tool_names:
            names_str = ", ".join(tool_names)
            # Include reasoning text if available
            text = ""
            if isinstance(content, str) and content:
                text = content.strip()
            elif isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "text":
                        text = block.get("text", "").strip()
                        if text:
                            break

            if text:
                preview = f"{names_str}: {text}"
            else:
                preview = names_str
            return (preview[:50] + "...") if len(preview) > 50 else preview

    return ""


def _is_tool_continuation(messages: list[dict]) -> bool:
    """Check if the **last** message contains tool results (tool-use continuation).

    Only the last message matters — earlier tool results in conversation history
    don't make a new user prompt into a continuation.  Scanning all messages
    causes false positives in multi-turn conversations where old tool results
    remain in history.
    """
    if not messages:
        return False
    last = messages[-1]
    # OpenAI format: role="tool" messages carry tool results
    if last.get("role") == "tool":
        return True
    # Anthropic format: user messages with tool_result content blocks
    content = last.get("content")
    if isinstance(content, list):
        for block in content:
            if isinstance(block, dict) and block.get("type") == "tool_result":
                return True
    return False


class CostTracker:
    """Tracks token usage and cost for each LLM call."""

    def __init__(
        self,
        pricing: PricingRegistry,
        cost_callback: Callable[[str, str, float, int], None] | None = None,
    ) -> None:
        self._pricing = pricing
        self._cost_callback = cost_callback

    async def process(
        self,
        ctx: MiddlewareContext,
        call_next: Callable[[MiddlewareContext], Awaitable[Any]],
    ) -> Any:
        result = await call_next(ctx)

        if ctx.is_streaming:
            ctx._on_stream_complete.append(lambda: self._track_cost(ctx))
        else:
            self._track_cost(ctx)

        return result

    def _track_cost(self, ctx: MiddlewareContext) -> None:
        """Shared cost tracking logic for both streaming and non-streaming."""
        # Cache hits are fully handled by CacheMiddleware — skip
        if any(isinstance(e, CacheHitEvent) for e in ctx.events):
            return

        # Use tokens already set by upstream middleware (e.g. auto-router)
        # before falling back to response extraction
        if ctx.prompt_tokens > 0 or ctx.completion_tokens > 0:
            prompt_tokens = ctx.prompt_tokens
            completion_tokens = ctx.completion_tokens
        else:
            prompt_tokens, completion_tokens = self._extract_tokens(ctx)
            ctx.prompt_tokens = prompt_tokens
            ctx.completion_tokens = completion_tokens

        ctx.total_tokens = prompt_tokens + completion_tokens

        # Calculate cost — dual tracking for subscription billing
        api_cost = self._pricing.calculate_cost(ctx.model, prompt_tokens, completion_tokens)
        billing_mode = ctx.session.metadata.get("billing_mode", "api")
        actual_cost = 0.0 if billing_mode == "subscription" else api_cost

        # Extract prompt preview and detect CLI internal requests
        messages = ctx.request_kwargs.get("messages", [])
        prompt_preview = ""
        is_cli_internal = False
        if isinstance(messages, list):
            try:
                prompt_preview = _extract_prompt_preview(messages)
            except Exception:
                logger.debug("Prompt preview extraction failed")
            # Explicit flag set by proxy handlers (e.g. Code Assist detecting
            # user_prompt_id="session-summary-generation")
            if ctx.request_kwargs.get("_cli_internal"):
                is_cli_internal = True

        # Detect tool-use continuations (responses to tool results)
        is_tool_continuation = (
            _is_tool_continuation(messages) if isinstance(messages, list) else False
        )

        # For tool continuations, show the tool call names + reasoning
        # instead of the original user prompt (which is the same for all).
        if is_tool_continuation and isinstance(messages, list):
            try:
                tool_preview = _extract_tool_preview(messages)
                if tool_preview:
                    prompt_preview = tool_preview
            except Exception:
                logger.debug("Tool preview extraction failed")

        # Track whether the session has ONLY cli-internal calls.
        # First non-internal call clears the flag.
        if is_cli_internal:
            ctx.session.metadata.setdefault("_cli_internal_only", True)
        else:
            ctx.session.metadata.pop("_cli_internal_only", None)

        # Capture user_prompt_id for dashboard grouping (Gemini CLI sends
        # the same ID for all calls belonging to a single user prompt,
        # including subagent delegations and tool continuations).
        user_prompt_id = ctx.request_kwargs.get("_user_prompt_id", "")

        # Create the LLM call event
        event = LLMCallEvent(
            session_id=ctx.session.id,
            step=ctx.session.step_counter,
            provider=ctx.provider,
            model=ctx.model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            cost=actual_cost,
            estimated_api_cost=api_cost,
            latency_ms=ctx.latency_ms,
            is_streaming=ctx.is_streaming,
            request_hash=ctx.request_hash,
            prompt_preview=prompt_preview,
            is_cli_internal=is_cli_internal,
            is_tool_continuation=is_tool_continuation,
        )
        if user_prompt_id:
            event.metadata["user_prompt_id"] = user_prompt_id

        # Flow framework context into event metadata for dashboard visibility
        if ctx._framework_context:
            event.metadata.update(ctx._framework_context)

        # Store response payload for durable session replay
        if ctx.session.durable:
            if ctx._durable_cached_json is not None:
                # Stream — chunks already serialized by stream wrapper
                event.cached_response_json = ctx._durable_cached_json
            elif ctx.response is not None:
                # Non-stream — serialize complete response
                try:
                    from stateloom.replay.schema import serialize_response

                    event.cached_response_json = serialize_response(ctx.response)
                except Exception:
                    logger.warning(
                        "Durable serialization failed for session '%s' step %d "
                        "— step will re-execute on resume",
                        ctx.session.id,
                        ctx.session.step_counter,
                        exc_info=True,
                    )

        ctx.events.append(event)

        logger.debug(
            "Cost tracked: session=%s model=%s tokens=%d+%d cost=%.6f api_cost=%.6f%s",
            ctx.session.id, ctx.model, prompt_tokens, completion_tokens,
            actual_cost, api_cost,
            " (tool_continuation)" if is_tool_continuation else "",
        )

        # Update session accumulators
        ctx.session.add_cost(
            actual_cost,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            estimated_api_cost=api_cost,
            model=ctx.model,
        )

        # Propagate cost to org/team accumulators
        if self._cost_callback:
            self._cost_callback(
                ctx.session.org_id,
                ctx.session.team_id,
                actual_cost,
                prompt_tokens + completion_tokens,
            )

    def _extract_tokens(self, ctx: MiddlewareContext) -> tuple[int, int]:
        """Extract token counts from response based on provider."""
        response = ctx.response
        if response is None:
            return (0, 0)

        # Dict responses from passthrough proxy
        if isinstance(response, dict):
            return self._extract_tokens_from_dict(response)

        # Use adapter if available
        from stateloom.intercept.provider_registry import get_adapter

        adapter = get_adapter(ctx.provider)
        if adapter:
            pt, ct, _tt = adapter.extract_tokens(response)
            return (pt, ct)

        # Fallback for custom providers with no registered adapter
        try:
            usage = response.usage
            if usage:
                # OpenAI format
                if hasattr(usage, "prompt_tokens"):
                    return (usage.prompt_tokens or 0, usage.completion_tokens or 0)
                # Anthropic format
                if hasattr(usage, "input_tokens"):
                    return (usage.input_tokens or 0, usage.output_tokens or 0)
        except AttributeError:
            pass

        return (0, 0)

    @staticmethod
    def _extract_tokens_from_dict(data: dict) -> tuple[int, int]:
        """Extract token counts from a dict response (passthrough proxy).

        Handles OpenAI, Anthropic, and Gemini response formats.
        """
        # OpenAI format: {"usage": {"prompt_tokens": N, "completion_tokens": N}}
        usage = data.get("usage", {})
        if isinstance(usage, dict):
            if "prompt_tokens" in usage:
                return (usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0))
            # Anthropic format: {"usage": {"input_tokens": N, "output_tokens": N}}
            if "input_tokens" in usage:
                return (usage.get("input_tokens", 0), usage.get("output_tokens", 0))

        # Gemini format: {"usageMetadata": {"promptTokenCount": N, ...}}
        meta = data.get("usageMetadata", {})
        if isinstance(meta, dict) and meta:
            return (
                meta.get("promptTokenCount", 0),
                meta.get("candidatesTokenCount", 0),
            )

        # Code Assist format: {"response": {"usageMetadata": {...}}}
        inner = data.get("response", {})
        if isinstance(inner, dict):
            inner_meta = inner.get("usageMetadata", {})
            if isinstance(inner_meta, dict) and inner_meta:
                return (
                    inner_meta.get("promptTokenCount", 0),
                    inner_meta.get("candidatesTokenCount", 0),
                )

        return (0, 0)
