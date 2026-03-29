"""Anthropic client monkey-patch — sync, async, and streaming.

.. deprecated::
    Legacy shim.  Canonical logic in ``generic_interceptor``.
    Kept for backward compatibility.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from stateloom.core.context import get_current_replay_engine
from stateloom.core.types import Provider
from stateloom.intercept.unpatch import register_patch

if TYPE_CHECKING:
    from stateloom.gate import Gate

logger = logging.getLogger("stateloom.intercept.anthropic")


def patch_anthropic(gate: Gate) -> list[str]:
    """Patch Anthropic client methods. Returns list of patched method descriptions.

    Delegates to the generic interceptor with the Anthropic adapter.
    """
    from stateloom.intercept.adapters.anthropic_adapter import AnthropicAdapter
    from stateloom.intercept.generic_interceptor import patch_provider

    return patch_provider(gate, AnthropicAdapter())


def _extract_model(kwargs: dict[str, Any]) -> str:
    """Extract model name from Anthropic request kwargs."""
    return kwargs.get("model", "unknown")


def _extract_tokens_from_response(response: Any) -> tuple[int, int, int]:
    """Extract token counts from an Anthropic response."""
    try:
        usage = response.usage
        if usage:
            prompt = usage.input_tokens or 0
            completion = usage.output_tokens or 0
            return (prompt, completion, prompt + completion)
    except AttributeError:
        pass
    return (0, 0, 0)


def _check_replay(gate: Gate, step: int) -> Any | None:
    """Check if the replay engine wants to mock this step.

    Returns the cached response if mocked, or None to proceed with the live call.
    """
    engine = get_current_replay_engine()
    if engine is not None and engine.is_active and engine.should_mock(step):
        logger.debug("Replay: returning cached response for step %d", step)
        return engine.get_cached_response(step)
    return None


def _intercept_sync(
    gate: Gate, original: Any, instance: Any, args: tuple, kwargs: dict[str, Any]
) -> Any:
    """Intercept a sync Anthropic call through the middleware pipeline."""
    model = _extract_model(kwargs)
    is_streaming = kwargs.get("stream", False)

    try:
        session = gate.get_or_create_session()
        step = session.next_step()

        # Check replay engine before making the live call
        cached = _check_replay(gate, step)
        if cached is not None:
            return cached

        if is_streaming:
            result = original(instance, *args, **kwargs)
            return _wrap_stream_sync(gate, result, session, model, step, kwargs)

        result = gate.pipeline.execute_sync(
            provider=Provider.ANTHROPIC,
            method="messages.create",
            model=model,
            request_kwargs=kwargs,
            session=session,
            config=gate.config,
            llm_call=lambda: original(instance, *args, **kwargs),
        )

        if result is not None:
            pt, ct, tt = _extract_tokens_from_response(result)
            session.add_cost(
                gate.pricing.calculate_cost(model, pt, ct),
                prompt_tokens=pt,
                completion_tokens=ct,
            )

        return result

    except Exception:
        if gate.config.fail_open:
            logger.error("Middleware error, failing open", exc_info=True)
            return original(instance, *args, **kwargs)
        raise


async def _intercept_async(
    gate: Gate, original: Any, instance: Any, args: tuple, kwargs: dict[str, Any]
) -> Any:
    """Intercept an async Anthropic call through the middleware pipeline."""
    model = _extract_model(kwargs)
    is_streaming = kwargs.get("stream", False)

    try:
        session = gate.get_or_create_session()
        step = session.next_step()

        # Check replay engine before making the live call
        cached = _check_replay(gate, step)
        if cached is not None:
            return cached

        if is_streaming:
            result = await original(instance, *args, **kwargs)
            return _wrap_stream_async(gate, result, session, model, step, kwargs)

        result = await gate.pipeline.execute_async(
            provider=Provider.ANTHROPIC,
            method="messages.create",
            model=model,
            request_kwargs=kwargs,
            session=session,
            config=gate.config,
            llm_call=lambda: original(instance, *args, **kwargs),
        )

        if result is not None:
            pt, ct, tt = _extract_tokens_from_response(result)
            session.add_cost(
                gate.pricing.calculate_cost(model, pt, ct),
                prompt_tokens=pt,
                completion_tokens=ct,
            )

        return result

    except Exception:
        if gate.config.fail_open:
            logger.error("Middleware error, failing open", exc_info=True)
            return await original(instance, *args, **kwargs)
        raise


def _wrap_stream_sync(
    gate: Gate, stream: Any, session: Any, model: str, step: int, kwargs: dict
) -> Any:
    """Wrap a sync Anthropic streaming response.

    Note: PII scanning runs on the request (pre-call). Response stream content
    is NOT scanned for PII in real-time. For PII in responses, use non-streaming
    mode or implement post-stream scanning in your application.
    """
    from stateloom.core.event import LLMCallEvent

    total_prompt = 0
    total_completion = 0

    try:
        for event in stream:
            # Anthropic streams emit message_start with usage
            try:
                if hasattr(event, "type"):
                    if event.type == "message_start" and hasattr(event, "message"):
                        usage = getattr(event.message, "usage", None)
                        if usage:
                            total_prompt = usage.input_tokens or 0
                    elif event.type == "message_delta":
                        usage = getattr(event, "usage", None)
                        if usage:
                            total_completion = usage.output_tokens or 0
            except AttributeError:
                pass
            yield event
    finally:
        cost = gate.pricing.calculate_cost(model, total_prompt, total_completion)
        session.add_cost(cost, prompt_tokens=total_prompt, completion_tokens=total_completion)

        event_obj = LLMCallEvent(
            session_id=session.id,
            step=step,
            provider=Provider.ANTHROPIC,
            model=model,
            prompt_tokens=total_prompt,
            completion_tokens=total_completion,
            total_tokens=total_prompt + total_completion,
            cost=cost,
            is_streaming=True,
            request_hash="",
        )
        gate.store.save_event(event_obj)


async def _wrap_stream_async(
    gate: Gate, stream: Any, session: Any, model: str, step: int, kwargs: dict
) -> Any:
    """Wrap an async Anthropic streaming response.

    Note: PII scanning runs on the request (pre-call). Response stream content
    is NOT scanned for PII in real-time. For PII in responses, use non-streaming
    mode or implement post-stream scanning in your application.
    """
    from stateloom.core.event import LLMCallEvent

    total_prompt = 0
    total_completion = 0

    try:
        async for event in stream:
            try:
                if hasattr(event, "type"):
                    if event.type == "message_start" and hasattr(event, "message"):
                        usage = getattr(event.message, "usage", None)
                        if usage:
                            total_prompt = usage.input_tokens or 0
                    elif event.type == "message_delta":
                        usage = getattr(event, "usage", None)
                        if usage:
                            total_completion = usage.output_tokens or 0
            except AttributeError:
                pass
            yield event
    finally:
        cost = gate.pricing.calculate_cost(model, total_prompt, total_completion)
        session.add_cost(cost, prompt_tokens=total_prompt, completion_tokens=total_completion)

        event_obj = LLMCallEvent(
            session_id=session.id,
            step=step,
            provider=Provider.ANTHROPIC,
            model=model,
            prompt_tokens=total_prompt,
            completion_tokens=total_completion,
            total_tokens=total_prompt + total_completion,
            cost=cost,
            is_streaming=True,
            request_hash="",
        )
        gate.store.save_event(event_obj)
