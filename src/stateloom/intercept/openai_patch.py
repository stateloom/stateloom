"""OpenAI client monkey-patch — sync, async, and streaming.

.. deprecated::
    Legacy shim.  Canonical logic in ``generic_interceptor``.
    Kept for backward compatibility.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, cast

from stateloom.core.context import get_current_replay_engine
from stateloom.core.types import Provider

if TYPE_CHECKING:
    from stateloom.gate import Gate

logger = logging.getLogger("stateloom.intercept.openai")


def patch_openai(gate: Gate) -> list[str]:
    """Patch OpenAI client methods. Returns list of patched method descriptions.

    Delegates to the generic interceptor with the OpenAI adapter.
    """
    from stateloom.intercept.adapters.openai_adapter import OpenAIAdapter
    from stateloom.intercept.generic_interceptor import patch_provider

    return patch_provider(gate, OpenAIAdapter())


def _extract_model(kwargs: dict[str, Any]) -> str:
    """Extract model name from OpenAI request kwargs."""
    return cast(str, kwargs.get("model", "unknown"))


def _extract_tokens_from_response(response: Any) -> tuple[int, int, int]:
    """Extract token counts from an OpenAI response."""
    try:
        usage = response.usage
        if usage:
            return (
                usage.prompt_tokens or 0,
                usage.completion_tokens or 0,
                usage.total_tokens or 0,
            )
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


_REPLAY_SENTINEL = object()


def _intercept_sync(
    gate: Gate, original: Any, instance: Any, args: tuple[Any, ...], kwargs: dict[str, Any]
) -> Any:
    """Intercept a sync OpenAI call through the middleware pipeline."""
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
            # For streaming, we wrap the iterator
            result = original(instance, *args, **kwargs)
            return _wrap_stream_sync(gate, result, session, model, step, kwargs)

        result = gate.pipeline.execute_sync(
            provider=Provider.OPENAI,
            method="chat.completions.create",
            model=model,
            request_kwargs=kwargs,
            session=session,
            config=gate.config,
            llm_call=lambda: original(instance, *args, **kwargs),
        )

        # Extract tokens from response and update session
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
    gate: Gate, original: Any, instance: Any, args: tuple[Any, ...], kwargs: dict[str, Any]
) -> Any:
    """Intercept an async OpenAI call through the middleware pipeline."""
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
            provider=Provider.OPENAI,
            method="chat.completions.create",
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
    gate: Gate, stream: Any, session: Any, model: str, step: int, kwargs: dict[str, Any]
) -> Any:
    """Wrap a sync streaming response to capture tokens on completion.

    Note: PII scanning runs on the request (pre-call). Response stream content
    is NOT scanned for PII in real-time. For PII in responses, use non-streaming
    mode or implement post-stream scanning in your application.
    """
    from stateloom.core.event import LLMCallEvent

    chunks = []
    try:
        for chunk in stream:
            chunks.append(chunk)
            yield chunk
    finally:
        # Stream complete — extract final usage if available
        total_prompt = 0
        total_completion = 0
        for chunk in chunks:
            try:
                if hasattr(chunk, "usage") and chunk.usage:
                    total_prompt = chunk.usage.prompt_tokens or 0
                    total_completion = chunk.usage.completion_tokens or 0
            except AttributeError:
                pass

        cost = gate.pricing.calculate_cost(model, total_prompt, total_completion)
        session.add_cost(cost, prompt_tokens=total_prompt, completion_tokens=total_completion)

        event = LLMCallEvent(
            session_id=session.id,
            step=step,
            provider=Provider.OPENAI,
            model=model,
            prompt_tokens=total_prompt,
            completion_tokens=total_completion,
            total_tokens=total_prompt + total_completion,
            cost=cost,
            is_streaming=True,
            request_hash="",
        )
        gate.store.save_event(event)


async def _wrap_stream_async(
    gate: Gate, stream: Any, session: Any, model: str, step: int, kwargs: dict[str, Any]
) -> Any:
    """Wrap an async streaming response to capture tokens on completion.

    Note: PII scanning runs on the request (pre-call). Response stream content
    is NOT scanned for PII in real-time. For PII in responses, use non-streaming
    mode or implement post-stream scanning in your application.
    """
    from stateloom.core.event import LLMCallEvent

    chunks = []
    try:
        async for chunk in stream:
            chunks.append(chunk)
            yield chunk
    finally:
        total_prompt = 0
        total_completion = 0
        for chunk in chunks:
            try:
                if hasattr(chunk, "usage") and chunk.usage:
                    total_prompt = chunk.usage.prompt_tokens or 0
                    total_completion = chunk.usage.completion_tokens or 0
            except AttributeError:
                pass

        cost = gate.pricing.calculate_cost(model, total_prompt, total_completion)
        session.add_cost(cost, prompt_tokens=total_prompt, completion_tokens=total_completion)

        event = LLMCallEvent(
            session_id=session.id,
            step=step,
            provider=Provider.OPENAI,
            model=model,
            prompt_tokens=total_prompt,
            completion_tokens=total_completion,
            total_tokens=total_prompt + total_completion,
            cost=cost,
            is_streaming=True,
            request_hash="",
        )
        gate.store.save_event(event)
