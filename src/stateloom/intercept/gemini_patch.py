"""Google Gemini client monkey-patch — sync, async, and streaming.

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

logger = logging.getLogger("stateloom.intercept.gemini")


def patch_gemini(gate: Gate) -> list[str]:
    """Patch Gemini client methods. Returns list of patched method descriptions.

    Delegates to the generic interceptor with the Gemini adapter.
    """
    from stateloom.intercept.adapters.gemini_adapter import GeminiAdapter
    from stateloom.intercept.generic_interceptor import patch_provider

    return patch_provider(gate, GeminiAdapter())


def _extract_model(instance: Any, kwargs: dict[str, Any]) -> str:
    """Extract model name from Gemini instance.

    Gemini stores the model name on the instance (instance.model_name),
    unlike OpenAI/Anthropic which pass it in kwargs.
    """
    return getattr(instance, "model_name", None) or kwargs.get("model", "unknown")


def _extract_tokens_from_response(response: Any) -> tuple[int, int, int]:
    """Extract token counts from a Gemini response."""
    try:
        usage = response.usage_metadata
        if usage:
            prompt = usage.prompt_token_count or 0
            completion = usage.candidates_token_count or 0
            total = usage.total_token_count or 0
            return (prompt, completion, total)
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
    """Intercept a sync Gemini call through the middleware pipeline."""
    model = _extract_model(instance, kwargs)
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
            provider=Provider.GEMINI,
            method="generate_content",
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
    """Intercept an async Gemini call through the middleware pipeline."""
    model = _extract_model(instance, kwargs)
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
            provider=Provider.GEMINI,
            method="generate_content",
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
    """Wrap a sync Gemini streaming response to capture tokens on completion.

    Note: PII scanning runs on the request (pre-call). Response stream content
    is NOT scanned for PII in real-time. For PII in responses, use non-streaming
    mode or implement post-stream scanning in your application.
    """
    from stateloom.core.event import LLMCallEvent

    total_prompt = 0
    total_completion = 0

    try:
        for chunk in stream:
            # Gemini: keep last non-None usage_metadata (final chunk has totals)
            try:
                usage = getattr(chunk, "usage_metadata", None)
                if usage:
                    total_prompt = usage.prompt_token_count or 0
                    total_completion = usage.candidates_token_count or 0
            except AttributeError:
                pass
            yield chunk
    finally:
        cost = gate.pricing.calculate_cost(model, total_prompt, total_completion)
        session.add_cost(cost, prompt_tokens=total_prompt, completion_tokens=total_completion)

        event = LLMCallEvent(
            session_id=session.id,
            step=step,
            provider=Provider.GEMINI,
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
    gate: Gate, stream: Any, session: Any, model: str, step: int, kwargs: dict
) -> Any:
    """Wrap an async Gemini streaming response to capture tokens on completion.

    Note: PII scanning runs on the request (pre-call). Response stream content
    is NOT scanned for PII in real-time. For PII in responses, use non-streaming
    mode or implement post-stream scanning in your application.
    """
    from stateloom.core.event import LLMCallEvent

    total_prompt = 0
    total_completion = 0

    try:
        async for chunk in stream:
            try:
                usage = getattr(chunk, "usage_metadata", None)
                if usage:
                    total_prompt = usage.prompt_token_count or 0
                    total_completion = usage.candidates_token_count or 0
            except AttributeError:
                pass
            yield chunk
    finally:
        cost = gate.pricing.calculate_cost(model, total_prompt, total_completion)
        session.add_cost(cost, prompt_tokens=total_prompt, completion_tokens=total_completion)

        event = LLMCallEvent(
            session_id=session.id,
            step=step,
            provider=Provider.GEMINI,
            model=model,
            prompt_tokens=total_prompt,
            completion_tokens=total_completion,
            total_tokens=total_prompt + total_completion,
            cost=cost,
            is_streaming=True,
            request_hash="",
        )
        gate.store.save_event(event)
