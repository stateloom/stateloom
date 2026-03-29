"""LiteLLM provider adapter.

LiteLLM uses module-level functions (not class methods), so this adapter
handles patching directly instead of going through the generic interceptor.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from stateloom.core.context import get_current_replay_engine
from stateloom.core.errors import StateLoomError
from stateloom.core.types import Provider
from stateloom.intercept.provider_adapter import BaseProviderAdapter, PatchTarget, TokenFieldMap
from stateloom.intercept.unpatch import register_patch

if TYPE_CHECKING:
    from stateloom.gate import Gate

logger = logging.getLogger("stateloom.intercept.litellm")

_TOKEN_FIELDS = TokenFieldMap()


class LiteLLMAdapter(BaseProviderAdapter):
    """Adapter for the LiteLLM Python library.

    LiteLLM provides a unified interface to 100+ LLM providers. This adapter
    patches ``litellm.completion`` and ``litellm.acompletion`` so all calls
    flow through StateLoom's middleware pipeline.

    Users keep full access to LiteLLM's own features (caching, budgets,
    fallbacks, routing, etc.) — StateLoom layers on top.
    """

    @property
    def name(self) -> str:
        return Provider.LITELLM

    @property
    def method_label(self) -> str:
        return "completion"

    def get_patch_targets(self) -> list[PatchTarget]:
        # Patching is handled by patch_litellm(), not the generic interceptor.
        return []

    def extract_model(self, instance: Any, args: tuple, kwargs: dict[str, Any]) -> str:
        return kwargs.get("model", "unknown")

    def extract_tokens(self, response: Any) -> tuple[int, int, int]:
        return self._extract_tokens_from_fields(response, _TOKEN_FIELDS)

    def to_openai_dict(self, response: Any, model: str, request_id: str) -> dict[str, Any]:
        """LiteLLM returns OpenAI-format responses — passthrough via model_dump."""
        if hasattr(response, "model_dump"):
            result = response.model_dump()
            result["id"] = request_id
            return result
        # Delegate to base class default
        return super().to_openai_dict(response, model, request_id)

    def is_streaming(self, kwargs: dict[str, Any]) -> bool:
        return kwargs.get("stream", False)

    def extract_stream_tokens(self, chunk: Any, accumulated: dict[str, int]) -> dict[str, int]:
        try:
            if hasattr(chunk, "usage") and chunk.usage:
                accumulated["prompt_tokens"] = chunk.usage.prompt_tokens or 0
                accumulated["completion_tokens"] = chunk.usage.completion_tokens or 0
        except AttributeError:
            pass
        return accumulated

    def extract_chunk_info(self, chunk: Any) -> StreamChunkInfo:
        from stateloom.middleware.base import StreamChunkInfo

        info = StreamChunkInfo()
        try:
            if hasattr(chunk, "choices") and chunk.choices:
                choice = chunk.choices[0]
                delta = getattr(choice, "delta", None)
                if delta and hasattr(delta, "content") and delta.content:
                    info.text_delta = delta.content
                info.finish_reason = getattr(choice, "finish_reason", None)
            if hasattr(chunk, "usage") and chunk.usage:
                info.prompt_tokens = chunk.usage.prompt_tokens or 0
                info.completion_tokens = chunk.usage.completion_tokens or 0
                info.has_usage = True
        except (AttributeError, IndexError):
            pass
        return info


def _check_replay(gate: Gate, step: int) -> Any | None:
    """Check if the replay engine wants to mock this step."""
    engine = get_current_replay_engine()
    if engine is not None and engine.is_active and engine.should_mock(step):
        return engine.get_cached_response(step)
    return None


def patch_litellm(gate: Gate) -> list[str]:
    """Patch litellm.completion and litellm.acompletion.

    Returns a list of human-readable descriptions of what was patched.
    """
    try:
        import litellm
        import litellm.main
    except ImportError:
        return []

    from stateloom.intercept.provider_registry import register_adapter

    adapter = LiteLLMAdapter()
    register_adapter(adapter)

    patched: list[str] = []

    # --- Patch sync completion ---
    original_completion = litellm.completion

    def _completion_wrapper(*args: Any, **kwargs: Any) -> Any:
        return _intercept_litellm_sync(gate, adapter, original_completion, args, kwargs)

    # Patch on both the package and the definition module
    for target in (litellm, litellm.main):
        if hasattr(target, "completion"):
            setattr(target, "completion", _completion_wrapper)
            register_patch(target, "completion", original_completion, "litellm.completion")

    patched.append("litellm.completion (sync)")

    # --- Patch async completion ---
    original_acompletion = litellm.acompletion

    async def _acompletion_wrapper(*args: Any, **kwargs: Any) -> Any:
        return await _intercept_litellm_async(gate, adapter, original_acompletion, args, kwargs)

    for target in (litellm, litellm.main):
        if hasattr(target, "acompletion"):
            setattr(target, "acompletion", _acompletion_wrapper)
            register_patch(target, "acompletion", original_acompletion, "litellm.acompletion")

    patched.append("litellm.acompletion (async)")

    logger.info("[StateLoom] Patched litellm.completion and litellm.acompletion")
    return patched


def _intercept_litellm_sync(
    gate: Gate,
    adapter: LiteLLMAdapter,
    original: Any,
    args: tuple,
    kwargs: dict[str, Any],
) -> Any:
    """Intercept a sync litellm.completion call through the middleware pipeline."""
    model = kwargs.get("model") or (args[0] if args else "unknown")
    is_streaming = kwargs.get("stream", False)

    try:
        session = gate.get_or_create_session()
        step = session.next_step()

        cached = _check_replay(gate, step)
        if cached is not None:
            return cached

        if is_streaming:
            ctx = gate.pipeline.execute_streaming_sync(
                provider=adapter.name,
                method=adapter.method_label,
                model=model,
                request_kwargs=kwargs,
                session=session,
                config=gate.config,
            )
            if ctx.skip_call and ctx.cached_response is not None:
                return ctx.cached_response
            result = original(*args, **kwargs)
            return _wrap_stream_sync(gate, adapter, result, session, model, step, ctx=ctx)

        result = gate.pipeline.execute_sync(
            provider=adapter.name,
            method=adapter.method_label,
            model=model,
            request_kwargs=kwargs,
            session=session,
            config=gate.config,
            llm_call=lambda: original(*args, **kwargs),
        )
        return result

    except StateLoomError:
        raise
    except Exception:
        if gate.config.fail_open:
            logger.error("Middleware error, failing open", exc_info=True)
            return original(*args, **kwargs)
        raise


async def _intercept_litellm_async(
    gate: Gate,
    adapter: LiteLLMAdapter,
    original: Any,
    args: tuple,
    kwargs: dict[str, Any],
) -> Any:
    """Intercept an async litellm.acompletion call through the middleware pipeline."""
    model = kwargs.get("model") or (args[0] if args else "unknown")
    is_streaming = kwargs.get("stream", False)

    try:
        session = gate.get_or_create_session()
        step = session.next_step()

        cached = _check_replay(gate, step)
        if cached is not None:
            return cached

        if is_streaming:
            ctx = await gate.pipeline.execute_streaming_async(
                provider=adapter.name,
                method=adapter.method_label,
                model=model,
                request_kwargs=kwargs,
                session=session,
                config=gate.config,
            )
            if ctx.skip_call and ctx.cached_response is not None:
                return ctx.cached_response
            result = await original(*args, **kwargs)
            return _wrap_stream_async(gate, adapter, result, session, model, step, ctx=ctx)

        result = await gate.pipeline.execute_async(
            provider=adapter.name,
            method=adapter.method_label,
            model=model,
            request_kwargs=kwargs,
            session=session,
            config=gate.config,
            llm_call=lambda: original(*args, **kwargs),
        )
        return result

    except StateLoomError:
        raise
    except Exception:
        if gate.config.fail_open:
            logger.error("Middleware error, failing open", exc_info=True)
            return await original(*args, **kwargs)
        raise


def _wrap_stream_sync(
    gate: Gate,
    adapter: LiteLLMAdapter,
    stream: Any,
    session: Any,
    model: str,
    step: int,
    ctx: Any = None,
) -> Any:
    """Wrap a sync streaming response to capture tokens on completion."""
    from stateloom.core.event import LLMCallEvent
    from stateloom.middleware.base import MiddlewareContext

    accumulated: dict[str, int] = {"prompt_tokens": 0, "completion_tokens": 0}
    error: BaseException | None = None
    try:
        for chunk in stream:
            accumulated = adapter.extract_stream_tokens(chunk, accumulated)
            yield chunk
    except BaseException as exc:
        error = exc
        raise
    finally:
        total_prompt = accumulated.get("prompt_tokens", 0)
        total_completion = accumulated.get("completion_tokens", 0)

        if isinstance(ctx, MiddlewareContext):
            ctx.prompt_tokens = total_prompt
            ctx.completion_tokens = total_completion
            ctx._stream_error = error
            for callback in ctx._on_stream_complete:
                try:
                    callback()
                except Exception:
                    logger.debug("Stream complete callback failed", exc_info=True)
        else:
            cost = gate.pricing.calculate_cost(model, total_prompt, total_completion)
            session.add_cost(
                cost,
                prompt_tokens=total_prompt,
                completion_tokens=total_completion,
            )
            event = LLMCallEvent(
                session_id=session.id,
                step=step,
                provider=adapter.name,
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
    gate: Gate,
    adapter: LiteLLMAdapter,
    stream: Any,
    session: Any,
    model: str,
    step: int,
    ctx: Any = None,
) -> Any:
    """Wrap an async streaming response to capture tokens on completion."""
    from stateloom.core.event import LLMCallEvent
    from stateloom.middleware.base import MiddlewareContext

    accumulated: dict[str, int] = {"prompt_tokens": 0, "completion_tokens": 0}
    error: BaseException | None = None
    try:
        async for chunk in stream:
            accumulated = adapter.extract_stream_tokens(chunk, accumulated)
            yield chunk
    except BaseException as exc:
        error = exc
        raise
    finally:
        total_prompt = accumulated.get("prompt_tokens", 0)
        total_completion = accumulated.get("completion_tokens", 0)

        if isinstance(ctx, MiddlewareContext):
            ctx.prompt_tokens = total_prompt
            ctx.completion_tokens = total_completion
            ctx._stream_error = error
            for callback in ctx._on_stream_complete:
                try:
                    callback()
                except Exception:
                    logger.debug("Stream complete callback failed", exc_info=True)
        else:
            cost = gate.pricing.calculate_cost(model, total_prompt, total_completion)
            session.add_cost(
                cost,
                prompt_tokens=total_prompt,
                completion_tokens=total_completion,
            )
            event = LLMCallEvent(
                session_id=session.id,
                step=step,
                provider=adapter.name,
                model=model,
                prompt_tokens=total_prompt,
                completion_tokens=total_completion,
                total_tokens=total_prompt + total_completion,
                cost=cost,
                is_streaming=True,
                request_hash="",
            )
            gate.store.save_event(event)
