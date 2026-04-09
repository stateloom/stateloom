"""Generic interceptor — shared intercept logic parameterized by adapter."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from stateloom.core.context import get_current_replay_engine
from stateloom.core.errors import StateLoomError
from stateloom.intercept.provider_adapter import BaseProviderAdapter
from stateloom.intercept.provider_registry import register_adapter
from stateloom.intercept.unpatch import register_patch

if TYPE_CHECKING:
    from stateloom.gate import Gate
    from stateloom.intercept.provider_adapter import ProviderAdapter

logger = logging.getLogger("stateloom.intercept.generic")


def _inject_api_key_if_needed(gate: Gate, adapter: ProviderAdapter, instance: Any) -> None:
    """Inject centrally-managed API key only if the SDK has no key set."""
    provider = adapter.name
    if provider == "openai":
        key = gate.config.provider_api_key_openai
    elif provider == "anthropic":
        key = gate.config.provider_api_key_anthropic
    elif provider == "gemini":
        key = gate.config.provider_api_key_google
    else:
        return

    if not key:
        return

    if provider in ("openai", "anthropic"):
        client = BaseProviderAdapter._unwrap_client(instance)
        existing = getattr(client, "api_key", None)
        if existing:
            return
        client.api_key = key
    elif provider == "gemini":
        import os as _os

        if _os.environ.get("GOOGLE_API_KEY"):
            return
        try:
            import google.generativeai as genai

            # WARNING: genai.configure() modifies global module state.
            # Safe for single-tenant/single-key but breaks under multi-key.
            # The google.generativeai SDK has no per-client key API.
            # Proper fix requires migrating to google-genai SDK (out of scope).
            genai.configure(api_key=key)  # type: ignore[attr-defined]
        except ImportError:
            pass


def _check_replay(gate: Gate, step: int, session: Any = None, request_hash: str = "") -> Any | None:
    """Check if the replay engine wants to mock this step.

    When a cached response is returned, a ``CacheHitEvent`` is persisted so
    replayed steps appear in the dashboard trace timeline.

    For durable replay, validates that the request hash matches the cached
    step's hash. A mismatch means LLM call order changed between runs
    (non-deterministic iteration) — raises ``StateLoomDurableReplayError``.
    """
    engine = get_current_replay_engine()
    if engine is not None and engine.is_active and engine.should_mock(step):
        # Validate request hash for durable replay (backward compat: skip if either is empty)
        if request_hash:
            record = engine._step_index.get(step)
            if record is not None and record.request_hash and record.request_hash != request_hash:
                from stateloom.core.errors import StateLoomDurableReplayError

                raise StateLoomDurableReplayError(
                    session_id=session.id if session else "",
                    step=step,
                    expected_hash=record.request_hash,
                    actual_hash=request_hash,
                )

        logger.debug("Replay: returning cached response for step %d", step)
        cached = engine.get_cached_response(step)
        if cached is not None and session is not None:
            _record_replay_event(gate, session, step, engine)
        return cached
    return None


def _record_replay_event(gate: Gate, session: Any, step: int, engine: Any) -> None:
    """Record a CacheHitEvent for a replayed step (fail-open)."""
    try:
        from stateloom.core.event import CacheHitEvent

        record = engine._step_index.get(step)
        model = getattr(record, "model", "") or ""
        prompt = getattr(record, "prompt_preview", "") or f"[replayed step {step}]"

        event = CacheHitEvent(
            session_id=session.id,
            step=step,
            original_model=model,
            saved_cost=0.0,
            match_type="replay",
            prompt_preview=prompt,
        )
        gate.store.save_event(event)
        session.cache_hits += 1
    except Exception:
        logger.debug("Failed to record replay event for step %d", step, exc_info=True)


def patch_provider(gate: Gate, adapter: ProviderAdapter) -> list[str]:
    """Apply monkey-patches for all targets declared by *adapter*.

    Registers each patch for teardown via ``unpatch_all()`` and ensures the
    adapter is in the provider registry.

    Returns a list of human-readable descriptions of what was patched.
    """
    register_adapter(adapter)

    targets = adapter.get_patch_targets()
    if not targets:
        logger.debug("Provider %s: no targets (SDK not installed?), skipping", adapter.name)
        return []

    patched: list[str] = []
    for target in targets:
        original = getattr(target.target_class, target.method_name)

        if target.is_async:
            async_wrapper = _build_async_wrapper(
                gate, adapter, original, always_streaming=target.always_streaming
            )
            setattr(target.target_class, target.method_name, async_wrapper)
        else:
            sync_wrapper = _build_sync_wrapper(
                gate, adapter, original, always_streaming=target.always_streaming
            )
            setattr(target.target_class, target.method_name, sync_wrapper)

        register_patch(
            target.target_class,
            target.method_name,
            original,
            target.description or f"{adapter.name}.{target.method_name}",
        )

        kind = "async" if target.is_async else "sync"
        desc = f"{adapter.name}.{target.method_name} ({kind})"
        patched.append(desc)
        logger.info("Patched %s", desc)

    return patched


def _build_sync_wrapper(
    gate: Gate,
    adapter: ProviderAdapter,
    original: Any,
    always_streaming: bool = False,
) -> Any:
    """Build a sync wrapper function for the given original method."""

    def wrapped(self: Any, *args: Any, **kwargs: Any) -> Any:
        return _intercept_sync(
            gate, adapter, original, self, args, kwargs, always_streaming=always_streaming
        )

    return wrapped


def _build_async_wrapper(
    gate: Gate,
    adapter: ProviderAdapter,
    original: Any,
    always_streaming: bool = False,
) -> Any:
    """Build an async wrapper function for the given original method."""

    async def wrapped(self: Any, *args: Any, **kwargs: Any) -> Any:
        return await _intercept_async(
            gate, adapter, original, self, args, kwargs, always_streaming=always_streaming
        )

    return wrapped


def _intercept_sync(
    gate: Gate,
    adapter: ProviderAdapter,
    original: Any,
    instance: Any,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    always_streaming: bool = False,
) -> Any:
    """Intercept a sync LLM call through the middleware pipeline."""
    _inject_api_key_if_needed(gate, adapter, instance)
    model = adapter.extract_model(instance, args, kwargs)

    try:
        session = gate.get_or_create_session(provider=adapter.name)
        is_durable = session.durable
        step = session.acquire_durable_step() if is_durable else session.next_step()
        logger.debug(
            "Intercept sync: provider=%s model=%s session=%s step=%d",
            adapter.name,
            model,
            session.id,
            step,
        )

        # Compute request hash early for durable sessions (needed for replay validation)
        durable_request_hash = ""
        if is_durable:
            try:
                normalized_for_hash = adapter.normalize_request(args, kwargs)
                durable_request_hash = gate.pipeline._hash_request(normalized_for_hash)
            except Exception:
                logger.debug("Failed to compute durable request hash", exc_info=True)

        try:
            cached = _check_replay(gate, step, session, request_hash=durable_request_hash)
            if cached is not None:
                return cached

            # Extract provider base URL for compliance checks (fail-open)
            try:
                provider_base_url = adapter.extract_base_url(instance)
            except Exception:
                provider_base_url = ""

            # Compliance: force non-streaming for regulated profiles.
            # Skip when always_streaming — those methods are inherently streaming
            # (e.g. genai generate_content_stream) and cannot be downgraded via kwarg.
            is_streaming = always_streaming or adapter.is_streaming(kwargs)
            profile = gate._get_compliance_profile(session.org_id, session.team_id)
            if profile and profile.block_streaming and is_streaming and not always_streaming:
                logger.debug("Compliance: forcing non-streaming for session=%s", session.id)
                kwargs["stream"] = False
                is_streaming = False

            if is_streaming:
                normalized_kwargs = adapter.normalize_request(args, kwargs)
                ctx = gate.pipeline.execute_streaming_sync(
                    provider=adapter.name,
                    method=adapter.method_label,
                    model=model,
                    request_kwargs=normalized_kwargs,
                    session=session,
                    config=gate.config,
                    provider_base_url=provider_base_url,
                )
                if ctx.skip_call and ctx.cached_response is not None:
                    return ctx.cached_response

                # Create stream PII buffer if enabled
                stream_buffer = None
                pii_mw = gate.pipeline.get_middleware("pii_scanner")
                if pii_mw and hasattr(pii_mw, "create_stream_buffer"):
                    stream_buffer = pii_mw.create_stream_buffer()

                # Rebuild call args so middleware modifications (PII redaction,
                # Phase 1 strip) are reflected in the actual provider call.
                live_args, live_kwargs = adapter.rebuild_call_args(
                    normalized_kwargs,
                    args,
                    kwargs,
                )
                result = original(instance, *live_args, **live_kwargs)
                # For streaming: durable step release happens in the stream wrapper's
                # finally block, after the stream is fully consumed.
                return _wrap_stream_sync(
                    gate,
                    adapter,
                    result,
                    session,
                    model,
                    step,
                    live_kwargs,
                    ctx=ctx,
                    stream_buffer=stream_buffer,
                )

            normalized_kwargs = adapter.normalize_request(args, kwargs)

            def _llm_call() -> Any:
                # Rebuild call args so middleware modifications (PII redaction,
                # Phase 1 strip) are reflected in the actual provider call.
                live_args, live_kwargs = adapter.rebuild_call_args(
                    normalized_kwargs,
                    args,
                    kwargs,
                )
                return original(instance, *live_args, **live_kwargs)

            result = gate.pipeline.execute_sync(
                provider=adapter.name,
                method=adapter.method_label,
                model=model,
                request_kwargs=normalized_kwargs,
                session=session,
                config=gate.config,
                llm_call=_llm_call,
                provider_base_url=provider_base_url,
            )

            # Cost tracking is handled by CostTracker middleware in the pipeline.
            # No need to add cost here — it would double-count.

            return result

        finally:
            # Release durable in-flight guard for non-streaming calls.
            # Streaming calls release in the stream wrapper's finally block.
            if is_durable and not (always_streaming or adapter.is_streaming(kwargs)):
                session.release_durable_step()

    except StateLoomError:
        raise  # Security errors (PII block, budget, loop) must propagate
    except Exception:
        if gate.config.fail_open:
            logger.error("Middleware error, failing open", exc_info=True)
            return original(instance, *args, **kwargs)
        raise


async def _intercept_async(
    gate: Gate,
    adapter: ProviderAdapter,
    original: Any,
    instance: Any,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    always_streaming: bool = False,
) -> Any:
    """Intercept an async LLM call through the middleware pipeline."""
    _inject_api_key_if_needed(gate, adapter, instance)
    model = adapter.extract_model(instance, args, kwargs)

    try:
        session = gate.get_or_create_session(provider=adapter.name)
        is_durable = session.durable
        step = session.acquire_durable_step() if is_durable else session.next_step()
        logger.debug(
            "Intercept async: provider=%s model=%s session=%s step=%d",
            adapter.name,
            model,
            session.id,
            step,
        )

        # Compute request hash early for durable sessions (needed for replay validation)
        durable_request_hash = ""
        if is_durable:
            try:
                normalized_for_hash = adapter.normalize_request(args, kwargs)
                durable_request_hash = gate.pipeline._hash_request(normalized_for_hash)
            except Exception:
                logger.debug("Failed to compute durable request hash", exc_info=True)

        try:
            cached = _check_replay(gate, step, session, request_hash=durable_request_hash)
            if cached is not None:
                return cached

            # Extract provider base URL for compliance checks (fail-open)
            try:
                provider_base_url = adapter.extract_base_url(instance)
            except Exception:
                provider_base_url = ""

            # Compliance: force non-streaming for regulated profiles.
            # Skip when always_streaming — those methods are inherently streaming
            # (e.g. genai generate_content_stream) and cannot be downgraded via kwarg.
            is_streaming = always_streaming or adapter.is_streaming(kwargs)
            profile = gate._get_compliance_profile(session.org_id, session.team_id)
            if profile and profile.block_streaming and is_streaming and not always_streaming:
                logger.debug("Compliance: forcing non-streaming for session=%s", session.id)
                kwargs["stream"] = False
                is_streaming = False

            if is_streaming:
                normalized_kwargs = adapter.normalize_request(args, kwargs)
                ctx = await gate.pipeline.execute_streaming_async(
                    provider=adapter.name,
                    method=adapter.method_label,
                    model=model,
                    request_kwargs=normalized_kwargs,
                    session=session,
                    config=gate.config,
                    provider_base_url=provider_base_url,
                )
                if ctx.skip_call and ctx.cached_response is not None:
                    return ctx.cached_response

                # Create stream PII buffer if enabled
                stream_buffer = None
                pii_mw = gate.pipeline.get_middleware("pii_scanner")
                if pii_mw and hasattr(pii_mw, "create_stream_buffer"):
                    stream_buffer = pii_mw.create_stream_buffer()

                # Rebuild call args so middleware modifications (PII redaction,
                # Phase 1 strip) are reflected in the actual provider call.
                live_args, live_kwargs = adapter.rebuild_call_args(
                    normalized_kwargs,
                    args,
                    kwargs,
                )
                result = await original(instance, *live_args, **live_kwargs)
                # For streaming: durable step release happens in the stream wrapper's
                # finally block, after the stream is fully consumed.
                return _wrap_stream_async(
                    gate,
                    adapter,
                    result,
                    session,
                    model,
                    step,
                    live_kwargs,
                    ctx=ctx,
                    stream_buffer=stream_buffer,
                )

            normalized_kwargs = adapter.normalize_request(args, kwargs)

            def _llm_call() -> Any:
                # Rebuild call args so middleware modifications (PII redaction,
                # Phase 1 strip) are reflected in the actual provider call.
                live_args, live_kwargs = adapter.rebuild_call_args(
                    normalized_kwargs,
                    args,
                    kwargs,
                )
                return original(instance, *live_args, **live_kwargs)

            result = await gate.pipeline.execute_async(
                provider=adapter.name,
                method=adapter.method_label,
                model=model,
                request_kwargs=normalized_kwargs,
                session=session,
                config=gate.config,
                llm_call=_llm_call,
                provider_base_url=provider_base_url,
            )

            # Cost tracking is handled by CostTracker middleware in the pipeline.

            return result

        finally:
            # Release durable in-flight guard for non-streaming calls.
            # Streaming calls release in the stream wrapper's finally block.
            if is_durable and not (always_streaming or adapter.is_streaming(kwargs)):
                session.release_durable_step()

    except StateLoomError:
        raise  # Security errors (PII block, budget, loop) must propagate
    except Exception:
        if gate.config.fail_open:
            logger.error("Middleware error, failing open", exc_info=True)
            return await original(instance, *args, **kwargs)
        raise


def _wrap_stream_sync(
    gate: Gate,
    adapter: ProviderAdapter,
    stream: Any,
    session: Any,
    model: str,
    step: int,
    kwargs: dict[str, Any],
    ctx: Any = None,
    stream_buffer: Any = None,
) -> Any:
    """Wrap a sync streaming response to capture tokens on completion.

    When *ctx* is provided (MiddlewareContext from pipeline), tokens are set
    on the context and ``_on_stream_complete`` callbacks are fired.  When
    *ctx* is ``None`` (legacy / backward-compat path), events are written
    directly to the store.

    When *stream_buffer* is provided (StreamPIIBuffer), text chunks are
    buffered and scanned for PII before being released.
    """
    from stateloom.core.event import LLMCallEvent
    from stateloom.middleware.base import MiddlewareContext

    accumulated: dict[str, int] = {"prompt_tokens": 0, "completion_tokens": 0}
    error: BaseException | None = None
    last_chunk: Any = None

    # Durable streaming: accumulate chunks for serialization after stream completes
    is_durable = session.durable
    durable_chunks: list[Any] | None = [] if is_durable else None

    # Buffered mode: buffer all chunks, persist event, then yield for crash safety
    use_buffer_mode = (
        is_durable and gate.config.durable_stream_buffer and isinstance(ctx, MiddlewareContext)
    )

    try:
        if use_buffer_mode:
            # Buffer all chunks without yielding
            buffered_chunks: list[Any] = []
            for chunk in stream:
                accumulated = adapter.extract_stream_tokens(chunk, accumulated)
                if stream_buffer is not None:
                    chunk_info = adapter.extract_chunk_info(chunk)
                    if chunk_info.text_delta:
                        clean = stream_buffer.feed(chunk_info.text_delta)
                        if clean is not None:
                            chunk = adapter.modify_chunk_text(chunk, clean)
                            last_chunk = chunk
                            buffered_chunks.append(chunk)
                    else:
                        buffered_chunks.append(chunk)
                else:
                    buffered_chunks.append(chunk)

            # Flush stream buffer
            if stream_buffer is not None:
                try:
                    final = stream_buffer.flush()
                    if final and last_chunk is not None:
                        flushed = adapter.modify_chunk_text(last_chunk, final)
                        buffered_chunks.append(flushed)
                except Exception:
                    logger.debug("Stream buffer flush failed", exc_info=True)

            # Persist event BEFORE yielding (crash safe)
            durable_chunks = buffered_chunks
            total_prompt = accumulated.get("prompt_tokens", 0)
            total_completion = accumulated.get("completion_tokens", 0)
            assert isinstance(ctx, MiddlewareContext)
            ctx.prompt_tokens = total_prompt
            ctx.completion_tokens = total_completion
            ctx._stream_error = None

            if durable_chunks:
                try:
                    from stateloom.replay.schema import serialize_stream_chunks

                    ctx._durable_cached_json = serialize_stream_chunks(durable_chunks, adapter.name)
                except Exception:
                    pass  # fail-open

            for callback in ctx._on_stream_complete:
                try:
                    callback()
                except Exception:
                    logger.debug("Stream complete callback failed", exc_info=True)

            # Now yield all buffered chunks
            yield from buffered_chunks
            return  # skip the normal finally path (already handled above)

        for chunk in stream:
            accumulated = adapter.extract_stream_tokens(chunk, accumulated)
            if stream_buffer is not None:
                chunk_info = adapter.extract_chunk_info(chunk)
                if chunk_info.text_delta:
                    clean = stream_buffer.feed(chunk_info.text_delta)
                    if clean is not None:
                        chunk = adapter.modify_chunk_text(chunk, clean)
                        last_chunk = chunk
                        if durable_chunks is not None:
                            durable_chunks.append(chunk)
                        yield chunk
                    # else: still buffering, don't yield
                else:
                    if durable_chunks is not None:
                        durable_chunks.append(chunk)
                    yield chunk  # non-text chunks pass through
            else:
                if durable_chunks is not None:
                    durable_chunks.append(chunk)
                yield chunk
    except BaseException as exc:
        error = exc
        raise
    finally:
        # Release durable in-flight guard (stream fully consumed or errored)
        if is_durable:
            session.release_durable_step()

        # Flush stream buffer
        if stream_buffer is not None and error is None:
            try:
                final = stream_buffer.flush()
                if final and last_chunk is not None:
                    flushed = adapter.modify_chunk_text(last_chunk, final)
                    if durable_chunks is not None:
                        durable_chunks.append(flushed)
                    yield flushed
            except Exception:
                logger.debug("Stream buffer flush failed", exc_info=True)

        total_prompt = accumulated.get("prompt_tokens", 0)
        total_completion = accumulated.get("completion_tokens", 0)

        if isinstance(ctx, MiddlewareContext):
            ctx.prompt_tokens = total_prompt
            ctx.completion_tokens = total_completion
            ctx._stream_error = error

            # Serialize accumulated chunks for durable caching (fail-open)
            if durable_chunks is not None and error is None:
                try:
                    from stateloom.replay.schema import serialize_stream_chunks

                    ctx._durable_cached_json = serialize_stream_chunks(durable_chunks, adapter.name)
                except Exception:
                    pass  # fail-open

            for callback in ctx._on_stream_complete:
                try:
                    callback()
                except Exception:
                    logger.debug("Stream complete callback failed", exc_info=True)
        else:
            # Legacy path — write directly to store
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
    adapter: ProviderAdapter,
    stream: Any,
    session: Any,
    model: str,
    step: int,
    kwargs: dict[str, Any],
    ctx: Any = None,
    stream_buffer: Any = None,
) -> Any:
    """Wrap an async streaming response to capture tokens on completion.

    When *ctx* is provided (MiddlewareContext from pipeline), tokens are set
    on the context and ``_on_stream_complete`` callbacks are fired.  When
    *ctx* is ``None`` (legacy / backward-compat path), events are written
    directly to the store.

    When *stream_buffer* is provided (StreamPIIBuffer), text chunks are
    buffered and scanned for PII before being released.
    """
    from stateloom.core.event import LLMCallEvent
    from stateloom.middleware.base import MiddlewareContext

    accumulated: dict[str, int] = {"prompt_tokens": 0, "completion_tokens": 0}
    error: BaseException | None = None
    last_chunk: Any = None

    # Durable streaming: accumulate chunks for serialization after stream completes
    is_durable = session.durable
    durable_chunks: list[Any] | None = [] if is_durable else None

    # Buffered mode: buffer all chunks, persist event, then yield for crash safety
    use_buffer_mode = (
        is_durable and gate.config.durable_stream_buffer and isinstance(ctx, MiddlewareContext)
    )

    try:
        if use_buffer_mode:
            # Buffer all chunks without yielding
            buffered_chunks: list[Any] = []
            async for chunk in stream:
                accumulated = adapter.extract_stream_tokens(chunk, accumulated)
                if stream_buffer is not None:
                    chunk_info = adapter.extract_chunk_info(chunk)
                    if chunk_info.text_delta:
                        clean = stream_buffer.feed(chunk_info.text_delta)
                        if clean is not None:
                            chunk = adapter.modify_chunk_text(chunk, clean)
                            last_chunk = chunk
                            buffered_chunks.append(chunk)
                    else:
                        buffered_chunks.append(chunk)
                else:
                    buffered_chunks.append(chunk)

            # Flush stream buffer
            if stream_buffer is not None:
                try:
                    final = stream_buffer.flush()
                    if final and last_chunk is not None:
                        flushed = adapter.modify_chunk_text(last_chunk, final)
                        buffered_chunks.append(flushed)
                except Exception:
                    logger.debug("Stream buffer flush failed", exc_info=True)

            # Persist event BEFORE yielding (crash safe)
            durable_chunks = buffered_chunks
            total_prompt = accumulated.get("prompt_tokens", 0)
            total_completion = accumulated.get("completion_tokens", 0)
            assert isinstance(ctx, MiddlewareContext)
            ctx.prompt_tokens = total_prompt
            ctx.completion_tokens = total_completion
            ctx._stream_error = None

            if durable_chunks:
                try:
                    from stateloom.replay.schema import serialize_stream_chunks

                    ctx._durable_cached_json = serialize_stream_chunks(durable_chunks, adapter.name)
                except Exception:
                    pass  # fail-open

            for callback in ctx._on_stream_complete:
                try:
                    callback()
                except Exception:
                    logger.debug("Stream complete callback failed", exc_info=True)

            # Now yield all buffered chunks
            for buffered_chunk in buffered_chunks:
                yield buffered_chunk
            return  # skip the normal finally path (already handled above)

        async for chunk in stream:
            accumulated = adapter.extract_stream_tokens(chunk, accumulated)
            if stream_buffer is not None:
                chunk_info = adapter.extract_chunk_info(chunk)
                if chunk_info.text_delta:
                    clean = stream_buffer.feed(chunk_info.text_delta)
                    if clean is not None:
                        chunk = adapter.modify_chunk_text(chunk, clean)
                        last_chunk = chunk
                        if durable_chunks is not None:
                            durable_chunks.append(chunk)
                        yield chunk
                    # else: still buffering, don't yield
                else:
                    if durable_chunks is not None:
                        durable_chunks.append(chunk)
                    yield chunk  # non-text chunks pass through
            else:
                if durable_chunks is not None:
                    durable_chunks.append(chunk)
                yield chunk
    except BaseException as exc:
        error = exc
        raise
    finally:
        # Release durable in-flight guard (stream fully consumed or errored)
        if is_durable:
            session.release_durable_step()

        # Flush stream buffer
        if stream_buffer is not None and error is None:
            try:
                final = stream_buffer.flush()
                if final and last_chunk is not None:
                    flushed = adapter.modify_chunk_text(last_chunk, final)
                    if durable_chunks is not None:
                        durable_chunks.append(flushed)
                    yield flushed
            except Exception:
                logger.debug("Stream buffer flush failed", exc_info=True)

        total_prompt = accumulated.get("prompt_tokens", 0)
        total_completion = accumulated.get("completion_tokens", 0)

        if isinstance(ctx, MiddlewareContext):
            ctx.prompt_tokens = total_prompt
            ctx.completion_tokens = total_completion
            ctx._stream_error = error

            # Serialize accumulated chunks for durable caching (fail-open)
            if durable_chunks is not None and error is None:
                try:
                    from stateloom.replay.schema import serialize_stream_chunks

                    ctx._durable_cached_json = serialize_stream_chunks(durable_chunks, adapter.name)
                except Exception:
                    pass  # fail-open

            for callback in ctx._on_stream_complete:
                try:
                    callback()
                except Exception:
                    logger.debug("Stream complete callback failed", exc_info=True)
        else:
            # Legacy path — write directly to store
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


def wrap_instance(gate: Gate, adapter: ProviderAdapter, client: Any) -> None:
    """Wrap a client instance's methods (for gate.wrap() without monkey-patching)."""
    targets = adapter.get_instance_targets(client)
    for sub_object, method_name in targets:
        original = getattr(sub_object, method_name)

        def _make_wrapper(_orig: Any = original, _sub: Any = sub_object) -> Any:
            def wrapper(*args: Any, **kwargs: Any) -> Any:
                return _intercept_sync(gate, adapter, _orig, _sub, args, kwargs)

            return wrapper

        setattr(sub_object, method_name, _make_wrapper())
    if targets:
        logger.info("[StateLoom] Wrapped %s instance", adapter.name)
