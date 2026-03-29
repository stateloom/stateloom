"""Ordered middleware chain executor."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time
from collections.abc import Callable
from typing import Any

from stateloom.core.config import StateLoomConfig
from stateloom.core.context import get_framework_context
from stateloom.core.session import Session
from stateloom.middleware.base import Middleware, MiddlewareContext

logger = logging.getLogger("stateloom.pipeline")


class Pipeline:
    """Executes an ordered chain of middleware around an LLM call."""

    def __init__(
        self,
        middlewares: list[Middleware] | None = None,
        normalizer: Any | None = None,
    ) -> None:
        """Initialize the pipeline.

        Args:
            middlewares: Pre-built middleware list, or None to start empty
                (middleware is added later via ``Gate._setup_middleware()``).
            normalizer: Optional ``RequestNormalizer`` used by
                ``_hash_request()`` to strip dynamic content (UUIDs,
                timestamps) before hashing for cache dedup.
        """
        self.middlewares: list[Middleware] = middlewares or []
        self._normalizer = normalizer

    def add(self, middleware: Middleware) -> None:
        """Add a middleware to the end of the chain."""
        self.middlewares.append(middleware)

    def insert(self, index: int, middleware: Middleware) -> None:
        """Insert a middleware at a specific position."""
        self.middlewares.insert(index, middleware)

    def remove(self, middleware: Middleware) -> None:
        """Remove a middleware from the chain."""
        self.middlewares.remove(middleware)

    def get_middleware(self, name: str) -> Any:
        """Look up a middleware instance by short name (e.g. 'pii_scanner')."""
        from stateloom.middleware.pii_scanner import PIIScannerMiddleware

        name_map: dict[str, type] = {"pii_scanner": PIIScannerMiddleware}
        target = name_map.get(name)
        if target:
            for mw in self.middlewares:
                if isinstance(mw, target):
                    return mw
        return None

    async def execute(
        self,
        ctx: MiddlewareContext,
        llm_call: Callable[..., Any],
    ) -> Any:
        """Execute the middleware chain around the LLM call.

        Args:
            ctx: Mutable context for this call.
            llm_call: The callable that performs the actual LLM API call.

        Returns:
            The LLM response (or cached response if short-circuited).

        The chain is built recursively: each middleware at index *i*
        receives a ``call_next`` that invokes middleware *i+1*.  The
        terminal node (past the last middleware) either returns
        ``ctx.cached_response`` when ``skip_call`` is set, or executes
        the real ``llm_call``.
        """

        logger.debug(
            "Pipeline execute: session=%s provider=%s model=%s streaming=%s middlewares=%d",
            ctx.session.id, ctx.provider, ctx.model, ctx.is_streaming, len(self.middlewares),
        )

        async def build_chain(index: int) -> Callable:
            async def call_next(c: MiddlewareContext) -> Any:
                if index >= len(self.middlewares):
                    # Terminal: make the actual LLM call (or return cached)
                    if c.skip_call and c.cached_response is not None:
                        logger.debug("Pipeline terminal: returning cached response (skip_call=True)")
                        return c.cached_response
                    return await self._execute_llm_call(c, llm_call)
                return await self.middlewares[index].process(c, await build_chain(index + 1))

            return call_next

        chain = await build_chain(0)
        result = await chain(ctx)
        logger.debug(
            "Pipeline complete: session=%s latency=%.1fms tokens=%d",
            ctx.session.id, ctx.latency_ms, ctx.prompt_tokens + ctx.completion_tokens,
        )
        return result

    async def _execute_llm_call(
        self,
        ctx: MiddlewareContext,
        llm_call: Callable[..., Any],
    ) -> Any:
        """Execute the actual LLM call with timing.

        Args:
            ctx: Pipeline context (``response`` and ``latency_ms`` are set).
            llm_call: The SDK callable (sync or async).

        Returns:
            The raw provider response object.

        Sync ``llm_call`` functions are offloaded to ``asyncio.to_thread()``
        so they don't block the async event loop (critical for proxy mode
        where uvicorn must stay responsive to concurrent requests).
        """
        start = time.perf_counter()
        try:
            if asyncio.iscoroutinefunction(llm_call):
                result = await llm_call()
            else:
                # Offload sync SDK calls to a thread to keep the event loop
                # responsive (uvicorn, dashboard, concurrent proxy requests).
                result = await asyncio.to_thread(llm_call)
                if asyncio.iscoroutine(result):
                    result = await result
            ctx.response = result
            return result
        except Exception as exc:
            logger.warning(
                "LLM call failed: session=%s model=%s error=%s",
                ctx.session.id, ctx.model, type(exc).__name__,
            )
            raise
        finally:
            ctx.latency_ms = (time.perf_counter() - start) * 1000

    @staticmethod
    def _check_request_size(request_kwargs: dict[str, Any]) -> None:
        """Log a warning if the request payload is very large (>1MB serialized)."""
        try:
            serialized = json.dumps(request_kwargs, default=str)
            size_bytes = len(serialized.encode("utf-8"))
            if size_bytes > 1_048_576:  # 1 MB
                size_mb = size_bytes / 1_048_576
                logger.warning(
                    "Large LLM request payload: %.1f MB. Consider truncating conversation history.",
                    size_mb,
                )
        except (TypeError, ValueError):
            pass

    def execute_sync(
        self,
        *,
        provider: str,
        method: str,
        model: str,
        request_kwargs: dict[str, Any],
        session: Session,
        config: StateLoomConfig,
        llm_call: Callable[..., Any],
        auto_route_eligible: bool = False,
        provider_base_url: str = "",
    ) -> Any:
        """Synchronous entry point for monkey-patched sync SDK methods.

        Args:
            provider: Provider name (e.g. ``"openai"``).
            method: SDK method path (e.g. ``"chat.completions.create"``).
            model: Model identifier.
            request_kwargs: Original kwargs passed to the SDK method.
            session: Active session for this call.
            config: Global StateLoom config.
            llm_call: The original (unpatched) SDK callable.
            auto_route_eligible: Whether auto-routing is allowed.
            provider_base_url: Upstream API base URL.

        Returns:
            The LLM response after flowing through the full pipeline.

        If a running event loop is detected (e.g. inside Jupyter or an
        async framework), the pipeline is dispatched to a
        ``ThreadPoolExecutor`` to avoid blocking the loop.
        """
        self._check_request_size(request_kwargs)
        ctx = MiddlewareContext(
            session=session,
            config=config,
            provider=provider,
            method=method,
            model=model,
            request_kwargs=request_kwargs,
            request_hash=self._hash_request(request_kwargs),
            auto_route_eligible=auto_route_eligible,
            provider_base_url=provider_base_url,
        )
        fw_ctx = get_framework_context()
        if fw_ctx:
            ctx._framework_context = fw_ctx.copy()

        # Run async pipeline in sync context
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            # Nested event loop detected (Jupyter, async framework, etc.).
            # asyncio.run() would fail, so offload to a fresh thread with
            # its own event loop via ThreadPoolExecutor.
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(asyncio.run, self.execute(ctx, llm_call))
                return future.result()
        return asyncio.run(self.execute(ctx, llm_call))

    async def execute_async(
        self,
        *,
        provider: str,
        method: str,
        model: str,
        request_kwargs: dict[str, Any],
        session: Session,
        config: StateLoomConfig,
        llm_call: Callable[..., Any],
        auto_route_eligible: bool = False,
        provider_base_url: str = "",
    ) -> Any:
        """Async entry point for monkey-patched async SDK methods.

        Args:
            provider: Provider name.
            method: SDK method path.
            model: Model identifier.
            request_kwargs: Original kwargs passed to the SDK method.
            session: Active session for this call.
            config: Global StateLoom config.
            llm_call: The original (unpatched) async SDK callable.
            auto_route_eligible: Whether auto-routing is allowed.
            provider_base_url: Upstream API base URL.

        Returns:
            The LLM response after flowing through the full pipeline.
        """
        self._check_request_size(request_kwargs)
        ctx = MiddlewareContext(
            session=session,
            config=config,
            provider=provider,
            method=method,
            model=model,
            request_kwargs=request_kwargs,
            request_hash=self._hash_request(request_kwargs),
            auto_route_eligible=auto_route_eligible,
            provider_base_url=provider_base_url,
        )
        fw_ctx = get_framework_context()
        if fw_ctx:
            ctx._framework_context = fw_ctx.copy()
        return await self.execute(ctx, llm_call)

    async def execute_streaming(
        self,
        ctx: MiddlewareContext,
    ) -> None:
        """Execute the middleware chain for streaming with a no-op terminal.

        Pre-call middleware runs normally. Post-call middleware registers
        callbacks on ``ctx._on_stream_complete``.  The caller inspects
        ``ctx.skip_call`` / ``ctx.cached_response`` and fires callbacks
        after stream exhaustion.
        """

        async def build_chain(index: int) -> Callable:
            async def call_next(c: MiddlewareContext) -> Any:
                if index >= len(self.middlewares):
                    # No-op terminal — streaming will be handled by the caller
                    if c.skip_call and c.cached_response is not None:
                        return c.cached_response
                    return None
                return await self.middlewares[index].process(c, await build_chain(index + 1))

            return call_next

        chain = await build_chain(0)
        await chain(ctx)

    def execute_streaming_sync(
        self,
        *,
        provider: str,
        method: str,
        model: str,
        request_kwargs: dict[str, Any],
        session: Session,
        config: StateLoomConfig,
        auto_route_eligible: bool = False,
        provider_base_url: str = "",
    ) -> MiddlewareContext:
        """Synchronous streaming entry point — runs pre-call middleware only.

        Args:
            provider: Provider name.
            method: SDK method path.
            model: Model identifier.
            request_kwargs: Original kwargs.
            session: Active session.
            config: Global config.
            auto_route_eligible: Whether auto-routing is allowed.
            provider_base_url: Upstream API base URL.

        Returns:
            The ``MiddlewareContext`` with pre-call middleware applied.
            The caller is responsible for streaming the actual response and
            firing ``ctx._on_stream_complete`` callbacks afterward.
        """
        self._check_request_size(request_kwargs)
        ctx = MiddlewareContext(
            session=session,
            config=config,
            provider=provider,
            method=method,
            model=model,
            request_kwargs=request_kwargs,
            request_hash="",  # cache skips streaming; avoid JSON serialize + SHA-256
            is_streaming=True,
            auto_route_eligible=auto_route_eligible,
            provider_base_url=provider_base_url,
        )
        fw_ctx = get_framework_context()
        if fw_ctx:
            ctx._framework_context = fw_ctx.copy()

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(asyncio.run, self.execute_streaming(ctx))
                future.result()
        else:
            asyncio.run(self.execute_streaming(ctx))

        return ctx

    async def execute_streaming_async(
        self,
        *,
        provider: str,
        method: str,
        model: str,
        request_kwargs: dict[str, Any],
        session: Session,
        config: StateLoomConfig,
        auto_route_eligible: bool = False,
        provider_base_url: str = "",
    ) -> MiddlewareContext:
        """Async streaming entry point — runs pre-call middleware only.

        Args:
            provider: Provider name.
            method: SDK method path.
            model: Model identifier.
            request_kwargs: Original kwargs.
            session: Active session.
            config: Global config.
            auto_route_eligible: Whether auto-routing is allowed.
            provider_base_url: Upstream API base URL.

        Returns:
            The ``MiddlewareContext`` with pre-call middleware applied.
        """
        self._check_request_size(request_kwargs)
        ctx = MiddlewareContext(
            session=session,
            config=config,
            provider=provider,
            method=method,
            model=model,
            request_kwargs=request_kwargs,
            request_hash="",  # cache skips streaming; avoid JSON serialize + SHA-256
            is_streaming=True,
            auto_route_eligible=auto_route_eligible,
            provider_base_url=provider_base_url,
        )
        fw_ctx = get_framework_context()
        if fw_ctx:
            ctx._framework_context = fw_ctx.copy()
        await self.execute_streaming(ctx)
        return ctx

    def notify_session_end(self, session_id: str) -> None:
        """Notify all middleware that a session has ended.

        Middleware can implement ``on_session_end(session_id)`` to clean up
        per-session state (e.g. failure counters, loop counts).
        """
        for mw in self.middlewares:
            if hasattr(mw, "on_session_end"):
                try:
                    mw.on_session_end(session_id)
                except Exception:
                    pass  # cleanup must not crash

    def _hash_request(self, kwargs: dict[str, Any]) -> str:
        """Hash the request for caching/dedup, normalizing dynamic content.

        Args:
            kwargs: Request kwargs (``messages``, ``model``, etc.).

        Returns:
            A 16-char hex SHA-256 prefix, or ``""`` on serialization failure.
        """
        try:
            hash_input = kwargs
            if self._normalizer is not None:
                hash_input = self._normalizer.normalize_kwargs(kwargs)
            serialized = json.dumps(hash_input, sort_keys=True, default=str)
            return hashlib.sha256(serialized.encode()).hexdigest()[:16]
        except (TypeError, ValueError):
            return ""
