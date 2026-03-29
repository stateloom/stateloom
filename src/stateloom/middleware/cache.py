"""Two-tier request cache middleware (exact hash + optional semantic matching)."""

from __future__ import annotations

import logging
import time
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any

from stateloom.cache.base import CacheEntry, CacheStore
from stateloom.core.config import StateLoomConfig
from stateloom.core.event import CacheHitEvent
from stateloom.middleware.base import MiddlewareContext
from stateloom.replay.schema import deserialize_response, serialize_response

if TYPE_CHECKING:
    from stateloom.cache.semantic import SemanticMatcher

logger = logging.getLogger("stateloom.middleware.cache")


def _is_error_response(result: Any) -> bool:
    """Return True if *result* looks like a provider error response.

    Passthrough proxy returns upstream errors as dicts:
      - Anthropic: ``{"type": "error", ...}``
      - OpenAI/Gemini: ``{"error": {...}}``
      - Internal sentinel: ``{"_upstream_error": True}``
    SDK objects that raise on errors never reach here, so only dict
    responses need checking.
    """
    if not isinstance(result, dict):
        return False
    if result.get("_upstream_error"):
        return True
    if result.get("type") == "error":
        return True
    if "error" in result and isinstance(result["error"], dict):
        return True
    return False


class CacheMiddleware:
    """Two-tier cache for LLM requests.

    Tier 1: Exact hash match (always available, <1ms).
    Tier 2: Semantic similarity via FAISS embeddings (optional, ~5-10ms).

    Supports pluggable storage backends and global/session scoping.
    """

    def __init__(
        self,
        config: StateLoomConfig,
        cache_store: CacheStore | None = None,
        semantic_matcher: SemanticMatcher | None = None,
        compliance_fn: Any = None,
    ) -> None:
        self._config = config
        self._max_size = config.cache.max_size
        self._ttl_seconds = config.cache.ttl_seconds
        self._scope = config.cache.scope
        self._similarity_threshold = config.cache.similarity_threshold
        self._compliance_fn = compliance_fn

        # Pluggable store — fall back to in-memory if none provided
        if cache_store is not None:
            self._store = cache_store
        else:
            from stateloom.cache.memory_store import MemoryCacheStore

            self._store = MemoryCacheStore(max_size=self._max_size)

        self._semantic: SemanticMatcher | None = semantic_matcher

    async def process(
        self,
        ctx: MiddlewareContext,
        call_next: Callable[[MiddlewareContext], Awaitable[Any]],
    ) -> Any:
        request_hash = ctx.request_hash
        session_id = ctx.session.id

        if not request_hash:
            return await call_next(ctx)

        # Skip cache for streaming requests — cached non-streaming responses
        # cannot be reliably converted to SSE format by all proxy endpoints.
        if ctx.is_streaming:
            return await call_next(ctx)

        # Compliance overrides
        ttl = self._ttl_seconds
        scope_session = session_id if self._scope == "session" else None

        if self._compliance_fn:
            profile = self._compliance_fn(ctx.session.org_id, ctx.session.team_id)
            if profile:
                # HIPAA no-cache: zero TTL + zero-retention = skip entirely
                if profile.cache_ttl_seconds == 0 and profile.zero_retention_logs:
                    return await call_next(ctx)
                # TTL override
                if profile.cache_ttl_seconds > 0:
                    ttl = profile.cache_ttl_seconds
                # Org-scoped cache isolation
                if ctx.session.org_id:
                    scope_session = f"org:{ctx.session.org_id}"

        # Evict expired entries if TTL is set
        if ttl > 0:
            self._store.evict_expired(ttl)

        # --- Tier 1: Exact hash match ---
        entry = self._store.get(request_hash, session_id=scope_session)
        if entry is not None:
            logger.debug("Cache hit (exact): hash=%s session=%s", request_hash, ctx.session.id)
            return await self._serve_cache_hit(
                ctx,
                call_next,
                entry,
                match_type="exact",
                similarity_score=None,
                matched_hash=request_hash,
            )

        # --- Tier 2: Semantic similarity (optional) ---
        if self._semantic is not None:
            try:
                # Use shared embedding cache to avoid recomputing
                cache_key = f"embed:{request_hash}"
                embedding = ctx._embedding_cache.get(cache_key)
                if embedding is None:
                    embedding = self._semantic.embed_request(ctx.request_kwargs)
                    ctx._embedding_cache[cache_key] = embedding
                matched_entry, score = self._semantic.search(
                    embedding,
                    session_id=scope_session,
                )
                if matched_entry is not None and score >= self._similarity_threshold:
                    logger.debug(
                        "Cache hit (semantic): score=%.3f hash=%s session=%s",
                        score, matched_entry.request_hash, ctx.session.id,
                    )
                    return await self._serve_cache_hit(
                        ctx,
                        call_next,
                        matched_entry,
                        match_type="semantic",
                        similarity_score=score,
                        matched_hash=matched_entry.request_hash,
                    )
            except Exception:
                logger.debug("Semantic cache search failed", exc_info=True)

        # --- Cache miss — make the call ---
        result = await call_next(ctx)

        # Store in cache for future hits (as JSON).
        # Never cache error responses — upstream errors (429, 529, etc.) returned
        # as dicts by the passthrough proxy must not pollute the cache.
        if _is_error_response(result):
            logger.debug("Cache skip: error response not cached for hash=%s", request_hash)
        if result is not None and not ctx.is_streaming and not _is_error_response(result):
            cost = 0.0
            for event in ctx.events:
                if hasattr(event, "cost"):
                    cost = event.cost
                    break

            response_json = serialize_response(result)

            # Compute embedding for semantic index (reuse from shared cache)
            embedding: list[float] | None = None
            if self._semantic is not None:
                try:
                    cache_key = f"embed:{request_hash}"
                    embedding = ctx._embedding_cache.get(cache_key)
                    if embedding is None:
                        embedding = self._semantic.embed_request(ctx.request_kwargs)
                        ctx._embedding_cache[cache_key] = embedding
                except Exception:
                    logger.debug("Semantic embedding failed", exc_info=True)

            entry = CacheEntry(
                request_hash=request_hash,
                session_id=session_id,
                response_json=response_json,
                model=ctx.model,
                provider=ctx.provider,
                cost=cost,
                created_at=time.time(),
                embedding=embedding,
            )
            self._store.put(entry)

            # Add to semantic index
            if self._semantic is not None and embedding is not None:
                try:
                    self._semantic.add(entry)
                except Exception:
                    logger.debug("Semantic index add failed", exc_info=True)

        return result

    async def _serve_cache_hit(
        self,
        ctx: MiddlewareContext,
        call_next: Callable[[MiddlewareContext], Awaitable[Any]],
        entry: CacheEntry,
        *,
        match_type: str,
        similarity_score: float | None,
        matched_hash: str,
    ) -> Any:
        """Shared logic for serving a cache hit (exact or semantic)."""
        event = CacheHitEvent(
            session_id=ctx.session.id,
            step=ctx.session.step_counter,
            original_model=entry.model,
            saved_cost=entry.cost,
            request_hash=ctx.request_hash,
            match_type=match_type,
            similarity_score=similarity_score,
            matched_hash=matched_hash,
        )
        ctx.events.append(event)
        ctx.session.add_cache_hit(entry.cost)

        response = deserialize_response(entry.response_json, ctx.provider)

        ctx.skip_call = True
        ctx.cached_response = response
        ctx.response = response
        return await call_next(ctx)
