"""Model testing middleware — test candidate models in parallel with cloud calls.

Formerly "shadow drafting". Runs one or more candidate local models in parallel
with every cloud LLM call, comparing responses via similarity scoring to generate
migration readiness reports. Never delays or modifies the cloud response.

Internal event type remains ``shadow_draft`` for backward compatibility.
"""

from __future__ import annotations

import copy
import logging
import random
import re
import threading
import time
from collections.abc import Awaitable, Callable
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from rich.console import Console
from rich.text import Text

from stateloom.core.config import StateLoomConfig
from stateloom.core.event import ShadowDraftEvent
from stateloom.core.types import Provider
from stateloom.local.client import OllamaClient
from stateloom.middleware.base import MiddlewareContext
from stateloom.middleware.similarity import (
    SemanticSimilarityScorer,
    SimilarityResult,
    compute_similarity_auto,
    extract_response_text,
)
from stateloom.store.base import Store

logger = logging.getLogger("stateloom.middleware.shadow")

# Known cloud model prefixes — fallback when adapter registry is empty
_CLOUD_PREFIX_RE = re.compile(r"^(gpt-|o1|o3|o4|chatgpt-|claude-|gemini-)")


def _is_cloud_model(model: str) -> bool:
    """Return True if *model* matches a cloud provider pattern."""
    from stateloom.intercept.provider_registry import resolve_provider

    provider = resolve_provider(model)
    if provider is not None:
        return True
    return bool(_CLOUD_PREFIX_RE.match(model))


_SIMILARITY_HIGH_THRESHOLD = 0.7
_SIMILARITY_MEDIUM_THRESHOLD = 0.4

_console = Console(stderr=True)

# Import realtime-data patterns from auto_router (shared heuristic)
try:
    from stateloom.middleware.auto_router import _REALTIME_PATTERNS
except ImportError:  # pragma: no cover
    _REALTIME_PATTERNS = re.compile(
        r"\b(?:weather|forecast|stock\s*(?:price|market)|current(?:ly)?|right\s+now|"
        r"latest\s+news|live\s+score|today'?s?\s+(?:date|news|price|score)|"
        r"real[- ]?time|breaking\s+news|trending|just\s+happened|happening\s+now)\b",
        re.IGNORECASE,
    )


class _SimilarityBridge:
    """Thread-safe bridge for computing similarity between cloud and shadow.

    Also controls event persistence — events are only saved under the lock,
    so cancel() can retroactively mark events that were already saved.
    Whoever finishes second (cloud or shadow) computes similarity.
    """

    def __init__(
        self,
        similarity_method: str = "difflib",
        similarity_scorer: SemanticSimilarityScorer | None = None,
    ) -> None:
        self.cloud_text: str = ""
        self.local_text: str = ""
        self.shadow_event: ShadowDraftEvent | None = None
        self.cancelled: bool = False
        # Cloud metrics — populated by set_cloud() so _compute_and_update can use them
        self.cloud_cost: float = 0.0
        self.cloud_tokens: int = 0
        self.cloud_latency_ms: float = 0.0
        self._lock = threading.Lock()
        self._cloud_ready = False
        self._local_ready = False
        self._similarity_method = similarity_method
        self._similarity_scorer = similarity_scorer

    def cancel(self, store: Store | None = None) -> None:
        """Cancel — call was routed locally or cached.

        If the shadow event was already saved, retroactively mark it cancelled.
        """
        with self._lock:
            self.cancelled = True
            if self.shadow_event is not None and store is not None:
                self.shadow_event.shadow_status = "cancelled"
                self.shadow_event.error_message = "Skipped: request routed locally"
                try:
                    store.save_event(self.shadow_event)
                except Exception:
                    logger.debug("Failed to cancel shadow event", exc_info=True)

    def set_cloud(
        self,
        text: str,
        store: Store,
        config: StateLoomConfig,
        *,
        cloud_cost: float = 0.0,
        cloud_tokens: int = 0,
        cloud_latency_ms: float = 0.0,
    ) -> None:
        """Called from process() after cloud response arrives."""
        with self._lock:
            if self.cancelled:
                return
            self.cloud_text = text
            self.cloud_cost = cloud_cost
            self.cloud_tokens = cloud_tokens
            self.cloud_latency_ms = cloud_latency_ms
            self._cloud_ready = True
            logger.debug(
                "Bridge: cloud ready (text_len=%d, local_ready=%s, cost=%.6f, tokens=%d)",
                len(text),
                self._local_ready,
                cloud_cost,
                cloud_tokens,
            )
            if not text:
                logger.warning("Shadow: cloud_text is empty — similarity cannot be computed")
            if self._local_ready and self.shadow_event is not None:
                self._compute_and_update(store, config)

    def set_local(
        self, text: str, event: ShadowDraftEvent, store: Store, config: StateLoomConfig
    ) -> None:
        """Called from shadow thread after local response.

        Saves the event to the store under the lock. If already cancelled,
        the event is not saved.
        """
        with self._lock:
            if self.cancelled:
                return
            # Persist the event (first save — without similarity)
            try:
                store.save_event(event)
            except Exception:
                logger.warning(
                    "Failed to persist shadow event for session %s: %s",
                    event.session_id,
                    event.local_model,
                    exc_info=True,
                )
            self.local_text = text
            self.shadow_event = event
            self._local_ready = True
            logger.debug(
                "Bridge: local ready (text_len=%d, cloud_ready=%s)",
                len(text),
                self._cloud_ready,
            )
            if not text:
                logger.warning("Shadow: local_text is empty — similarity cannot be computed")
            if self._cloud_ready:
                self._compute_and_update(store, config)

    def set_local_error(self, event: ShadowDraftEvent, store: Store) -> None:
        """Called from shadow thread on error. Saves event under the lock."""
        with self._lock:
            if self.cancelled:
                return
            try:
                store.save_event(event)
            except Exception:
                logger.warning(
                    "Failed to persist shadow error event for session %s",
                    event.session_id,
                    exc_info=True,
                )
            self.shadow_event = event

    def _compute_and_update(self, store: Store, config: StateLoomConfig) -> None:
        """Compute similarity and update the event. Must be called under lock."""
        if not self.cloud_text or not self.local_text or self.shadow_event is None:
            logger.debug(
                "Bridge: skipping similarity (cloud_len=%d, local_len=%d, event=%s)",
                len(self.cloud_text),
                len(self.local_text),
                self.shadow_event is not None,
            )
            return

        sim = compute_similarity_auto(
            self.cloud_text,
            self.local_text,
            method=self._similarity_method,
            scorer=self._similarity_scorer,
        )
        if sim is None:
            logger.debug("Bridge: compute_similarity returned None")
            return

        event = self.shadow_event
        event.similarity_score = sim.score
        event.similarity_method = sim.method
        event.cloud_preview = sim.cloud_preview
        event.local_preview = sim.local_preview
        event.length_ratio = sim.length_ratio

        # Populate cloud metrics from bridge state
        event.cloud_cost = self.cloud_cost
        event.cloud_tokens = self.cloud_tokens
        event.cloud_latency_ms = self.cloud_latency_ms
        event.latency_ratio = (
            event.local_latency_ms / event.cloud_latency_ms if event.cloud_latency_ms > 0 else 0.0
        )
        event.cost_saved = self.cloud_cost  # Local is free

        logger.debug(
            "Bridge: similarity computed — score=%.3f method=%s cloud_len=%d local_len=%d",
            sim.score,
            sim.method,
            sim.cloud_length,
            sim.local_length,
        )

        # Re-save the updated event
        try:
            store.save_event(event)
        except Exception:
            logger.warning("Failed to update shadow event with similarity", exc_info=True)

        # Print similarity to console
        if config.console_output and event.similarity_score is not None:
            _print_similarity(event)


def _friendly_error(exc: Exception, model: str) -> str:
    """Convert raw exceptions into user-friendly shadow error messages."""
    exc_name = type(exc).__name__
    msg = str(exc)
    # Cloud-specific errors
    if "AuthenticationError" in exc_name or "401" in msg:
        return f"API key not configured for shadow model '{model}'"
    if "RateLimitError" in exc_name or "429" in msg:
        return f"rate limited on shadow model '{model}'"
    # Ollama returns 404 when the model isn't pulled
    if "404" in msg:
        return f"model '{model}' not found — run: ollama pull {model}"
    # Connection refused = Ollama not running
    if "connect" in msg.lower() and ("refused" in msg.lower() or "error" in msg.lower()):
        return "Ollama not reachable — is it running?"
    # Keep original message but truncate
    return msg[:120]


def _print_similarity(event: ShadowDraftEvent) -> None:
    """Print a similarity update line to console."""
    if event.similarity_score is None:
        return
    pct = int(event.similarity_score * 100)
    if event.similarity_score >= _SIMILARITY_HIGH_THRESHOLD:
        style = "bold green"
    elif event.similarity_score >= _SIMILARITY_MEDIUM_THRESHOLD:
        style = "bold yellow"
    else:
        style = "bold red"

    line = Text()
    line.append("[StateLoom] ", style="bold cyan")
    line.append("  MODEL TEST ", style="bold magenta")
    line.append(f"| {event.local_model} ", style="bold")
    line.append(f"| sim={pct}%", style=style)
    _console.print(line)


class ShadowMiddleware:
    """Model testing middleware — test candidate models against cloud calls.

    Runs one or more candidate local models in parallel with every cloud LLM
    call, comparing responses via similarity scoring. Smart eligibility filtering
    skips tool continuations, unsupported features, images, realtime data
    requests, and oversized contexts. Supports multi-model candidate testing
    and configurable sampling rates.

    Never delays or modifies the cloud response. All errors are caught and
    logged — fail-open by design.
    """

    def __init__(
        self,
        config: StateLoomConfig,
        store: Store,
        compliance_fn: Any = None,
        provider_keys: dict[str, str] | None = None,
    ) -> None:
        self._config = config
        self._store = store
        self._compliance_fn = compliance_fn
        self._provider_keys: dict[str, str] = provider_keys or {}
        self._client = OllamaClient(
            host=config.local_model_host,
            timeout=config.shadow.timeout,
        )
        self._executor: ThreadPoolExecutor | None = None
        self._similarity_scorer: SemanticSimilarityScorer | None = None
        if config.shadow.similarity_method in ("semantic", "auto"):
            try:
                self._similarity_scorer = SemanticSimilarityScorer(
                    model_name=config.shadow.similarity_model,
                )
            except Exception:
                logger.debug("SemanticSimilarityScorer creation failed", exc_info=True)

        # Skip counters for eligibility filter stats
        self._skip_counts: dict[str, int] = {}
        self._skip_lock = threading.Lock()

    def _get_executor(self, min_workers: int = 0) -> ThreadPoolExecutor:
        """Lazy-initialize the thread pool executor.

        Auto-adjusts pool size if more workers are needed for multi-model testing.
        """
        desired = max(self._config.shadow.max_workers, min_workers)
        if self._executor is None:
            self._executor = ThreadPoolExecutor(
                max_workers=desired,
                thread_name_prefix="stateloom-shadow",
            )
        return self._executor

    def _increment_skip(self, reason: str) -> None:
        """Increment a skip counter for a given reason."""
        with self._skip_lock:
            self._skip_counts[reason] = self._skip_counts.get(reason, 0) + 1

    def get_skip_stats(self) -> dict[str, int]:
        """Return a copy of the skip counter dict."""
        with self._skip_lock:
            return dict(self._skip_counts)

    def _check_eligibility(
        self, ctx: MiddlewareContext, *, has_cloud_candidate: bool = False
    ) -> tuple[bool, str]:
        """Check whether this call is eligible for model testing.

        Returns (eligible, skip_reason). If eligible is False, skip_reason
        describes why the call was filtered out.

        When *has_cloud_candidate* is True, local-model-specific filters
        (tool continuations, unsupported features, images, realtime data)
        are skipped because cloud models can handle them.
        """
        kwargs = ctx.request_kwargs
        messages = kwargs.get("messages", [])

        if not has_cloud_candidate:
            # 1. Tool-continuation — last message has role="tool" or contains tool_result
            if messages:
                last = messages[-1]
                if isinstance(last, dict):
                    if last.get("role") == "tool":
                        return False, "tool_continuation"
                    content = last.get("content")
                    if isinstance(content, list):
                        for block in content:
                            if isinstance(block, dict) and block.get("type") == "tool_result":
                                return False, "tool_continuation"

            # 2. Unsupported features — tools, functions, response_format, logprobs
            for key in ("tools", "functions", "response_format", "logprobs"):
                if kwargs.get(key):
                    return False, "unsupported_features"

            # 3. Image content
            for msg in messages:
                if not isinstance(msg, dict):
                    continue
                content = msg.get("content")
                if isinstance(content, list):
                    for block in content:
                        if isinstance(block, dict) and block.get("type") in (
                            "image_url",
                            "image",
                        ):
                            return False, "images"

            # 4. Realtime data — last user message matches realtime patterns
            for msg in reversed(messages):
                if isinstance(msg, dict) and msg.get("role") == "user":
                    user_content = msg.get("content", "")
                    if isinstance(user_content, str) and _REALTIME_PATTERNS.search(user_content):
                        return False, "realtime_data"
                    break

        # 5. Context too large — estimate tokens as len(text)/4
        max_ctx = self._config.shadow.max_context_tokens
        if max_ctx > 0:
            total_chars = 0
            for msg in messages:
                if isinstance(msg, dict):
                    c = msg.get("content", "")
                    if isinstance(c, str):
                        total_chars += len(c)
                    elif isinstance(c, list):
                        for block in c:
                            if isinstance(block, dict):
                                total_chars += len(block.get("text", ""))
            estimated_tokens = total_chars / 4
            if estimated_tokens > max_ctx:
                return False, "context_too_large"

        # 6. Sampling — checked last so skip stats reflect real filtering
        sample_rate = self._config.shadow.sample_rate
        if sample_rate < 1.0 and random.random() > sample_rate:  # noqa: S311
            return False, "sampling"

        return True, ""

    def _resolve_shadow_models(self, ctx: MiddlewareContext) -> list[str]:
        """Resolve candidate models for testing.

        Priority: session metadata → config.shadow.models → config.shadow.model
        """
        session = ctx.session

        # Per-session override
        if session.metadata.get("shadow_enabled") is False:
            return []
        session_model = session.metadata.get("shadow_model", "")
        if session_model:
            return [session_model]

        # Global config
        if not self._config.shadow.enabled:
            return []

        # Multi-model list takes priority
        if self._config.shadow.models:
            return list(self._config.shadow.models)

        # Single model fallback
        if self._config.shadow.model:
            return [self._config.shadow.model]

        return []

    async def process(
        self,
        ctx: MiddlewareContext,
        call_next: Callable[[MiddlewareContext], Awaitable[Any]],
    ) -> Any:
        """Launch model test call(s) fire-and-forget, then continue the pipeline."""
        shadow_models = self._resolve_shadow_models(ctx)

        # Skip if no models configured, already a local call,
        # streaming (comparison requires full response text),
        # or this call is itself a shadow call (prevent recursion)
        if not shadow_models or ctx.provider == Provider.LOCAL or ctx.is_streaming:
            return await call_next(ctx)

        # Compliance check: block_shadow only applies to local model candidates.
        # Cloud-to-cloud shadow testing (e.g. sonnet vs haiku) keeps data within
        # cloud providers, so compliance restrictions on local data leakage
        # don't apply.
        if self._compliance_fn:
            profile = self._compliance_fn(ctx.session.org_id, ctx.session.team_id)
            if profile and profile.block_shadow:
                shadow_models = [m for m in shadow_models if _is_cloud_model(m)]
                if not shadow_models:
                    return await call_next(ctx)

        # PII safety: if PII scanning is enabled and request contains PII,
        # skip shadow to prevent sending PII to the local model before the
        # downstream PII scanner can block/redact the request.
        if self._has_pii_risk(ctx):
            logger.debug("Model test: skipping due to PII risk (session=%s)", ctx.session.id)
            return await call_next(ctx)

        # Smart eligibility filter — relax local-only filters when any
        # candidate is a cloud model (cloud models handle tools/images/etc.)
        has_cloud = any(_is_cloud_model(m) for m in shadow_models)
        eligible, skip_reason = self._check_eligibility(ctx, has_cloud_candidate=has_cloud)
        if not eligible:
            self._increment_skip(skip_reason)
            logger.debug("Model test: skipping — %s (session=%s)", skip_reason, ctx.session.id)
            return await call_next(ctx)

        # Snapshot request kwargs before downstream middleware mutates them
        request_snapshot = copy.deepcopy(ctx.request_kwargs)
        session_id = ctx.session.id
        cloud_provider = ctx.provider
        cloud_model = ctx.model

        # Extract prompt preview for the shadow event (same as CostTracker)
        prompt_preview = ""
        messages = ctx.request_kwargs.get("messages", [])
        if isinstance(messages, list):
            for msg in reversed(messages):
                if isinstance(msg, dict) and msg.get("role") == "user":
                    content = msg.get("content", "")
                    if isinstance(content, str):
                        prompt_preview = content[:120]
                    break

        # Create one bridge per candidate model
        bridges: list[tuple[str, _SimilarityBridge]] = []
        for model in shadow_models:
            bridge = _SimilarityBridge(
                similarity_method=self._config.shadow.similarity_method,
                similarity_scorer=self._similarity_scorer,
            )
            bridges.append((model, bridge))

        # Fire-and-forget: one thread per candidate model
        executor = self._get_executor(min_workers=len(shadow_models))
        for model, bridge in bridges:
            future = executor.submit(
                self._shadow_sync,
                session_id=session_id,
                cloud_provider=cloud_provider,
                cloud_model=cloud_model,
                shadow_model=model,
                request_kwargs=request_snapshot,
                bridge=bridge,
                prompt_preview=prompt_preview,
            )
            future.add_done_callback(self._on_shadow_done)

        # Continue pipeline immediately — cloud call is never delayed
        try:
            result = await call_next(ctx)
        except Exception:
            # Downstream error (e.g. PII block) — cancel all bridges
            for _, bridge in bridges:
                bridge.cancel(store=self._store)
            raise

        # If the call was routed locally or served from cache, cancel —
        # comparing local-vs-local or cached results is meaningless.
        if ctx.provider == Provider.LOCAL or ctx.skip_call:
            logger.debug(
                "Model test: cancelling bridges (provider=%s, skip_call=%s)",
                ctx.provider,
                ctx.skip_call,
            )
            for _, bridge in bridges:
                bridge.cancel(store=self._store)
        else:
            try:
                cloud_text = extract_response_text(result, cloud_provider)
            except Exception:
                logger.debug(
                    "Model test: extract_response_text failed for provider=%s, type=%s",
                    cloud_provider,
                    type(result).__name__,
                    exc_info=True,
                )
                cloud_text = ""

            # Pass cloud metrics to all bridges
            cloud_cost = getattr(ctx, "cost", 0.0) or 0.0
            cloud_tokens = (getattr(ctx, "prompt_tokens", 0) or 0) + (
                getattr(ctx, "completion_tokens", 0) or 0
            )
            cloud_latency_ms = getattr(ctx, "latency_ms", 0.0) or 0.0
            for _, bridge in bridges:
                bridge.set_cloud(
                    cloud_text,
                    self._store,
                    self._config,
                    cloud_cost=cloud_cost,
                    cloud_tokens=cloud_tokens,
                    cloud_latency_ms=cloud_latency_ms,
                )

        return result

    def _has_pii_risk(self, ctx: MiddlewareContext) -> bool:
        """Check if the request might contain PII that shouldn't be sent to local.

        Performs a lightweight pre-scan using the PII scanner. This prevents
        PII data from being sent to the local model via the shadow call before
        the downstream PII scanner has a chance to block or redact it.
        """
        try:
            from stateloom.pii.scanner import PIIScanner

            scanner = PIIScanner()
            messages = ctx.request_kwargs.get("messages", [])
            for msg in messages:
                content = msg.get("content", "")
                if isinstance(content, str) and content:
                    detections = scanner.scan(content)
                    if detections:
                        return True
        except ImportError:
            pass
        except Exception:
            logger.warning(
                "Model test PII pre-scan failed — skipping to prevent PII leak",
                exc_info=True,
            )
            return True
        return False

    # Keep backward-compatible single-model resolver for tests
    def _resolve_shadow_model(self, ctx: MiddlewareContext) -> str:
        """Resolve shadow model from session metadata or config (compat)."""
        models = self._resolve_shadow_models(ctx)
        return models[0] if models else ""

    def _cloud_shadow_call(
        self, model: str, request_kwargs: dict[str, Any]
    ) -> tuple[str, int, int]:
        """Execute a shadow call directly against a cloud provider API.

        Uses httpx to call the provider's HTTP API directly, bypassing the
        intercepted SDK pipeline.  This avoids creating a separate session
        for the shadow call — the result is recorded as a ``ShadowDraftEvent``
        in the primary session instead.

        Returns ``(response_text, prompt_tokens, completion_tokens)``.
        """

        provider = self._resolve_cloud_provider(model)
        messages = request_kwargs.get("messages", [])
        timeout = self._config.shadow.timeout

        if provider == "anthropic":
            return self._call_anthropic(model, messages, request_kwargs, timeout)
        elif provider in ("google", "gemini"):
            return self._call_google(model, messages, request_kwargs, timeout)
        else:
            return self._call_openai(model, messages, request_kwargs, timeout)

    def _call_openai(
        self,
        model: str,
        messages: list[dict[str, Any]],
        request_kwargs: dict[str, Any],
        timeout: float,
    ) -> tuple[str, int, int]:
        import httpx

        key = self._provider_keys.get("openai", "")
        body: dict[str, Any] = {"model": model, "messages": messages}
        for k in ("temperature", "max_tokens", "top_p"):
            if k in request_kwargs:
                body[k] = request_kwargs[k]

        resp = httpx.post(
            "https://api.openai.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            json=body,
            timeout=timeout,
        )
        if resp.status_code >= 400:
            raise RuntimeError(f"{resp.status_code}: {resp.text[:200]}")
        data = resp.json()
        choices = data.get("choices", [])
        text = choices[0]["message"]["content"] if choices else ""
        usage = data.get("usage", {})
        return text or "", usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0)

    def _call_anthropic(
        self,
        model: str,
        messages: list[dict[str, Any]],
        request_kwargs: dict[str, Any],
        timeout: float,
    ) -> tuple[str, int, int]:
        import httpx

        key = self._provider_keys.get("anthropic", "")
        system = ""
        api_messages: list[dict[str, Any]] = []
        for msg in messages:
            if isinstance(msg, dict):
                if msg.get("role") == "system":
                    system = msg.get("content", "")
                else:
                    api_messages.append(
                        {"role": msg.get("role", "user"), "content": msg.get("content", "")}
                    )

        max_tokens = request_kwargs.get("max_tokens", 1024)
        body: dict[str, Any] = {
            "model": model,
            "messages": api_messages,
            "max_tokens": max_tokens,
        }
        if system:
            body["system"] = system
        for k in ("temperature", "top_p"):
            if k in request_kwargs:
                body[k] = request_kwargs[k]

        resp = httpx.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": key,
                "anthropic-version": "2023-06-01",
                "Content-Type": "application/json",
            },
            json=body,
            timeout=timeout,
        )
        if resp.status_code >= 400:
            raise RuntimeError(f"{resp.status_code}: {resp.text[:200]}")
        data = resp.json()
        content = data.get("content", [])
        text = "".join(b.get("text", "") for b in content if b.get("type") == "text")
        usage = data.get("usage", {})
        return text, usage.get("input_tokens", 0), usage.get("output_tokens", 0)

    def _call_google(
        self,
        model: str,
        messages: list[dict[str, Any]],
        request_kwargs: dict[str, Any],
        timeout: float,
    ) -> tuple[str, int, int]:
        import httpx

        key = self._provider_keys.get("google", "")
        system_text = ""
        contents: list[dict[str, Any]] = []
        for msg in messages:
            if not isinstance(msg, dict):
                continue
            role = msg.get("role", "user")
            content_text = msg.get("content", "")
            if role == "system":
                system_text = content_text
                continue
            gemini_role = "model" if role == "assistant" else "user"
            contents.append({"role": gemini_role, "parts": [{"text": content_text}]})

        body: dict[str, Any] = {"contents": contents}
        if system_text:
            body["systemInstruction"] = {"parts": [{"text": system_text}]}

        gen_config: dict[str, Any] = {}
        if "temperature" in request_kwargs:
            gen_config["temperature"] = request_kwargs["temperature"]
        if "max_tokens" in request_kwargs:
            gen_config["maxOutputTokens"] = request_kwargs["max_tokens"]
        if gen_config:
            body["generationConfig"] = gen_config

        resp = httpx.post(
            f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
            params={"key": key},
            headers={"Content-Type": "application/json"},
            json=body,
            timeout=timeout,
        )
        if resp.status_code >= 400:
            raise RuntimeError(f"{resp.status_code}: {resp.text[:200]}")
        data = resp.json()
        candidates = data.get("candidates", [])
        text = ""
        if candidates:
            parts = candidates[0].get("content", {}).get("parts", [])
            text = "".join(p.get("text", "") for p in parts)
        usage = data.get("usageMetadata", {})
        return text, usage.get("promptTokenCount", 0), usage.get("candidatesTokenCount", 0)

    def _shadow_sync(
        self,
        *,
        session_id: str,
        cloud_provider: str,
        cloud_model: str,
        shadow_model: str,
        request_kwargs: dict[str, Any],
        bridge: _SimilarityBridge,
        prompt_preview: str = "",
    ) -> None:
        """Run the shadow call synchronously (for thread pool)."""
        try:
            is_cloud = _is_cloud_model(shadow_model)
            start = time.perf_counter()

            if is_cloud:
                # Direct httpx call — returns (text, prompt_tokens, completion_tokens)
                local_text, prompt_tokens, completion_tokens = self._cloud_shadow_call(
                    shadow_model, request_kwargs
                )
                elapsed_ms = (time.perf_counter() - start) * 1000

                if bridge.cancelled:
                    return

                event = self._build_shadow_event(
                    session_id=session_id,
                    cloud_provider=cloud_provider,
                    cloud_model=cloud_model,
                    shadow_model=shadow_model,
                    response=None,
                    elapsed_ms=elapsed_ms,
                    status="success",
                    cloud_shadow_tokens=(prompt_tokens, completion_tokens),
                    prompt_preview=prompt_preview,
                )
            else:
                response = self._client.chat(
                    provider=cloud_provider,
                    model=shadow_model,
                    request_kwargs=request_kwargs,
                )
                elapsed_ms = (time.perf_counter() - start) * 1000

                if bridge.cancelled:
                    return

                event = self._build_shadow_event(
                    session_id=session_id,
                    cloud_provider=cloud_provider,
                    cloud_model=cloud_model,
                    shadow_model=shadow_model,
                    response=response,
                    elapsed_ms=elapsed_ms,
                    status="success",
                    prompt_preview=prompt_preview,
                )
                local_text = extract_response_text(response, "local")

            # Console output (immediate feedback, before bridge saves)
            self._print_shadow(event)

            # Bridge saves the event under its lock — if cancelled, event is not saved
            bridge.set_local(local_text, event, self._store, self._config)

        except Exception as e:
            if bridge.cancelled:
                return
            error_msg = _friendly_error(e, shadow_model)
            status = "timeout" if "timeout" in str(e).lower() else "error"
            event = self._build_shadow_event(
                session_id=session_id,
                cloud_provider=cloud_provider,
                cloud_model=cloud_model,
                shadow_model=shadow_model,
                response=None,
                elapsed_ms=0.0,
                status=status,
                error=error_msg,
                prompt_preview=prompt_preview,
            )
            self._print_shadow(event)
            bridge.set_local_error(event, self._store)

    @staticmethod
    def _resolve_cloud_provider(model: str) -> str:
        """Resolve a cloud model name to its provider string for text extraction."""
        from stateloom.chat import _resolve_provider

        return _resolve_provider(model)

    def _build_shadow_event(
        self,
        *,
        session_id: str,
        cloud_provider: str,
        cloud_model: str,
        shadow_model: str,
        response: Any,
        elapsed_ms: float,
        status: str,
        error: str = "",
        similarity: SimilarityResult | None = None,
        cloud_shadow_tokens: tuple[int, int] | None = None,
        prompt_preview: str = "",
    ) -> ShadowDraftEvent:
        """Create a ShadowDraftEvent (does NOT persist — bridge handles that)."""
        from stateloom.local.client import OllamaResponse

        local_prompt_tokens = 0
        local_completion_tokens = 0
        local_tokens = 0

        if cloud_shadow_tokens is not None:
            # Direct httpx cloud call — token counts returned explicitly
            local_prompt_tokens, local_completion_tokens = cloud_shadow_tokens
            local_tokens = local_prompt_tokens + local_completion_tokens
        elif isinstance(response, OllamaResponse):
            local_prompt_tokens = response.prompt_tokens
            local_completion_tokens = response.completion_tokens
            local_tokens = response.total_tokens

        # Deterministic ID: same session + shadow model → overwrites on rerun
        # (SQLite uses INSERT OR REPLACE on event id)
        import hashlib

        det_id = hashlib.sha256(f"{session_id}:shadow:{shadow_model}".encode()).hexdigest()[:16]

        event = ShadowDraftEvent(
            id=det_id,
            session_id=session_id,
            cloud_provider=cloud_provider,
            cloud_model=cloud_model,
            local_model=shadow_model,
            local_latency_ms=elapsed_ms,
            local_prompt_tokens=local_prompt_tokens,
            local_completion_tokens=local_completion_tokens,
            prompt_preview=prompt_preview,
            local_tokens=local_tokens,
            shadow_status=status,
            error_message=error,
        )

        # Populate similarity fields
        if similarity is not None:
            event.similarity_score = similarity.score
            event.similarity_method = similarity.method
            event.cloud_preview = similarity.cloud_preview
            event.local_preview = similarity.local_preview
            event.length_ratio = similarity.length_ratio

        return event

    # Keep backward-compatible name for tests that call _record_shadow_event
    def _record_shadow_event(self, **kwargs: Any) -> ShadowDraftEvent:
        """Create, persist, and print a ShadowDraftEvent."""
        event = self._build_shadow_event(**kwargs)
        try:
            self._store.save_event(event)
        except Exception:
            logger.debug("Failed to persist shadow event", exc_info=True)
        self._print_shadow(event)
        return event

    def _print_shadow(self, event: ShadowDraftEvent) -> None:
        """Print model test result to console."""
        if not self._config.console_output:
            return

        line = Text()
        line.append("[StateLoom] ", style="bold cyan")

        if event.shadow_status == "success":
            line.append("  MODEL TEST ", style="bold magenta")
            line.append(f"| {event.local_model} ", style="bold")
            line.append(f"| {event.local_latency_ms:.0f}ms ", style="dim")
            line.append(f"| {event.local_tokens} tok ", style="dim")
            if event.cost_saved > 0:
                line.append(f"| saved ${event.cost_saved:.4f}", style="yellow")
            if event.similarity_score is not None:
                pct = int(event.similarity_score * 100)
                if event.similarity_score >= _SIMILARITY_HIGH_THRESHOLD:
                    style = "bold green"
                elif event.similarity_score >= _SIMILARITY_MEDIUM_THRESHOLD:
                    style = "bold yellow"
                else:
                    style = "bold red"
                line.append(f" | sim={pct}%", style=style)
        elif event.shadow_status == "timeout":
            line.append("  TEST TIMEOUT ", style="bold yellow")
            line.append(f"| {event.local_model}", style="bold")
        else:
            line.append("  TEST ERROR ", style="bold red")
            line.append(f"| {event.local_model} ", style="bold")
            if event.error_message:
                line.append(f"| {event.error_message[:80]}", style="dim")

        _console.print(line)

    def _on_shadow_done(self, future: Any) -> None:
        """Callback for thread pool futures — log unexpected errors."""
        exc = future.exception()
        if exc:
            logger.warning("Model test thread error: %s", exc, exc_info=True)

    def shutdown(self) -> None:
        """Clean up the thread pool executor and HTTP client."""
        if self._executor is not None:
            self._executor.shutdown(wait=True)
            self._executor = None
        self._client.close()
