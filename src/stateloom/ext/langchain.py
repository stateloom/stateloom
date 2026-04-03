"""LangChain callback handler — first-class StateLoom observability for LangChain.

Provides tool-call tracking and LangChain-specific metadata alongside the full
middleware pipeline (PII scanning, guardrails, budget enforcement, etc.).

**Recommended usage** (auto_patch=True + callback handler)::

    import stateloom
    from stateloom.ext.langchain import StateLoomCallbackHandler

    stateloom.init()  # auto_patch=True by default
    handler = StateLoomCallbackHandler()  # auto-detects tools_only mode
    chain.invoke(input, config={"callbacks": [handler]})

With ``auto_patch=True``, the underlying SDK calls flow through the full middleware
pipeline. The callback handler adds LangChain-specific observability (tool names,
chain tracking) without duplicating LLM events.

**Standalone usage** (callback handler only, no middleware)::

    stateloom.init(auto_patch=False)
    handler = StateLoomCallbackHandler(tools_only=False)
    chain.invoke(input, config={"callbacks": [handler]})

This mode records LLM events directly to the store, bypassing the middleware
pipeline. Use it when you only need observability without PII/guardrail/budget
enforcement.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any
from uuid import UUID

from stateloom.core.event import LLMCallEvent, ToolCallEvent
from stateloom.core.types import Provider

if TYPE_CHECKING:
    from stateloom.gate import Gate

logger = logging.getLogger("stateloom.ext.langchain")

try:
    from langchain_core.callbacks import BaseCallbackHandler  # type: ignore[import-not-found]
    from langchain_core.outputs import LLMResult  # type: ignore[import-not-found]

    _HAS_LANGCHAIN = True
except ImportError:
    _HAS_LANGCHAIN = False
    BaseCallbackHandler = object  # type: ignore[assignment,misc]
    LLMResult = Any  # type: ignore[assignment,misc]

# ---------------------------------------------------------------------------
# Provider mapping from LangChain serialized id path
# ---------------------------------------------------------------------------
_PROVIDER_MAP: dict[str, str] = {
    "openai": Provider.OPENAI,
    "anthropic": Provider.ANTHROPIC,
    "google_genai": Provider.GEMINI,
    "google_vertexai": Provider.GEMINI,
    "fireworks": "fireworks",
    "together": "together",
    "cohere": "cohere",
    "mistralai": "mistral",
    "groq": "groq",
    "bedrock": "bedrock",
}


# ---------------------------------------------------------------------------
# Run-state dataclasses
# ---------------------------------------------------------------------------
@dataclass
class _LLMRunState:
    model: str = ""
    provider: str = ""
    start_time: float = 0.0
    step: int = 0
    session_id: str = ""
    is_streaming: bool = False
    prompt_preview: str = ""


@dataclass
class _ToolRunState:
    tool_name: str = ""
    start_time: float = 0.0
    step: int = 0
    session_id: str = ""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _extract_model_and_provider(serialized: dict[str, Any]) -> tuple[str, str]:
    """Extract (model, provider) from a LangChain ``serialized`` dict."""
    # --- model ---
    model = ""
    kwargs = serialized.get("kwargs", {})
    for key in ("model_name", "model", "model_id"):
        model = kwargs.get(key, "")
        if model:
            break
    if not model:
        invocation = kwargs.get("invocation_params", {})
        model = invocation.get("model_name", "") or invocation.get("model", "")

    # --- provider ---
    provider = "unknown"
    id_path: list[str] = serialized.get("id", [])
    if len(id_path) >= 3:
        raw = id_path[2]
        provider = _PROVIDER_MAP.get(raw, raw)

    return model, provider


def _extract_tokens(llm_output: dict[str, Any] | None) -> tuple[int, int]:
    """Return (prompt_tokens, completion_tokens) from LLMResult.llm_output."""
    if not llm_output:
        return 0, 0

    # OpenAI-style
    token_usage = llm_output.get("token_usage", {})
    if token_usage:
        return (
            token_usage.get("prompt_tokens", 0),
            token_usage.get("completion_tokens", 0),
        )

    # Anthropic-style
    usage = llm_output.get("usage", {})
    if usage:
        return (
            usage.get("input_tokens", 0),
            usage.get("output_tokens", 0),
        )

    return 0, 0


def _is_langchain_cache_hit(response: Any) -> bool:
    """Check if an LLMResult came from LangChain's cache layer.

    LangChain's cache layer modifies generations in detectable ways:
    - Strips ``usage_metadata`` to ``None`` (most common in recent versions)
    - Sets ``usage_metadata["total_cost"] = 0`` (explicit marker)
    - Zeros out token counts in ``usage_metadata``

    Fresh API responses from all major providers (OpenAI, Anthropic, Google)
    always populate ``usage_metadata``, so its absence is a reliable signal.
    """
    gens = getattr(response, "generations", None)
    if not gens:
        return False
    for gen_list in gens:
        for gen in gen_list:
            msg = getattr(gen, "message", None)
            if msg is None:
                continue
            usage_meta = getattr(msg, "usage_metadata", None)
            # Cache layer strips usage_metadata → None
            if usage_meta is None:
                return True
            # Explicit total_cost == 0 marker (dict or object)
            if isinstance(usage_meta, dict):
                if usage_meta.get("total_cost") == 0:
                    return True
                # Zero token counts — cache layer zeroed them out
                if (
                    usage_meta.get("input_tokens", -1) == 0
                    and usage_meta.get("output_tokens", -1) == 0
                ):
                    return True
            else:
                if getattr(usage_meta, "total_cost", None) == 0:
                    return True
                if (
                    getattr(usage_meta, "input_tokens", -1) == 0
                    and getattr(usage_meta, "output_tokens", -1) == 0
                ):
                    return True
    return False


# ---------------------------------------------------------------------------
# Callback handler
# ---------------------------------------------------------------------------
if _HAS_LANGCHAIN:

    class StateLoomCallbackHandler(BaseCallbackHandler):  # type: ignore[misc]
        """LangChain callback handler that records events in StateLoom.

        Tracks LLM calls (with cost / token / latency data) and tool calls.
        All callback methods are wrapped in try/except so they never crash the
        user's chain (consistent with StateLoom's *fail_open* philosophy).

        Args:
            gate: Explicit Gate reference. Resolved lazily from the global
                singleton when ``None``.
            tools_only: Controls whether LLM events are recorded by the callback.
                ``True`` — only record ``ToolCallEvent``, skip all LLM events
                (use when ``auto_patch=True`` handles LLM tracking via middleware).
                ``False`` — record both LLM and tool events (standalone mode).
                ``None`` (default) — auto-detect from ``gate.config.auto_patch``.
        """

        def __init__(
            self,
            gate: Gate | None = None,
            *,
            tools_only: bool | None = None,
        ) -> None:
            super().__init__()
            self._gate_ref: Gate | None = gate
            self._tools_only: bool | None = tools_only
            self._llm_runs: dict[UUID, _LLMRunState] = {}
            self._tool_runs: dict[UUID, _ToolRunState] = {}
            self._lock = threading.Lock()

            # Warn on likely misconfiguration: tools_only=True but auto_patch=False
            # means LLM events won't be recorded by either path.
            if tools_only is True:
                g = self._gate
                if g is not None and not g.config.auto_patch:
                    logger.warning(
                        "tools_only=True but auto_patch=False — LLM events "
                        "won't be recorded by either the callback handler or "
                        "the middleware pipeline"
                    )

        # -- gate resolution (lazy) ------------------------------------
        @property
        def _gate(self) -> Gate | None:
            if self._gate_ref is None:
                try:
                    import stateloom

                    self._gate_ref = stateloom.get_gate()
                except Exception:
                    return None
            return self._gate_ref

        @property
        def _effective_tools_only(self) -> bool:
            """Resolve the effective tools_only mode.

            Explicit ``True``/``False`` is honored. ``None`` auto-detects from
            ``gate.config.auto_patch`` — when auto_patch is on, the SDK
            interceptor already records LLM events through the middleware
            pipeline, so the callback should only add tool events.
            """
            if self._tools_only is not None:
                return self._tools_only
            gate = self._gate
            if gate is not None:
                return gate.config.auto_patch
            return False  # safe fallback: record everything

        # -- LLM callbacks ---------------------------------------------
        def on_llm_start(
            self,
            serialized: dict[str, Any],
            prompts: list[str],
            *,
            run_id: UUID,
            **kwargs: Any,
        ) -> None:
            try:
                # Set framework context for the middleware pipeline (even in tools_only mode)
                from stateloom.core.context import set_framework_context

                model, provider = _extract_model_and_provider(serialized)
                set_framework_context(
                    {
                        "langchain": {
                            "run_id": str(run_id),
                            "chain_name": serialized.get("name", ""),
                            "tags": kwargs.get("tags", []),
                            "model": model,
                            "provider": provider,
                        }
                    }
                )
                if self._effective_tools_only:
                    return
                gate = self._gate
                if gate is None:
                    return
                session = gate.get_or_create_session()
                step = session.next_step()
                prompt_preview = ""
                if prompts:
                    last = prompts[-1]
                    prompt_preview = (last[:50] + "...") if len(last) > 50 else last
                state = _LLMRunState(
                    model=model,
                    provider=provider,
                    start_time=time.perf_counter(),
                    step=step,
                    session_id=session.id,
                    prompt_preview=prompt_preview,
                )
                with self._lock:
                    self._llm_runs[run_id] = state
            except Exception:
                logger.debug("on_llm_start failed", exc_info=True)

        def on_chat_model_start(
            self,
            serialized: dict[str, Any],
            messages: list[list[Any]],
            *,
            run_id: UUID,
            **kwargs: Any,
        ) -> None:
            try:
                # Set framework context for the middleware pipeline (even in tools_only mode)
                from stateloom.core.context import set_framework_context

                model, provider = _extract_model_and_provider(serialized)
                set_framework_context(
                    {
                        "langchain": {
                            "run_id": str(run_id),
                            "chain_name": serialized.get("name", ""),
                            "tags": kwargs.get("tags", []),
                            "model": model,
                            "provider": provider,
                        }
                    }
                )
                if self._effective_tools_only:
                    return
                gate = self._gate
                if gate is None:
                    return
                session = gate.get_or_create_session()
                step = session.next_step()
                prompt_preview = ""
                if messages and messages[-1]:
                    for msg in reversed(messages[-1]):
                        if getattr(msg, "type", "") == "human":
                            content = getattr(msg, "content", "")
                            if isinstance(content, str):
                                prompt_preview = (
                                    (content[:50] + "...") if len(content) > 50 else content
                                )
                            break
                state = _LLMRunState(
                    model=model,
                    provider=provider,
                    start_time=time.perf_counter(),
                    step=step,
                    session_id=session.id,
                    prompt_preview=prompt_preview,
                )
                with self._lock:
                    self._llm_runs[run_id] = state
            except Exception:
                logger.debug("on_chat_model_start failed", exc_info=True)

        def on_llm_new_token(
            self,
            token: str,
            *,
            run_id: UUID,
            **kwargs: Any,
        ) -> None:
            try:
                with self._lock:
                    state = self._llm_runs.get(run_id)
                if state is not None:
                    state.is_streaming = True
            except Exception:
                logger.debug("on_llm_new_token failed", exc_info=True)

        def on_llm_end(
            self,
            response: LLMResult,
            *,
            run_id: UUID,
            **kwargs: Any,
        ) -> None:
            try:
                from stateloom.core.context import clear_framework_context

                clear_framework_context()
                with self._lock:
                    state = self._llm_runs.pop(run_id, None)
                if state is None:
                    return
                gate = self._gate
                if gate is None:
                    return

                latency_ms = (time.perf_counter() - state.start_time) * 1000
                llm_output = getattr(response, "llm_output", None) or {}

                # Detect LangChain-level cache hits (InMemoryCache, SQLiteCache,
                # etc.).  Three detection layers:
                #   1. llm_output is empty (no provider response metadata)
                #   2. usage_metadata signals (None, total_cost==0, zero tokens)
                #   3. Latency fallback — some LangChain versions preserve the
                #      full original response (llm_output + usage_metadata) on
                #      cache hits, making them indistinguishable by structure.
                #      Real API calls need network + generation time (>200ms);
                #      cache lookups complete in 1-5ms.  on_chat_model_start
                #      fires before the cache check, so latency is reliable.
                #      Only applied when response has LangChain's LLMResult
                #      structure (generations) to avoid false positives on raw
                #      SDK responses.
                is_cache_hit = not llm_output or _is_langchain_cache_hit(response)
                if not is_cache_hit and latency_ms < 100:
                    if getattr(response, "generations", None):
                        is_cache_hit = True
                if is_cache_hit:
                    prompt_tokens = completion_tokens = total_tokens = 0
                    cost = 0.0
                else:
                    prompt_tokens, completion_tokens = _extract_tokens(llm_output)
                    total_tokens = prompt_tokens + completion_tokens
                    cost = gate.pricing.calculate_cost(
                        state.model, prompt_tokens, completion_tokens
                    )

                event = LLMCallEvent(
                    session_id=state.session_id,
                    step=state.step,
                    provider=state.provider,
                    model=state.model,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=total_tokens,
                    cost=cost,
                    latency_ms=latency_ms,
                    is_streaming=state.is_streaming,
                    prompt_preview=state.prompt_preview,
                )

                session = gate.session_manager.get(state.session_id)
                if session is not None:
                    session.add_cost(cost, prompt_tokens, completion_tokens)
                gate.store.save_event(event)
            except Exception:
                logger.debug("on_llm_end failed", exc_info=True)

        def on_llm_error(
            self,
            error: BaseException,
            *,
            run_id: UUID,
            **kwargs: Any,
        ) -> None:
            try:
                from stateloom.core.context import clear_framework_context

                clear_framework_context()
                with self._lock:
                    self._llm_runs.pop(run_id, None)
            except Exception:
                logger.debug("on_llm_error failed", exc_info=True)

        # -- Tool callbacks --------------------------------------------
        def on_tool_start(
            self,
            serialized: dict[str, Any],
            input_str: str,
            *,
            run_id: UUID,
            **kwargs: Any,
        ) -> None:
            try:
                gate = self._gate
                if gate is None:
                    return
                tool_name = serialized.get("name", "") or kwargs.get("name", "unknown")
                session = gate.get_or_create_session()
                step = session.next_step()
                state = _ToolRunState(
                    tool_name=tool_name,
                    start_time=time.perf_counter(),
                    step=step,
                    session_id=session.id,
                )
                with self._lock:
                    self._tool_runs[run_id] = state
            except Exception:
                logger.debug("on_tool_start failed", exc_info=True)

        def on_tool_end(
            self,
            output: str,
            *,
            run_id: UUID,
            **kwargs: Any,
        ) -> None:
            try:
                with self._lock:
                    state = self._tool_runs.pop(run_id, None)
                if state is None:
                    return
                gate = self._gate
                if gate is None:
                    return

                latency_ms = (time.perf_counter() - state.start_time) * 1000
                event = ToolCallEvent(
                    session_id=state.session_id,
                    step=state.step,
                    tool_name=state.tool_name,
                    mutates_state=False,
                    latency_ms=latency_ms,
                )
                session = gate.session_manager.get(state.session_id)
                if session is not None:
                    session.call_count += 1
                gate.store.save_event(event)
            except Exception:
                logger.debug("on_tool_end failed", exc_info=True)

        def on_tool_error(
            self,
            error: BaseException,
            *,
            run_id: UUID,
            **kwargs: Any,
        ) -> None:
            try:
                with self._lock:
                    self._tool_runs.pop(run_id, None)
            except Exception:
                logger.debug("on_tool_error failed", exc_info=True)

        # -- Chain callbacks (no-op) -----------------------------------
        def on_chain_start(
            self, serialized: dict[str, Any], inputs: dict[str, Any], **kwargs: Any
        ) -> None:
            pass

        def on_chain_end(self, outputs: dict[str, Any], **kwargs: Any) -> None:
            pass

        def on_chain_error(self, error: BaseException, **kwargs: Any) -> None:
            pass

else:

    class StateLoomCallbackHandler:  # type: ignore[no-redef]  # Stub when langchain not installed
        """Stub that raises ImportError when langchain_core is not installed."""

        def __init__(
            self,
            *args: Any,
            gate: Any = None,
            tools_only: bool | None = None,
            **kwargs: Any,
        ) -> None:
            raise ImportError(
                "langchain_core is required to use StateLoomCallbackHandler. "
                "Install with: pip install stateloom[langchain]"
            )
