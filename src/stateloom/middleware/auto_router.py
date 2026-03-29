"""Intelligent auto-routing middleware — route simple requests to local models."""

from __future__ import annotations

import inspect
import logging
import re
import threading
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from stateloom.core.config import StateLoomConfig
from stateloom.core.event import LocalRoutingEvent
from stateloom.core.types import Provider
from stateloom.local.client import OllamaClient, OllamaResponse
from stateloom.middleware.base import MiddlewareContext
from stateloom.middleware.response_converter import convert_response
from stateloom.store.base import Store

logger = logging.getLogger("stateloom.middleware.auto_router")

# Realtime / external-data indicators — requests matching these should go to cloud
_REALTIME_PATTERNS = re.compile(
    r"\b(?:weather|forecast|stock\s*(?:price|market)|current(?:ly)?|right\s+now|"
    r"latest\s+news|live\s+score|today'?s?\s+(?:date|news|price|score)|"
    r"real[- ]?time|breaking\s+news|trending|just\s+happened|happening\s+now)\b",
    re.IGNORECASE,
)

# Complexity keyword indicators — short prompts containing these are non-trivial
_COMPLEXITY_INDICATORS = re.compile(
    r"\b(?:"
    r"design|architect|implement|compare|analy[sz]e|evaluate|"
    r"explain\s+(?:how|why)|trade[- ]?offs?|step[- ]by[- ]step|comprehensive|"
    r"distributed|algorithm|optimiz(?:e|ation)|refactor|migrat(?:e|ion)|"
    r"security|authentication|"
    r"database\s+(?:schema|design)|api\s+design|"
    r"system\s+(?:design|architecture)|"
    r"data\s+(?:model|structure|pipeline)|"
    r"machine\s+learning|neural\s+network|"
    r"write\s+(?:a\s+)?(?:function|class|code|program|script|module)|"
    r"debug|troubleshoot|diagnose"
    r")\b",
    re.IGNORECASE,
)

# Local-model refusal/inability phrases — triggers cloud reroute
_INADEQUATE_PATTERNS = re.compile(
    r"(?:i\s+(?:don't|do\s+not|cannot|can't|am\s+unable\s+to)\s+"
    r"(?:have\s+access\s+to|access|provide|check|browse|retrieve)\s+"
    r"(?:real[- ]?time|current|live|the\s+internet|the\s+web|up[- ]to[- ]date|latest))|"
    r"(?:my\s+(?:knowledge|training)\s+(?:cutoff|data)\b)|"
    r"(?:i\s+(?:don't|do\s+not)\s+have\s+(?:current|up[- ]to[- ]date|real[- ]?time)\s+"
    r"(?:information|data|access))",
    re.IGNORECASE,
)

# Models considered "high tier" (complex tasks)
_HIGH_TIER_MODELS = frozenset(
    {
        "gpt-4",
        "gpt-4-turbo",
        "gpt-4o",
        "gpt-4o-mini",
        "claude-3-opus",
        "claude-3-5-sonnet",
        "claude-3.5-sonnet",
        "claude-opus-4",
        "claude-sonnet-4",
        "gemini-1.5-pro",
        "gemini-2.0-pro",
    }
)

# Models considered "budget tier" (simple tasks)
_BUDGET_MODELS = frozenset(
    {
        "gpt-3.5-turbo",
        "gpt-4o-mini",
        "claude-3-haiku",
        "claude-3-5-haiku",
        "claude-haiku-4",
        "gemini-1.5-flash",
        "gemini-2.0-flash",
    }
)


@dataclass
class ComplexitySignals:
    """Breakdown of complexity heuristic analysis."""

    estimated_tokens: float = 0.0
    conversation_depth: float = 0.0
    message_count: float = 0.0
    max_message_length: float = 0.0
    system_prompt: float = 0.0
    model_tier: float = 0.0
    complexity_keywords: float = 0.0


@dataclass
class RoutingDecision:
    """Result of the routing decision process."""

    route_local: bool = False
    reason: str = ""
    complexity_score: float = 0.0
    budget_pressure: float = 0.0
    probed: bool = False
    probe_confidence: float | None = None
    historical_success_rate: float | None = None
    semantic_complexity: float | None = None
    custom_scorer_used: bool = False


@dataclass(frozen=True)
class RoutingContext:
    """Context provided to custom routing scorers."""

    prompt: str
    messages: list[dict[str, Any]]
    model: str
    provider: str
    session_id: str
    session_metadata: dict[str, Any]
    org_id: str
    team_id: str
    total_cost: float
    budget: float | None
    call_count: int
    is_streaming: bool
    has_tools: bool
    has_images: bool
    local_model: str


@dataclass
class _ModelStats:
    """In-memory routing outcome stats for a cloud model."""

    successes: int = 0
    failures: int = 0

    @property
    def total(self) -> int:
        return self.successes + self.failures

    @property
    def success_rate(self) -> float:
        if self.total == 0:
            return 0.0
        return self.successes / self.total


class AutoRouterMiddleware:
    """Routes simple requests to a local model to save cost.

    Analyzes complexity, factors budget pressure and historical success,
    optionally probes the local model, and silently falls back to cloud
    on any failure.
    """

    def __init__(
        self,
        config: StateLoomConfig,
        store: Store,
        pricing: Any = None,
        semantic_matcher: Any = None,
        compliance_fn: Any = None,
        semantic_classifier: Any = None,
        custom_scorer: Callable[..., bool | float | None] | None = None,
    ) -> None:
        self._config = config
        self._store = store
        self._pricing = pricing
        self._compliance_fn = compliance_fn
        self._custom_scorer = custom_scorer
        self._custom_scorer_wants_context = False
        if custom_scorer is not None:
            self._introspect_scorer(custom_scorer)
        self._client = OllamaClient(
            host=config.local_model_host,
            timeout=config.auto_route.timeout,
        )
        self._probe_client = OllamaClient(
            host=config.local_model_host,
            timeout=config.auto_route.probe_timeout,
        )
        # In-memory routing stats keyed by cloud model name.
        # Bounded by distinct model count (typically <20), no eviction needed.
        self._stats: dict[str, _ModelStats] = {}
        # Lock: _stats_lock guards _stats dict only (no nesting)
        self._stats_lock = threading.Lock()
        # Cached Ollama availability check (60s TTL)
        self._ollama_available: bool | None = None
        self._ollama_check_time: float = 0.0

        # Semantic complexity classifier (optional)
        # Use shared instance if provided (from PrecomputeMiddleware),
        # otherwise create own instance
        self._semantic_classifier = semantic_classifier
        if self._semantic_classifier is None and config.auto_route.semantic_enabled:
            try:
                from stateloom.middleware.semantic_router import SemanticComplexityClassifier

                self._semantic_classifier = SemanticComplexityClassifier(
                    model_name=config.auto_route.semantic_model,
                )
            except Exception:
                logger.debug("Semantic classifier init failed", exc_info=True)

        # Load historical stats from store
        self._load_historical_stats()

    def _introspect_scorer(self, scorer: Callable) -> None:
        """Determine whether the scorer expects a RoutingContext or plain str."""
        try:
            sig = inspect.signature(scorer)
            params = [
                p
                for p in sig.parameters.values()
                if p.default is inspect.Parameter.empty
                and p.kind
                in (
                    inspect.Parameter.POSITIONAL_ONLY,
                    inspect.Parameter.POSITIONAL_OR_KEYWORD,
                )
            ]
            if params:
                ann = params[0].annotation
                self._custom_scorer_wants_context = ann is RoutingContext or (
                    isinstance(ann, str) and "RoutingContext" in ann
                )
            else:
                self._custom_scorer_wants_context = False
        except (ValueError, TypeError):
            self._custom_scorer_wants_context = False

    def _build_routing_context(self, ctx: MiddlewareContext) -> RoutingContext:
        """Build a RoutingContext from the current middleware context."""
        kwargs = ctx.request_kwargs
        return RoutingContext(
            prompt=self._extract_last_user_text(ctx),
            messages=list(kwargs.get("messages", [])),
            model=ctx.model,
            provider=ctx.provider,
            session_id=ctx.session.id,
            session_metadata=dict(ctx.session.metadata),
            org_id=ctx.session.org_id,
            team_id=ctx.session.team_id,
            total_cost=ctx.session.total_cost,
            budget=ctx.session.budget,
            call_count=ctx.session.call_count,
            is_streaming=ctx.is_streaming,
            has_tools=any(
                k in kwargs for k in ("tools", "functions", "tool_choice", "function_call")
            ),
            has_images=self._has_images(kwargs),
            local_model=self._resolve_local_model(ctx),
        )

    def _run_custom_scorer(self, ctx: MiddlewareContext) -> bool | float | None:
        """Run the custom scorer callback (fail-open)."""
        if self._custom_scorer is None:
            return None
        try:
            if self._custom_scorer_wants_context:
                return self._custom_scorer(self._build_routing_context(ctx))
            return self._custom_scorer(self._extract_last_user_text(ctx))
        except Exception:
            logger.debug("Custom scorer raised, falling through", exc_info=True)
            return None

    def _load_historical_stats(self) -> None:
        """Load routing stats from persisted LocalRoutingEvents (fail-open)."""
        try:
            events = self._store.get_session_events("", event_type="local_routing", limit=10000)
            for event in events:
                if not isinstance(event, LocalRoutingEvent):
                    continue
                cloud_model = event.original_cloud_model
                if not cloud_model:
                    continue
                if cloud_model not in self._stats:
                    self._stats[cloud_model] = _ModelStats()
                if event.routing_success:
                    self._stats[cloud_model].successes += 1
                else:
                    self._stats[cloud_model].failures += 1
            if self._stats:
                logger.debug("Loaded historical routing stats for %d models", len(self._stats))
        except Exception:
            logger.debug("Failed to load historical routing stats", exc_info=True)

    async def process(
        self,
        ctx: MiddlewareContext,
        call_next: Callable[[MiddlewareContext], Awaitable[Any]],
    ) -> Any:
        """Route to local if appropriate, otherwise continue to cloud."""
        # Resolve pre-computed complexity score if available
        if ctx._precomputed_complexity_future is not None:
            try:
                ctx._precomputed_complexity_score = await ctx._precomputed_complexity_future
            except Exception:
                pass  # Fail-open, falls back to inline computation

        # Compliance check: block local routing if profile says so
        if self._compliance_fn:
            profile = self._compliance_fn(ctx.session.org_id, ctx.session.team_id)
            if profile and profile.block_local_routing:
                return await call_next(ctx)

        decision = self._should_route_local(ctx)

        if decision.route_local:
            # Save originals before ctx gets mutated by _route_to_local
            original_provider = ctx.provider
            original_model = ctx.model

            result = self._route_to_local(ctx, decision)
            if result is not None:
                # Record success event
                self._record_event(
                    ctx,
                    decision,
                    success=True,
                    original_provider=original_provider,
                    original_model=original_model,
                )
                self._update_stats(original_model, success=True)
                return await call_next(ctx)

            # Local call failed — record failure, fall through to cloud
            self._record_event(
                ctx,
                decision,
                success=False,
                original_provider=original_provider,
                original_model=original_model,
            )
            self._update_stats(original_model, success=False)
        elif decision.reason and decision.reason not in (
            "not eligible (SDK direct call)",
            "auto_route disabled",
            "already local",
            "streaming not supported",
            "skip_call set (cache hit)",
        ):
            # Record cloud decision so trace shows complexity score/reason
            self._record_event(
                ctx,
                decision,
                success=False,
                original_provider=ctx.provider,
                original_model=ctx.model,
            )

        return await call_next(ctx)

    def _should_route_local(self, ctx: MiddlewareContext) -> RoutingDecision:
        """Determine whether this request should be routed to local."""
        # Gate checks — any of these skip routing
        if not ctx.auto_route_eligible:
            return RoutingDecision(reason="not eligible (SDK direct call)")

        if not self._config.auto_route.enabled:
            return RoutingDecision(reason="auto_route disabled")

        if ctx.provider == Provider.LOCAL:
            return RoutingDecision(reason="already local")

        if ctx.is_streaming:
            return RoutingDecision(reason="streaming not supported")

        if ctx.skip_call:
            return RoutingDecision(reason="skip_call set (cache hit)")

        # Force-local mode: admin override, skip complexity analysis
        if self._config.auto_route.force_local:
            kwargs = ctx.request_kwargs
            if ctx.is_streaming:
                return RoutingDecision(reason="force-local: streaming unsupported")
            if any(k in kwargs for k in ("tools", "functions", "tool_choice", "function_call")):
                return RoutingDecision(reason="force-local: tools unsupported")
            if any(k in kwargs for k in ("response_format", "logprobs")):
                return RoutingDecision(reason="force-local: response_format/logprobs unsupported")
            if self._has_images(kwargs):
                return RoutingDecision(reason="force-local: images unsupported")
            if not self._check_ollama_available():
                return RoutingDecision(reason="force-local: Ollama unavailable")
            return RoutingDecision(
                route_local=True,
                reason="force-local mode",
                complexity_score=0.0,
                budget_pressure=0.0,
            )

        # Session opt-out
        if ctx.session.metadata.get("auto_route_enabled") is False:
            return RoutingDecision(reason="session opt-out")

        # Check for tools/functions
        kwargs = ctx.request_kwargs
        if any(k in kwargs for k in ("tools", "functions", "tool_choice", "function_call")):
            return RoutingDecision(reason="tools/functions present")

        # Check for response_format or logprobs
        if any(k in kwargs for k in ("response_format", "logprobs")):
            return RoutingDecision(reason="response_format/logprobs present")

        # Check for image content blocks
        if self._has_images(kwargs):
            return RoutingDecision(reason="images in messages")

        # Check for realtime/external data need
        last_text = self._extract_last_user_text(ctx)
        if last_text and self._needs_realtime_data(last_text):
            logger.debug(
                "Skipping local route: request requires realtime data (%r)", last_text[:80]
            )
            return RoutingDecision(reason="realtime data request")

        # Check Ollama availability (cached, 60s TTL)
        if not self._check_ollama_available():
            return RoutingDecision(reason="Ollama unavailable")

        # Custom scorer (highest scoring priority)
        custom_result = self._run_custom_scorer(ctx)
        if custom_result is not None:
            if isinstance(custom_result, bool):
                return RoutingDecision(
                    route_local=custom_result,
                    reason="custom scorer" if custom_result else "custom scorer (cloud)",
                    complexity_score=0.0 if custom_result else 1.0,
                    budget_pressure=self._compute_budget_pressure(ctx),
                    custom_scorer_used=True,
                )
            elif isinstance(custom_result, (int, float)):
                # Use as complexity score — fall through to threshold logic below
                complexity_score = float(max(0.0, min(1.0, custom_result)))
                semantic_score: float | None = None
                budget_pressure = self._compute_budget_pressure(ctx)
                historical_rate = self._get_historical_rate(ctx.model)
                threshold = self._config.auto_route.complexity_threshold
                complex_floor = self._config.auto_route.complex_floor
                threshold -= budget_pressure * 0.3
                if historical_rate is not None:
                    if historical_rate < 0.3:
                        threshold *= 0.7
                    elif historical_rate > 0.8:
                        threshold *= 1.2
                if complexity_score >= complex_floor:
                    return RoutingDecision(
                        reason="custom scorer: complexity above floor",
                        complexity_score=complexity_score,
                        budget_pressure=budget_pressure,
                        historical_success_rate=historical_rate,
                        custom_scorer_used=True,
                    )
                if complexity_score < threshold:
                    return RoutingDecision(
                        route_local=True,
                        reason="custom scorer: low complexity",
                        complexity_score=complexity_score,
                        budget_pressure=budget_pressure,
                        historical_success_rate=historical_rate,
                        custom_scorer_used=True,
                    )
                if self._config.auto_route.probe_enabled:
                    probe_confidence = self._probe_local(ctx)
                    if (
                        probe_confidence is not None
                        and probe_confidence >= self._config.auto_route.probe_threshold
                    ):
                        return RoutingDecision(
                            route_local=True,
                            reason="custom scorer: probe confident",
                            complexity_score=complexity_score,
                            budget_pressure=budget_pressure,
                            probed=True,
                            probe_confidence=probe_confidence,
                            historical_success_rate=historical_rate,
                            custom_scorer_used=True,
                        )
                    return RoutingDecision(
                        reason="custom scorer: probe not confident"
                        if probe_confidence is not None
                        else "custom scorer: probe failed",
                        complexity_score=complexity_score,
                        budget_pressure=budget_pressure,
                        probed=True,
                        probe_confidence=probe_confidence,
                        historical_success_rate=historical_rate,
                        custom_scorer_used=True,
                    )
                return RoutingDecision(
                    reason="custom scorer: uncertain zone",
                    complexity_score=complexity_score,
                    budget_pressure=budget_pressure,
                    historical_success_rate=historical_rate,
                    custom_scorer_used=True,
                )

        # Compute complexity score — pre-computed first, then semantic, heuristic fallback
        semantic_score = ctx._precomputed_complexity_score
        if semantic_score is None and self._semantic_classifier is not None:
            last_text = self._extract_last_user_text(ctx)
            if last_text:
                semantic_score = self._semantic_classifier.classify(last_text)

        if semantic_score is not None:
            complexity_score = semantic_score
        else:
            signals = self._analyze_complexity(ctx)
            complexity_score = self._compute_score(signals)

        # Compute budget pressure
        budget_pressure = self._compute_budget_pressure(ctx)

        # Get historical success rate
        historical_rate = self._get_historical_rate(ctx.model)

        # Compute effective threshold (adjusted by budget and history)
        threshold = self._config.auto_route.complexity_threshold
        complex_floor = self._config.auto_route.complex_floor

        # Budget pressure lowers the threshold (more aggressive local routing)
        threshold -= budget_pressure * 0.3

        # Historical learning adjusts threshold
        if historical_rate is not None:
            if historical_rate < 0.3:
                threshold *= 0.7  # Harder to route local (poor history)
            elif historical_rate > 0.8:
                threshold *= 1.2  # Easier to route local (good history)

        # Always cloud if above complex floor
        if complexity_score >= complex_floor:
            return RoutingDecision(
                reason="complexity above floor",
                complexity_score=complexity_score,
                budget_pressure=budget_pressure,
                historical_success_rate=historical_rate,
                semantic_complexity=semantic_score,
            )

        # Clearly simple — route local
        if complexity_score < threshold:
            return RoutingDecision(
                route_local=True,
                reason="low complexity",
                complexity_score=complexity_score,
                budget_pressure=budget_pressure,
                historical_success_rate=historical_rate,
                semantic_complexity=semantic_score,
            )

        # Uncertain zone — probe if enabled
        if self._config.auto_route.probe_enabled:
            probe_confidence = self._probe_local(ctx)
            if (
                probe_confidence is not None
                and probe_confidence >= self._config.auto_route.probe_threshold
            ):
                return RoutingDecision(
                    route_local=True,
                    reason="probe confident",
                    complexity_score=complexity_score,
                    budget_pressure=budget_pressure,
                    probed=True,
                    probe_confidence=probe_confidence,
                    historical_success_rate=historical_rate,
                    semantic_complexity=semantic_score,
                )
            return RoutingDecision(
                reason="probe not confident" if probe_confidence is not None else "probe failed",
                complexity_score=complexity_score,
                budget_pressure=budget_pressure,
                probed=True,
                probe_confidence=probe_confidence,
                historical_success_rate=historical_rate,
                semantic_complexity=semantic_score,
            )

        # No probe — uncertain zone defaults to cloud
        return RoutingDecision(
            reason="uncertain zone, probe disabled",
            complexity_score=complexity_score,
            budget_pressure=budget_pressure,
            historical_success_rate=historical_rate,
            semantic_complexity=semantic_score,
        )

    @staticmethod
    def _needs_realtime_data(text: str) -> bool:
        """Check if the text indicates a need for realtime or external data."""
        return bool(_REALTIME_PATTERNS.search(text))

    @staticmethod
    def _is_inadequate_response(content: str) -> bool:
        """Check if the local model response indicates inability to answer."""
        return bool(_INADEQUATE_PATTERNS.search(content))

    @staticmethod
    def _normalize_messages(ctx: MiddlewareContext) -> list[dict[str, str]]:
        """Extract messages in a provider-agnostic way.

        Handles OpenAI (messages), Anthropic (system + messages), and
        Gemini (system_instruction + contents with parts) formats.
        """
        kwargs = ctx.request_kwargs
        messages: list[dict[str, str]] = []

        # Gemini format: contents + system_instruction
        contents = kwargs.get("contents", [])
        if contents:
            sys_instr = kwargs.get("system_instruction")
            if sys_instr:
                messages.append({"role": "system", "content": str(sys_instr)})
            for entry in contents:
                if isinstance(entry, dict):
                    gemini_role = entry.get("role", "user")
                    role = "assistant" if gemini_role == "model" else gemini_role
                    parts = entry.get("parts", [])
                    text_parts = [
                        p["text"] if isinstance(p, dict) and "text" in p else str(p)
                        for p in parts
                        if isinstance(p, (dict, str))
                    ]
                    messages.append({"role": role, "content": " ".join(text_parts)})
            return messages

        # Anthropic: top-level system param
        system = kwargs.get("system")
        if system:
            if isinstance(system, str):
                messages.append({"role": "system", "content": system})
            elif isinstance(system, list):
                parts = [
                    b.get("text", "")
                    for b in system
                    if isinstance(b, dict) and b.get("type") == "text"
                ]
                if parts:
                    messages.append({"role": "system", "content": "\n".join(parts)})

        # OpenAI / standard format
        for msg in kwargs.get("messages", []):
            if isinstance(msg, dict):
                role = msg.get("role", "user")
                content = msg.get("content", "")
                if isinstance(content, list):
                    content = " ".join(
                        b.get("text", "")
                        for b in content
                        if isinstance(b, dict) and b.get("type") == "text"
                    )
                messages.append({"role": role, "content": str(content)})

        return messages

    @staticmethod
    def _extract_last_user_text(ctx: MiddlewareContext) -> str:
        """Extract the last user message text from request kwargs."""
        messages = AutoRouterMiddleware._normalize_messages(ctx)
        for msg in reversed(messages):
            if msg.get("role") == "user":
                return msg.get("content", "")
        return ""

    def _analyze_complexity(self, ctx: MiddlewareContext) -> ComplexitySignals:
        """Analyze request complexity using heuristics."""
        messages = self._normalize_messages(ctx)
        system = ""

        # Estimate total input tokens (rough: 1 token ≈ 4 chars)
        total_chars = 0
        max_msg_len = 0
        user_assistant_pairs = 0
        prev_role = ""

        for msg in messages:
            content = msg.get("content", "")
            msg_len = len(content)
            total_chars += msg_len
            max_msg_len = max(max_msg_len, msg_len)

            role = msg.get("role", "")
            if role == "assistant" and prev_role == "user":
                user_assistant_pairs += 1
            prev_role = role

        estimated_tokens = total_chars / 4.0

        # Factor: estimated input tokens (0.0 at <200, 1.0 at >4000)
        token_factor = min(1.0, max(0.0, (estimated_tokens - 200) / 3800))

        # Factor: conversation depth (0.0 at 0 pairs, 1.0 at 5+)
        depth_factor = min(1.0, user_assistant_pairs / 5.0)

        # Factor: message count (0.0 at 1-2, 1.0 at 10+)
        msg_count = len(messages)
        count_factor = min(1.0, max(0.0, (msg_count - 2) / 8.0))

        # Factor: max single message length (0.0 at <500, 1.0 at >8000)
        length_factor = min(1.0, max(0.0, (max_msg_len - 500) / 7500))

        # Factor: system prompt present (0.0 or 0.3)
        has_system = bool(system) or any(
            (isinstance(m, dict) and m.get("role") == "system") for m in messages
        )
        system_factor = 0.3 if has_system else 0.0

        # Factor: model tier hint
        model_lower = ctx.model.lower()
        if any(m in model_lower for m in ("gpt-4", "opus", "pro")):
            tier_factor = 0.5
        elif any(m in model_lower for m in ("gpt-3.5", "haiku", "flash", "mini")):
            tier_factor = 0.0
        else:
            tier_factor = 0.25

        # Factor: complexity keywords in last user message
        last_user_text = self._extract_last_user_text(ctx)
        keyword_matches = len(_COMPLEXITY_INDICATORS.findall(last_user_text))
        keyword_factor = min(1.0, keyword_matches * 0.3)

        return ComplexitySignals(
            estimated_tokens=token_factor,
            conversation_depth=depth_factor,
            message_count=count_factor,
            max_message_length=length_factor,
            system_prompt=system_factor,
            model_tier=tier_factor,
            complexity_keywords=keyword_factor,
        )

    def _compute_score(self, signals: ComplexitySignals) -> float:
        """Compute weighted complexity score from signals."""
        return (
            signals.estimated_tokens * 0.20
            + signals.conversation_depth * 0.05
            + signals.message_count * 0.05
            + signals.max_message_length * 0.10
            + signals.system_prompt * 0.10
            + signals.model_tier * 0.05
            + signals.complexity_keywords * 0.45
        )

    def _compute_budget_pressure(self, ctx: MiddlewareContext) -> float:
        """Compute budget pressure (0.0 to 1.0)."""
        budget = ctx.session.budget
        if not budget or budget <= 0:
            return 0.0
        spent = ctx.session.total_cost
        ratio = spent / budget
        if ratio <= 0.5:
            return 0.0
        return min(1.0, (ratio - 0.5) / 0.5)

    def _get_historical_rate(self, cloud_model: str) -> float | None:
        """Get historical success rate for routing this cloud model locally."""
        with self._stats_lock:
            stats = self._stats.get(cloud_model)
            if stats is None or stats.total < 5:
                return None  # Cold start
            return stats.success_rate

    def _update_stats(self, cloud_model: str, *, success: bool) -> None:
        """Update in-memory routing stats."""
        with self._stats_lock:
            if cloud_model not in self._stats:
                self._stats[cloud_model] = _ModelStats()
            if success:
                self._stats[cloud_model].successes += 1
            else:
                self._stats[cloud_model].failures += 1

    def _has_images(self, kwargs: dict[str, Any]) -> bool:
        """Check if request contains image content blocks."""
        messages = kwargs.get("messages", [])
        for msg in messages:
            if not isinstance(msg, dict):
                continue
            content = msg.get("content")
            if isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and block.get("type") in ("image_url", "image"):
                        return True
        return False

    def _check_ollama_available(self) -> bool:
        """Check Ollama availability with 60s TTL cache."""
        now = time.monotonic()
        if self._ollama_available is not None and (now - self._ollama_check_time) < 60.0:
            return self._ollama_available
        self._ollama_available = self._client.is_available()
        self._ollama_check_time = now
        return self._ollama_available

    def _probe_local(self, ctx: MiddlewareContext) -> float | None:
        """Probe the local model for self-assessed confidence.

        Returns confidence as float [0, 1] or None on failure.
        """
        try:
            # Extract last user message, truncated
            messages = ctx.request_kwargs.get("messages", [])
            last_user_msg = ""
            for msg in reversed(messages):
                if isinstance(msg, dict) and msg.get("role") == "user":
                    content = msg.get("content", "")
                    if isinstance(content, list):
                        content = " ".join(
                            b.get("text", "")
                            for b in content
                            if isinstance(b, dict) and b.get("type") == "text"
                        )
                    last_user_msg = str(content)[:500]
                    break

            if not last_user_msg:
                return None

            local_model = self._resolve_local_model(ctx)
            if not local_model:
                return None

            probe_kwargs = {
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "Rate your confidence in answering the following question "
                            "on a scale of 0 to 10, where 0 means you cannot answer "
                            "and 10 means you are highly confident. "
                            "Respond with ONLY a single number."
                        ),
                    },
                    {"role": "user", "content": last_user_msg},
                ],
                "temperature": 0,
                "max_tokens": 5,
            }

            response = self._probe_client.chat(
                provider=ctx.provider,
                model=local_model,
                request_kwargs=probe_kwargs,
            )

            # Parse first number from response
            match = re.search(r"\d+(?:\.\d+)?", response.content)
            if match:
                value = float(match.group())
                return min(1.0, value / 10.0)
            return None
        except Exception:
            logger.debug("Probe failed", exc_info=True)
            return None

    def _resolve_local_model(self, ctx: MiddlewareContext) -> str:
        """Resolve the local model to use for routing."""
        # Session override
        session_model = ctx.session.metadata.get("auto_route_model", "")
        if session_model:
            return session_model
        # Config auto_route_model → local_model_default
        return self._config.auto_route.model or self._config.local_model_default

    def _route_to_local(self, ctx: MiddlewareContext, decision: RoutingDecision) -> Any | None:
        """Attempt to route the request to local model.

        Returns the converted response on success, None on failure.
        """
        local_model = self._resolve_local_model(ctx)
        if not local_model:
            return None

        try:
            start = time.perf_counter()
            response = self._client.chat(
                provider=ctx.provider,
                model=local_model,
                request_kwargs=ctx.request_kwargs,
            )
            elapsed_ms = (time.perf_counter() - start) * 1000
            decision.reason = f"{decision.reason} → routed to {local_model}"

            if not response.content:
                return None

            # Check if the local model admitted inability to answer
            if self._is_inadequate_response(response.content):
                decision.reason = f"{decision.reason} → inadequate response, rerouting to cloud"
                logger.debug(
                    "Local model indicated inability, falling back to cloud (%r)",
                    response.content[:120],
                )
                return None

            # Convert response to provider-native format
            converted = convert_response(ctx.provider, ctx.model, response)
            if converted is None:
                return None

            # Mutate context for downstream middleware
            ctx.skip_call = True
            ctx.cached_response = converted
            ctx.response = converted
            ctx.model = local_model
            ctx.provider = Provider.LOCAL
            ctx.prompt_tokens = response.prompt_tokens
            ctx.completion_tokens = response.completion_tokens
            ctx.total_tokens = response.total_tokens

            # Store latency on decision for event recording
            decision.complexity_score = decision.complexity_score  # already set
            self._last_local_latency = elapsed_ms

            return converted
        except Exception:
            logger.debug("Local routing failed", exc_info=True)
            return None

    def _estimate_cloud_cost(self, model: str, prompt_tokens: int, completion_tokens: int) -> float:
        """Estimate what the cloud call would have cost."""
        if not self._pricing:
            return 0.0
        try:
            input_cost, output_cost = self._pricing.get_price(model)
            return (prompt_tokens * input_cost) + (completion_tokens * output_cost)
        except Exception:
            return 0.0

    def _record_event(
        self,
        ctx: MiddlewareContext,
        decision: RoutingDecision,
        *,
        success: bool,
        original_provider: str = "",
        original_model: str = "",
    ) -> None:
        """Record a LocalRoutingEvent."""
        local_model = self._resolve_local_model(ctx)
        latency = getattr(self, "_last_local_latency", 0.0) if success else 0.0

        # Estimate what the cloud call would have cost
        estimated_cost = 0.0
        if success:
            estimated_cost = self._estimate_cloud_cost(
                original_model or ctx.model,
                ctx.prompt_tokens or 0,
                ctx.completion_tokens or 0,
            )

        event = LocalRoutingEvent(
            session_id=ctx.session.id,
            step=ctx.session.step_counter,
            original_cloud_provider=original_provider or ctx.provider,
            original_cloud_model=original_model or ctx.model,
            local_model=local_model,
            complexity_score=decision.complexity_score,
            budget_pressure=decision.budget_pressure,
            routing_reason=decision.reason,
            routing_success=success,
            probed=decision.probed,
            probe_confidence=decision.probe_confidence,
            historical_success_rate=decision.historical_success_rate,
            local_latency_ms=latency,
            estimated_cloud_cost=estimated_cost,
            semantic_complexity=decision.semantic_complexity,
            custom_scorer_used=decision.custom_scorer_used,
        )
        ctx.events.append(event)

    def shutdown(self) -> None:
        """Clean up resources."""
        self._client.close()
        self._probe_client.close()
