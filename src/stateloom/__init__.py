"""StateLoom — The first stateful AI gateway.

Session-aware middleware for LLM agents. Cost tracking, PII detection,
loop detection, budget enforcement, time-travel debugging.

Usage:
    import stateloom
    stateloom.init()  # zero-config, auto-patches OpenAI/Anthropic
"""

from __future__ import annotations

import logging
from collections.abc import AsyncGenerator, Callable, Generator
from contextlib import asynccontextmanager, contextmanager
from typing import Any, cast

from stateloom._version import __version__
from stateloom.agent.models import Agent, AgentVersion
from stateloom.chat import ChatResponse, Client, achat, chat
from stateloom.consensus.models import ConsensusResult, DebateRound
from stateloom.core.config import ComplianceProfile, KillSwitchRule, PIIRule, StateLoomConfig
from stateloom.core.errors import (
    StateLoomAuthError,
    StateLoomBlastRadiusError,
    StateLoomBudgetError,
    StateLoomCancellationError,
    StateLoomCircuitBreakerError,
    StateLoomComplianceError,
    StateLoomConfigLockedError,
    StateLoomError,
    StateLoomFeatureError,
    StateLoomGuardrailError,
    StateLoomJobError,
    StateLoomKillSwitchError,
    StateLoomLicenseError,
    StateLoomLoopError,
    StateLoomPermissionError,
    StateLoomPIIBlockedError,
    StateLoomRateLimitError,
    StateLoomReplayError,
    StateLoomRetryError,
    StateLoomSecurityError,
    StateLoomSideEffectError,
    StateLoomSuspendedError,
    StateLoomTimeoutError,
)
from stateloom.core.organization import Organization, Team
from stateloom.core.session import Session
from stateloom.core.types import (
    AgentStatus,
    ComplianceStandard,
    ConsensusStrategy,
    DataRegion,
    FailureAction,
)
from stateloom.gate import Gate
from stateloom.middleware.auto_router import RoutingContext
from stateloom.mock import MockSession

logger = logging.getLogger("stateloom")

# Global gate instance
_gate: Gate | None = None

# Deferred pricing: provider pricing registered before init() is called
_deferred_pricing: dict[str, tuple[float, float]] = {}


def register_provider(
    adapter: Any,
    *,
    pricing: dict[str, tuple[float, float]] | None = None,
) -> None:
    """Register a custom LLM provider adapter.

    Call this before ``init()`` to make a custom provider's LLM calls flow
    through StateLoom's middleware pipeline automatically.

    Args:
        adapter: A ``ProviderAdapter`` or ``BaseProviderAdapter`` instance.
        pricing: Optional dict mapping model names to ``(input_cost_per_token,
            output_cost_per_token)`` tuples.

    Example::

        stateloom.register_provider(
            MistralAdapter(),
            pricing={"mistral-large-latest": (0.000002, 0.000006)},
        )
        stateloom.init()
    """
    from stateloom.intercept.provider_registry import register_adapter

    register_adapter(adapter)

    if pricing:
        if _gate is not None:
            for model_name, (inp, out) in pricing.items():
                _gate.pricing.register(model_name, inp, out)
        else:
            _deferred_pricing.update(pricing)


_UNSET = object()


def init(
    *,
    auto_patch: bool = True,
    default_model: str = "",
    budget: float | None = None,
    budget_on_middleware_failure: str | FailureAction | None = None,
    pii: bool = False,
    pii_rules: list[PIIRule | dict[str, Any]] | None = None,
    dashboard: bool = True,
    dashboard_port: int = 4782,
    console_output: bool = True,
    fail_open: bool = True,
    store_backend: str = "sqlite",
    cache: bool = True,
    loop_detection: bool = False,
    loop_threshold: int | None = None,
    cache_backend: str | None = None,
    cache_scope: str | None = None,
    cache_semantic: bool | None = None,
    cache_similarity_threshold: float | None = None,
    cache_embedding_model: str | None = None,
    cache_vector_backend: str | None = None,
    cache_normalize_patterns: list[str] | None = None,
    local_model: str | None | object = _UNSET,
    shadow: bool | object = _UNSET,
    shadow_model: str | None = None,
    shadow_similarity: str | None = None,
    auto_route: bool | None = None,
    auto_route_model: str | None = None,
    auto_route_semantic: bool | None = None,
    auto_route_semantic_model: str | None = None,
    auto_route_scorer: Callable[..., bool | float | None] | None = None,
    circuit_breaker: bool = False,
    circuit_breaker_fallback_map: dict[str, str] | None = None,
    metrics_enabled: bool = False,
    async_jobs_enabled: bool = False,
    async_jobs_max_workers: int = 4,
    with_ollama: bool = False,
    proxy: bool = False,
    proxy_require_virtual_key: bool = True,
    prompts_dir: str = "",
    compliance: str | None = None,
    security_audit_hooks_enabled: bool = False,
    security_secret_vault_enabled: bool = False,
    durable_stream_delay_ms: float = 0,
    debug: bool = False,
    **kwargs: Any,
) -> Gate:
    """Initialize StateLoom. This is the main entry point.

    Args:
        auto_patch: Auto-detect and patch installed LLM clients (OpenAI, Anthropic).
        default_model: Default model for stateloom.chat() when model is omitted.
            When set, calls without an explicit model use this and are eligible
            for auto-routing. Required if using stateloom.chat() without model.
        budget: Default per-session budget in USD. None = unlimited.
        budget_on_middleware_failure: What to do when budget middleware fails ('block'/'pass').
        pii: Enable PII detection (audit mode by default).
        pii_rules: List of PIIRule configs. When provided, these take precedence
            over any PII rules persisted via the dashboard API. When omitted,
            dashboard/API-configured rules apply (cross-process sync).
        dashboard: Start the local dashboard at localhost:{dashboard_port}.
        dashboard_port: Port for the dashboard server.
        console_output: Print one-liner for every LLM call.
        fail_open: If True, observability middleware errors never break LLM calls.
        store_backend: "sqlite" (default) or "memory".
        cache: Enable request caching.
        loop_detection: Enable loop detection (blocks repeated identical requests).
        loop_threshold: Number of identical requests before blocking (default 5).
        cache_backend: Cache storage backend — "memory" (default), "sqlite", or "redis".
        cache_scope: Cache scope — "global" (cross-session, default) or "session".
        cache_semantic: Enable semantic similarity matching
            (requires faiss-cpu + sentence-transformers).
        cache_similarity_threshold: Cosine similarity threshold for
            semantic cache hits (default 0.95).
        cache_embedding_model: Sentence-transformers model for semantic
            matching (default "all-MiniLM-L6-v2").
        local_model: Enable local model with this default model name (e.g. "llama3.2").
            If not provided, auto-detects Ollama and uses the first available model.
            Pass None to explicitly disable auto-detection.
        shadow: Enable model testing (run candidate model in parallel with cloud).
            Defaults to auto-enable when a local model is available. Set to False
            to explicitly disable.
        shadow_model: Which local model to test against (defaults to local_model).
        auto_route: Enable intelligent auto-routing to local models for simple requests.
            Defaults to None (auto-enable when local_model is set). Set to False to
            explicitly disable.
        auto_route_model: Which local model to route to (defaults to local_model).
        auto_route_semantic: Enable semantic complexity classification for auto-routing.
            Uses embeddings instead of heuristics when available. Defaults to True.
        auto_route_semantic_model: Embedding model for semantic classification
            (default "all-MiniLM-L6-v2").
        auto_route_scorer: Custom callback for routing decisions. Receives either
            a prompt string or a RoutingContext. Return True (local), False (cloud),
            float (complexity score 0-1), or None (fall through to default scoring).
        metrics_enabled: Enable Prometheus metrics collection. Requires
            ``prometheus_client`` (``pip install stateloom[metrics]``).
        with_ollama: Start a managed Ollama instance for local models.
            Downloads Ollama if not installed. Runs on a separate port (11435)
            to avoid conflicts with system-wide Ollama.
        proxy: Enable OpenAI-compatible proxy mode at /v1 on the dashboard port.
            Requires dashboard=True.
        proxy_require_virtual_key: If True, proxy requests must provide a valid
            virtual API key. Set to False for development/testing.
        compliance: Global compliance profile preset name ("gdpr", "hipaa", "ccpa")
            or None. Applies to all sessions unless overridden by org/team profiles.
        **kwargs: Additional config overrides.

    Returns:
        The Gate singleton instance.
    """
    global _gate

    if _gate is not None:
        logger.info("Re-initializing StateLoom with new config.")
        shutdown()

    # Convert pii_rules dicts to PIIRule objects
    resolved_pii_rules: list[PIIRule] = []
    if pii_rules:
        for r in pii_rules:
            if isinstance(r, dict):
                resolved_pii_rules.append(PIIRule(**r))
            else:
                resolved_pii_rules.append(r)

    # Convert string failure action to enum
    resolved_budget_failure = None
    if budget_on_middleware_failure is not None:
        if isinstance(budget_on_middleware_failure, str):
            resolved_budget_failure = FailureAction(budget_on_middleware_failure)
        else:
            resolved_budget_failure = budget_on_middleware_failure

    # Local model is opt-in only — user must explicitly pass a model name.
    # _UNSET / None = no local model, str = user explicitly chose a model.
    if local_model is _UNSET or local_model is None:
        detected_local_model = None
    else:
        detected_local_model = local_model

    # Shadow is opt-in only — enabled when user passes shadow=True or
    # when a local_model is explicitly provided.
    if shadow is _UNSET:
        shadow = bool(detected_local_model)
    shadow = bool(shadow)

    # Resolve shadow model
    resolved_shadow_model = shadow_model or detected_local_model or ""

    config_kwargs: dict[str, Any] = {
        "auto_patch": auto_patch,
        "default_model": default_model,
        "budget_per_session": budget,
        "pii_enabled": pii or bool(resolved_pii_rules),
        "dashboard": dashboard,
        "dashboard_port": dashboard_port,
        "console_output": console_output,
        "fail_open": fail_open,
        "store_backend": store_backend,
        "cache_enabled": cache,
        "loop_detection_enabled": loop_detection,
        "metrics_enabled": metrics_enabled,
        "durable_stream_delay_ms": durable_stream_delay_ms,
        "debug": debug,
        **kwargs,
    }

    # Cache config overrides
    if cache_backend is not None:
        config_kwargs["cache_backend"] = cache_backend
    if cache_scope is not None:
        config_kwargs["cache_scope"] = cache_scope
    if cache_semantic is not None:
        config_kwargs["cache_semantic_enabled"] = cache_semantic
    if cache_similarity_threshold is not None:
        config_kwargs["cache_similarity_threshold"] = cache_similarity_threshold
    if cache_embedding_model is not None:
        config_kwargs["cache_embedding_model"] = cache_embedding_model
    if cache_vector_backend is not None:
        config_kwargs["cache_vector_backend"] = cache_vector_backend
    if cache_normalize_patterns is not None:
        config_kwargs["cache_normalize_patterns"] = cache_normalize_patterns

    # Loop detection config
    if loop_threshold is not None:
        config_kwargs["loop_exact_threshold"] = loop_threshold

    # Local model config
    if detected_local_model:
        config_kwargs["local_model_enabled"] = True
        config_kwargs["local_model_default"] = detected_local_model
    if shadow:
        config_kwargs["shadow_enabled"] = True
        config_kwargs["shadow_model"] = resolved_shadow_model
    if shadow_similarity is not None:
        config_kwargs["shadow_similarity_method"] = shadow_similarity

    # Auto-routing: opt-in only — must explicitly pass auto_route=True
    auto_route_enabled = auto_route if auto_route is not None else False
    if auto_route_enabled:
        config_kwargs["auto_route_enabled"] = True
        if auto_route_model:
            config_kwargs["auto_route_model"] = auto_route_model
        if auto_route_semantic is not None:
            config_kwargs["auto_route_semantic_enabled"] = auto_route_semantic
        if auto_route_semantic_model is not None:
            config_kwargs["auto_route_semantic_model"] = auto_route_semantic_model
        # Auto-enable local model if not already set
        if not detected_local_model:
            config_kwargs.setdefault("local_model_enabled", True)
    if resolved_pii_rules:
        config_kwargs["pii_rules"] = resolved_pii_rules
    if resolved_budget_failure is not None:
        config_kwargs["budget_on_middleware_failure"] = resolved_budget_failure

    # Circuit breaker config
    if circuit_breaker:
        config_kwargs["circuit_breaker_enabled"] = True
        if circuit_breaker_fallback_map:
            config_kwargs["circuit_breaker_fallback_map"] = circuit_breaker_fallback_map

    # Async jobs config
    if async_jobs_enabled:
        config_kwargs["async_jobs_enabled"] = True
        config_kwargs["async_jobs_max_workers"] = async_jobs_max_workers

    # Managed Ollama config
    if with_ollama:
        config_kwargs["ollama_managed"] = True

    # Proxy config
    if proxy:
        config_kwargs["proxy_enabled"] = True
        config_kwargs["proxy_require_virtual_key"] = proxy_require_virtual_key

    # Prompt file watcher
    if prompts_dir:
        config_kwargs["prompts_dir"] = prompts_dir

    # Compliance profile
    if compliance is not None:
        from stateloom.compliance.profiles import resolve_profile

        config_kwargs["compliance_profile"] = resolve_profile(compliance)

    # Security engine
    if security_audit_hooks_enabled:
        config_kwargs["security_audit_hooks_enabled"] = True
    if security_secret_vault_enabled:
        config_kwargs["security_secret_vault_enabled"] = True

    config = StateLoomConfig(**config_kwargs)

    _gate = Gate(config)

    # Enforce admin-locked settings — raise if developer tries to override
    _gate.check_locked_settings(config_kwargs)

    if auto_route_scorer is not None:
        _gate._custom_scorer = auto_route_scorer

    # Load EE FIRST — registers hooks (middleware, startup, shutdown)
    _gate._load_ee()

    # _setup_middleware() fires EE hooks during execution
    _gate._setup_middleware()

    # Flush deferred pricing (registered before init)
    if _deferred_pricing:
        for model_name, (inp, out) in _deferred_pricing.items():
            _gate.pricing.register(model_name, inp, out)
        _deferred_pricing.clear()

    if auto_patch:
        _gate._auto_patch()

    logger.info("Initialized")
    return _gate


def get_gate() -> Gate:
    """Get the current Gate instance. Raises if not initialized."""
    if _gate is None:
        raise StateLoomError(
            "StateLoom not initialized. Call stateloom.init() first.",
            details="Add `import stateloom; stateloom.init()` to your application startup.",
        )
    return _gate


@contextmanager
def session(
    session_id: str | None = None,
    name: str | None = None,
    budget: float | None = None,
    experiment: str | None = None,
    variant: str | None = None,
    org_id: str = "",
    team_id: str = "",
    durable: bool = False,
    parent: str | None = None,
    timeout: float | None = None,
    idle_timeout: float | None = None,
) -> Generator[Session, None, None]:
    """Context manager for session scoping.

    Usage:
        with stateloom.session("task-123", budget=5.0) as s:
            response = client.chat.completions.create(...)
            print(s.total_cost)

    Durable mode (crash recovery):
        with stateloom.session(id="task-123", durable=True) as s:
            res1 = client.chat.completions.create(...)  # Cache hit on resume
            res2 = client.chat.completions.create(...)  # Live call (picks up here)
    """
    gate = get_gate()
    with gate.session(
        session_id=session_id,
        name=name,
        budget=budget,
        experiment=experiment,
        variant=variant,
        org_id=org_id,
        team_id=team_id,
        durable=durable,
        parent=parent,
        timeout=timeout,
        idle_timeout=idle_timeout,
    ) as s:
        yield s


@asynccontextmanager
async def async_session(
    session_id: str | None = None,
    name: str | None = None,
    budget: float | None = None,
    experiment: str | None = None,
    variant: str | None = None,
    org_id: str = "",
    team_id: str = "",
    durable: bool = False,
    parent: str | None = None,
    timeout: float | None = None,
    idle_timeout: float | None = None,
) -> AsyncGenerator[Session, None]:
    """Async context manager for session scoping.

    Usage:
        async with stateloom.async_session("task-123", budget=5.0) as s:
            response = await client.chat.completions.create(...)
            print(s.total_cost)

    Durable mode (crash recovery):
        async with stateloom.async_session(session_id="task-123", durable=True) as s:
            res1 = await client.chat.completions.create(...)  # Cache hit on resume
            res2 = await client.chat.completions.create(...)  # Live call
    """
    gate = get_gate()
    async with gate.async_session(
        session_id=session_id,
        name=name,
        budget=budget,
        experiment=experiment,
        variant=variant,
        org_id=org_id,
        team_id=team_id,
        durable=durable,
        parent=parent,
        timeout=timeout,
        idle_timeout=idle_timeout,
    ) as s:
        yield s


def wrap(client: Any) -> Any:
    """Wrap an LLM client for interception (explicit, no monkey-patching).

    Usage:
        gate = stateloom.init(auto_patch=False)
        client = stateloom.wrap(openai.OpenAI())
    """
    gate = get_gate()
    return gate.wrap(client)


def tool(
    *,
    mutates_state: bool = False,
    name: str | None = None,
    session_root: bool = False,
) -> Any:
    """Decorator for tool functions (sync and async).

    Enables tool execution visibility, safe replay, and loop detection.

    Args:
        mutates_state: If True, marks this tool as having side effects.
        name: Override the tool name (defaults to function name).
        session_root: If True, automatically create a scoped session per call.

    Usage:
        @stateloom.tool(mutates_state=True)
        def create_ticket(title: str) -> dict: ...

        @stateloom.tool(session_root=True)
        async def run_agent(prompt: str) -> str: ...
    """
    gate = get_gate()
    return gate.tool(mutates_state=mutates_state, name=name, session_root=session_root)


def set_session_id(session_id: str) -> None:
    """Set the current session ID (for distributed context propagation)."""
    gate = get_gate()
    gate.set_session_id(session_id)


def get_session_id() -> str | None:
    """Get the current session ID."""
    gate = get_gate()
    return gate.get_session_id()


def _replay_session(
    session: str,
    mock_until_step: int,
    strict: bool = True,
    allow_hosts: list[str] | None = None,
) -> Any:
    """Time-travel debugging: replay a session, mocking steps up to mock_until_step.

    Returns the ``ReplayEngine`` (also a context manager). Call
    ``engine.stop()`` to clean up, or use as a context manager::

        with stateloom.replay(session="ticket-123", mock_until_step=13) as engine:
            # steps 1-13 return cached responses, 14+ execute live
            run_pipeline()

    Args:
        session: The session ID to replay.
        mock_until_step: Mock steps 1 through N with cached responses.
        strict: If True, block outbound HTTP calls not captured via @gate.tool().
        allow_hosts: Hosts to allow through the network blocker in strict mode.
    """
    gate = get_gate()
    return gate.replay(
        session=session,
        mock_until_step=mock_until_step,
        strict=strict,
        allow_hosts=allow_hosts,
    )


def share(session: str) -> str:
    """Share a session for collaborative debugging.

    Returns a shareable URL. Requires control plane connection.
    """
    gate = get_gate()
    return gate.share(session=session)


def export_session(
    session_id: str,
    path: str | None = None,
    *,
    include_children: bool = False,
    scrub_pii: bool = False,
) -> dict[str, Any]:
    """Export a session as a portable JSON bundle.

    Args:
        session_id: The session ID to export.
        path: Optional file path to write the bundle to.
        include_children: Include child sessions recursively.
        scrub_pii: Redact PII from event fields before export.

    Returns:
        The bundle dict.
    """
    gate = get_gate()
    return gate.export_session(
        session_id,
        path,
        include_children=include_children,
        scrub_pii=scrub_pii,
    )


def import_session(
    source: Any,
    *,
    session_id_override: str | None = None,
) -> Session:
    """Import a session bundle into the store.

    Args:
        source: A bundle dict, or path to a ``.json``/``.json.gz`` file.
        session_id_override: Replace the session ID to avoid collisions.

    Returns:
        The imported Session object.
    """
    gate = get_gate()
    return gate.import_session(source, session_id_override=session_id_override)


def pin(session: str, name: str) -> None:
    """Pin a session as a regression test baseline."""
    gate = get_gate()
    gate.pin(session=session, name=name)


def langchain_callback(
    gate: Gate | None = None,
    *,
    tools_only: bool | None = None,
) -> Any:
    """Create a LangChain callback handler for StateLoom observability.

    Recommended usage (``auto_patch=True`` + callback)::

        stateloom.init()  # auto_patch=True by default
        handler = stateloom.langchain_callback()
        chain.invoke(input, config={"callbacks": [handler]})

    Args:
        gate: Explicit Gate reference. Uses the global singleton when ``None``.
        tools_only: Controls whether LLM events are recorded by the callback.
            ``True`` — only record tool events (LLM tracking via middleware).
            ``False`` — record both LLM and tool events (standalone mode).
            ``None`` (default) — auto-detect from ``gate.config.auto_patch``.

    Requires ``langchain_core``. Install with: ``pip install stateloom[langchain]``
    """
    from stateloom.ext.langchain import StateLoomCallbackHandler

    if gate is None:
        gate = get_gate()
    return StateLoomCallbackHandler(gate=gate, tools_only=tools_only)


def patch_threading() -> None:
    """Patch threading.Thread to propagate StateLoom session context to child threads.

    Call this once at startup if your application spawns threads inside sessions.
    Use ``stateloom.shutdown()`` (which calls ``unpatch_all()``) to restore the
    original ``threading.Thread``.
    """
    from stateloom.concurrency import patch_threading as _do_patch

    _do_patch()


# --- Experiment API ---


def create_experiment(
    name: str,
    variants: list[dict[str, Any]],
    *,
    description: str = "",
    strategy: str = "random",
    metadata: dict[str, Any] | None = None,
    agent_id: str | None = None,
) -> Any:
    """Create an experiment in DRAFT status.

    Args:
        name: Experiment name.
        variants: List of variant config dicts with 'name', 'weight',
            optional 'model', 'request_overrides', 'agent_version_id'.
        description: Optional experiment description.
        strategy: Assignment strategy — 'random', 'hash', or 'manual'.
        metadata: Optional metadata dict.
        agent_id: Optional agent ID to scope the experiment to.

    Returns:
        The Experiment object.
    """
    from stateloom.experiment.models import VariantConfig

    gate = get_gate()
    return gate.experiment_manager.create_experiment(
        name=name,
        variants=cast(list[dict[str, Any] | VariantConfig], variants),
        strategy=strategy,
        description=description,
        metadata=metadata,
        agent_id=agent_id,
    )


def start_experiment(experiment_id: str) -> Any:
    """Start an experiment — begin assigning sessions to variants."""
    gate = get_gate()
    return gate.experiment_manager.start_experiment(experiment_id)


def pause_experiment(experiment_id: str) -> Any:
    """Pause an experiment — stop assigning new sessions."""
    gate = get_gate()
    return gate.experiment_manager.pause_experiment(experiment_id)


def conclude_experiment(experiment_id: str) -> dict[str, Any]:
    """Conclude an experiment and return final metrics."""
    gate = get_gate()
    return gate.experiment_manager.conclude_experiment(experiment_id)


def update_experiment(
    experiment_id: str,
    *,
    name: str | None = None,
    description: str | None = None,
    variants: list[dict[str, Any]] | None = None,
    strategy: str | None = None,
    metadata: dict[str, Any] | None = None,
    agent_id: str | None = None,
) -> Any:
    """Update a DRAFT experiment.

    Args:
        experiment_id: The experiment ID to update.
        name: New experiment name.
        description: New description.
        variants: New list of variant config dicts.
        strategy: New assignment strategy.
        metadata: New metadata dict.
        agent_id: New agent ID.

    Returns:
        The updated Experiment object.

    Raises:
        ValueError: If experiment not found or not in DRAFT status.
    """
    from stateloom.experiment.models import VariantConfig

    gate = get_gate()
    return gate.experiment_manager.update_experiment(
        experiment_id,
        name=name,
        description=description,
        variants=cast(list[dict[str, Any] | VariantConfig] | None, variants),
        strategy=strategy,
        metadata=metadata,
        agent_id=agent_id,
    )


def feedback(
    session_id: str,
    rating: str,
    *,
    score: float | None = None,
    comment: str = "",
) -> None:
    """Record feedback (success/failure/partial) for a session."""
    gate = get_gate()
    gate.feedback(session_id=session_id, rating=rating, score=score, comment=comment)


def experiment_metrics(experiment_id: str) -> dict[str, Any]:
    """Get per-variant aggregated metrics for an experiment."""
    gate = get_gate()
    return gate.experiment_manager.get_metrics(experiment_id)


def leaderboard() -> list[dict[str, Any]]:
    """Get cross-experiment variant ranking sorted by success_rate desc, avg_cost asc."""
    gate = get_gate()
    return gate.experiment_manager.get_leaderboard()


def backtest(
    sessions: list[str],
    variants: list[dict[str, Any]],
    agent_fn: Any,
    *,
    mock_until_step: int | None = None,
    strict: bool = False,
    evaluator: Any = None,
) -> list[dict[str, Any]]:
    """Run backtest: replay sessions with different variant configs.

    Args:
        sessions: Source session IDs whose recorded steps provide cached
            responses for the mocked portion of the replay.
        variants: Variant configs to test (each source session is replayed
            once per variant).
        agent_fn: A callable that re-executes the agent logic.  Receives the
            replay Session as its only argument.  All LLM calls made inside
            agent_fn flow through the middleware pipeline where
            ExperimentMiddleware applies the variant overrides.
        mock_until_step: Mock steps 1..N with cached responses.  Defaults to
            all recorded steps.
        strict: Block outbound HTTP during mocked steps.
        evaluator: Optional scoring callback.

    Returns:
        List of result dicts with metrics per (session, variant) pair.
    """
    from stateloom.experiment.backtest import BacktestRunner
    from stateloom.experiment.models import VariantConfig

    gate = get_gate()
    runner = BacktestRunner(gate, experiment_manager=gate.experiment_manager)
    results = runner.run_backtest(
        source_session_ids=sessions,
        variants=cast(list[dict[str, Any] | VariantConfig], variants),
        agent_fn=agent_fn,
        mock_until_step=mock_until_step,
        strict=strict,
        evaluator=evaluator,
    )
    return [r.to_dict() for r in results]


def pull_model(model: str, *, progress: Any = None) -> None:
    """Download a local model via Ollama.

    Args:
        model: Model name (e.g. "llama3.2", "mistral:7b").
        progress: Optional callback receiving progress dicts.
    """
    from stateloom.local.client import OllamaClient

    gate = get_gate()
    client = OllamaClient(
        host=gate.config.local_model_host,
        timeout=600.0,
    )
    try:
        client.pull_model(model, progress_callback=progress)
    finally:
        client.close()


def list_local_models() -> list[dict[str, Any]]:
    """List locally downloaded Ollama models."""
    from stateloom.local.client import OllamaClient

    gate = get_gate()
    client = OllamaClient(host=gate.config.local_model_host)
    try:
        return client.list_models()
    finally:
        client.close()


def recommend_models() -> list[dict[str, Any]]:
    """Get hardware-aware model recommendations for local inference."""
    from stateloom.local.hardware import detect_hardware
    from stateloom.local.hardware import recommend_models as _recommend

    hardware = detect_hardware()
    return _recommend(hardware)


def delete_local_model(model: str) -> None:
    """Delete a locally downloaded Ollama model."""
    from stateloom.local.client import OllamaClient

    gate = get_gate()
    client = OllamaClient(host=gate.config.local_model_host)
    try:
        client.delete_model(model)
    finally:
        client.close()


def force_local(enabled: bool = True) -> None:
    """Force all LLM traffic to route through local models.

    Auto-enables auto_route if not already enabled.
    Incompatible requests (streaming, tools, images) fall back to cloud.
    """
    gate = get_gate()
    gate.config.auto_route_force_local = enabled
    if enabled and not gate.config.auto_route_enabled:
        gate.config.auto_route_enabled = True
        if not gate.config.local_model_enabled:
            gate.config.local_model_enabled = True


def hot_swap_model(new_model: str, *, delete_old: bool = True, progress: Any = None) -> None:
    """Hot-swap local model with zero downtime.

    Pulls new model, atomically switches config, then deletes old model.
    Blocking call (use in background thread if needed).
    """
    from stateloom.local.client import OllamaClient

    gate = get_gate()
    old_model = gate.config.local_model_default

    client = OllamaClient(host=gate.config.local_model_host, timeout=600.0)
    try:
        client.pull_model(new_model, progress_callback=progress)

        # Atomically switch
        gate.config.local_model_default = new_model
        if gate.config.auto_route_model == old_model:
            gate.config.auto_route_model = new_model
        if gate.config.shadow_model == old_model:
            gate.config.shadow_model = new_model

        # Delete old (fail-open)
        if delete_old and old_model:
            try:
                client.delete_model(old_model)
            except Exception:
                pass
    finally:
        client.close()


def set_local_model(model: str) -> None:
    """Set the active local model for auto-routing and model testing.

    Updates local_model_default, auto_route_model, and shadow_model.
    """
    gate = get_gate()
    gate.config.local_model_default = model
    gate.config.auto_route_model = model
    gate.config.shadow_model = model


def set_auto_route_scorer(scorer: Callable[..., Any] | None) -> None:
    """Set or clear a custom routing scorer for auto-routing decisions.

    Simple:  def my_scorer(prompt: str) -> bool | float | None
    Rich:    def my_scorer(ctx: stateloom.RoutingContext) -> bool | float | None

    Returns: True=local, False=cloud, float=score, None=fallthrough.
    Fail-open: exceptions fall through to default scoring.
    """
    gate = get_gate()
    gate.set_custom_scorer(scorer)


def kill_switch(active: bool = True, *, message: str | None = None) -> None:
    """Activate or deactivate the global kill switch.

    Args:
        active: True to block all LLM traffic, False to resume.
        message: Optional custom message for blocked requests.
    """
    gate = get_gate()
    gate.config.kill_switch_active = active
    if message is not None:
        gate.config.kill_switch_message = message


def kill_switch_rules() -> list[dict[str, Any]]:
    """Get the current kill switch rules as a list of dicts."""
    gate = get_gate()
    return [r.model_dump() for r in gate.config.kill_switch_rules]


def add_kill_switch_rule(
    *,
    model: str | None = None,
    provider: str | None = None,
    environment: str | None = None,
    agent_version: str | None = None,
    message: str = "",
    reason: str = "",
) -> None:
    """Add a granular kill switch rule."""
    gate = get_gate()
    gate.config.kill_switch_rules.append(
        KillSwitchRule(
            model=model,
            provider=provider,
            environment=environment,
            agent_version=agent_version,
            message=message,
            reason=reason,
        )
    )


def clear_kill_switch_rules() -> None:
    """Remove all kill switch rules."""
    gate = get_gate()
    gate.config.kill_switch_rules.clear()


def blast_radius_status() -> dict[str, Any]:
    """Get blast radius containment status (paused sessions/agents, counts)."""
    gate = get_gate()
    if gate._blast_radius is None:
        return {
            "enabled": False,
            "paused_sessions": [],
            "paused_agents": [],
            "session_failure_counts": {},
            "agent_failure_counts": {},
            "session_budget_violations": {},
            "agent_budget_violations": {},
        }
    status: dict[str, Any] = gate._blast_radius.get_status()
    status["enabled"] = True
    return status


def guardrails_status() -> dict[str, Any]:
    """Get guardrails middleware status."""
    gate = get_gate()
    if gate._guardrails is None:
        return {"enabled": False}
    return {
        "enabled": True,
        "mode": gate.config.guardrails_mode.value,
        "heuristic_enabled": gate.config.guardrails_heuristic_enabled,
        "nli_enabled": gate.config.guardrails_nli_enabled,
        "nli_available": (
            gate._guardrails._nli_classifier.is_available
            if gate._guardrails._nli_classifier
            else False
        ),
        "local_model_enabled": gate.config.guardrails_local_model_enabled,
        "local_model": gate.config.guardrails_local_model,
        "local_model_available": (
            gate._guardrails._local_validator.is_available
            if gate._guardrails._local_validator
            else False
        ),
        "output_scanning_enabled": gate.config.guardrails_output_scanning_enabled,
        "pattern_count": len(gate._guardrails._heuristic_patterns),
    }


def configure_guardrails(
    *,
    nli_enabled: bool | None = None,
    nli_threshold: float | None = None,
    heuristic_enabled: bool | None = None,
    mode: str | None = None,
) -> dict[str, Any]:
    """Configure guardrails at runtime (hot-reload, no restart needed).

    Args:
        nli_enabled: Enable/disable NLI injection classifier.
        nli_threshold: NLI score threshold for detection (0.0-1.0).
        heuristic_enabled: Enable/disable heuristic regex patterns.
        mode: Guardrail mode ("audit" or "enforce").

    Returns:
        Updated guardrails status dict.
    """
    gate = get_gate()
    if nli_enabled is not None:
        gate.config.guardrails_nli_enabled = bool(nli_enabled)
    if nli_threshold is not None:
        gate.config.guardrails_nli_threshold = float(nli_threshold)
    if heuristic_enabled is not None:
        gate.config.guardrails_heuristic_enabled = bool(heuristic_enabled)
    if mode is not None:
        from stateloom.core.types import GuardrailMode

        gate.config.guardrails_mode = GuardrailMode(mode)
    return guardrails_status()


def shadow_status() -> dict[str, Any]:
    """Get model testing (shadow drafting) status."""
    gate = get_gate()
    return gate.shadow_status()


def configure_shadow(
    *,
    enabled: bool | None = None,
    model: str | None = None,
    sample_rate: float | None = None,
    max_context_tokens: int | None = None,
    models: list[str] | None = None,
) -> dict[str, Any]:
    """Configure model testing (shadow drafting) at runtime. Enterprise-gated."""
    gate = get_gate()
    return gate.configure_shadow(
        enabled=enabled,
        model=model,
        sample_rate=sample_rate,
        max_context_tokens=max_context_tokens,
        models=models,
    )


def unpause_session(session_id: str) -> bool:
    """Unpause a blast-radius-paused session. Returns True if it was paused."""
    gate = get_gate()
    if gate._blast_radius is None:
        return False
    return cast(bool, gate._blast_radius.unpause_session(session_id))


def unpause_agent(agent_id: str) -> bool:
    """Unpause a blast-radius-paused agent. Returns True if it was paused."""
    gate = get_gate()
    if gate._blast_radius is None:
        return False
    return cast(bool, gate._blast_radius.unpause_agent(agent_id))


# --- Organization & Team API ---


def create_organization(
    name: str = "",
    *,
    budget: float | None = None,
    pii_rules: list[PIIRule | dict[str, Any]] | None = None,
    compliance_profile: ComplianceProfile | str | None = None,
    **kwargs: Any,
) -> Organization:
    """Create a new organization.

    Args:
        name: Organization name.
        budget: Org-wide budget cap in USD.
        pii_rules: Org-level PII rules (floor — strictest mode wins).
        compliance_profile: Compliance profile or preset name ("gdpr", "hipaa", "ccpa").
        **kwargs: Additional metadata fields.
    """
    gate = get_gate()
    resolved_rules: list[PIIRule] = []
    if pii_rules:
        for r in pii_rules:
            if isinstance(r, dict):
                resolved_rules.append(PIIRule(**r))
            else:
                resolved_rules.append(r)
    return gate.create_organization(
        name=name,
        budget=budget,
        pii_rules=resolved_rules,
        metadata=kwargs.get("metadata"),
        compliance_profile=compliance_profile,
    )


def get_organization(org_id: str) -> Organization | None:
    """Get an organization by ID."""
    gate = get_gate()
    return gate.get_organization(org_id)


def list_organizations() -> list[Organization]:
    """List all organizations."""
    gate = get_gate()
    return gate.list_organizations()


def create_team(
    org_id: str,
    name: str = "",
    *,
    budget: float | None = None,
    compliance_profile: ComplianceProfile | str | None = None,
    **kwargs: Any,
) -> Team:
    """Create a new team within an organization.

    Args:
        org_id: Parent organization ID.
        name: Team name.
        budget: Team-level budget cap in USD.
        compliance_profile: Compliance profile or preset name ("gdpr", "hipaa", "ccpa").
        **kwargs: Additional metadata fields.
    """
    gate = get_gate()
    return gate.create_team(
        org_id=org_id,
        name=name,
        budget=budget,
        metadata=kwargs.get("metadata"),
        compliance_profile=compliance_profile,
    )


def get_team(team_id: str) -> Team | None:
    """Get a team by ID."""
    gate = get_gate()
    return gate.get_team(team_id)


def list_teams(org_id: str | None = None) -> list[Team]:
    """List teams, optionally filtered by org_id."""
    gate = get_gate()
    return gate.list_teams(org_id=org_id)


def org_stats(org_id: str) -> dict[str, Any]:
    """Get aggregated stats for an organization."""
    gate = get_gate()
    return gate.store.get_org_stats(org_id)


def team_stats(team_id: str) -> dict[str, Any]:
    """Get aggregated stats for a team."""
    gate = get_gate()
    return gate.store.get_team_stats(team_id)


# --- Agent API ---


def create_agent(
    slug: str,
    team_id: str = "",
    *,
    name: str = "",
    model: str = "",
    system_prompt: str = "",
    description: str = "",
    request_overrides: dict[str, Any] | None = None,
    budget_per_session: float | None = None,
    metadata: dict[str, Any] | None = None,
    created_by: str = "",
) -> Any:
    """Create a managed agent definition with an initial version (v1).

    Args:
        slug: URL-friendly identifier (3-64 chars, lowercase alphanumeric + hyphens).
        team_id: Team that owns this agent (optional).
        name: Human-readable name.
        model: Model for the initial version (e.g. "gpt-4o").
        system_prompt: System prompt for the initial version.
        description: Agent description.
        request_overrides: Default request overrides (temperature, max_tokens, etc.).
        budget_per_session: Per-session budget cap in USD.
        metadata: Arbitrary metadata.
        created_by: Who created this version.

    Returns:
        The Agent object.
    """
    gate = get_gate()
    return gate.create_agent(
        slug=slug,
        team_id=team_id,
        name=name,
        model=model,
        system_prompt=system_prompt,
        description=description,
        request_overrides=request_overrides,
        budget_per_session=budget_per_session,
        metadata=metadata,
        created_by=created_by,
    )


def get_agent(agent_id: str) -> Any:
    """Get an agent by ID."""
    gate = get_gate()
    return gate.get_agent(agent_id)


def list_agents(
    team_id: str | None = None,
    org_id: str | None = None,
) -> list[Any]:
    """List agents, optionally filtered by team_id or org_id."""
    gate = get_gate()
    return gate.list_agents(team_id=team_id, org_id=org_id)


def create_agent_version(
    agent_id: str,
    *,
    model: str = "",
    system_prompt: str = "",
    request_overrides: dict[str, Any] | None = None,
    budget_per_session: float | None = None,
    metadata: dict[str, Any] | None = None,
    created_by: str = "",
) -> Any:
    """Create a new version for an agent.

    Args:
        agent_id: The agent to create a version for.
        model: Model for this version.
        system_prompt: System prompt for this version.
        request_overrides: Request overrides (temperature, max_tokens, etc.).
        budget_per_session: Per-session budget cap in USD.
        metadata: Arbitrary metadata.
        created_by: Who created this version.

    Returns:
        The AgentVersion object.
    """
    gate = get_gate()
    return gate.create_agent_version(
        agent_id,
        model=model,
        system_prompt=system_prompt,
        request_overrides=request_overrides,
        budget_per_session=budget_per_session,
        metadata=metadata,
        created_by=created_by,
    )


def list_agent_versions(agent_id: str) -> list[Any]:
    """List all versions for an agent, ordered by version number.

    Args:
        agent_id: The agent ID.

    Returns:
        List of AgentVersion objects.
    """
    gate = get_gate()
    return gate.store.list_agent_versions(agent_id)


def activate_agent_version(agent_id: str, version_id: str) -> Any:
    """Activate a specific version for an agent (rollback).

    Args:
        agent_id: The agent ID.
        version_id: The version ID to activate.

    Returns:
        The updated Agent object.
    """
    gate = get_gate()
    return gate.activate_agent_version(agent_id, version_id)


def purge_user_data(user_identifier: str, standard: str = "gdpr") -> dict[str, Any]:
    """Purge all data matching a user identifier (Right to Be Forgotten).

    Args:
        user_identifier: The user identifier to purge.
        standard: Compliance standard for the audit event ("gdpr", "hipaa", "ccpa").

    Returns:
        Dict with counts of deleted items and the audit event ID.
    """
    from stateloom.compliance.purge import PurgeEngine

    gate = get_gate()
    engine = PurgeEngine(gate.store, cache_store=gate._cache_store)
    result = engine.purge(user_identifier, standard=standard)
    return {
        "user_identifier": result.user_identifier,
        "sessions_deleted": result.sessions_deleted,
        "events_deleted": result.events_deleted,
        "cache_entries_deleted": result.cache_entries_deleted,
        "jobs_deleted": result.jobs_deleted,
        "virtual_keys_deleted": result.virtual_keys_deleted,
        "audit_event_id": result.audit_event_id,
    }


def compliance_cleanup() -> int:
    """Run session TTL enforcement for all compliance-configured orgs.

    Returns the number of sessions purged.
    """
    gate = get_gate()
    return gate.compliance_cleanup()


def mock(
    session_id: str | None = None,
    *,
    force_record: bool = False,
    network_block: bool = True,
    allow_hosts: list[str] | None = None,
) -> Any:
    """VCR-cassette mock: record once, replay forever. Zero-cost testing.

    Returns a ``MockSession`` usable as a decorator or context manager.

    Usage::

        @stateloom.mock()
        def test_my_agent():
            response = openai.chat.completions.create(...)

        with stateloom.mock("my-cassette") as m:
            response = openai.chat.completions.create(...)
            print(m.is_replay)
    """
    from stateloom.mock import mock as _mock

    return _mock(
        session_id=session_id,
        force_record=force_record,
        network_block=network_block,
        allow_hosts=allow_hosts,
    )


def durable_task(
    retries: int = 3,
    **kwargs: Any,
) -> Any:
    """Decorator: durable session + automatic retry on exception.

    Usage::

        @stateloom.durable_task(retries=3)
        def generate_report(prompt: str) -> dict:
            response = openai.chat.completions.create(...)
            return json.loads(response.choices[0].message.content)
    """
    from stateloom.retry import durable_task as _dt

    return _dt(retries=retries, **kwargs)


def retry_loop(
    retries: int = 3,
    **kwargs: Any,
) -> Any:
    """Create an iterable retry loop for LLM call blocks.

    Usage::

        for attempt in stateloom.retry_loop(retries=3):
            with attempt:
                response = openai.chat.completions.create(...)
                result = json.loads(response.choices[0].message.content)
    """
    from stateloom.retry import RetryLoop

    return RetryLoop(retries=retries, **kwargs)


def checkpoint(label: str, description: str = "") -> None:
    """Create a named checkpoint in the current session.

    Checkpoints are recorded as events and visible in the waterfall trace.

    Args:
        label: Short name for the checkpoint (e.g., "data_fetched", "plan_complete").
        description: Optional longer description.
    """
    gate = get_gate()
    gate.checkpoint(label=label, description=description)


async def consensus(
    prompt: str = "",
    *,
    messages: list[dict[str, Any]] | None = None,
    models: list[str] | None = None,
    rounds: int = 2,
    strategy: str = "debate",
    budget: float | None = None,
    session_id: str | None = None,
    greedy: bool = False,
    samples: int = 5,
    temperature: float = 0.7,
    judge_model: str | None = None,
    aggregation: str = "confidence_weighted",
    early_stop_enabled: bool = True,
    early_stop_threshold: float = 0.9,
    greedy_agreement_threshold: float = 0.7,
    agent: str | None = None,
    **kwargs: Any,
) -> Any:
    """Run a multi-agent consensus session.

    Multiple LLMs debate a question to reduce hallucinations and improve
    reasoning. Supports vote, debate, and self-consistency strategies.

    Args:
        prompt: The question or task for consensus.
        messages: Alternative to prompt — OpenAI-style messages list.
        models: List of model names to use as debaters.
        rounds: Number of debate rounds (for debate strategy).
        strategy: "vote", "debate", or "self_consistency".
        budget: Maximum total cost in USD.
        session_id: Named session for durable replay on crash.
        greedy: Auto-downgrade to cheaper models when consensus is easy.
        samples: Number of samples (for self_consistency strategy).
        temperature: Sampling temperature (for self_consistency).
        judge_model: Model for final synthesis (defaults to first model).
        aggregation: "confidence_weighted" or "majority_vote".
        early_stop_enabled: Stop early when all agents agree.
        early_stop_threshold: Confidence threshold for early stop.
        greedy_agreement_threshold: Agreement threshold for greedy downgrade.
        agent: Agent slug or ID — uses the agent's system prompt for debaters.

    Returns:
        A ``ConsensusResult`` with answer, confidence, cost, and rounds.
    """
    gate = get_gate()
    return await gate.consensus(
        prompt=prompt,
        messages=messages,
        models=models,
        rounds=rounds,
        strategy=strategy,
        budget=budget,
        session_id=session_id,
        greedy=greedy,
        samples=samples,
        temperature=temperature,
        judge_model=judge_model,
        aggregation=aggregation,
        early_stop_enabled=early_stop_enabled,
        early_stop_threshold=early_stop_threshold,
        greedy_agreement_threshold=greedy_agreement_threshold,
        agent=agent,
        **kwargs,
    )


def consensus_sync(
    prompt: str = "",
    *,
    models: list[str] | None = None,
    rounds: int = 2,
    strategy: str = "debate",
    budget: float | None = None,
    session_id: str | None = None,
    greedy: bool = False,
    agent: str | None = None,
    **kwargs: Any,
) -> Any:
    """Synchronous wrapper for ``consensus()``.

    Convenience for non-async code. See ``consensus()`` for full docs.
    """
    import asyncio

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    coro = consensus(
        prompt=prompt,
        models=models,
        rounds=rounds,
        strategy=strategy,
        budget=budget,
        session_id=session_id,
        greedy=greedy,
        agent=agent,
        **kwargs,
    )

    if loop and loop.is_running():
        import concurrent.futures

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(asyncio.run, coro).result()
    return asyncio.run(coro)


def cancel_session(session_id: str) -> bool:
    """Cancel an active session.

    The next LLM call in the session will raise StateLoomCancellationError.

    Args:
        session_id: The session ID to cancel.

    Returns:
        True if the session was found and cancelled.
    """
    gate = get_gate()
    return gate.cancel_session(session_id)


def suspend_session(
    session_id: str,
    reason: str = "",
    data: dict[str, Any] | None = None,
) -> bool:
    """Suspend an active session (human-in-the-loop).

    The next LLM call in the session will raise StateLoomSuspendedError.
    Use ``signal_session()`` to resume execution.

    Args:
        session_id: The session ID to suspend.
        reason: Why the session is being suspended.
        data: Arbitrary context data for the human reviewer.

    Returns:
        True if the session was found and suspended.
    """
    gate = get_gate()
    return gate.suspend_session(session_id, reason=reason, data=data)


def signal_session(session_id: str, payload: Any = None) -> bool:
    """Resume a suspended session with an optional payload.

    Args:
        session_id: The session to resume.
        payload: Arbitrary data (approval decision, human feedback, etc.)
            accessible via ``session.signal_payload`` after resumption.

    Returns:
        True if the session was found and signaled.
    """
    gate = get_gate()
    return gate.signal_session(session_id, payload)


def suspend(
    reason: str = "",
    data: dict[str, Any] | None = None,
    timeout: float | None = None,
) -> Any:
    """Suspend the current session and block until signaled.

    Use this inside a session to pause execution and wait for human input
    (e.g., approval, review, feedback).

    Args:
        reason: Why the session is being suspended (shown in dashboard).
        data: Arbitrary context data for the human reviewer.
        timeout: Max seconds to wait for signal. None = wait indefinitely.

    Returns:
        The signal payload from the human, or None on timeout.
    """
    gate = get_gate()
    return gate.suspend(reason=reason, data=data, timeout=timeout)


async def async_suspend(
    reason: str = "",
    data: dict[str, Any] | None = None,
    timeout: float | None = None,
) -> Any:
    """Async version of suspend(). Non-blocking wait.

    Args:
        reason: Why the session is being suspended (shown in dashboard).
        data: Arbitrary context data for the human reviewer.
        timeout: Max seconds to wait for signal. None = wait indefinitely.

    Returns:
        The signal payload from the human, or None on timeout.
    """
    gate = get_gate()
    return await gate.async_suspend(reason=reason, data=data, timeout=timeout)


def circuit_breaker_status() -> dict[str, Any]:
    """Get circuit breaker status for all tracked providers."""
    gate = get_gate()
    return gate.circuit_breaker_status()


def reset_circuit_breaker(provider: str) -> bool:
    """Reset a provider's circuit breaker to closed.

    Args:
        provider: Provider name (e.g. "openai", "anthropic", "gemini").

    Returns:
        True if the circuit was found and reset.
    """
    gate = get_gate()
    return gate.reset_circuit_breaker(provider)


def rate_limiter_status() -> dict[str, Any]:
    """Get rate limiter status (per-team queue depths, token counts)."""
    gate = get_gate()
    if gate._rate_limiter is None:
        return {"teams": {}}
    return cast(dict[str, Any], gate._rate_limiter.get_status())


# --- Async Jobs API ---


def submit_job(
    *,
    provider: str = "",
    model: str = "",
    messages: list[dict[str, Any]] | None = None,
    request_kwargs: dict[str, Any] | None = None,
    webhook_url: str = "",
    webhook_secret: str = "",
    session_id: str = "",
    org_id: str = "",
    team_id: str = "",
    max_retries: int = 3,
    ttl_seconds: int | None = None,
    metadata: dict[str, Any] | None = None,
    agent: str | None = None,
) -> Any:
    """Submit an async job for background processing.

    Returns the Job object with a unique ID. The job will be processed
    by the background worker pool through the full middleware pipeline.

    Args:
        agent: Agent slug or ID — resolves the agent's model and system prompt.
    """
    gate = get_gate()
    return gate.submit_job(
        provider=provider,
        model=model,
        messages=messages,
        request_kwargs=request_kwargs,
        webhook_url=webhook_url,
        webhook_secret=webhook_secret,
        session_id=session_id,
        org_id=org_id,
        team_id=team_id,
        max_retries=max_retries,
        ttl_seconds=ttl_seconds,
        metadata=metadata,
        agent=agent,
    )


def get_job(job_id: str) -> Any:
    """Get a job by ID."""
    gate = get_gate()
    return gate.get_job(job_id)


def list_jobs(
    status: str | None = None,
    session_id: str | None = None,
    limit: int = 100,
    offset: int = 0,
) -> list[Any]:
    """List jobs, optionally filtered by status or session_id."""
    gate = get_gate()
    return gate.list_jobs(
        status=status,
        session_id=session_id,
        limit=limit,
        offset=offset,
    )


def cancel_job(job_id: str) -> bool:
    """Cancel a pending job. Returns True if cancelled."""
    gate = get_gate()
    return gate.cancel_job(job_id)


def job_stats() -> dict[str, Any]:
    """Get aggregate job statistics."""
    gate = get_gate()
    return gate.job_stats()


def create_virtual_key(
    team_id: str,
    name: str = "",
    *,
    scopes: list[str] | None = None,
    allowed_models: list[str] | None = None,
    budget_limit: float | None = None,
    rate_limit_tps: float | None = None,
    rate_limit_max_queue: int = 100,
    rate_limit_queue_timeout: float = 30.0,
    agent_ids: list[str] | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Create a virtual API key for proxy authentication.

    Args:
        team_id: Team to associate the key with.
        name: Human-readable name for the key.
        scopes: Optional list of scopes (reserved for future use).
        allowed_models: Optional glob patterns restricting which models
            this key can access (e.g. ``["gpt-4o-mini", "claude-*"]``).
            Empty list = all models allowed.
        budget_limit: Optional per-key spend cap in USD.
        rate_limit_tps: Optional per-key TPS limit.
        rate_limit_max_queue: Max queued requests when TPS is exceeded (default 100).
        rate_limit_queue_timeout: Seconds to wait in queue before rejection (default 30).
        agent_ids: Optional list of agent IDs this key can access.
            Empty list = all agents for the team.
        metadata: Optional metadata dict.

    Returns:
        Dict with 'id', 'key' (full key, shown only once), 'key_preview',
        'team_id', 'org_id', 'name', 'created_at'.
    """
    from stateloom.proxy.virtual_key import (
        VirtualKey,
        generate_virtual_key,
        make_key_preview,
        make_virtual_key_id,
    )

    gate = get_gate()
    team = gate.get_team(team_id)
    org_id = team.org_id if team else ""

    full_key, key_hash = generate_virtual_key()
    vk = VirtualKey(
        id=make_virtual_key_id(),
        key_hash=key_hash,
        key_preview=make_key_preview(full_key),
        team_id=team_id,
        org_id=org_id,
        name=name,
        scopes=scopes or [],
        allowed_models=allowed_models or [],
        budget_limit=budget_limit,
        rate_limit_tps=rate_limit_tps,
        rate_limit_max_queue=rate_limit_max_queue,
        rate_limit_queue_timeout=rate_limit_queue_timeout,
        agent_ids=agent_ids or [],
        metadata=metadata or {},
    )
    gate.store.save_virtual_key(vk)
    return {
        "id": vk.id,
        "key": full_key,
        "key_preview": vk.key_preview,
        "team_id": vk.team_id,
        "org_id": vk.org_id,
        "name": vk.name,
        "created_at": vk.created_at.isoformat(),
    }


def list_virtual_keys(team_id: str | None = None) -> list[dict[str, Any]]:
    """List virtual keys (previews only, never full keys).

    Args:
        team_id: Optional team filter.

    Returns:
        List of key info dicts (no full key values).
    """
    gate = get_gate()
    keys = gate.store.list_virtual_keys(team_id=team_id)
    return [
        {
            "id": vk.id,
            "key_preview": vk.key_preview,
            "team_id": vk.team_id,
            "org_id": vk.org_id,
            "name": vk.name,
            "created_at": vk.created_at.isoformat(),
            "revoked": vk.revoked,
        }
        for vk in keys
    ]


def revoke_virtual_key(key_id: str) -> bool:
    """Revoke a virtual API key.

    Returns:
        True if the key was found and revoked.
    """
    gate = get_gate()
    return gate.store.revoke_virtual_key(key_id)


def lock_setting(setting: str, value: Any = None, *, reason: str = "") -> dict[str, Any]:
    """Lock a config setting (admin only).

    Args:
        setting: Config field name to lock (e.g. "blast_radius_enabled").
        value: Value to lock at. If None, locks at the current config value.
        reason: Optional reason for the lock.

    Returns:
        The lock record dict.
    """
    return get_gate().lock_setting(setting, value, reason=reason)


def unlock_setting(setting: str) -> bool:
    """Unlock a config setting.

    Args:
        setting: Config field name to unlock.

    Returns:
        True if a lock was removed, False if no lock existed.
    """
    return get_gate().unlock_setting(setting)


def list_locked_settings() -> list[dict[str, Any]]:
    """List all admin-locked settings."""
    return get_gate().list_locked_settings()


def prompt_watcher_status() -> dict[str, Any]:
    """Get prompt watcher status (tracked files, errors, etc.).

    Returns:
        Status dict with 'enabled', 'prompts_dir', 'tracked_files', etc.
    """
    gate = get_gate()
    if gate._prompt_watcher is None:
        return {"enabled": False}
    return cast(dict[str, Any], gate._prompt_watcher.get_status())


def rescan_prompts() -> dict[str, Any]:
    """Force an immediate full scan of the prompts directory.

    Returns:
        Updated watcher status dict.
    """
    gate = get_gate()
    if gate._prompt_watcher is None:
        return {"enabled": False}
    gate._prompt_watcher.scan()
    return cast(dict[str, Any], gate._prompt_watcher.get_status())


def security_status() -> dict[str, Any]:
    """Get security engine status (audit hooks + secret vault)."""
    gate = get_gate()
    return gate.security_status()


def vault_store(name: str, value: str) -> None:
    """Store a secret in the security vault."""
    gate = get_gate()
    if gate._secret_vault is None:
        from stateloom.security.vault import SecretVault

        gate._secret_vault = SecretVault()
        gate._secret_vault.configure(enabled=True)
    gate._secret_vault.store(name, value)


def vault_retrieve(name: str) -> str | None:
    """Retrieve a secret from the security vault. Returns None if not found."""
    gate = get_gate()
    if gate._secret_vault is None:
        return None
    return cast(str | None, gate._secret_vault.retrieve(name))


def server_logs(limit: int = 200, level: str | None = None) -> list[dict[str, Any]]:
    """Get recent server logs (debug mode only).

    Returns an empty list if debug mode is not enabled or the log buffer
    is not installed.

    Args:
        limit: Maximum number of log entries to return.
        level: Optional level filter (e.g. "DEBUG", "INFO", "WARNING", "ERROR").

    Returns:
        List of log entry dicts with timestamp, level, logger, message, module, lineno.
    """
    from stateloom.dashboard.log_buffer import get_log_buffer

    buf = get_log_buffer()
    if buf is None:
        return []
    return buf.get_logs(limit=limit, level=level)


def shutdown() -> None:
    """Shutdown StateLoom and clean up resources."""
    global _gate
    if _gate is not None:
        _gate.shutdown()
        _gate = None
        logger.info("Shutdown complete")

    # Reset default chat client
    import sys

    _chat_mod = sys.modules.get("stateloom.chat")
    if _chat_mod is not None:
        setattr(_chat_mod, "_default_client", None)


def feature_status() -> dict[str, Any]:
    """Get status of loaded enterprise features.

    Returns:
        Dict with 'features' (name → metadata) and 'count'.
    """
    gate = get_gate()
    return gate._feature_registry.status()


__all__ = [
    "StateLoomAuthError",
    "StateLoomBlastRadiusError",
    "StateLoomBudgetError",
    "StateLoomCancellationError",
    "StateLoomCircuitBreakerError",
    "StateLoomComplianceError",
    "StateLoomConfigLockedError",
    "StateLoomError",
    "StateLoomFeatureError",
    "StateLoomGuardrailError",
    "StateLoomJobError",
    "StateLoomKillSwitchError",
    "StateLoomLicenseError",
    "StateLoomLoopError",
    "StateLoomPermissionError",
    "StateLoomPIIBlockedError",
    "StateLoomRateLimitError",
    "StateLoomReplayError",
    "StateLoomRetryError",
    "StateLoomSecurityError",
    "StateLoomSideEffectError",
    "StateLoomSuspendedError",
    "StateLoomTimeoutError",
    "ChatResponse",
    "Client",
    "ComplianceProfile",
    "ComplianceStandard",
    "DataRegion",
    "FailureAction",
    "Gate",
    "KillSwitchRule",
    "Organization",
    "PIIRule",
    "Session",
    "Team",
    "__version__",
    "achat",
    "activate_agent_version",
    "add_kill_switch_rule",
    "Agent",
    "AgentStatus",
    "AgentVersion",
    "async_session",
    "async_suspend",
    "backtest",
    "blast_radius_status",
    "cancel_job",
    "cancel_session",
    "chat",
    "checkpoint",
    "circuit_breaker_status",
    "consensus",
    "consensus_sync",
    "ConsensusResult",
    "ConsensusStrategy",
    "clear_kill_switch_rules",
    "compliance_cleanup",
    "conclude_experiment",
    "create_agent",
    "create_agent_version",
    "create_experiment",
    "create_organization",
    "create_team",
    "create_virtual_key",
    "delete_local_model",
    "DebateRound",
    "durable_task",
    "experiment_metrics",
    "export_session",
    "feature_status",
    "feedback",
    "force_local",
    "get_agent",
    "get_gate",
    "get_job",
    "guardrails_status",
    "configure_guardrails",
    "configure_shadow",
    "import_session",
    "get_organization",
    "get_session_id",
    "get_team",
    "hot_swap_model",
    "init",
    "job_stats",
    "kill_switch",
    "kill_switch_rules",
    "langchain_callback",
    "leaderboard",
    "list_agents",
    "lock_setting",
    "list_jobs",
    "list_local_models",
    "list_locked_settings",
    "list_organizations",
    "mock",
    "MockSession",
    "list_teams",
    "list_virtual_keys",
    "org_stats",
    "patch_threading",
    "pause_experiment",
    "pin",
    "prompt_watcher_status",
    "pull_model",
    "purge_user_data",
    "rate_limiter_status",
    "recommend_models",
    "reset_circuit_breaker",
    "retry_loop",
    "register_provider",
    "replay",
    "rescan_prompts",
    "revoke_virtual_key",
    "RoutingContext",
    "security_status",
    "server_logs",
    "session",
    "set_auto_route_scorer",
    "set_local_model",
    "set_session_id",
    "shadow_status",
    "share",
    "shutdown",
    "signal_session",
    "start_experiment",
    "suspend",
    "suspend_session",
    "submit_job",
    "team_stats",
    "tool",
    "unlock_setting",
    "unpause_agent",
    "unpause_session",
    "vault_retrieve",
    "vault_store",
    "wrap",
]


# ---------------------------------------------------------------------------
# Prevent the `replay` subpackage from shadowing the public `replay()` function.
#
# When `from stateloom.replay.engine import …` runs (e.g. inside
# Gate.session()), Python's import system binds `stateloom.replay` to the
# subpackage module, overwriting the function defined above.  We fix this by
# swapping the module's __class__ to a custom subclass whose property
# intercepts attribute access for `replay` and always returns the callable.
# ---------------------------------------------------------------------------
import sys as _sys  # noqa: E402
import types as _types  # noqa: E402


class _StateLoomModule(_types.ModuleType):
    """Custom module class that protects `replay` from subpackage shadowing."""

    @property
    def replay(self) -> Any:
        return _replay_session

    @replay.setter
    def replay(self, value: Any) -> None:
        # Silently ignore when Python's import system tries to set the
        # subpackage module — the property getter always wins.
        pass


_sys.modules[__name__].__class__ = _StateLoomModule
