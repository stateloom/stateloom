"""StateLoom singleton runtime — the central coordinator."""

from __future__ import annotations

import functools
import inspect
import logging
import threading
import time
import uuid
from collections.abc import AsyncGenerator, Callable, Generator
from contextlib import asynccontextmanager, contextmanager
from pathlib import Path
from typing import Any, cast

from stateloom.core.config import LOCKABLE_SETTINGS, ComplianceProfile, PIIRule, StateLoomConfig
from stateloom.core.context import (
    get_current_replay_engine,
    get_current_session,
    get_current_session_id,
    set_current_replay_engine,
    set_current_session,
    set_current_session_id,
)
from stateloom.core.errors import (
    StateLoomConfigLockedError,
    StateLoomError,
    StateLoomFeatureError,
    StateLoomJobError,
    StateLoomLoopError,
)
from stateloom.core.event import CheckpointEvent, SuspensionEvent, ToolCallEvent
from stateloom.core.feature_registry import FeatureRegistry
from stateloom.core.interfaces import (
    AlertManagerProtocol,
    MetricsCollectorProtocol,
    TimeSeriesAggregatorProtocol,
)
from stateloom.core.job import Job
from stateloom.core.observability_protocol import (
    NullAlertManager,
    NullMetricsCollector,
    NullTimeSeriesAggregator,
)
from stateloom.core.organization import Organization, Team
from stateloom.core.session import Session, SessionManager
from stateloom.core.signals import Signal
from stateloom.core.types import BudgetAction, FailureAction, JobStatus, PIIMode, SessionStatus
from stateloom.middleware.pipeline import Pipeline
from stateloom.pricing.registry import PricingRegistry
from stateloom.store.base import Store
from stateloom.store.memory_store import MemoryStore
from stateloom.store.sqlite_store import SQLiteStore

logger = logging.getLogger("stateloom")


class Gate:
    """The StateLoom runtime singleton.

    Holds config, middleware pipeline, session manager, store,
    pricing registry, and dashboard server.
    """

    def __init__(self, config: StateLoomConfig) -> None:
        """Initialize the Gate runtime.

        Args:
            config: Validated ``StateLoomConfig`` with all middleware,
                store, observability, and provider settings.

        Sets up the store backend, pricing registry, session manager,
        experiment manager, and loads the org/team/agent hierarchy from
        the store.  Does **not** set up the middleware pipeline — call
        ``_load_ee()`` then ``_setup_middleware()`` and ``_auto_patch()``
        separately (done by ``init()`` / ``from_config()``).
        """
        self.config = config
        self._validate_config()
        self.session_manager = SessionManager()
        self.pipeline = self._create_pipeline()
        self.pricing = PricingRegistry()
        self._patched_methods: list[dict[str, Any]] = []
        self._dashboard_server: Any = None
        # tool-call loop detection: session_id -> {tool_name -> count}
        self._tool_call_counts: dict[str, dict[str, int]] = {}
        # Shutdown coordination
        self._shutting_down = False
        self._active_requests = 0
        self._active_requests_lock = threading.Lock()

        # Job processor (async jobs)
        self._job_processor: Any = None

        # Custom routing scorer callback
        self._custom_scorer: Any = None

        # Prompt file watcher
        self._prompt_watcher: Any = None

        # Extension signals (receivers registered by ee/setup.py:register_ee or any plugin)
        self.on_middleware_setup: Signal[Gate] = Signal("middleware_setup")
        self.on_startup: Signal[Gate] = Signal("startup")
        self.on_shutdown: Signal[Gate] = Signal("shutdown")

        # Feature registry (tracks loaded EE features)
        self._feature_registry = FeatureRegistry()
        self._define_features()

        # EE middleware instances (set by _setup_ee_middleware hook, None if EE not loaded)
        self._kill_switch: Any = None
        self._blast_radius: Any = None
        self._rate_limiter: Any = None
        self._circuit_breaker: Any = None

        # Security (audit hooks + secret vault)
        self._audit_hook_manager: Any = None
        self._secret_vault: Any = None

        # Consensus orchestrator (lazy-init)
        self._consensus_orchestrator: Any = None

        # Managed Ollama
        self._ollama_manager: Any = None

        # Throttled prompt payload cleanup
        self._last_prompt_cleanup: float = 0.0

        # In-memory caches for orgs, teams, and agents
        self._orgs: dict[str, Organization] = {}
        self._teams: dict[str, Team] = {}

        # Auto-infer username from OS if not set
        if not config.username:
            import getpass

            try:
                config.username = getpass.getuser()
            except Exception:
                config.username = ""

        # Set default budget
        self.session_manager.set_default_budget(config.budget_per_session)

        # Initialize store
        if config.store_backend == "sqlite":
            self.store: Store = SQLiteStore(
                config.store_path,
                auto_migrate=config.store_auto_migrate,
            )
        elif config.store_backend == "postgres":
            from stateloom.store.postgres_store import PostgresStore

            self.store = PostgresStore(  # type: ignore[assignment]
                url=config.store_postgres_url,
                pool_min=config.store_postgres_pool_min,
                pool_max=config.store_postgres_pool_max,
                auto_migrate=config.store_auto_migrate,
            )
        else:
            self.store = MemoryStore()

        # Load persisted provider API keys
        for provider in ("openai", "anthropic", "google"):
            field = f"provider_api_key_{provider}"
            if not getattr(self.config, field):
                stored = self.store.get_secret(f"provider_key_{provider}")
                if stored:
                    setattr(self.config, field, stored)

        # Load persisted local routing config (dashboard settings survive re-init)
        self._load_local_routing_config()

        # Register builtin adapters (idempotent — needed for Client.chat() routing)
        from stateloom.intercept.provider_registry import register_builtin_adapters

        register_builtin_adapters()

        # Load orgs/teams from store into in-memory caches
        self._load_hierarchy()

        # Managed Ollama: start if configured
        if config.ollama_managed:
            from stateloom.local.manager import OllamaManager

            self._ollama_manager = OllamaManager()
            self._ollama_manager.ensure_running(port=config.ollama_managed_port)
            config.local_model_host = f"http://127.0.0.1:{config.ollama_managed_port}"
            config.local_model_enabled = True
            if config.ollama_auto_pull and config.local_model_default:
                self._ollama_manager.ensure_model(
                    config.local_model_default,
                    port=config.ollama_managed_port,
                )

        # Initialize cache store (separate from event store)
        self._cache_store, self._semantic_matcher = self._init_cache_store()

        # Initialize experiment manager
        from stateloom.experiment.manager import ExperimentManager

        self.experiment_manager = ExperimentManager(self.store)
        self.experiment_manager.restore_running_experiments()

        # Initialize observability with null implementations.
        # EE replaces these with real implementations via register_ee().
        self._metrics_collector: MetricsCollectorProtocol = NullMetricsCollector()
        self._observability_aggregator: TimeSeriesAggregatorProtocol = NullTimeSeriesAggregator()
        self._alert_manager: AlertManagerProtocol = NullAlertManager()

        # Debug mode: install log buffer + set log level to DEBUG
        self._log_buffer: Any = None
        if self.config.debug:
            from stateloom.dashboard.log_buffer import install_log_buffer

            self._log_buffer = install_log_buffer()
            logging.getLogger("stateloom").setLevel(logging.DEBUG)

        # Set up logging
        logging.basicConfig(
            level=getattr(logging, config.log_level.upper(), logging.INFO),
            format="%(message)s",
        )

    def _validate_config(self) -> None:
        """Auto-default on_middleware_failure to 'block' (fail-closed) for
        security-critical config that doesn't specify it explicitly.
        """
        for rule in self.config.pii.rules:
            if rule.mode == PIIMode.BLOCK and rule.on_middleware_failure is None:
                rule.on_middleware_failure = FailureAction.BLOCK

        if (
            self.config.budget_action == BudgetAction.HARD_STOP
            and self.config.budget_on_middleware_failure is None
        ):
            self.config.budget_on_middleware_failure = FailureAction.BLOCK

    def _define_features(self) -> None:
        """Define all features with their tier (community / enterprise)."""
        r = self._feature_registry
        # Community (auto-enabled)
        r.define("local_models", tier="community", description="Local LLM support via Ollama")
        r.define("exact_cache", tier="community", description="Exact-match response caching")
        r.define("cli", tier="community", description="CLI tools (doctor, tail, stats)")
        r.define("pii_scanner", tier="community", description="Regex-based PII scanning")
        r.define("budget_enforcer", tier="community", description="Per-session budget limits")

        # Enterprise (need license or dev mode)
        r.define("oidc", tier="enterprise", description="OIDC/SSO federation")
        r.define("multi_tenant", tier="enterprise", description="Multi-tenant org/team hierarchy")
        r.define("semantic_cache", tier="enterprise", description="Semantic similarity caching")
        r.define("guardrails_local", tier="enterprise", description="Llama-Guard local validation")
        r.define(
            "audit_logs", tier="enterprise", description="Tamper-proof compliance audit trails"
        )
        r.define(
            "observability",
            tier="enterprise",
            description="Prometheus metrics, alerting, time-series",
        )
        r.define("compliance", tier="enterprise", description="GDPR/HIPAA/CCPA enforcement")
        r.define(
            "consensus_advanced",
            tier="enterprise",
            description=(
                "Advanced consensus: >3 models, judge synthesis, greedy downgrade, durable replay"
            ),
        )
        r.define(
            "model_testing",
            tier="enterprise",
            description="Runtime model testing configuration (shadow drafting)",
        )
        r.define(
            "model_override",
            tier="enterprise",
            description="Emergency model override (force all traffic to a specific model)",
        )

    def _create_pipeline(self) -> Pipeline:
        """Create the middleware pipeline with optional request normalizer."""
        from stateloom.cache.normalizer import RequestNormalizer

        normalizer = RequestNormalizer(
            custom_patterns=self.config.cache.normalize_patterns or None,
        )
        return Pipeline(normalizer=normalizer)

    # Well-known embedding model dimensions (used for remote backends
    # that need dimension before the model is loaded).
    _EMBEDDING_DIMS: dict[str, int] = {
        "all-MiniLM-L6-v2": 384,
        "all-mpnet-base-v2": 768,
        "all-distilroberta-v1": 768,
        "paraphrase-MiniLM-L6-v2": 384,
        "paraphrase-mpnet-base-v2": 768,
    }

    def _create_vector_backend(self) -> Any:
        """Create a vector backend based on config. Returns None for default (FAISS)."""
        if self.config.cache.vector_backend == "redis":
            from stateloom.cache.redis_vector_backend import RedisVectorBackend

            dim = self._EMBEDDING_DIMS.get(self.config.cache.embedding_model, 384)
            return RedisVectorBackend(
                url=self.config.cache.redis_url,
                dimension=dim,
            )
        return None  # SemanticMatcher defaults to FaissBackend

    def _init_cache_store(self) -> tuple[Any, Any]:
        """Initialize the cache storage backend and optional semantic matcher."""
        from stateloom.cache.memory_store import MemoryCacheStore

        cache_store: Any = None
        semantic_matcher: Any = None

        backend = self.config.cache.backend

        if backend == "sqlite":
            try:
                from stateloom.cache.sqlite_store import SQLiteCacheStore

                cache_store = SQLiteCacheStore(
                    path=self.config.cache.db_path,
                    max_size=self.config.cache.max_size,
                )
            except Exception:
                logger.warning("SQLite cache store unavailable, falling back to memory")
                cache_store = MemoryCacheStore(max_size=self.config.cache.max_size)
        elif backend == "redis":
            try:
                from stateloom.cache.redis_store import RedisCacheStore

                cache_store = RedisCacheStore(url=self.config.cache.redis_url)
            except (ImportError, NotImplementedError):
                logger.warning("Redis cache store unavailable, falling back to memory")
                cache_store = MemoryCacheStore(max_size=self.config.cache.max_size)
        else:
            cache_store = MemoryCacheStore(max_size=self.config.cache.max_size)

        # Semantic matcher (optional — enterprise feature)
        if self.config.cache.semantic_enabled and self._feature_registry.is_available(
            "semantic_cache"
        ):
            try:
                from stateloom.cache.semantic import SemanticMatcher, is_semantic_available

                if is_semantic_available():
                    vector_backend = self._create_vector_backend()
                    semantic_matcher = SemanticMatcher(
                        model_name=self.config.cache.embedding_model,
                        vector_backend=vector_backend,
                    )
                    # Rebuild FAISS index from persistent store
                    entries = cache_store.get_all_entries()
                    entries_with_embeddings = [e for e in entries if e.embedding]
                    if entries_with_embeddings:
                        semantic_matcher.rebuild_from_entries(entries_with_embeddings)
                        logger.info(
                            "Rebuilt semantic index with %d entries",
                            len(entries_with_embeddings),
                        )
                else:
                    logger.warning(
                        "Semantic cache enabled but faiss-cpu/sentence-transformers "
                        "not installed. Falling back to exact-only cache."
                    )
            except Exception:
                logger.warning(
                    "Semantic matcher initialization failed, using exact-only cache",
                    exc_info=True,
                )
        elif self.config.cache.semantic_enabled:
            logger.warning(
                "Semantic cache enabled but 'semantic_cache' feature requires an "
                "enterprise license. Using exact-only cache."
            )

        return cache_store, semantic_matcher

    def _setup_security(self) -> None:
        """Set up security engine (audit hooks + secret vault)."""
        if not (
            self.config.security.audit_hooks_enabled or self.config.security.secret_vault_enabled
        ):
            return

        from stateloom.security.audit_hook import AuditHookManager
        from stateloom.security.vault import SecretVault

        # Vault first (move keys before audit hooks start monitoring)
        if self.config.security.secret_vault_enabled:
            self._secret_vault = SecretVault()
            self._secret_vault.configure(
                enabled=True,
                scrub_environ=self.config.security.secret_vault_scrub_environ,
                keys=self.config.security.secret_vault_keys or None,
            )

        # Audit hooks (interpreter-level, NOT middleware)
        if self.config.security.audit_hooks_enabled:
            from stateloom.core.context import _current_session

            self._audit_hook_manager = AuditHookManager()
            self._audit_hook_manager._store = self.store
            self._audit_hook_manager._session_fn = lambda: _current_session.get(None)
            self._audit_hook_manager.configure(
                enabled=True,
                mode=(
                    self.config.security.audit_hooks_mode.value
                    if hasattr(self.config.security.audit_hooks_mode, "value")
                    else self.config.security.audit_hooks_mode
                ),
                deny_events=self.config.security.audit_hooks_deny_events,
                allow_paths=self.config.security.audit_hooks_allow_paths,
            )
            self._audit_hook_manager.install()

    def security_status(self) -> dict[str, Any]:
        """Get security engine status (audit hooks + vault)."""
        result: dict[str, Any] = {
            "audit_hooks": {"installed": False, "enabled": False},
            "secret_vault": {"enabled": False},
        }
        if self._audit_hook_manager:
            result["audit_hooks"] = self._audit_hook_manager.get_status()
        if self._secret_vault:
            result["secret_vault"] = self._secret_vault.get_status()
        return result

    def shadow_status(self) -> dict[str, Any]:
        """Get model testing (shadow drafting) status. Ungated — read-only."""
        try:
            from stateloom.cache.semantic import is_semantic_available

            semantic_available = is_semantic_available()
        except Exception:
            semantic_available = False
        return {
            "enabled": self.config.shadow_enabled,
            "model": self.config.shadow_model,
            "sample_rate": self.config.shadow_sample_rate,
            "max_context_tokens": self.config.shadow_max_context_tokens,
            "models": self.config.shadow_models,
            "similarity_method": self.config.shadow_similarity_method,
            "semantic_available": semantic_available,
            "middleware_active": (
                hasattr(self, "_shadow_middleware") and self._shadow_middleware is not None
            ),
        }

    def configure_shadow(
        self,
        *,
        enabled: bool | None = None,
        model: str | None = None,
        sample_rate: float | None = None,
        max_context_tokens: int | None = None,
        models: list[str] | None = None,
    ) -> dict[str, Any]:
        """Configure model testing (shadow drafting) at runtime. Enterprise-gated."""
        self._feature_registry.require("model_testing")
        if model is not None:
            self.config.shadow_model = model
        if sample_rate is not None:
            self.config.shadow_sample_rate = sample_rate
        if max_context_tokens is not None:
            self.config.shadow_max_context_tokens = max_context_tokens
        if models is not None:
            self.config.shadow_models = models
        if enabled is not None:
            if enabled and not self.config.shadow_enabled:
                from stateloom.middleware.shadow import ShadowMiddleware

                self.config.shadow_enabled = True
                mw = ShadowMiddleware(
                    self.config, self.store, provider_keys=self._get_provider_keys()
                )
                self._shadow_middleware = mw
                self.pipeline.insert(1, mw)
            elif not enabled and self.config.shadow_enabled:
                self.config.shadow_enabled = False
                if hasattr(self, "_shadow_middleware"):
                    self.pipeline.remove(self._shadow_middleware)
                    self._shadow_middleware.shutdown()
                    del self._shadow_middleware
        return self.shadow_status()

    def _setup_middleware(self) -> None:
        """Set up the default middleware chain based on config."""
        # Import here to avoid circular imports
        from stateloom.middleware.cost_tracker import CostTracker
        from stateloom.middleware.event_recorder import EventRecorder
        from stateloom.middleware.experiment import ExperimentMiddleware
        from stateloom.middleware.latency_tracker import LatencyTracker

        # Security engine (before middleware, interpreter-level)
        self._setup_security()

        # Extension middleware hooks (kill switch, compliance, blast radius, rate limiter, etc.)
        # Receivers registered by ee/setup.py:register_ee() which runs before _setup_middleware().
        self.on_middleware_setup.emit(self)

        # Timeout checker (after circuit breaker, before experiment)
        from stateloom.middleware.timeout_checker import TimeoutCheckerMiddleware

        self.pipeline.add(TimeoutCheckerMiddleware(store=self.store))

        # Experiment middleware (before shadow/PII/budget)
        self.pipeline.add(ExperimentMiddleware(store=self.store))

        # Parallel pre-computation (after Experiment overrides model/kwargs)
        self._shared_semantic_classifier = None
        if (
            self.config.parallel_precompute
            and self.config.auto_route.enabled
            and self.config.local_model_enabled
            and self.config.auto_route.semantic_enabled
        ):
            try:
                from stateloom.middleware.precompute import PrecomputeMiddleware
                from stateloom.middleware.semantic_router import SemanticComplexityClassifier

                shared_classifier = SemanticComplexityClassifier(
                    model_name=self.config.auto_route.semantic_model,
                )
                self._shared_semantic_classifier = shared_classifier
                self.pipeline.add(
                    PrecomputeMiddleware(
                        shared_classifier,
                        compliance_fn=self._get_compliance_profile,
                    )
                )
            except Exception:
                logger.debug("PrecomputeMiddleware init failed", exc_info=True)

        # Model testing at position 1 (observation-only, never modifies ctx)
        if self.config.shadow.enabled:
            from stateloom.middleware.shadow import ShadowMiddleware

            self._shadow_middleware = ShadowMiddleware(
                self.config,
                self.store,
                compliance_fn=self._get_compliance_profile,
                provider_keys=self._get_provider_keys(),
            )
            self.pipeline.add(self._shadow_middleware)

        # Pre-call middleware
        if self.config.pii.enabled:
            try:
                from stateloom.middleware.pii_scanner import PIIScannerMiddleware

                self.pipeline.add(
                    PIIScannerMiddleware(
                        self.config,
                        store=self.store,
                        org_rules_fn=self._get_org_pii_rules,
                    )
                )
            except ImportError:
                logger.warning("PII scanner not available")

        # Guardrails (after PII, before budget)
        self._guardrails = None
        if self.config.guardrails.enabled:
            from stateloom.middleware.guardrails import GuardrailMiddleware

            self._guardrails = GuardrailMiddleware(
                self.config,
                self.store,
                metrics=self._metrics_collector,
                registry=self._feature_registry,
            )
            self.pipeline.add(self._guardrails)

        # Always add BudgetEnforcer — per-session budgets can be set at
        # runtime via gate.session(budget=...) without config-level defaults.
        try:
            from stateloom.middleware.budget_enforcer import BudgetEnforcer

            self.pipeline.add(
                BudgetEnforcer(
                    config=self.config,
                    hierarchy_budget_check=self._check_hierarchy_budget,
                    store=self.store,
                )
            )
        except ImportError:
            logger.warning("Budget enforcer not available")

        if self.config.cache.enabled:
            try:
                from stateloom.middleware.cache import CacheMiddleware

                self.pipeline.add(
                    CacheMiddleware(
                        self.config,
                        cache_store=self._cache_store,
                        semantic_matcher=self._semantic_matcher,
                        compliance_fn=self._get_compliance_profile,
                    )
                )
            except ImportError:
                logger.debug("Cache middleware not available")

        if self.config.loop_detection_enabled and self.config.loop_exact_threshold > 0:
            try:
                from stateloom.middleware.loop_detector import LoopDetector

                self.pipeline.add(LoopDetector(self.config, store=self.store))
            except ImportError:
                logger.debug("Loop detector not available")

        # Auto-routing (after cache/loop, before cost tracking)
        if self.config.auto_route.enabled and self.config.local_model_enabled:
            from stateloom.middleware.auto_router import AutoRouterMiddleware

            self._auto_router = AutoRouterMiddleware(
                self.config,
                self.store,
                pricing=self.pricing,
                semantic_matcher=self._semantic_matcher,
                compliance_fn=self._get_compliance_profile,
                semantic_classifier=self._shared_semantic_classifier,
                custom_scorer=self._custom_scorer,
            )
            self.pipeline.add(self._auto_router)

        # Post-call middleware — chain position determines post-call order:
        # Earlier in chain = post-call runs LATER (outermost wrapper).
        # We need: LatencyTracker → CostTracker → ConsoleOutput → EventRecorder
        # So add them in reverse post-call order:
        self.pipeline.add(
            EventRecorder(
                self.store,
                metrics=self._metrics_collector,
                alert_manager=self._alert_manager,
            )
        )

        if self.config.console_output:
            try:
                from stateloom.export.console import ConsoleOutput

                self.pipeline.add(ConsoleOutput(self.config))
            except ImportError:
                logger.debug("Console output not available")

        self.pipeline.add(CostTracker(self.pricing, cost_callback=self._on_cost))
        self.pipeline.add(LatencyTracker())

        # Run extension startup hooks (dashboard, job processor, prompt watcher)
        # Receivers registered by ee/setup.py:register_ee() which runs before _setup_middleware().
        self.on_startup.emit(self)

    def _auto_patch(self) -> None:
        """Auto-detect and patch installed LLM clients."""
        from stateloom.intercept.auto_patch import auto_patch

        self._patched_methods = auto_patch(self)

    def _load_ee(self) -> None:
        """Load enterprise edition if available and licensed."""
        try:
            from stateloom.ee import _validate_license

            _validate_license(self.config.license_key)

            from stateloom.ee.setup import register_ee

            register_ee(self)
        except ImportError:
            pass
        except Exception:
            logger.debug("EE loading failed", exc_info=True)

    # --- Hierarchy methods ---

    def _load_local_routing_config(self) -> None:
        """Load persisted local routing config from the store.

        Dashboard settings (force-local, active model, auto-route enabled)
        are persisted to the store so they survive ``init()`` re-creation.
        Only applied when the user didn't explicitly set these in ``init()``.
        """
        import json

        try:
            blob = self.store.get_secret("local_routing_config_json")
            if not blob:
                return
            data = json.loads(blob)
            # Only apply if the user didn't explicitly set these in init()
            if not self.config.auto_route_force_local and data.get("auto_route_force_local"):
                self.config.auto_route_force_local = True
            if not self.config.auto_route_enabled and data.get("auto_route_enabled"):
                self.config.auto_route_enabled = True
            if not self.config.local_model_default and data.get("local_model_default"):
                self.config.local_model_default = data["local_model_default"]
            if not self.config.local_model_enabled and data.get("local_model_enabled"):
                self.config.local_model_enabled = True
        except Exception:
            logger.debug("Failed to load local routing config from store", exc_info=True)

    def _load_hierarchy(self) -> None:
        """Load orgs/teams/agents from store into in-memory caches."""
        try:
            for org in self.store.list_organizations():
                self._orgs[org.id] = org
            for team in self.store.list_teams():
                self._teams[team.id] = team
        except Exception:
            logger.debug("Failed to load hierarchy from store", exc_info=True)

    def _on_cost(self, org_id: str, team_id: str, cost: float, tokens: int) -> None:
        """Cost propagation callback — called by CostTracker after each LLM call."""
        if team_id and team_id in self._teams:
            self._teams[team_id].add_cost(cost, tokens)
        if org_id and org_id in self._orgs:
            self._orgs[org_id].add_cost(cost, tokens)

    def _check_hierarchy_budget(
        self, org_id: str, team_id: str
    ) -> tuple[float | None, float, str] | None:
        """Hierarchy budget check callback — called by BudgetEnforcer.

        Returns (budget_limit, spent, level) where level is "team" or "org",
        or None if no budget is exceeded.
        """
        # Check team budget first, then org
        if team_id and team_id in self._teams:
            team = self._teams[team_id]
            if team.budget is not None and team.total_cost >= team.budget:
                return (team.budget, team.total_cost, "team")
        if org_id and org_id in self._orgs:
            org = self._orgs[org_id]
            if org.budget is not None and org.total_cost >= org.budget:
                return (org.budget, org.total_cost, "org")
        return None

    def _get_org_pii_rules(self, org_id: str) -> list[PIIRule]:
        """Org PII rule lookup — called by PIIScannerMiddleware.

        Merges org-level PII rules with compliance profile PII rules.
        """
        rules: list[PIIRule] = []
        org = self._orgs.get(org_id)
        if org:
            rules.extend(org.pii_rules)
            if org.compliance_profile:
                rules.extend(org.compliance_profile.pii_rules)
        return rules

    def _get_compliance_profile(self, org_id: str, team_id: str) -> ComplianceProfile | None:
        """Resolve active compliance profile: team > org > global."""
        if team_id and team_id in self._teams:
            team = self._teams[team_id]
            if team.compliance_profile:
                return team.compliance_profile
        if org_id and org_id in self._orgs:
            org = self._orgs[org_id]
            if org.compliance_profile:
                return org.compliance_profile
        return self.config.compliance_profile

    def _get_provider_keys(self) -> dict[str, str]:
        """Collect provider API keys from config and environment.

        Checks config fields first, then falls back to standard env vars
        so that cloud shadow calls can authenticate even when keys are
        only set via environment (the most common setup).
        """
        import os

        env_keys = {
            "openai": "OPENAI_API_KEY",
            "anthropic": "ANTHROPIC_API_KEY",
            "google": "GOOGLE_API_KEY",
        }
        cfg_keys = {
            "openai": self.config.provider_api_key_openai,
            "anthropic": self.config.provider_api_key_anthropic,
            "google": self.config.provider_api_key_google,
        }
        keys: dict[str, str] = {}
        for provider, cfg_val in cfg_keys.items():
            val = cfg_val or os.environ.get(env_keys[provider], "")
            if val:
                keys[provider] = val
        return keys

    def create_organization(
        self,
        org_id: str | None = None,
        name: str = "",
        budget: float | None = None,
        pii_rules: list[PIIRule] | None = None,
        metadata: dict[str, Any] | None = None,
        compliance_profile: ComplianceProfile | str | None = None,
    ) -> Organization:
        """Create a new organization. Returns existing org if one with the same name exists."""
        # Return existing org with same name to prevent duplicates
        if name:
            for existing in self._orgs.values():
                if existing.name == name:
                    return existing

        resolved_compliance = None
        if compliance_profile is not None:
            from stateloom.compliance.profiles import resolve_profile

            resolved_compliance = resolve_profile(compliance_profile)
        org = Organization(
            name=name,
            budget=budget,
            pii_rules=pii_rules or [],
            metadata=metadata or {},
            compliance_profile=resolved_compliance,
        )
        if org_id:
            org.id = org_id
        self._orgs[org.id] = org
        self.store.save_organization(org)
        return org

    def get_organization(self, org_id: str) -> Organization | None:
        """Get an organization by ID."""
        return self._orgs.get(org_id)

    def list_organizations(self) -> list[Organization]:
        """List all organizations."""
        return list(self._orgs.values())

    def create_team(
        self,
        org_id: str,
        team_id: str | None = None,
        name: str = "",
        budget: float | None = None,
        metadata: dict[str, Any] | None = None,
        compliance_profile: ComplianceProfile | str | None = None,
    ) -> Team:
        """Create a new team within an organization.

        Returns existing team if one with the same name exists in the org.
        """
        # Return existing team with same name in same org to prevent duplicates
        if name:
            for existing in self._teams.values():
                if existing.org_id == org_id and existing.name == name:
                    return existing

        resolved_compliance = None
        if compliance_profile is not None:
            from stateloom.compliance.profiles import resolve_profile

            resolved_compliance = resolve_profile(compliance_profile)
        team = Team(
            org_id=org_id,
            name=name,
            budget=budget,
            metadata=metadata or {},
            compliance_profile=resolved_compliance,
        )
        if team_id:
            team.id = team_id
        self._teams[team.id] = team
        self.store.save_team(team)
        return team

    def get_team(self, team_id: str) -> Team | None:
        """Get a team by ID."""
        return self._teams.get(team_id)

    def list_teams(self, org_id: str | None = None) -> list[Team]:
        """List teams, optionally filtered by org_id."""
        teams = list(self._teams.values())
        if org_id is not None:
            teams = [t for t in teams if t.org_id == org_id]
        return teams

    # --- Admin lock methods ---

    def lock_setting(
        self, setting: str, value: Any = None, *, locked_by: str = "", reason: str = ""
    ) -> dict[str, Any]:
        """Lock a config setting at its current (or specified) value."""
        import json as json_mod

        if setting not in LOCKABLE_SETTINGS:
            raise StateLoomError(
                f"Unknown config setting: '{setting}'",
                details=f"Valid settings: {', '.join(sorted(LOCKABLE_SETTINGS))}",
            )

        if value is None:
            value = getattr(self.config, setting, None)

        value_json = json_mod.dumps(value)
        self.store.save_admin_lock(setting, value_json, locked_by=locked_by, reason=reason)

        lock = self.store.get_admin_lock(setting)
        assert lock is not None
        return lock

    def unlock_setting(self, setting: str) -> bool:
        """Remove an admin lock. Returns True if a lock was removed."""
        existing = self.store.get_admin_lock(setting)
        if existing is None:
            return False
        self.store.delete_admin_lock(setting)
        return True

    def list_locked_settings(self) -> list[dict[str, Any]]:
        """Return all admin-locked settings with metadata."""
        return self.store.list_admin_locks()

    def check_locked_settings(self, config_kwargs: dict[str, Any]) -> None:
        """Check if any kwargs conflict with admin-locked settings. Raises on conflict."""
        import json as json_mod

        locks = self.store.list_admin_locks()
        for lock in locks:
            setting = lock["setting"]
            if setting in config_kwargs:
                locked_value = json_mod.loads(lock["value"])
                proposed_value = config_kwargs[setting]
                if proposed_value != locked_value:
                    raise StateLoomConfigLockedError(setting, locked_value, lock.get("reason", ""))

    # --- Agent management methods ---

    def create_agent(
        self,
        slug: str,
        team_id: str = "",
        *,
        name: str = "",
        description: str = "",
        model: str = "",
        system_prompt: str = "",
        request_overrides: dict[str, Any] | None = None,
        compliance_profile_json: str = "",
        budget_per_session: float | None = None,
        metadata: dict[str, Any] | None = None,
        created_by: str = "",
        org_id: str = "",
    ) -> Any:
        """Create an agent with an initial version (v1)."""
        from stateloom.agent.models import Agent, AgentVersion, validate_slug
        from stateloom.core.types import AgentStatus

        if not validate_slug(slug):
            raise StateLoomError(
                f"Invalid agent slug: '{slug}'",
                details="Slug must be 3-64 chars, lowercase alphanumeric"
                " + hyphens, no leading/trailing hyphen.",
            )

        # Return existing agent with same slug in same team to prevent duplicates
        existing = self.get_agent_by_slug(slug, team_id)
        if existing is not None:
            return existing

        # Resolve org_id from team if not given
        resolved_org = org_id
        if not resolved_org and team_id in self._teams:
            resolved_org = self._teams[team_id].org_id

        agent = Agent(
            slug=slug,
            team_id=team_id,
            org_id=resolved_org,
            name=name,
            description=description,
            status=AgentStatus.ACTIVE,
            metadata=metadata or {},
        )

        version = AgentVersion(
            agent_id=agent.id,
            version_number=1,
            model=model,
            system_prompt=system_prompt,
            request_overrides=request_overrides or {},
            compliance_profile_json=compliance_profile_json,
            budget_per_session=budget_per_session,
            created_by=created_by,
        )

        agent.active_version_id = version.id

        self.store.save_agent(agent)
        self.store.save_agent_version(version)
        return agent

    def get_agent(self, agent_id: str) -> Any:
        """Get an agent by ID."""
        return self.store.get_agent(agent_id)

    def get_agent_by_slug(self, slug: str, team_id: str) -> Any:
        """Get an agent by slug within a team."""
        return self.store.get_agent_by_slug(slug, team_id)

    def list_agents(
        self,
        team_id: str | None = None,
        org_id: str | None = None,
        status: str | None = None,
    ) -> list[Any]:
        """List agents, optionally filtered."""
        return self.store.list_agents(team_id=team_id, org_id=org_id, status=status)

    def update_agent(
        self,
        agent_id: str,
        *,
        name: str | None = None,
        description: str | None = None,
        status: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Any:
        """Update agent fields."""
        from datetime import datetime, timezone

        from stateloom.core.types import AgentStatus

        agent = self.store.get_agent(agent_id)
        if agent is None:
            raise StateLoomError(f"Agent not found: {agent_id}")

        if name is not None:
            agent.name = name
        if description is not None:
            agent.description = description
        if status is not None:
            agent.status = AgentStatus(status)
        if metadata is not None:
            agent.metadata = metadata
        agent.updated_at = datetime.now(timezone.utc)

        self.store.save_agent(agent)
        return agent

    def create_agent_version(
        self,
        agent_id: str,
        *,
        model: str = "",
        system_prompt: str = "",
        request_overrides: dict[str, Any] | None = None,
        compliance_profile_json: str = "",
        budget_per_session: float | None = None,
        metadata: dict[str, Any] | None = None,
        created_by: str = "",
    ) -> Any:
        """Create a new version for an agent (auto-increment version number)."""
        from stateloom.agent.models import AgentVersion

        agent = self.store.get_agent(agent_id)
        if agent is None:
            raise StateLoomError(f"Agent not found: {agent_id}")

        version_number = self.store.get_next_version_number(agent_id)

        version = AgentVersion(
            agent_id=agent_id,
            version_number=version_number,
            model=model,
            system_prompt=system_prompt,
            request_overrides=request_overrides or {},
            compliance_profile_json=compliance_profile_json,
            budget_per_session=budget_per_session,
            metadata=metadata or {},
            created_by=created_by,
        )

        self.store.save_agent_version(version)
        return version

    def activate_agent_version(self, agent_id: str, version_id: str) -> Any:
        """Set the active version for an agent (rollback)."""
        from datetime import datetime, timezone

        agent = self.store.get_agent(agent_id)
        if agent is None:
            raise StateLoomError(f"Agent not found: {agent_id}")

        version = self.store.get_agent_version(version_id)
        if version is None or version.agent_id != agent_id:
            raise StateLoomError(f"Version not found: {version_id}")

        agent.active_version_id = version_id
        agent.updated_at = datetime.now(timezone.utc)
        self.store.save_agent(agent)
        return agent

    def archive_agent(self, agent_id: str) -> Any:
        """Archive an agent (soft delete)."""
        from datetime import datetime, timezone

        from stateloom.core.types import AgentStatus

        agent = self.store.get_agent(agent_id)
        if agent is None:
            raise StateLoomError(f"Agent not found: {agent_id}")

        agent.status = AgentStatus.ARCHIVED
        agent.updated_at = datetime.now(timezone.utc)
        self.store.save_agent(agent)
        return agent

    def compliance_cleanup(self) -> int:
        """Run session TTL enforcement for compliance-configured orgs.

        Finds sessions older than their org's session_ttl_days and purges them.
        Returns the number of sessions purged.
        """
        from datetime import datetime, timezone

        now = datetime.now(timezone.utc)
        purged = 0
        for org in self._orgs.values():
            if not org.compliance_profile or org.compliance_profile.session_ttl_days <= 0:
                continue
            ttl_days = org.compliance_profile.session_ttl_days
            sessions = self.store.list_sessions(org_id=org.id, limit=10000)
            for s in sessions:
                if s.ended_at:
                    age_days = (now - s.ended_at).days
                    if age_days > ttl_days:
                        self.store.purge_session(s.id)
                        purged += 1
        return purged

    def wrap(self, client: Any) -> Any:
        """Wrap an LLM client for interception without monkey-patching.

        Returns the same client object, but with methods wrapped.
        """
        from stateloom.intercept.auto_patch import wrap_client

        wrap_client(self, client)
        return client

    @contextmanager
    def session(
        self,
        session_id: str | None = None,
        name: str | None = None,
        budget: float | None = None,
        experiment: str | None = None,
        variant: str | None = None,
        org_id: str = "",
        team_id: str = "",
        durable: bool = False,
        durable_cache_tools: bool = False,
        parent: str | None = None,
        timeout: float | None = None,
        idle_timeout: float | None = None,
    ) -> Generator[Session, None, None]:
        """Context manager for session scoping.

        Args:
            session_id: Explicit ID, or None to auto-generate.  When an
                existing ID is given, session accumulators are restored from
                the store (cost, tokens, step_counter, etc.).
            name: Human-readable session label (shown in dashboard).
            budget: Per-session budget in USD.
            experiment: Experiment ID for variant assignment.
            variant: Explicit variant name override.
            org_id: Organization scope (inherited from parent if empty).
            team_id: Team scope (inherited from parent if empty).
            durable: Enable crash-recovery replay — LLM responses are
                serialized to the store and replayed on resume.
            durable_cache_tools: Also replay tool results on durable resume.
            parent: Explicit parent session ID.  If None, auto-derived from
                the current active session in the ContextVar.
            timeout: Max session duration in seconds.
            idle_timeout: Max idle time between calls in seconds.

        Yields:
            The active ``Session`` instance.
        """
        # Auto-derive parent from current session if not given
        # (ContextVar lookup — child inherits org/team from parent).
        resolved_parent = parent
        if resolved_parent is None:
            current = get_current_session()
            if current is not None:
                resolved_parent = current.id

        # Inherit org_id/team_id from parent if not set
        resolved_org = org_id
        resolved_team = team_id
        if resolved_parent and (not resolved_org or not resolved_team):
            parent_session = self.session_manager.get(resolved_parent)
            if parent_session is not None:
                if not resolved_org:
                    resolved_org = parent_session.org_id
                if not resolved_team:
                    resolved_team = parent_session.team_id

        session = self.session_manager.create(
            session_id=session_id,
            name=name,
            budget=budget,
            org_id=resolved_org,
            team_id=resolved_team,
            parent_session_id=resolved_parent,
            timeout=timeout,
            idle_timeout=idle_timeout,
        )

        # Resume accumulators from a previous run with the same session ID.
        # Proxy clients reuse session IDs across HTTP requests, so we must
        # restore cost/token/step counters to avoid starting from zero each
        # time the context manager is re-entered for the same session.
        if session_id is not None:
            existing = self.store.get_session(session_id)
            if existing is not None:
                session.total_cost = existing.total_cost
                session.total_tokens = existing.total_tokens
                session.total_prompt_tokens = existing.total_prompt_tokens
                session.total_completion_tokens = existing.total_completion_tokens
                session.call_count = existing.call_count
                session.cache_hits = existing.cache_hits
                session.cache_savings = existing.cache_savings
                session.pii_detections = existing.pii_detections
                session.step_counter = existing.step_counter
                session.estimated_api_cost = existing.estimated_api_cost
                # Metadata merge order: stored values first, then current-request
                # overrides layered on top.  This preserves middleware state
                # (e.g. _pii_scanned_msg_count) across HTTP requests while
                # letting caller-supplied metadata win on conflicts.
                stored_meta = dict(existing.metadata)
                stored_meta.update(session.metadata)
                session.metadata = stored_meta

        # Durable resumption: activate replay engine if resuming with cached steps
        _durable_engine = None
        if durable and session_id is not None and session.step_counter > 0:
            from stateloom.replay.engine import DurableReplayEngine, _load_durable_steps

            durable_steps = _load_durable_steps(self, session_id)
            if durable_steps:
                # Reset step_counter to 0 so the code re-executes from step 1.
                # next_step() increments before returning, so the first call
                # yields step 1, matching the cached response for step 1.
                # Without this reset, step numbers would continue from where
                # the previous run left off and miss every cached entry.
                session.step_counter = 0
                _durable_engine = DurableReplayEngine(
                    durable_steps,
                    cache_tools=durable_cache_tools,
                    stream_delay_ms=self.config.durable_stream_delay_ms,
                )
                set_current_replay_engine(_durable_engine)
                logger.info(
                    "Durable resumption: session '%s', %d cached steps",
                    session_id,
                    len(durable_steps),
                )

        if durable:
            session.durable = True
            session.metadata["durable"] = True
            if self.config.durable_stream_delay_ms > 0:
                session.metadata["durable_stream_delay_ms"] = self.config.durable_stream_delay_ms

        previous_session = get_current_session()
        set_current_session(session)

        # Experiment assignment (fail-open)
        try:
            assignment = self.experiment_manager.assign_session(
                session_id=session.id,
                experiment_id=experiment,
                variant_name=variant,
            )
            if assignment:
                session.metadata["experiment_id"] = assignment.experiment_id
                session.metadata["variant"] = assignment.variant_name
                session.metadata["experiment_variant_config"] = assignment.variant_config
                exp = self.store.get_experiment(assignment.experiment_id)
                if exp:
                    session.metadata["experiment_name"] = exp.name
        except Exception:
            logger.debug("Experiment assignment failed", exc_info=True)

        self.store.save_session(session)
        try:
            yield session
        finally:
            if _durable_engine is not None:
                _durable_engine.stop()
            if session.status in (SessionStatus.ACTIVE, SessionStatus.SUSPENDED):
                session.end(SessionStatus.COMPLETED)
            self.store.save_session(session)
            self._tool_call_counts.pop(session.id, None)
            self.pipeline.notify_session_end(session.id)
            self._maybe_cleanup_prompt_payloads()
            set_current_session(previous_session)

    @asynccontextmanager
    async def async_session(
        self,
        session_id: str | None = None,
        name: str | None = None,
        budget: float | None = None,
        experiment: str | None = None,
        variant: str | None = None,
        org_id: str = "",
        team_id: str = "",
        durable: bool = False,
        durable_cache_tools: bool = False,
        parent: str | None = None,
        timeout: float | None = None,
        idle_timeout: float | None = None,
    ) -> AsyncGenerator[Session, None]:
        """Async context manager for session scoping.

        Async mirror of :meth:`session` — see that method for full
        parameter documentation and behavioral notes.
        """
        # Auto-derive parent (see sync session() for detailed comments)
        resolved_parent = parent
        if resolved_parent is None:
            current = get_current_session()
            if current is not None:
                resolved_parent = current.id

        # Inherit org_id/team_id from parent if not set
        resolved_org = org_id
        resolved_team = team_id
        if resolved_parent and (not resolved_org or not resolved_team):
            parent_session = self.session_manager.get(resolved_parent)
            if parent_session is not None:
                if not resolved_org:
                    resolved_org = parent_session.org_id
                if not resolved_team:
                    resolved_team = parent_session.team_id

        session = self.session_manager.create(
            session_id=session_id,
            name=name,
            budget=budget,
            org_id=resolved_org,
            team_id=resolved_team,
            parent_session_id=resolved_parent,
            timeout=timeout,
            idle_timeout=idle_timeout,
        )

        # Resume accumulators — see sync session() for rationale.
        if session_id is not None:
            existing = self.store.get_session(session_id)
            if existing is not None:
                session.total_cost = existing.total_cost
                session.total_tokens = existing.total_tokens
                session.total_prompt_tokens = existing.total_prompt_tokens
                session.total_completion_tokens = existing.total_completion_tokens
                session.call_count = existing.call_count
                session.cache_hits = existing.cache_hits
                session.cache_savings = existing.cache_savings
                session.pii_detections = existing.pii_detections
                session.step_counter = existing.step_counter
                session.estimated_api_cost = existing.estimated_api_cost
                # Metadata merge: stored first, current overlays on top
                # (see sync session() for detailed merge-order rationale).
                stored_meta = dict(existing.metadata)
                stored_meta.update(session.metadata)
                session.metadata = stored_meta

        # Durable resumption — see sync session() for step_counter reset rationale.
        _durable_engine = None
        if durable and session_id is not None and session.step_counter > 0:
            from stateloom.replay.engine import DurableReplayEngine, _load_durable_steps

            durable_steps = _load_durable_steps(self, session_id)
            if durable_steps:
                session.step_counter = 0  # Reset for replay correctness
                _durable_engine = DurableReplayEngine(
                    durable_steps,
                    cache_tools=durable_cache_tools,
                    stream_delay_ms=self.config.durable_stream_delay_ms,
                )
                set_current_replay_engine(_durable_engine)
                logger.info(
                    "Durable resumption: session '%s', %d cached steps",
                    session_id,
                    len(durable_steps),
                )

        if durable:
            session.durable = True
            session.metadata["durable"] = True
            if self.config.durable_stream_delay_ms > 0:
                session.metadata["durable_stream_delay_ms"] = self.config.durable_stream_delay_ms

        previous_session = get_current_session()
        set_current_session(session)

        # Experiment assignment (fail-open)
        try:
            assignment = self.experiment_manager.assign_session(
                session_id=session.id,
                experiment_id=experiment,
                variant_name=variant,
            )
            if assignment:
                session.metadata["experiment_id"] = assignment.experiment_id
                session.metadata["variant"] = assignment.variant_name
                session.metadata["experiment_variant_config"] = assignment.variant_config
                exp = self.store.get_experiment(assignment.experiment_id)
                if exp:
                    session.metadata["experiment_name"] = exp.name
        except Exception:
            logger.debug("Experiment assignment failed", exc_info=True)

        self.store.save_session(session)
        try:
            yield session
        finally:
            if _durable_engine is not None:
                _durable_engine.stop()
            if session.status in (SessionStatus.ACTIVE, SessionStatus.SUSPENDED):
                session.end(SessionStatus.COMPLETED)
            self.store.save_session(session)
            self._tool_call_counts.pop(session.id, None)
            self.pipeline.notify_session_end(session.id)
            self._maybe_cleanup_prompt_payloads()
            set_current_session(previous_session)

    def get_or_create_session(
        self,
        *,
        provider: str = "",
    ) -> Session:
        """Get the current session or create a default one.

        When *provider* is given, the implicit session ID is
        ``default-{provider}`` so that different providers land in
        separate sessions.
        """
        session = get_current_session()
        if session is None:
            _prov = getattr(provider, "value", provider) or ""
            session_id = f"default-{_prov}" if _prov else "default"
            session = self.session_manager.get_or_create(session_id)
            set_current_session(session)
            self.store.save_session(session)
        return session

    def set_session_id(self, session_id: str) -> None:
        """Set the current session ID (for distributed context propagation)."""
        session = self.session_manager.get_or_create(session_id)
        set_current_session(session)
        set_current_session_id(session_id)

    def get_session_id(self) -> str | None:
        """Get the current session ID."""
        return get_current_session_id()

    def feedback(
        self,
        session_id: str,
        rating: str,
        *,
        score: float | None = None,
        comment: str = "",
    ) -> None:
        """Record feedback for a session."""
        self.experiment_manager.record_feedback(
            session_id=session_id,
            rating=rating,
            score=score,
            comment=comment,
        )

    def checkpoint(self, label: str, description: str = "") -> None:
        """Create a named checkpoint in the current session."""
        session = get_current_session()
        if session is None:
            logger.warning("checkpoint() called outside of a session")
            return
        event = CheckpointEvent(
            session_id=session.id,
            step=session.step_counter,
            label=label,
            description=description,
        )
        self.store.save_event(event)

    def suspend(
        self,
        reason: str = "",
        data: dict[str, Any] | None = None,
        timeout: float | None = None,
    ) -> Any:
        """Suspend the current session and block until signaled.

        Args:
            reason: Why the session is being suspended (shown in dashboard).
            data: Arbitrary context data for the human reviewer.
            timeout: Max seconds to wait for signal. None = wait indefinitely.

        Returns:
            The signal payload from the human, or None on timeout.
        """
        session = get_current_session()
        if session is None:
            logger.warning("suspend() called outside of a session")
            return None

        # Record suspension event
        suspend_event = SuspensionEvent(
            session_id=session.id,
            step=session.step_counter,
            action="suspended",
            reason=reason,
            suspend_data=data or {},
        )
        self.store.save_event(suspend_event)

        # Suspend the session
        session.suspend(reason=reason, data=data)
        self.store.save_session(session)

        # Block until signaled
        start = time.perf_counter()
        payload = session.wait_for_signal(timeout=timeout)
        elapsed_ms = (time.perf_counter() - start) * 1000

        # Record resumption event
        resume_event = SuspensionEvent(
            session_id=session.id,
            step=session.step_counter,
            action="resumed",
            reason=reason,
            signal_payload=payload,
            suspended_duration_ms=elapsed_ms,
        )
        self.store.save_event(resume_event)
        self.store.save_session(session)

        return payload

    async def async_suspend(
        self,
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
        import asyncio

        session = get_current_session()
        if session is None:
            logger.warning("async_suspend() called outside of a session")
            return None

        # Record suspension event
        suspend_event = SuspensionEvent(
            session_id=session.id,
            step=session.step_counter,
            action="suspended",
            reason=reason,
            suspend_data=data or {},
        )
        self.store.save_event(suspend_event)

        # Suspend the session
        session.suspend(reason=reason, data=data)
        self.store.save_session(session)

        # Block (non-blocking for event loop) until signaled
        start = time.perf_counter()
        loop = asyncio.get_running_loop()
        payload = await loop.run_in_executor(
            None,
            session.wait_for_signal,
            timeout,
        )
        elapsed_ms = (time.perf_counter() - start) * 1000

        # Record resumption event
        resume_event = SuspensionEvent(
            session_id=session.id,
            step=session.step_counter,
            action="resumed",
            reason=reason,
            signal_payload=payload,
            suspended_duration_ms=elapsed_ms,
        )
        self.store.save_event(resume_event)
        self.store.save_session(session)

        return payload

    def cancel_session(self, session_id: str) -> bool:
        """Cancel a session by ID. Returns True if the session was found and cancelled."""
        session = self.session_manager.get(session_id)
        if session is None:
            # Try the store
            stored = self.store.get_session(session_id)
            if stored is None:
                return False
            # Can't cancel a non-active session
            if stored.status != SessionStatus.ACTIVE:
                return False
            stored.cancel()
            return True
        if session.status != SessionStatus.ACTIVE:
            return False
        session.cancel()
        return True

    def end_session(self, session_id: str) -> bool:
        """End a session by ID, marking it as completed.

        Works on ACTIVE and SUSPENDED sessions. Returns True if the session
        was found and ended, False otherwise.
        """
        session = self.session_manager.get(session_id)
        if session is None:
            stored = self.store.get_session(session_id)
            if stored is None:
                return False
            if stored.status not in (SessionStatus.ACTIVE, SessionStatus.SUSPENDED):
                return False
            stored.end(SessionStatus.COMPLETED)
            self.store.save_session(stored)
            self._tool_call_counts.pop(session_id, None)
            self.pipeline.notify_session_end(session_id)
            return True
        if session.status not in (SessionStatus.ACTIVE, SessionStatus.SUSPENDED):
            return False
        session.end(SessionStatus.COMPLETED)
        self.store.save_session(session)
        self._tool_call_counts.pop(session_id, None)
        self.pipeline.notify_session_end(session_id)
        return True

    def suspend_session(
        self,
        session_id: str,
        reason: str = "",
        data: dict[str, Any] | None = None,
    ) -> bool:
        """Suspend a session (human-in-the-loop).

        The session will block further LLM calls until ``signal_session()``
        is called. Returns True if the session was found and suspended.

        Args:
            session_id: The session to suspend.
            reason: Why the session is being suspended.
            data: Arbitrary context data for the human reviewer.
        """
        session = self.session_manager.get(session_id)
        if session is None:
            return False
        if session.status != SessionStatus.ACTIVE:
            return False
        session.suspend(reason=reason, data=data)
        self.store.save_session(session)

        # Record suspension event
        event = SuspensionEvent(
            session_id=session.id,
            step=session.step_counter,
            action="suspended",
            reason=reason,
            suspend_data=data or {},
        )
        self.store.save_event(event)
        return True

    def signal_session(self, session_id: str, payload: Any = None) -> bool:
        """Resume a suspended session with an optional payload.

        Args:
            session_id: The session to resume.
            payload: Arbitrary data (approval decision, human feedback, etc.).

        Returns:
            True if the session was found and signaled.
        """
        session = self.session_manager.get(session_id)
        if session is None:
            return False
        if session.status != SessionStatus.SUSPENDED:
            return False
        session.signal(payload)
        self.store.save_session(session)

        # Record resumption event
        event = SuspensionEvent(
            session_id=session.id,
            step=session.step_counter,
            action="resumed",
            signal_payload=payload,
        )
        self.store.save_event(event)
        return True

    def circuit_breaker_status(self) -> dict[str, Any]:
        """Get circuit breaker status for all tracked providers."""
        if self._circuit_breaker is None:
            return {"enabled": False, "providers": {}}
        status = self._circuit_breaker.get_status()
        return {"enabled": True, "providers": status}

    def reset_circuit_breaker(self, provider: str) -> bool:
        """Reset a provider's circuit breaker to closed."""
        if self._circuit_breaker is None:
            return False
        return cast(bool, self._circuit_breaker.reset(provider))

    def set_custom_scorer(self, scorer: Any) -> None:
        """Set or clear the custom routing scorer callback."""
        self._custom_scorer = scorer
        if hasattr(self, "_auto_router") and self._auto_router is not None:
            self._auto_router._custom_scorer = scorer
            if scorer is not None:
                self._auto_router._introspect_scorer(scorer)
            else:
                self._auto_router._custom_scorer_wants_context = False

    def _check_tool_loop(self, session_id: str, tool_name: str) -> None:
        """Check if a tool is being called in a loop within a session.

        Raises StateLoomLoopError if the same tool exceeds the threshold.
        """
        threshold = self.config.loop_exact_threshold
        if threshold <= 0:
            return

        if session_id not in self._tool_call_counts:
            self._tool_call_counts[session_id] = {}
        counts = self._tool_call_counts[session_id]
        counts[tool_name] = counts.get(tool_name, 0) + 1
        count = counts[tool_name]

        if count >= threshold:
            logger.warning(
                "Tool loop detected: %s called %d times in session %s",
                tool_name,
                count,
                session_id,
            )
            raise StateLoomLoopError(
                session_id=session_id,
                pattern=tool_name,
                count=count,
            )

    def tool(
        self,
        *,
        mutates_state: bool = False,
        name: str | None = None,
        session_root: bool = False,
    ) -> Callable[..., Any]:
        """Decorator for tool functions (sync and async).

        Enables tool execution visibility, safe replay, and loop detection.

        Args:
            mutates_state: If True, marks this tool as having side effects.
                During replay, mutates_state tools return cached results
                instead of re-executing.
            name: Override the tool name (defaults to function name).
            session_root: If True, automatically create a scoped session for
                each invocation. Eliminates the need for ``with stateloom.session()``.
        """

        def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
            tool_name = name or func.__name__

            if inspect.iscoroutinefunction(func):

                @functools.wraps(func)
                async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                    previous_session = None

                    if session_root:
                        previous_session = get_current_session()
                        sid = f"{tool_name}-{uuid.uuid4().hex[:8]}"
                        session = self.session_manager.create(session_id=sid, name=tool_name)
                        set_current_session(session)
                        self.store.save_session(session)
                    else:
                        session = self.get_or_create_session()

                    step = session.next_step()

                    # Tool loop detection
                    self._check_tool_loop(session.id, tool_name)

                    # Replay dry-run for mutating tools
                    _engine = get_current_replay_engine()
                    if (
                        mutates_state
                        and _engine is not None
                        and _engine.is_active
                        and _engine.should_mock_tool(step)
                    ):
                        return _engine.get_cached_response(step)

                    start = time.perf_counter()
                    result = None
                    try:
                        result = await func(*args, **kwargs)
                        return result
                    finally:
                        elapsed_ms = (time.perf_counter() - start) * 1000
                        event = ToolCallEvent(
                            session_id=session.id,
                            step=step,
                            tool_name=tool_name,
                            mutates_state=mutates_state,
                            latency_ms=elapsed_ms,
                        )
                        # Cache tool result for durable replay
                        if session.durable and result is not None:
                            try:
                                import json as _json

                                event.cached_response_json = _json.dumps(
                                    result,
                                    default=str,
                                )
                            except Exception:
                                pass
                        session.call_count += 1
                        self.store.save_event(event)

                        if session_root:
                            session.end(SessionStatus.COMPLETED)
                            self.store.save_session(session)
                            set_current_session(previous_session)

                setattr(async_wrapper, "_stateloom_tool", True)
                setattr(async_wrapper, "_stateloom_mutates_state", mutates_state)
                setattr(async_wrapper, "_stateloom_tool_name", tool_name)
                return async_wrapper

            @functools.wraps(func)
            def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
                previous_session = None

                if session_root:
                    previous_session = get_current_session()
                    sid = f"{tool_name}-{uuid.uuid4().hex[:8]}"
                    session = self.session_manager.create(session_id=sid, name=tool_name)
                    set_current_session(session)
                    self.store.save_session(session)
                else:
                    session = self.get_or_create_session()

                step = session.next_step()

                # Tool loop detection
                self._check_tool_loop(session.id, tool_name)

                # Replay dry-run for mutating tools
                _engine = get_current_replay_engine()
                if (
                    mutates_state
                    and _engine is not None
                    and _engine.is_active
                    and _engine.should_mock_tool(step)
                ):
                    return _engine.get_cached_response(step)

                start = time.perf_counter()
                result = None
                try:
                    result = func(*args, **kwargs)
                    return result
                finally:
                    elapsed_ms = (time.perf_counter() - start) * 1000
                    event = ToolCallEvent(
                        session_id=session.id,
                        step=step,
                        tool_name=tool_name,
                        mutates_state=mutates_state,
                        latency_ms=elapsed_ms,
                    )
                    # Cache tool result for durable replay
                    if session.durable and result is not None:
                        try:
                            import json as _json

                            event.cached_response_json = _json.dumps(
                                result,
                                default=str,
                            )
                        except Exception:
                            pass
                    session.call_count += 1
                    self.store.save_event(event)

                    if session_root:
                        session.end(SessionStatus.COMPLETED)
                        self.store.save_session(session)
                        set_current_session(previous_session)

            setattr(sync_wrapper, "_stateloom_tool", True)
            setattr(sync_wrapper, "_stateloom_mutates_state", mutates_state)
            setattr(sync_wrapper, "_stateloom_tool_name", tool_name)
            return sync_wrapper

        return decorator

    def durable_task(
        self,
        retries: int = 3,
        **kwargs: Any,
    ) -> Any:
        """Decorator: durable session + automatic retry on exception."""
        from stateloom.retry import durable_task as _durable_task

        return _durable_task(retries=retries, **kwargs)

    def replay(
        self,
        session: str,
        mock_until_step: int,
        strict: bool = True,
        allow_hosts: list[str] | None = None,
    ) -> Any:
        """Time-travel debugging: replay a session, mocking until step N.

        Args:
            session: The session ID to replay.
            mock_until_step: Mock steps 1 through N with cached responses.
            strict: If True, block outbound HTTP calls not captured via @gate.tool().
            allow_hosts: Hosts to allow through the network blocker in strict mode.

        Returns:
            The ``ReplayEngine`` instance (also a context manager). Call
            ``engine.stop()`` when done to clean up ContextVars, or use
            as ``with gate.replay(...) as engine:``.
        """
        from stateloom.replay.engine import ReplayEngine

        engine = ReplayEngine(
            self,
            session_id=session,
            mock_until_step=mock_until_step,
            strict=strict,
            allow_hosts=allow_hosts,
        )
        engine.start()
        return engine

    def share(self, session: str) -> str:
        """Share a session for collaborative debugging.

        Returns a shareable URL. Requires control plane connection.
        """
        logger.warning("Control plane not configured. share() is a stub.")
        return f"https://app.stateloom.io/share/{session}"

    def pin(self, session: str, name: str) -> None:
        """Pin a session as a regression test baseline.

        Saves pin metadata to the store for later use with `stateloom test`.
        """
        stored_session = self.store.get_session(session)
        if stored_session is None:
            logger.warning("Session '%s' not found, cannot pin.", session)
            return
        # Store pin metadata
        logger.info("Pinned session '%s' as '%s'", session, name)

    # --- Session Export/Import ---

    def export_session(
        self,
        session_id: str,
        path: str | Path | None = None,
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
        from stateloom.export.bundle import export_session as _export
        from stateloom.export.bundle import write_bundle_to_file

        pii_scanner = None
        if scrub_pii:
            from stateloom.pii.scanner import PIIScanner

            pii_scanner = PIIScanner(self.config)

        bundle = _export(
            self.store,
            session_id,
            include_children=include_children,
            scrub_pii=scrub_pii,
            pii_scanner=pii_scanner,
        )
        if path is not None:
            write_bundle_to_file(bundle, path)
        return bundle

    def import_session(
        self,
        source: str | Path | dict[str, Any],
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
        from stateloom.export.bundle import import_session as _import

        return _import(self.store, source, session_id_override=session_id_override)

    # --- Async Jobs ---

    def submit_job(
        self,
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
        priority: int = 0,
        max_retries: int = 3,
        ttl_seconds: int | None = None,
        metadata: dict[str, Any] | None = None,
        agent: str | None = None,
    ) -> Job:
        """Submit an async job. Returns the Job immediately (status=PENDING)."""
        if not self.config.async_jobs_enabled:
            raise StateLoomJobError("", "Async jobs are not enabled")

        resolved_messages = list(messages) if messages else []
        resolved_model = model
        meta = dict(metadata) if metadata else {}

        if agent:
            from stateloom.agent.resolver import apply_agent_overrides
            from stateloom.chat import _resolve_agent_for_chat

            agent_obj, version = _resolve_agent_for_chat(self, agent)
            override_model, resolved_messages, extra_kwargs = apply_agent_overrides(
                version, resolved_messages, body={}
            )
            resolved_model = override_model or resolved_model
            req_kwargs = dict(request_kwargs) if request_kwargs else {}
            req_kwargs.update(extra_kwargs)
            request_kwargs = req_kwargs
            meta.update(
                {
                    "agent_id": agent_obj.id,
                    "agent_slug": agent_obj.slug,
                    "agent_version_id": version.id,
                    "agent_version_number": version.version_number,
                }
            )

        job = Job(
            provider=provider,
            model=resolved_model,
            messages=resolved_messages,
            request_kwargs=request_kwargs or {},
            webhook_url=webhook_url,
            webhook_secret=webhook_secret,
            session_id=session_id,
            org_id=org_id,
            team_id=team_id,
            priority=priority,
            max_retries=max_retries,
            ttl_seconds=ttl_seconds or self.config.async_jobs_default_ttl,
            metadata=meta,
        )
        self.store.save_job(job)
        return job

    def get_job(self, job_id: str) -> Job | None:
        """Get a job by ID."""
        return self.store.get_job(job_id)

    def list_jobs(
        self,
        status: str | None = None,
        session_id: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[Job]:
        """List jobs with optional filters."""
        return self.store.list_jobs(
            status=status, session_id=session_id, limit=limit, offset=offset
        )

    def cancel_job(self, job_id: str) -> bool:
        """Cancel a pending job. Returns True if cancelled."""
        job = self.store.get_job(job_id)
        if job is None:
            return False
        if job.status != JobStatus.PENDING:
            return False
        job.status = JobStatus.CANCELLED
        self.store.save_job(job)
        return True

    def list_dead_jobs(self, limit: int = 100) -> list[Job]:
        """List jobs in the dead letter queue."""
        if self._job_processor is not None:
            return cast(list[Job], self._job_processor.queue.list_dead(limit))
        return self.store.list_jobs(status="dead", limit=limit)

    def requeue_dead_job(self, job_id: str) -> bool:
        """Manually requeue a dead job for retry."""
        if self._job_processor is not None:
            return cast(bool, self._job_processor.queue.requeue_dead(job_id))
        return False

    def job_stats(self) -> dict[str, Any]:
        """Get aggregate job statistics."""
        return self.store.get_job_stats()

    async def consensus(self, **kwargs: Any) -> Any:
        """Run a multi-agent consensus session.

        Args:
            **kwargs: Forwarded to ``ConsensusConfig`` and
                ``ConsensusOrchestrator.run()``.

        Returns:
            A ``ConsensusResult`` with the synthesized answer, confidence,
            cost, and round-by-round details.
        """
        from stateloom.consensus.models import ConsensusConfig
        from stateloom.consensus.orchestrator import ConsensusOrchestrator

        if self._consensus_orchestrator is None:
            self._consensus_orchestrator = ConsensusOrchestrator(self)

        # Build config from kwargs + gate defaults
        defaults = self.config.consensus_defaults
        # Resolve agent if provided
        agent_ref = kwargs.get("agent")
        agent_system_prompt = ""
        agent_id = ""
        agent_slug = ""
        agent_version_id = ""
        agent_version_number = 0
        agent_model: str | None = None

        if agent_ref:
            from stateloom.chat import _resolve_agent_for_chat

            agent_obj, version = _resolve_agent_for_chat(self, agent_ref)
            agent_system_prompt = version.system_prompt or ""
            agent_id = agent_obj.id
            agent_slug = agent_obj.slug
            agent_version_id = version.id
            agent_version_number = version.version_number
            agent_model = version.model

        # Use agent model as default when models not explicitly provided
        explicit_models = kwargs.get("models")
        if not explicit_models and agent_model:
            resolved_models = [agent_model]
        else:
            resolved_models = explicit_models or defaults.default_models

        config = ConsensusConfig(
            prompt=kwargs.get("prompt", ""),
            messages=kwargs.get("messages") or [],
            models=resolved_models,
            rounds=kwargs.get("rounds", defaults.default_rounds),
            strategy=kwargs.get("strategy", defaults.default_strategy),
            budget=kwargs.get("budget", defaults.default_budget),
            session_id=kwargs.get("session_id"),
            greedy=kwargs.get("greedy", defaults.greedy),
            greedy_agreement_threshold=kwargs.get(
                "greedy_agreement_threshold", defaults.greedy_agreement_threshold
            ),
            early_stop_enabled=kwargs.get("early_stop_enabled", defaults.early_stop_enabled),
            early_stop_threshold=kwargs.get("early_stop_threshold", defaults.early_stop_threshold),
            temperature=kwargs.get("temperature", 0.7),
            samples=kwargs.get("samples", 5),
            judge_model=kwargs.get("judge_model"),
            aggregation=kwargs.get("aggregation", "confidence_weighted"),
            agent=agent_ref,
            agent_system_prompt=agent_system_prompt,
            agent_id=agent_id,
            agent_slug=agent_slug,
            agent_version_id=agent_version_id,
            agent_version_number=agent_version_number,
        )

        # EE gating — check consensus_advanced availability
        ee_available = self._feature_registry.is_available("consensus_advanced")
        config.ee_consensus = ee_available

        if not ee_available:
            ee_hint = (
                "Set STATELOOM_LICENSE_KEY or use STATELOOM_ENV=development for local development."
            )
            if len(config.models) > 3:
                raise StateLoomFeatureError(
                    "consensus_advanced",
                    f"Consensus with {len(config.models)} models requires Enterprise. "
                    f"Core supports up to 3 models. {ee_hint}",
                )
            if config.greedy:
                raise StateLoomFeatureError(
                    "consensus_advanced",
                    f"Greedy model downgrade requires Enterprise. {ee_hint}",
                )
            if config.aggregation == "judge_synthesis":
                raise StateLoomFeatureError(
                    "consensus_advanced",
                    "Judge synthesis aggregation requires Enterprise. "
                    f"Core supports 'majority_vote' and 'confidence_weighted'. {ee_hint}",
                )
            if config.judge_model is not None:
                raise StateLoomFeatureError(
                    "consensus_advanced",
                    f"Custom judge model requires Enterprise. {ee_hint}",
                )

        return await self._consensus_orchestrator.run(config)

    _PROMPT_CLEANUP_INTERVAL = 600.0  # 10 minutes between cleanup runs

    def _maybe_cleanup_prompt_payloads(self) -> None:
        """Run throttled cleanup of expired request message payloads."""
        if not self.config.store_payloads:
            return
        now = time.monotonic()
        if now - self._last_prompt_cleanup < self._PROMPT_CLEANUP_INTERVAL:
            return
        self._last_prompt_cleanup = now
        try:
            self.store.cleanup_request_messages(
                retention_hours=self.config.store_prompt_retention_hours,
            )
        except Exception:
            logger.warning("Prompt payload cleanup failed", exc_info=True)

    def cleanup_prompt_payloads(self) -> int:
        """Manually trigger cleanup of expired request message payloads.

        Returns:
            Number of events cleaned up.
        """
        return self.store.cleanup_request_messages(
            retention_hours=self.config.store_prompt_retention_hours,
        )

    def shutdown(self, drain_timeout: float = 5.0) -> None:
        """Clean shutdown of StateLoom.

        Args:
            drain_timeout: Max seconds to wait for in-flight requests to complete.
        """
        self._shutting_down = True

        # Wait for in-flight requests to drain
        deadline = time.monotonic() + drain_timeout
        while time.monotonic() < deadline:
            with self._active_requests_lock:
                if self._active_requests <= 0:
                    break
            time.sleep(0.1)

        # Disable security engine
        if self._audit_hook_manager:
            self._audit_hook_manager.configure(enabled=False)
        if self._secret_vault:
            self._secret_vault.restore_environ()

        # Stop dev mode nagware thread
        if hasattr(self, "_devmode_shutdown_event"):
            self._devmode_shutdown_event.set()

        # Run extension shutdown hooks
        self.on_shutdown.emit(self)

        # Shutdown job processor
        if self._job_processor is not None:
            self._job_processor.shutdown(drain_timeout=drain_timeout)

        # Shutdown shadow middleware
        if hasattr(self, "_shadow_middleware"):
            self._shadow_middleware.shutdown()

        # Shutdown auto-router middleware
        if hasattr(self, "_auto_router") and self._auto_router is not None:
            self._auto_router.shutdown()

        # Stop managed Ollama
        if self._ollama_manager is not None:
            self._ollama_manager.stop()

        # Stop prompt watcher
        if self._prompt_watcher is not None:
            self._prompt_watcher.stop()

        # Unpatch
        from stateloom.intercept.unpatch import unpatch_all

        unpatch_all()
        self._patched_methods.clear()

        # Clear provider registry
        from stateloom.intercept.provider_registry import clear_adapters

        clear_adapters()

        # Stop dashboard
        if self._dashboard_server:
            self._dashboard_server.stop()

        # Close cache store
        if hasattr(self, "_cache_store") and self._cache_store is not None:
            self._cache_store.close()

        # Close store
        if hasattr(self.store, "close"):
            self.store.close()

        # Clear tool call counts
        self._tool_call_counts.clear()

    @classmethod
    def from_config(cls, path: str | Path) -> Gate:
        """Create a Gate from a YAML config file."""
        config = StateLoomConfig.from_yaml(str(path))
        gate = cls(config)
        gate._load_ee()  # FIRST — registers hooks before middleware setup
        gate._setup_middleware()  # hooks fire during this
        if config.auto_patch:
            gate._auto_patch()
        return gate
