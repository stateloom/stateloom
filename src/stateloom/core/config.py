"""StateLoom configuration using Pydantic BaseSettings."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import yaml  # type: ignore[import-untyped]
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from stateloom.core.types import BudgetAction, FailureAction, GuardrailMode, PIIMode


class PIIRule(BaseModel):
    """A single PII detection rule."""

    pattern: str  # "email", "credit_card", "ssn", "api_key", etc.
    mode: PIIMode = PIIMode.AUDIT
    on_middleware_failure: FailureAction | None = None


class KillSwitchRule(BaseModel):
    """A granular kill switch rule — blocks traffic matching specific criteria."""

    model: str | None = None  # glob match, e.g. "gpt-4", "claude-*"
    provider: str | None = None  # e.g. "openai", "anthropic"
    environment: str | None = None  # e.g. "production"
    agent_version: str | None = None  # e.g. "v2.1.0"
    message: str = ""  # override message for this rule
    reason: str = ""  # human-readable reason (audit trail)


class ComplianceProfile(BaseModel):
    """Declarative compliance profile for an org or team."""

    standard: str = "none"
    region: str = "global"
    session_ttl_days: int = 0
    cache_ttl_seconds: int = 0
    zero_retention_logs: bool = False
    block_local_routing: bool = False
    block_shadow: bool = False
    block_streaming: bool = False
    allowed_endpoints: list[str] = Field(default_factory=list)  # regex patterns
    audit_salt: str = ""
    pii_rules: list[PIIRule] = Field(default_factory=list)
    default_consent: bool = False


# ── Nested config view dataclasses ────────────────────────────────────
# Frozen dataclasses that provide grouped, read-only views into the flat
# StateLoomConfig fields.  These are NOT stored — they are created on the
# fly by @property methods on StateLoomConfig.


@dataclass(frozen=True)
class GuardrailsConfig:
    """Grouped view of guardrails_* config fields."""

    enabled: bool
    mode: GuardrailMode
    heuristic_enabled: bool
    local_model_enabled: bool
    local_model: str
    local_model_timeout: float
    output_scanning_enabled: bool
    system_prompt_leak_threshold: float
    disabled_rules: list[str]
    webhook_url: str
    nli_enabled: bool
    nli_model: str
    nli_threshold: float


@dataclass(frozen=True)
class CacheConfig:
    """Grouped view of cache_* config fields."""

    enabled: bool
    max_size: int
    ttl_seconds: int
    backend: str
    scope: str
    semantic_enabled: bool
    similarity_threshold: float
    embedding_model: str
    redis_url: str
    vector_backend: str
    normalize_patterns: list[str]
    db_path: str


@dataclass(frozen=True)
class SecurityConfig:
    """Grouped view of security_* config fields."""

    audit_hooks_enabled: bool
    audit_hooks_mode: GuardrailMode
    audit_hooks_deny_events: list[str]
    audit_hooks_allow_paths: list[str]
    secret_vault_enabled: bool
    secret_vault_scrub_environ: bool
    secret_vault_keys: list[str]
    webhook_url: str


@dataclass(frozen=True)
class AutoRouteConfig:
    """Grouped view of auto_route_* config fields."""

    enabled: bool
    force_local: bool
    model: str
    timeout: float
    complexity_threshold: float
    complex_floor: float
    probe_enabled: bool
    probe_timeout: float
    probe_threshold: float
    semantic_enabled: bool
    semantic_model: str


@dataclass(frozen=True)
class ShadowConfig:
    """Grouped view of shadow_* config fields (Model Testing)."""

    enabled: bool
    model: str
    timeout: float
    max_workers: int
    similarity_timeout: float
    similarity_method: str
    similarity_model: str
    sample_rate: float
    max_context_tokens: int
    models: list[str]


@dataclass(frozen=True)
class PIIConfig:
    """Grouped view of pii_* config fields."""

    enabled: bool
    default_mode: PIIMode
    rules: list[PIIRule]
    ner_enabled: bool
    ner_model: str
    ner_labels: list[str]
    ner_threshold: float
    stream_buffer_enabled: bool
    stream_buffer_size: int


@dataclass(frozen=True)
class ProxyConfig:
    """Grouped view of proxy_* config fields."""

    enabled: bool
    require_virtual_key: bool
    upstream_anthropic: str
    upstream_openai: str
    upstream_gemini: str
    upstream_code_assist: str
    upstream_chatgpt: str
    timeout: float


@dataclass(frozen=True)
class ConsensusDefaultsConfig:
    """Grouped view of consensus_* config fields."""

    default_models: list[str]
    default_strategy: str
    default_rounds: int
    default_budget: float | None
    max_rounds: int
    max_models: int
    early_stop_enabled: bool
    early_stop_threshold: float
    greedy: bool
    greedy_agreement_threshold: float


@dataclass(frozen=True)
class AuthConfig:
    """Grouped view of auth_* config fields."""

    enabled: bool
    jwt_algorithm: str
    jwt_access_ttl: int
    jwt_refresh_ttl: int


@dataclass(frozen=True)
class BlastRadiusConfig:
    """Grouped view of blast_radius_* config fields."""

    enabled: bool
    consecutive_failures: int
    budget_violations_per_hour: int
    webhook_url: str


@dataclass(frozen=True)
class DashboardConfig:
    """Grouped view of dashboard_* config fields."""

    enabled: bool
    port: int
    host: str
    api_key: str


class StateLoomConfig(BaseSettings):
    """Full configuration for StateLoom. Loads from env vars, kwargs, or YAML."""

    model_config = SettingsConfigDict(env_prefix="STATELOOM_")

    # Core
    auto_patch: bool = True
    fail_open: bool = True
    log_level: str = "INFO"
    debug: bool = False  # Enable debug mode (verbose logging, log buffer for dashboard)
    default_model: str = ""  # Default model for stateloom.chat() when model is omitted
    username: str = ""  # Auto-inferred from OS if empty
    license_key: str = ""  # EE license key (STATELOOM_LICENSE_KEY env var)

    # Dashboard
    dashboard: bool = True
    dashboard_port: int = 4782
    dashboard_host: str = "127.0.0.1"
    dashboard_api_key: str = ""

    # Budget
    budget_per_session: float | None = None
    budget_global: float | None = None
    budget_action: BudgetAction = BudgetAction.HARD_STOP
    budget_on_middleware_failure: FailureAction = FailureAction.BLOCK

    # PII
    pii_enabled: bool = False
    pii_default_mode: PIIMode = PIIMode.AUDIT
    pii_rules: list[PIIRule] = Field(default_factory=list)

    # PII NER (Named Entity Recognition)
    pii_ner_enabled: bool = False
    pii_ner_model: str = "urchade/gliner_small-v2.1"
    pii_ner_labels: list[str] = Field(
        default_factory=lambda: ["person", "location", "organization", "address"]
    )
    pii_ner_threshold: float = 0.5

    # PII Stream Buffer
    pii_stream_buffer_enabled: bool = False
    pii_stream_buffer_size: int = 3  # number of chunks to hold back

    # Guardrails (prompt injection / jailbreak protection)
    guardrails_enabled: bool = False
    guardrails_mode: GuardrailMode = GuardrailMode.AUDIT
    guardrails_heuristic_enabled: bool = True
    guardrails_local_model_enabled: bool = False
    guardrails_local_model: str = "llama-guard3:1b"
    guardrails_local_model_timeout: float = 10.0
    guardrails_output_scanning_enabled: bool = True
    guardrails_system_prompt_leak_threshold: float = 0.6
    guardrails_disabled_rules: list[str] = Field(default_factory=list)
    guardrails_webhook_url: str = ""
    guardrails_nli_enabled: bool = False
    guardrails_nli_model: str = "cross-encoder/nli-MiniLM2-L6-H768"
    guardrails_nli_threshold: float = 0.75

    # Cache
    cache_enabled: bool = True
    cache_max_size: int = 1000
    cache_ttl_seconds: int = 0  # 0 = no TTL (entries never expire)
    cache_backend: Literal["memory", "sqlite", "redis"] = "memory"
    cache_scope: Literal["session", "global"] = "global"
    cache_semantic_enabled: bool = False
    cache_similarity_threshold: float = 0.95
    cache_embedding_model: str = "all-MiniLM-L6-v2"
    cache_redis_url: str = "redis://localhost:6379"
    cache_vector_backend: Literal["faiss", "redis"] = "faiss"
    cache_normalize_patterns: list[str] = Field(default_factory=list)
    cache_db_path: str = ".stateloom/cache.db"

    # Loop detection
    loop_detection_enabled: bool = False
    loop_exact_threshold: int = 5
    loop_semantic_enabled: bool = False
    loop_semantic_threshold: float = 0.92

    # Storage
    store_backend: Literal["sqlite", "memory", "postgres"] = "sqlite"
    store_path: str = ".stateloom/data.db"
    store_retention_days: int = 30
    store_postgres_url: str = "postgresql://localhost:5432/stateloom"
    store_postgres_pool_min: int = 2
    store_postgres_pool_max: int = 10
    store_auto_migrate: bool = True  # Use Alembic for schema migrations when available

    # Request size limits
    max_request_body_mb: float = 10.0  # Max request body size for dashboard API (MB)

    # Console
    console_output: bool = True
    console_verbose: bool = False

    # Context injection
    context_injection: bool = False

    # Payload storage (privacy-sensitive)
    store_payloads: bool = False

    # Local model (Ollama)
    local_model_enabled: bool = False
    local_model_host: str = "http://localhost:11434"
    local_model_default: str = ""
    local_model_timeout: float = 30.0

    # Managed Ollama (download-on-demand)
    ollama_managed: bool = False
    ollama_managed_port: int = 11435
    ollama_auto_pull: bool = True

    # Model Testing (candidate model evaluation)
    shadow_enabled: bool = False
    shadow_model: str = ""
    shadow_timeout: float = 30.0
    shadow_max_workers: int = 2
    shadow_similarity_timeout: float = 5.0
    shadow_similarity_method: str = "auto"  # "auto", "semantic", or "difflib"
    shadow_similarity_model: str = "all-MiniLM-L6-v2"
    shadow_sample_rate: float = 1.0  # 0.0–1.0: fraction of eligible calls to test
    shadow_max_context_tokens: int = 8192  # Skip prompts exceeding this token estimate
    shadow_models: list[str] = Field(default_factory=list)  # Multiple candidate models

    # Kill switch
    kill_switch_active: bool = False
    kill_switch_message: str = "Service temporarily unavailable. Please try again later."
    kill_switch_response_mode: Literal["error", "response"] = "error"
    kill_switch_rules: list[KillSwitchRule] = Field(default_factory=list)
    kill_switch_environment: str = ""
    kill_switch_agent_version: str = ""
    kill_switch_webhook_url: str = ""

    # Blast radius containment
    blast_radius_enabled: bool = False
    blast_radius_consecutive_failures: int = 5
    blast_radius_budget_violations_per_hour: int = 10
    blast_radius_webhook_url: str = ""

    # Compliance
    compliance_profile: ComplianceProfile | None = None

    # Observability / Prometheus metrics
    metrics_enabled: bool = False

    # Async jobs
    async_jobs_enabled: bool = False
    async_jobs_max_workers: int = 4
    async_jobs_default_ttl: int = 3600
    async_jobs_webhook_timeout: float = 30.0
    async_jobs_webhook_retries: int = 3
    async_jobs_webhook_secret: str = ""
    async_jobs_queue_backend: str = "in_process"  # "in_process" or "redis"
    async_jobs_redis_url: str = "redis://localhost:6379"

    # Parallel pre-computation
    parallel_precompute: bool = True

    # Provider API keys (centrally managed, injected into SDK clients as fallback)
    provider_api_key_openai: str = ""
    provider_api_key_anthropic: str = ""
    provider_api_key_google: str = ""

    # Proxy mode
    proxy_enabled: bool = False
    proxy_require_virtual_key: bool = True
    proxy_upstream_anthropic: str = "https://api.anthropic.com"
    proxy_upstream_openai: str = "https://api.openai.com"
    proxy_upstream_gemini: str = "https://generativelanguage.googleapis.com"
    proxy_upstream_code_assist: str = "https://cloudcode-pa.googleapis.com"
    proxy_upstream_chatgpt: str = ""
    proxy_timeout: float = 600.0

    # Rate limiting (master switch)
    rate_limiting_enabled: bool = True  # Set False to skip all rate limiting (individual dev mode)

    # Circuit breaker
    circuit_breaker_enabled: bool = False
    circuit_breaker_failure_threshold: int = 5
    circuit_breaker_window_seconds: int = 300
    circuit_breaker_recovery_timeout: int = 60
    circuit_breaker_fallback_map: dict[str, str] = Field(default_factory=dict)

    # Durable streaming replay delay (milliseconds between chunks on replay)
    durable_stream_delay_ms: float = 0

    # Prompt file watching (individual agent management)
    prompts_dir: str = ""  # empty = disabled. e.g., "prompts/" or absolute path
    prompts_poll_interval: float = 2.0  # seconds (passed to watchdog Observer timeout)

    # Authentication (OAuth2/OIDC)
    auth_enabled: bool = False
    auth_jwt_algorithm: str = "HS256"
    auth_jwt_access_ttl: int = 900  # 15 min
    auth_jwt_refresh_ttl: int = 604800  # 7 days

    # Security (CPython audit hooks + secret vault)
    security_audit_hooks_enabled: bool = False
    security_audit_hooks_mode: GuardrailMode = GuardrailMode.AUDIT
    security_audit_hooks_deny_events: list[str] = Field(default_factory=list)
    security_audit_hooks_allow_paths: list[str] = Field(default_factory=list)
    security_secret_vault_enabled: bool = False
    security_secret_vault_scrub_environ: bool = False
    security_secret_vault_keys: list[str] = Field(default_factory=list)
    security_webhook_url: str = ""

    # Consensus (multi-agent debate)
    consensus_default_models: list[str] = Field(default_factory=list)
    consensus_default_strategy: str = "debate"
    consensus_default_rounds: int = 2
    consensus_default_budget: float | None = None
    consensus_max_rounds: int = 10
    consensus_max_models: int = 10
    consensus_early_stop_enabled: bool = True
    consensus_early_stop_threshold: float = 0.9
    consensus_greedy: bool = False
    consensus_greedy_agreement_threshold: float = 0.7

    # Auto-routing
    auto_route_enabled: bool = False
    auto_route_force_local: bool = False
    auto_route_model: str = ""
    auto_route_timeout: float = 30.0
    auto_route_complexity_threshold: float = 0.15
    auto_route_complex_floor: float = 0.15
    auto_route_probe_enabled: bool = True
    auto_route_probe_timeout: float = 5.0
    auto_route_probe_threshold: float = 0.6
    auto_route_semantic_enabled: bool = True
    auto_route_semantic_model: str = "cross-encoder/nli-MiniLM2-L6-H768"

    # ── Nested config views ─────────────────────────────────────────────
    # Read-only grouped views into the flat fields above.  These create a
    # new frozen dataclass on every access — no caching, so the view
    # always reflects the current field values.

    @property
    def guardrails(self) -> GuardrailsConfig:
        """Grouped guardrails_* config."""
        return GuardrailsConfig(
            enabled=self.guardrails_enabled,
            mode=self.guardrails_mode,
            heuristic_enabled=self.guardrails_heuristic_enabled,
            local_model_enabled=self.guardrails_local_model_enabled,
            local_model=self.guardrails_local_model,
            local_model_timeout=self.guardrails_local_model_timeout,
            output_scanning_enabled=self.guardrails_output_scanning_enabled,
            system_prompt_leak_threshold=self.guardrails_system_prompt_leak_threshold,
            disabled_rules=self.guardrails_disabled_rules,
            webhook_url=self.guardrails_webhook_url,
            nli_enabled=self.guardrails_nli_enabled,
            nli_model=self.guardrails_nli_model,
            nli_threshold=self.guardrails_nli_threshold,
        )

    @property
    def cache(self) -> CacheConfig:
        """Grouped cache_* config."""
        return CacheConfig(
            enabled=self.cache_enabled,
            max_size=self.cache_max_size,
            ttl_seconds=self.cache_ttl_seconds,
            backend=self.cache_backend,
            scope=self.cache_scope,
            semantic_enabled=self.cache_semantic_enabled,
            similarity_threshold=self.cache_similarity_threshold,
            embedding_model=self.cache_embedding_model,
            redis_url=self.cache_redis_url,
            vector_backend=self.cache_vector_backend,
            normalize_patterns=self.cache_normalize_patterns,
            db_path=self.cache_db_path,
        )

    @property
    def security(self) -> SecurityConfig:
        """Grouped security_* config."""
        return SecurityConfig(
            audit_hooks_enabled=self.security_audit_hooks_enabled,
            audit_hooks_mode=self.security_audit_hooks_mode,
            audit_hooks_deny_events=self.security_audit_hooks_deny_events,
            audit_hooks_allow_paths=self.security_audit_hooks_allow_paths,
            secret_vault_enabled=self.security_secret_vault_enabled,
            secret_vault_scrub_environ=self.security_secret_vault_scrub_environ,
            secret_vault_keys=self.security_secret_vault_keys,
            webhook_url=self.security_webhook_url,
        )

    @property
    def auto_route(self) -> AutoRouteConfig:
        """Grouped auto_route_* config."""
        return AutoRouteConfig(
            enabled=self.auto_route_enabled,
            force_local=self.auto_route_force_local,
            model=self.auto_route_model,
            timeout=self.auto_route_timeout,
            complexity_threshold=self.auto_route_complexity_threshold,
            complex_floor=self.auto_route_complex_floor,
            probe_enabled=self.auto_route_probe_enabled,
            probe_timeout=self.auto_route_probe_timeout,
            probe_threshold=self.auto_route_probe_threshold,
            semantic_enabled=self.auto_route_semantic_enabled,
            semantic_model=self.auto_route_semantic_model,
        )

    @property
    def shadow(self) -> ShadowConfig:
        """Grouped shadow_* config (Model Testing)."""
        return ShadowConfig(
            enabled=self.shadow_enabled,
            model=self.shadow_model,
            timeout=self.shadow_timeout,
            max_workers=self.shadow_max_workers,
            similarity_timeout=self.shadow_similarity_timeout,
            similarity_method=self.shadow_similarity_method,
            similarity_model=self.shadow_similarity_model,
            sample_rate=self.shadow_sample_rate,
            max_context_tokens=self.shadow_max_context_tokens,
            models=self.shadow_models,
        )

    @property
    def pii(self) -> PIIConfig:
        """Grouped pii_* config."""
        return PIIConfig(
            enabled=self.pii_enabled,
            default_mode=self.pii_default_mode,
            rules=self.pii_rules,
            ner_enabled=self.pii_ner_enabled,
            ner_model=self.pii_ner_model,
            ner_labels=self.pii_ner_labels,
            ner_threshold=self.pii_ner_threshold,
            stream_buffer_enabled=self.pii_stream_buffer_enabled,
            stream_buffer_size=self.pii_stream_buffer_size,
        )

    @property
    def proxy(self) -> ProxyConfig:
        """Grouped proxy_* config."""
        return ProxyConfig(
            enabled=self.proxy_enabled,
            require_virtual_key=self.proxy_require_virtual_key,
            upstream_anthropic=self.proxy_upstream_anthropic,
            upstream_openai=self.proxy_upstream_openai,
            upstream_gemini=self.proxy_upstream_gemini,
            upstream_code_assist=self.proxy_upstream_code_assist,
            upstream_chatgpt=self.proxy_upstream_chatgpt,
            timeout=self.proxy_timeout,
        )

    @property
    def auth(self) -> AuthConfig:
        """Grouped auth_* config."""
        return AuthConfig(
            enabled=self.auth_enabled,
            jwt_algorithm=self.auth_jwt_algorithm,
            jwt_access_ttl=self.auth_jwt_access_ttl,
            jwt_refresh_ttl=self.auth_jwt_refresh_ttl,
        )

    @property
    def blast_radius(self) -> BlastRadiusConfig:
        """Grouped blast_radius_* config."""
        return BlastRadiusConfig(
            enabled=self.blast_radius_enabled,
            consecutive_failures=self.blast_radius_consecutive_failures,
            budget_violations_per_hour=self.blast_radius_budget_violations_per_hour,
            webhook_url=self.blast_radius_webhook_url,
        )

    @property
    def consensus_defaults(self) -> ConsensusDefaultsConfig:
        """Grouped consensus_* config."""
        return ConsensusDefaultsConfig(
            default_models=self.consensus_default_models,
            default_strategy=self.consensus_default_strategy,
            default_rounds=self.consensus_default_rounds,
            default_budget=self.consensus_default_budget,
            max_rounds=self.consensus_max_rounds,
            max_models=self.consensus_max_models,
            early_stop_enabled=self.consensus_early_stop_enabled,
            early_stop_threshold=self.consensus_early_stop_threshold,
            greedy=self.consensus_greedy,
            greedy_agreement_threshold=self.consensus_greedy_agreement_threshold,
        )

    @property
    def dashboard_config(self) -> DashboardConfig:
        """Grouped dashboard_* config."""
        return DashboardConfig(
            enabled=self.dashboard,
            port=self.dashboard_port,
            host=self.dashboard_host,
            api_key=self.dashboard_api_key,
        )

    @classmethod
    def lockable_fields(cls) -> frozenset[str]:
        """Return the set of config fields that can be admin-locked."""
        return LOCKABLE_SETTINGS

    @classmethod
    def from_yaml(cls, path: str | Path) -> StateLoomConfig:
        """Load config from a YAML file, merging with env vars."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        with open(path) as f:
            data = yaml.safe_load(f) or {}

        # Flatten nested YAML structure
        flat: dict[str, Any] = {}

        # Top-level keys
        for key in ("auto_patch", "fail_open", "log_level", "store_payloads", "debug"):
            if key in data:
                flat[key] = data[key]

        # Dashboard section
        if "dashboard" in data and isinstance(data["dashboard"], dict):
            for k, v in data["dashboard"].items():
                flat[f"dashboard_{k}" if k != "enabled" else "dashboard"] = v
        elif "dashboard" in data:
            flat["dashboard"] = data["dashboard"]

        # Budget section
        if "budget" in data and isinstance(data["budget"], dict):
            for k, v in data["budget"].items():
                flat[f"budget_{k}"] = v

        # PII section
        if "pii" in data and isinstance(data["pii"], dict):
            pii = data["pii"]
            if "default_mode" in pii:
                flat["pii_default_mode"] = pii["default_mode"]
            if "rules" in pii:
                flat["pii_rules"] = [PIIRule(**r) for r in pii["rules"]]
            if "ner_enabled" in pii:
                flat["pii_ner_enabled"] = pii["ner_enabled"]
            for k in (
                "ner_model",
                "ner_labels",
                "ner_threshold",
                "stream_buffer_enabled",
                "stream_buffer_size",
            ):
                if k in pii:
                    flat[f"pii_{k}"] = pii[k]
            flat["pii_enabled"] = True

        # Cache section
        if "cache" in data and isinstance(data["cache"], dict):
            for k, v in data["cache"].items():
                if k == "enabled":
                    flat["cache_enabled"] = v
                else:
                    flat[f"cache_{k}"] = v

        # Loop section
        if "loop" in data and isinstance(data["loop"], dict):
            for k, v in data["loop"].items():
                flat[f"loop_{k}"] = v

        # Store section
        if "store" in data and isinstance(data["store"], dict):
            for k, v in data["store"].items():
                flat[f"store_{k}"] = v

        # Console section
        if "console" in data and isinstance(data["console"], dict):
            for k, v in data["console"].items():
                flat[f"console_{k}" if k != "output" else "console_output"] = v

        if "context_injection" in data:
            flat["context_injection"] = data["context_injection"]

        # Local model section
        if "local" in data and isinstance(data["local"], dict):
            local = data["local"]
            if "enabled" in local:
                flat["local_model_enabled"] = local["enabled"]
            for k in ("host", "default", "timeout"):
                if k in local:
                    flat[f"local_model_{k}"] = local[k]
            if "managed" in local:
                flat["ollama_managed"] = local["managed"]
            if "managed_port" in local:
                flat["ollama_managed_port"] = local["managed_port"]
            if "auto_pull" in local:
                flat["ollama_auto_pull"] = local["auto_pull"]

        # Shadow drafting / Model Testing section
        if "shadow" in data and isinstance(data["shadow"], dict):
            shadow = data["shadow"]
            if "enabled" in shadow:
                flat["shadow_enabled"] = shadow["enabled"]
            for k in (
                "model",
                "timeout",
                "max_workers",
                "similarity_timeout",
                "similarity_method",
                "similarity_model",
                "sample_rate",
                "max_context_tokens",
                "models",
            ):
                if k in shadow:
                    flat[f"shadow_{k}"] = shadow[k]

        # Circuit breaker section
        if "circuit_breaker" in data and isinstance(data["circuit_breaker"], dict):
            cb = data["circuit_breaker"]
            if "enabled" in cb:
                flat["circuit_breaker_enabled"] = cb["enabled"]
            for k in (
                "failure_threshold",
                "window_seconds",
                "recovery_timeout",
                "fallback_map",
            ):
                if k in cb:
                    flat[f"circuit_breaker_{k}"] = cb[k]

        # Auto-routing section
        if "auto_route" in data and isinstance(data["auto_route"], dict):
            ar = data["auto_route"]
            if "enabled" in ar:
                flat["auto_route_enabled"] = ar["enabled"]
            for k in (
                "model",
                "timeout",
                "complexity_threshold",
                "complex_floor",
                "probe_enabled",
                "probe_timeout",
                "probe_threshold",
                "force_local",
                "semantic_enabled",
                "semantic_model",
            ):
                if k in ar:
                    flat[f"auto_route_{k}"] = ar[k]

        # Kill switch section
        if "kill_switch" in data and isinstance(data["kill_switch"], dict):
            ks = data["kill_switch"]
            if "active" in ks:
                flat["kill_switch_active"] = ks["active"]
            for k in ("message", "response_mode", "environment", "agent_version", "webhook_url"):
                if k in ks:
                    flat[f"kill_switch_{k}"] = ks[k]
            if "rules" in ks and isinstance(ks["rules"], list):
                flat["kill_switch_rules"] = [KillSwitchRule(**r) for r in ks["rules"]]

        # Proxy section
        if "proxy" in data and isinstance(data["proxy"], dict):
            proxy = data["proxy"]
            if "enabled" in proxy:
                flat["proxy_enabled"] = proxy["enabled"]
            if "require_virtual_key" in proxy:
                flat["proxy_require_virtual_key"] = proxy["require_virtual_key"]
            for k in (
                "upstream_anthropic",
                "upstream_openai",
                "upstream_gemini",
                "upstream_code_assist",
                "timeout",
            ):
                if k in proxy:
                    flat[f"proxy_{k}"] = proxy[k]

        # Blast radius section
        if "blast_radius" in data and isinstance(data["blast_radius"], dict):
            br = data["blast_radius"]
            if "enabled" in br:
                flat["blast_radius_enabled"] = br["enabled"]
            for k in ("consecutive_failures", "budget_violations_per_hour", "webhook_url"):
                if k in br:
                    flat[f"blast_radius_{k}"] = br[k]

        # Async jobs section
        if "async_jobs" in data and isinstance(data["async_jobs"], dict):
            aj = data["async_jobs"]
            if "enabled" in aj:
                flat["async_jobs_enabled"] = aj["enabled"]
            for k in (
                "max_workers",
                "default_ttl",
                "webhook_timeout",
                "webhook_retries",
                "webhook_secret",
                "queue_backend",
                "redis_url",
            ):
                if k in aj:
                    flat[f"async_jobs_{k}"] = aj[k]

        # Compliance profile section
        if "compliance" in data and isinstance(data["compliance"], dict):
            flat["compliance_profile"] = ComplianceProfile(**data["compliance"])

        # Metrics
        if "metrics_enabled" in data:
            flat["metrics_enabled"] = data["metrics_enabled"]

        # Parallel pre-computation
        if "parallel_precompute" in data:
            flat["parallel_precompute"] = data["parallel_precompute"]

        # Provider API keys section
        if "provider_keys" in data and isinstance(data["provider_keys"], dict):
            pk = data["provider_keys"]
            for provider in ("openai", "anthropic", "google"):
                if provider in pk:
                    flat[f"provider_api_key_{provider}"] = pk[provider]

        # Request size limit
        if "max_request_body_mb" in data:
            flat["max_request_body_mb"] = data["max_request_body_mb"]

        # Default model
        if "default_model" in data:
            flat["default_model"] = data["default_model"]

        # Prompts section
        if "prompts" in data and isinstance(data["prompts"], dict):
            p = data["prompts"]
            for k in ("dir", "poll_interval"):
                if k in p:
                    flat[f"prompts_{k}"] = p[k]
        elif "prompts_dir" in data:
            flat["prompts_dir"] = data["prompts_dir"]

        # Guardrails section
        if "guardrails" in data and isinstance(data["guardrails"], dict):
            gr = data["guardrails"]
            if "enabled" in gr:
                flat["guardrails_enabled"] = gr["enabled"]
            for k in (
                "mode",
                "heuristic_enabled",
                "local_model_enabled",
                "local_model",
                "local_model_timeout",
                "output_scanning_enabled",
                "system_prompt_leak_threshold",
                "disabled_rules",
                "webhook_url",
                "nli_enabled",
                "nli_model",
                "nli_threshold",
            ):
                if k in gr:
                    flat[f"guardrails_{k}"] = gr[k]

        # Security section
        if "security" in data and isinstance(data["security"], dict):
            sec = data["security"]
            if "audit_hooks_enabled" in sec:
                flat["security_audit_hooks_enabled"] = sec["audit_hooks_enabled"]
            for k in (
                "audit_hooks_mode",
                "audit_hooks_deny_events",
                "audit_hooks_allow_paths",
                "secret_vault_enabled",
                "secret_vault_scrub_environ",
                "secret_vault_keys",
                "webhook_url",
            ):
                if k in sec:
                    flat[f"security_{k}"] = sec[k]

        # Auth section
        if "auth" in data and isinstance(data["auth"], dict):
            auth = data["auth"]
            if "enabled" in auth:
                flat["auth_enabled"] = auth["enabled"]
            for k in ("jwt_algorithm", "jwt_access_ttl", "jwt_refresh_ttl"):
                if k in auth:
                    flat[f"auth_{k}"] = auth[k]

        return cls(**flat)


# Fields that can be admin-locked via lock_setting() / dashboard API.
# Extracted from dashboard/api.py ConfigUpdate model fields so gate.py
# doesn't need to import from dashboard code.
LOCKABLE_SETTINGS: frozenset[str] = frozenset(
    {
        "default_model",
        "shadow_enabled",
        "shadow_model",
        "local_model_enabled",
        "local_model_default",
        "pii_enabled",
        "pii_default_mode",
        "budget_per_session",
        "budget_global",
        "budget_action",
        "cache_max_size",
        "cache_ttl_seconds",
        "cache_semantic_enabled",
        "cache_similarity_threshold",
        "cache_scope",
        "loop_exact_threshold",
        "store_retention_days",
        "console_verbose",
        "auto_route_enabled",
        "auto_route_force_local",
        "auto_route_model",
        "auto_route_complexity_threshold",
        "auto_route_probe_enabled",
        "kill_switch_active",
        "kill_switch_message",
        "kill_switch_response_mode",
        "blast_radius_enabled",
        "blast_radius_consecutive_failures",
        "blast_radius_budget_violations_per_hour",
        "provider_api_key_openai",
        "provider_api_key_anthropic",
        "provider_api_key_google",
        "security_audit_hooks_enabled",
        "security_audit_hooks_mode",
        "security_secret_vault_enabled",
        "security_secret_vault_scrub_environ",
    }
)
