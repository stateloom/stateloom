"""Enterprise Edition registration — wires EE features into the Gate runtime.

Called by ``Gate._load_ee()`` when the EE package is available and licensed.
This module is the **single** point of integration between core and EE.
All enterprise wiring lives here; gate.py never imports from ee/ directly
(except ``_load_ee()`` which calls this module).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from stateloom.gate import Gate

logger = logging.getLogger("stateloom.ee.setup")


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def register_ee(gate: Gate) -> None:
    """Register all enterprise features into the Gate runtime.

    Called from ``Gate._load_ee()`` after license validation.  This is
    invoked **before** ``_setup_middleware()`` so hooks registered here
    fire at the correct time.

    Core features (middleware, dashboard, jobs, prompts) are registered
    unconditionally — they were always available before the refactor.
    License-gated features (observability replacement, event callbacks)
    require a valid license or dev mode.
    """
    # --- Unconditional: core features always registered ---

    # 1. Load persisted compliance profile (was in Gate.__init__)
    _load_compliance_profile(gate)

    # 2. Bootstrap admin user from env vars (was Gate._bootstrap_admin)
    _bootstrap_admin(gate)

    # 3. Register middleware hook (fires during _setup_middleware)
    gate.on_middleware_setup.connect(_setup_ee_middleware)

    # 4. Register startup hooks (fire at end of _setup_middleware)
    gate.on_startup.connect(_start_dashboard)
    gate.on_startup.connect(_start_job_processor)
    gate.on_startup.connect(_start_prompt_watcher)

    # 5. Register shutdown hooks
    gate.on_shutdown.connect(_shutdown_job_processor)
    gate.on_shutdown.connect(_shutdown_prompt_watcher)
    gate.on_shutdown.connect(_shutdown_dashboard)

    # 6. Register features in registry
    gate._feature_registry.register("compliance")
    gate._feature_registry.register("auth")
    gate._feature_registry.register("agents")
    gate._feature_registry.register("dashboard")
    gate._feature_registry.register("jobs")
    gate._feature_registry.register("prompts")
    gate._feature_registry.register("consensus_advanced")
    gate._feature_registry.register("model_testing")

    # --- Restricted dev mode guardrails ---
    from stateloom.ee import is_restricted_dev_mode

    if is_restricted_dev_mode():
        _enforce_dev_mode_restrictions(gate)

    # --- License-gated: provide enterprise features per-feature ---
    from stateloom.ee import is_dev_mode, is_licensed, license_has_feature

    if is_licensed() or is_dev_mode():
        for name in gate._feature_registry.enterprise_feature_names():
            if is_dev_mode() or license_has_feature(name):
                gate._feature_registry.provide(name)

    # Observability — always init (aggregator just queries persisted events,
    # Prometheus/alerting are no-ops without config). Every developer should
    # see the full dashboard experience.
    _init_observability(gate)

    # Audit logs now check registry
    if gate._feature_registry.is_available("audit_logs"):
        _register_event_callbacks()

    logger.info("Enterprise features registered")


# ---------------------------------------------------------------------------
# Observability
# ---------------------------------------------------------------------------


def _init_observability(gate: Gate) -> None:
    """Replace null observability with real implementations."""
    if gate.config.metrics_enabled:
        try:
            from stateloom.observability.collector import MetricsCollector

            gate._metrics_collector = MetricsCollector(enabled=True)
        except Exception:
            logger.debug("MetricsCollector init failed", exc_info=True)

    try:
        from stateloom.observability.aggregator import TimeSeriesAggregator

        gate._observability_aggregator = TimeSeriesAggregator(gate.store)
    except Exception:
        logger.debug("TimeSeriesAggregator init failed", exc_info=True)

    try:
        from stateloom.observability.alerting import AlertManager

        gate._alert_manager = AlertManager(
            webhook_url=gate.config.blast_radius_webhook_url,
            webhook_urls=[u for u in [gate.config.kill_switch_webhook_url] if u],
        )
    except Exception:
        logger.debug("AlertManager init failed", exc_info=True)


# ---------------------------------------------------------------------------
# Compliance
# ---------------------------------------------------------------------------


def _load_compliance_profile(gate: Gate) -> None:
    """Load persisted compliance profile from store. Was in Gate.__init__()."""
    if gate.config.compliance_profile:
        return  # Already set via config/YAML

    stored_standard = gate.store.get_secret("global_compliance_standard")
    if not stored_standard:
        return

    try:
        from stateloom.compliance.profiles import resolve_profile

        gate.config.compliance_profile = resolve_profile(stored_standard)
        # Enable PII scanning with compliance rules
        if gate.config.compliance_profile.pii_rules:
            gate.config.pii_enabled = True
            existing = {r.pattern for r in gate.config.pii_rules}
            for rule in gate.config.compliance_profile.pii_rules:
                if rule.pattern not in existing:
                    gate.config.pii_rules.append(rule)
                    existing.add(rule.pattern)
    except Exception:
        logger.debug("Failed to load compliance profile", exc_info=True)


# ---------------------------------------------------------------------------
# Admin bootstrap
# ---------------------------------------------------------------------------


def _bootstrap_admin(gate: Gate) -> None:
    """Auto-create admin user from env vars for headless K8s deployments.

    If STATELOOM_ADMIN_EMAIL and STATELOOM_ADMIN_PASSWORD_HASH (or
    STATELOOM_ADMIN_PASSWORD) are set, upsert an admin user as org_admin
    of the Default Organization.  Skipped if the user already exists.

    Was ``Gate._bootstrap_admin()``.
    """
    import os

    email = os.environ.get("STATELOOM_ADMIN_EMAIL", "")
    password_hash = os.environ.get("STATELOOM_ADMIN_PASSWORD_HASH", "")
    password = os.environ.get("STATELOOM_ADMIN_PASSWORD", "")
    if not email:
        return

    # Resolve password hash
    if not password_hash and password:
        try:
            from stateloom.auth.password import hash_password

            password_hash = hash_password(password)
        except Exception:
            logger.warning("Failed to hash STATELOOM_ADMIN_PASSWORD")
            return
    if not password_hash:
        logger.warning("STATELOOM_ADMIN_EMAIL set but no password/hash provided")
        return

    # Check if user already exists
    existing = gate.store.get_user_by_email(email)
    if existing:
        logger.debug("Admin user %s already exists, skipping bootstrap", email)
        return

    # Ensure a default org exists
    from stateloom.core.organization import Organization
    from stateloom.core.types import Role

    orgs = gate.store.list_organizations()
    if orgs:
        org_id = orgs[0].id
    else:
        org = Organization(name="Default Organization")
        gate.store.save_organization(org)
        gate._orgs[org.id] = org
        org_id = org.id

    # Create the admin user
    try:
        from stateloom.auth.models import User

        user = User(
            email=email,
            password_hash=password_hash,
            email_verified=True,
            org_id=org_id,
            org_role=Role.ORG_ADMIN,
            is_active=True,
        )
        gate.store.save_user(user)
        logger.info("Bootstrapped admin user: %s", email)
    except Exception:
        logger.warning("Failed to bootstrap admin user", exc_info=True)


# ---------------------------------------------------------------------------
# EE Middleware
# ---------------------------------------------------------------------------


def _setup_ee_middleware(gate: Gate) -> None:
    """Insert EE middleware into the pipeline.

    Moved from ``Gate._setup_ee_middleware()``.  Called via
    ``on_middleware_setup`` signal during ``Gate._setup_middleware()``.
    """
    from stateloom.middleware.kill_switch import KillSwitchMiddleware

    # Kill switch at position 0 (before everything — full short-circuit)
    gate._kill_switch = KillSwitchMiddleware(
        gate.config, gate.store, metrics=gate._metrics_collector
    )
    gate.pipeline.add(gate._kill_switch)

    # Compliance middleware at position 1 (after kill switch)
    from stateloom.middleware.compliance import ComplianceMiddleware

    gate.pipeline.add(
        ComplianceMiddleware(
            gate.config,
            store=gate.store,
            compliance_fn=gate._get_compliance_profile,
        )
    )

    # Blast radius (after compliance, before experiment)
    gate._blast_radius = None
    if gate.config.blast_radius_enabled:
        from stateloom.middleware.blast_radius import BlastRadiusMiddleware

        gate._blast_radius = BlastRadiusMiddleware(
            gate.config, gate.store, metrics=gate._metrics_collector
        )
        gate.pipeline.add(gate._blast_radius)

    # Rate limiter (after blast radius, before experiment)
    from stateloom.middleware.rate_limiter import RateLimiterMiddleware

    gate._rate_limiter = RateLimiterMiddleware(
        team_lookup=gate.get_team,
        store=gate.store,
        metrics=gate._metrics_collector,
    )
    gate.pipeline.add(gate._rate_limiter)

    # Circuit breaker (after rate limiter, before timeout checker)
    gate._circuit_breaker = None
    if gate.config.circuit_breaker_enabled:
        from stateloom.middleware.circuit_breaker import ProviderCircuitBreakerMiddleware

        gate._circuit_breaker = ProviderCircuitBreakerMiddleware(
            gate.config, gate.store, metrics=gate._metrics_collector
        )
        gate.pipeline.add(gate._circuit_breaker)


# ---------------------------------------------------------------------------
# Dashboard
# ---------------------------------------------------------------------------


def _start_dashboard(gate: Gate) -> None:
    """Start dashboard server. Was ``Gate._start_dashboard()``."""
    if not gate.config.dashboard:
        return
    try:
        from stateloom.dashboard.server import DashboardServer

        gate._dashboard_server = DashboardServer(gate)
        gate._dashboard_server.start()
    except ImportError:
        logger.debug("Dashboard not available")
    except Exception as e:
        logger.warning("Failed to start dashboard: %s", e)


def _shutdown_dashboard(gate: Gate) -> None:
    """Stop dashboard server."""
    if gate._dashboard_server:
        gate._dashboard_server.stop()


# ---------------------------------------------------------------------------
# Job processor
# ---------------------------------------------------------------------------


def _start_job_processor(gate: Gate) -> None:
    """Start background job processor. Was ``Gate._start_job_processor()``."""
    if not gate.config.async_jobs_enabled:
        return
    try:
        from stateloom.jobs.processor import JobProcessor

        if gate.config.async_jobs_queue_backend == "redis":
            from stateloom.jobs.redis_queue import RedisJobQueue

            queue: Any = RedisJobQueue(gate.store, url=gate.config.async_jobs_redis_url)
        else:
            from stateloom.jobs.queue import InProcessJobQueue

            queue = InProcessJobQueue(gate.store)

        gate._job_processor = JobProcessor(
            gate, queue=queue, max_workers=gate.config.async_jobs_max_workers
        )
        gate._job_processor.start()
    except Exception:
        logger.warning("Failed to start job processor", exc_info=True)


def _shutdown_job_processor(gate: Gate) -> None:
    """Shutdown job processor."""
    if gate._job_processor is not None:
        gate._job_processor.shutdown(drain_timeout=5.0)


# ---------------------------------------------------------------------------
# Prompt watcher
# ---------------------------------------------------------------------------


def _start_prompt_watcher(gate: Gate) -> None:
    """Start prompt file watcher. Was ``Gate._start_prompt_watcher()``."""
    if not gate.config.prompts_dir:
        return
    try:
        from stateloom.agent.prompt_watcher import PromptWatcher

        prompts_path = Path(gate.config.prompts_dir)
        if not prompts_path.is_absolute():
            prompts_path = Path.cwd() / prompts_path
        gate._prompt_watcher = PromptWatcher(
            gate=gate,
            prompts_dir=prompts_path,
            poll_interval=gate.config.prompts_poll_interval,
            default_model=gate.config.default_model,
        )
        gate._prompt_watcher.start()
    except Exception:
        logger.warning("Failed to start prompt watcher", exc_info=True)


def _shutdown_prompt_watcher(gate: Gate) -> None:
    """Stop prompt watcher."""
    if gate._prompt_watcher is not None:
        gate._prompt_watcher.stop()


# ---------------------------------------------------------------------------
# Restricted dev mode guardrails
# ---------------------------------------------------------------------------

_NAGWARE_MSG = (
    "\n"
    "\u2554\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2557\n"
    "\u2551  StateLoom is running in UNLICENSED DEV MODE               \u2551\n"
    "\u2551                                                            \u2551\n"
    "\u2551  Restrictions: 3 teams, 5 users, 5 VKs, 5 TPS,            \u2551\n"
    "\u2551  loopback-only, no PostgreSQL.                             \u2551\n"
    "\u2551                                                            \u2551\n"
    "\u2551  Purchase a license: https://stateloom.io/pricing          \u2551\n"
    "\u2551  Contact: sales@stateloom.io                               \u2551\n"
    "\u255a\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u255d\n"
)


def _enforce_dev_mode_restrictions(gate: Gate) -> None:
    """Enforce guardrails for unlicensed dev mode.

    Called during ``register_ee()`` when ``is_restricted_dev_mode()`` is True.
    Hard errors prevent the server from starting in invalid configurations.
    """
    from stateloom.core.errors import StateLoomLicenseError

    # Guardrail 1: Network Lock — reject non-loopback binding
    host = gate.config.dashboard_host
    if host not in ("127.0.0.1", "localhost", "::1"):
        raise StateLoomLicenseError(
            "network_bind",
            f"Dev mode requires binding to loopback (got '{host}'). "
            "A license is required to bind to 0.0.0.0 or external addresses.",
        )

    # Guardrail 2: Database Lock — reject PostgreSQL
    if gate.config.store_backend == "postgres":
        raise StateLoomLicenseError(
            "postgres_store",
            "Dev mode does not support PostgreSQL. "
            "Use SQLite (default) or purchase a license for production databases.",
        )

    # Guardrail 4: Nagware — startup banner + periodic reminder
    _start_devmode_nagware(gate)


def _check_dev_scale_cap(gate: Gate, entity: str) -> None:
    """Raise if entity count exceeds dev mode limits.

    Called from dashboard API endpoints that create teams, users, or virtual keys.
    """
    from stateloom.ee import DEV_MODE_LIMITS, is_restricted_dev_mode

    if not is_restricted_dev_mode():
        return

    limit = DEV_MODE_LIMITS.get(f"max_{entity}")
    if limit is None:
        return

    counts: dict[str, int] = {}
    if entity == "teams":
        orgs = gate.store.list_organizations()
        total = 0
        for org in orgs:
            total += len(gate.store.list_teams(org.id))
        counts["teams"] = total
    elif entity == "users":
        counts["users"] = len(gate.store.list_users())
    elif entity == "virtual_keys":
        counts["virtual_keys"] = len(gate.store.list_virtual_keys())

    current = counts.get(entity, 0)
    if current >= limit:
        from stateloom.core.errors import StateLoomLicenseError

        raise StateLoomLicenseError(
            f"scale_cap_{entity}",
            f"Dev mode limit: max {int(limit)} {entity} (currently {current}). "
            "Purchase a license to remove this limit.",
        )


def _start_devmode_nagware(gate: Gate) -> None:
    """Print startup warning + periodic reminder every 60 min."""
    import sys
    import threading

    sys.stdout.write(_NAGWARE_MSG)
    sys.stdout.flush()

    shutdown_event = threading.Event()
    setattr(gate, "_devmode_shutdown_event", shutdown_event)

    def _nag_loop() -> None:
        while not shutdown_event.wait(timeout=3600):
            sys.stdout.write(_NAGWARE_MSG)
            sys.stdout.flush()

    t = threading.Thread(target=_nag_loop, daemon=True, name="stateloom-devmode-nag")
    t.start()


# ---------------------------------------------------------------------------
# EventRecorder callbacks
# ---------------------------------------------------------------------------


def _register_event_callbacks() -> None:
    """Set callback slots on EventRecorder for EE functions."""
    from stateloom.middleware.event_recorder import EventRecorder

    # WebSocket broadcast callback
    try:
        from stateloom.dashboard.ws import broadcast_event

        EventRecorder.broadcast_fn = broadcast_event
    except ImportError:
        pass

    # Compliance audit hash callback
    try:
        from stateloom.compliance.audit import compute_audit_hash

        EventRecorder.audit_hash_fn = compute_audit_hash
    except ImportError:
        pass
