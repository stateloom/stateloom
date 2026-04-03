"""PostgreSQL store for StateLoom persistence.

Uses psycopg3 with ConnectionPool for multi-process safe access.
Install with: pip install stateloom[postgres]
"""

from __future__ import annotations

import base64
import hashlib
import json
import logging
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from stateloom.core.config import ComplianceProfile, PIIRule
from stateloom.core.event import (
    AnyEvent,
    Event,
)
from stateloom.core.job import Job
from stateloom.core.organization import Organization, Team
from stateloom.core.session import Session
from stateloom.core.types import (
    AssignmentStrategy,
    EventType,
    ExperimentStatus,
    JobStatus,
    OrgStatus,
    SessionStatus,
    TeamStatus,
)
from stateloom.experiment.models import (
    Experiment,
    ExperimentAssignment,
    SessionFeedback,
    VariantConfig,
)

try:
    import psycopg  # type: ignore[import-not-found]  # noqa: F401
    from psycopg.rows import dict_row  # type: ignore[import-not-found]
    from psycopg_pool import ConnectionPool  # type: ignore[import-not-found]

    _PSYCOPG_AVAILABLE = True
except ImportError:
    _PSYCOPG_AVAILABLE = False

if TYPE_CHECKING:
    from stateloom.agent.models import Agent, AgentVersion
    from stateloom.proxy.virtual_key import VirtualKey

logger = logging.getLogger("stateloom.store.postgres")

# Advisory lock ID for schema migrations (arbitrary constant)
_MIGRATION_LOCK_ID = 42

_SCHEMA_TABLES = [
    # sessions
    """
    CREATE TABLE IF NOT EXISTS sessions (
        id TEXT PRIMARY KEY,
        name TEXT,
        started_at TEXT NOT NULL,
        ended_at TEXT,
        status TEXT NOT NULL DEFAULT 'active',
        total_cost DOUBLE PRECISION NOT NULL DEFAULT 0.0,
        total_tokens INTEGER NOT NULL DEFAULT 0,
        total_prompt_tokens INTEGER NOT NULL DEFAULT 0,
        total_completion_tokens INTEGER NOT NULL DEFAULT 0,
        call_count INTEGER NOT NULL DEFAULT 0,
        cache_hits INTEGER NOT NULL DEFAULT 0,
        cache_savings DOUBLE PRECISION NOT NULL DEFAULT 0.0,
        pii_detections INTEGER NOT NULL DEFAULT 0,
        budget DOUBLE PRECISION,
        step_counter INTEGER NOT NULL DEFAULT 0,
        metadata TEXT,
        org_id TEXT DEFAULT '',
        team_id TEXT DEFAULT '',
        parent_session_id TEXT,
        timeout DOUBLE PRECISION,
        idle_timeout DOUBLE PRECISION,
        last_heartbeat TEXT,
        estimated_api_cost DOUBLE PRECISION DEFAULT 0.0,
        s_billing_mode TEXT DEFAULT '',
        s_durable INTEGER DEFAULT 0,
        s_agent_id TEXT DEFAULT '',
        s_agent_slug TEXT DEFAULT '',
        s_agent_version_id TEXT DEFAULT '',
        s_agent_version_number INTEGER DEFAULT 0,
        s_agent_name TEXT DEFAULT '',
        s_transport TEXT DEFAULT '',
        cost_by_model_json TEXT,
        tokens_by_model_json TEXT,
        guardrail_detections INTEGER NOT NULL DEFAULT 0,
        end_user TEXT DEFAULT ''
    )
    """,
    # events
    """
    CREATE TABLE IF NOT EXISTS events (
        id TEXT PRIMARY KEY,
        session_id TEXT NOT NULL REFERENCES sessions(id),
        step INTEGER NOT NULL,
        event_type TEXT NOT NULL,
        timestamp TEXT NOT NULL,
        provider TEXT,
        model TEXT,
        prompt_tokens INTEGER,
        completion_tokens INTEGER,
        total_tokens INTEGER,
        cost DOUBLE PRECISION,
        latency_ms DOUBLE PRECISION,
        is_streaming INTEGER,
        request_hash TEXT,
        tool_name TEXT,
        mutates_state INTEGER,
        pii_type TEXT,
        pii_mode TEXT,
        pii_field TEXT,
        action_taken TEXT,
        budget_limit DOUBLE PRECISION,
        budget_spent DOUBLE PRECISION,
        pattern_hash TEXT,
        repeat_count INTEGER,
        original_model TEXT,
        saved_cost DOUBLE PRECISION,
        cached_response_json TEXT,
        metadata TEXT,
        rating TEXT,
        score DOUBLE PRECISION,
        comment TEXT,
        extra_json TEXT
    )
    """,
    # experiments
    """
    CREATE TABLE IF NOT EXISTS experiments (
        id TEXT PRIMARY KEY,
        name TEXT NOT NULL,
        description TEXT NOT NULL DEFAULT '',
        status TEXT NOT NULL DEFAULT 'draft',
        strategy TEXT NOT NULL DEFAULT 'random',
        variants_json TEXT NOT NULL DEFAULT '[]',
        assignment_counts_json TEXT NOT NULL DEFAULT '{}',
        metadata TEXT NOT NULL DEFAULT '{}',
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL
    )
    """,
    # experiment_assignments
    """
    CREATE TABLE IF NOT EXISTS experiment_assignments (
        session_id TEXT PRIMARY KEY REFERENCES sessions(id),
        experiment_id TEXT NOT NULL REFERENCES experiments(id),
        variant_name TEXT NOT NULL,
        variant_config_json TEXT NOT NULL DEFAULT '{}',
        assigned_at TEXT NOT NULL
    )
    """,
    # session_feedback
    """
    CREATE TABLE IF NOT EXISTS session_feedback (
        session_id TEXT PRIMARY KEY REFERENCES sessions(id),
        rating TEXT NOT NULL,
        score DOUBLE PRECISION,
        comment TEXT NOT NULL DEFAULT '',
        created_at TEXT NOT NULL
    )
    """,
    # organizations
    """
    CREATE TABLE IF NOT EXISTS organizations (
        id TEXT PRIMARY KEY,
        name TEXT NOT NULL DEFAULT '',
        status TEXT NOT NULL DEFAULT 'active',
        created_at TEXT NOT NULL,
        budget DOUBLE PRECISION,
        total_cost DOUBLE PRECISION NOT NULL DEFAULT 0.0,
        total_tokens INTEGER NOT NULL DEFAULT 0,
        pii_rules_json TEXT NOT NULL DEFAULT '[]',
        metadata TEXT NOT NULL DEFAULT '{}',
        compliance_profile_json TEXT DEFAULT ''
    )
    """,
    # teams
    """
    CREATE TABLE IF NOT EXISTS teams (
        id TEXT PRIMARY KEY,
        org_id TEXT NOT NULL DEFAULT '' REFERENCES organizations(id),
        name TEXT NOT NULL DEFAULT '',
        status TEXT NOT NULL DEFAULT 'active',
        created_at TEXT NOT NULL,
        budget DOUBLE PRECISION,
        total_cost DOUBLE PRECISION NOT NULL DEFAULT 0.0,
        total_tokens INTEGER NOT NULL DEFAULT 0,
        metadata TEXT NOT NULL DEFAULT '{}',
        compliance_profile_json TEXT DEFAULT '',
        rate_limit_tps DOUBLE PRECISION,
        rate_limit_priority INTEGER NOT NULL DEFAULT 0,
        rate_limit_max_queue INTEGER NOT NULL DEFAULT 100,
        rate_limit_queue_timeout DOUBLE PRECISION NOT NULL DEFAULT 30.0
    )
    """,
    # jobs
    """
    CREATE TABLE IF NOT EXISTS jobs (
        id TEXT PRIMARY KEY,
        session_id TEXT NOT NULL DEFAULT '',
        org_id TEXT NOT NULL DEFAULT '',
        team_id TEXT NOT NULL DEFAULT '',
        status TEXT NOT NULL DEFAULT 'pending',
        provider TEXT NOT NULL DEFAULT '',
        model TEXT NOT NULL DEFAULT '',
        messages_json TEXT NOT NULL DEFAULT '[]',
        request_kwargs_json TEXT NOT NULL DEFAULT '{}',
        webhook_url TEXT NOT NULL DEFAULT '',
        webhook_secret TEXT NOT NULL DEFAULT '',
        result_json TEXT,
        error TEXT NOT NULL DEFAULT '',
        error_code TEXT NOT NULL DEFAULT '',
        created_at TEXT NOT NULL,
        started_at TEXT,
        completed_at TEXT,
        retry_count INTEGER NOT NULL DEFAULT 0,
        max_retries INTEGER NOT NULL DEFAULT 3,
        ttl_seconds INTEGER NOT NULL DEFAULT 3600,
        metadata_json TEXT NOT NULL DEFAULT '{}',
        webhook_status TEXT NOT NULL DEFAULT '',
        webhook_attempts INTEGER NOT NULL DEFAULT 0,
        webhook_last_error TEXT NOT NULL DEFAULT ''
    )
    """,
    # secrets
    """
    CREATE TABLE IF NOT EXISTS secrets (
        key TEXT PRIMARY KEY,
        value TEXT NOT NULL,
        updated_at TEXT NOT NULL
    )
    """,
    # virtual_keys
    """
    CREATE TABLE IF NOT EXISTS virtual_keys (
        id TEXT PRIMARY KEY,
        key_hash TEXT NOT NULL UNIQUE,
        key_preview TEXT NOT NULL,
        team_id TEXT NOT NULL,
        org_id TEXT NOT NULL,
        name TEXT NOT NULL DEFAULT '',
        created_at TEXT NOT NULL,
        revoked INTEGER NOT NULL DEFAULT 0,
        scopes_json TEXT NOT NULL DEFAULT '[]',
        metadata TEXT NOT NULL DEFAULT '{}',
        allowed_models_json TEXT NOT NULL DEFAULT '[]',
        budget_limit DOUBLE PRECISION,
        budget_spent DOUBLE PRECISION NOT NULL DEFAULT 0.0,
        rate_limit_tps DOUBLE PRECISION,
        rate_limit_max_queue INTEGER NOT NULL DEFAULT 100,
        rate_limit_queue_timeout DOUBLE PRECISION NOT NULL DEFAULT 30.0,
        agent_ids_json TEXT NOT NULL DEFAULT '[]',
        billing_mode TEXT NOT NULL DEFAULT 'api'
    )
    """,
    # agents
    """
    CREATE TABLE IF NOT EXISTS agents (
        id TEXT PRIMARY KEY,
        slug TEXT NOT NULL,
        team_id TEXT NOT NULL,
        org_id TEXT NOT NULL DEFAULT '',
        name TEXT NOT NULL DEFAULT '',
        description TEXT NOT NULL DEFAULT '',
        status TEXT NOT NULL DEFAULT 'active',
        active_version_id TEXT,
        metadata TEXT NOT NULL DEFAULT '{}',
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        UNIQUE(slug, team_id)
    )
    """,
    # agent_versions
    """
    CREATE TABLE IF NOT EXISTS agent_versions (
        id TEXT PRIMARY KEY,
        agent_id TEXT NOT NULL REFERENCES agents(id),
        version_number INTEGER NOT NULL,
        model TEXT NOT NULL DEFAULT '',
        system_prompt TEXT NOT NULL DEFAULT '',
        request_overrides_json TEXT NOT NULL DEFAULT '{}',
        compliance_profile_json TEXT NOT NULL DEFAULT '',
        budget_per_session DOUBLE PRECISION,
        metadata TEXT NOT NULL DEFAULT '{}',
        created_at TEXT NOT NULL,
        created_by TEXT NOT NULL DEFAULT '',
        UNIQUE(agent_id, version_number)
    )
    """,
    # admin_locks
    """
    CREATE TABLE IF NOT EXISTS admin_locks (
        setting TEXT PRIMARY KEY,
        value TEXT NOT NULL,
        locked_by TEXT NOT NULL DEFAULT '',
        reason TEXT NOT NULL DEFAULT '',
        locked_at TEXT NOT NULL
    )
    """,
]

_SCHEMA_INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_events_session ON events(session_id)",
    "CREATE INDEX IF NOT EXISTS idx_events_type ON events(event_type)",
    "CREATE INDEX IF NOT EXISTS idx_events_timestamp ON events(timestamp)",
    "CREATE INDEX IF NOT EXISTS idx_sessions_status ON sessions(status)",
    "CREATE INDEX IF NOT EXISTS idx_sessions_started ON sessions(started_at)",
    "CREATE INDEX IF NOT EXISTS idx_sessions_org ON sessions(org_id)",
    "CREATE INDEX IF NOT EXISTS idx_sessions_team ON sessions(team_id)",
    "CREATE INDEX IF NOT EXISTS idx_sessions_parent ON sessions(parent_session_id)",
    "CREATE INDEX IF NOT EXISTS idx_experiments_status ON experiments(status)",
    "CREATE INDEX IF NOT EXISTS idx_assignments_experiment"
    " ON experiment_assignments(experiment_id)",
    "CREATE INDEX IF NOT EXISTS idx_teams_org ON teams(org_id)",
    "CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status)",
    "CREATE INDEX IF NOT EXISTS idx_jobs_created ON jobs(created_at)",
    "CREATE INDEX IF NOT EXISTS idx_vk_hash ON virtual_keys(key_hash)",
    "CREATE INDEX IF NOT EXISTS idx_vk_team ON virtual_keys(team_id)",
    "CREATE INDEX IF NOT EXISTS idx_agents_slug_team ON agents(slug, team_id)",
    "CREATE INDEX IF NOT EXISTS idx_agents_team ON agents(team_id)",
    "CREATE INDEX IF NOT EXISTS idx_agent_versions_agent ON agent_versions(agent_id)",
]


class PostgresStore:
    """PostgreSQL-backed persistence using psycopg3 connection pool.

    Implements the full Store protocol with PostgreSQL MVCC for concurrency.
    No application-level locks needed — PostgreSQL handles concurrent access.
    """

    def __init__(
        self,
        url: str = "postgresql://localhost:5432/stateloom",
        pool_min: int = 2,
        pool_max: int = 10,
        auto_migrate: bool = True,
    ) -> None:
        if not _PSYCOPG_AVAILABLE:
            raise ImportError(
                "PostgreSQL store requires psycopg3. Install with: pip install stateloom[postgres]"
            )
        self._url = url
        self._pool = ConnectionPool(
            conninfo=url,
            min_size=pool_min,
            max_size=pool_max,
            kwargs={"row_factory": dict_row},
        )

        if auto_migrate:
            alembic_ran = False
            try:
                from stateloom.store.migrator import run_migrations

                alembic_ran = run_migrations(url)
            except Exception:
                logger.debug("Alembic migration failed, falling back to legacy", exc_info=True)

            if not alembic_ran:
                self._migrate_schema()
        else:
            self._migrate_schema()

    def _migrate_schema(self) -> None:
        """Create tables and indexes, using advisory lock to prevent races."""
        with self._pool.connection() as conn:
            # Acquire advisory lock so only one worker runs migrations
            conn.execute("SELECT pg_advisory_lock(%s)", (_MIGRATION_LOCK_ID,))
            try:
                for ddl in _SCHEMA_TABLES:
                    conn.execute(ddl)
                for ddl in _SCHEMA_INDEXES:
                    conn.execute(ddl)
                conn.commit()
            finally:
                conn.execute("SELECT pg_advisory_unlock(%s)", (_MIGRATION_LOCK_ID,))

    # --- Session methods ---

    def save_session(self, session: Session) -> None:
        with self._pool.connection() as conn:
            conn.execute(
                """INSERT INTO sessions
                   (id, name, started_at, ended_at, status, total_cost, total_tokens,
                    total_prompt_tokens, total_completion_tokens, call_count,
                    cache_hits, cache_savings, pii_detections, budget, step_counter,
                    metadata, org_id, team_id,
                    parent_session_id, timeout, idle_timeout, last_heartbeat,
                    estimated_api_cost,
                    cost_by_model_json, tokens_by_model_json,
                    s_billing_mode, s_durable, s_agent_id, s_agent_slug,
                    s_agent_version_id, s_agent_version_number, s_agent_name, s_transport)
                   VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                           %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                           %s, %s, %s, %s, %s, %s, %s, %s)
                   ON CONFLICT (id) DO UPDATE SET
                    name = EXCLUDED.name,
                    started_at = EXCLUDED.started_at,
                    ended_at = EXCLUDED.ended_at,
                    status = EXCLUDED.status,
                    total_cost = EXCLUDED.total_cost,
                    total_tokens = EXCLUDED.total_tokens,
                    total_prompt_tokens = EXCLUDED.total_prompt_tokens,
                    total_completion_tokens = EXCLUDED.total_completion_tokens,
                    call_count = EXCLUDED.call_count,
                    cache_hits = EXCLUDED.cache_hits,
                    cache_savings = EXCLUDED.cache_savings,
                    pii_detections = EXCLUDED.pii_detections,
                    budget = EXCLUDED.budget,
                    step_counter = EXCLUDED.step_counter,
                    metadata = EXCLUDED.metadata,
                    org_id = EXCLUDED.org_id,
                    team_id = EXCLUDED.team_id,
                    parent_session_id = EXCLUDED.parent_session_id,
                    timeout = EXCLUDED.timeout,
                    idle_timeout = EXCLUDED.idle_timeout,
                    last_heartbeat = EXCLUDED.last_heartbeat,
                    estimated_api_cost = EXCLUDED.estimated_api_cost,
                    cost_by_model_json = EXCLUDED.cost_by_model_json,
                    tokens_by_model_json = EXCLUDED.tokens_by_model_json,
                    s_billing_mode = EXCLUDED.s_billing_mode,
                    s_durable = EXCLUDED.s_durable,
                    s_agent_id = EXCLUDED.s_agent_id,
                    s_agent_slug = EXCLUDED.s_agent_slug,
                    s_agent_version_id = EXCLUDED.s_agent_version_id,
                    s_agent_version_number = EXCLUDED.s_agent_version_number,
                    s_agent_name = EXCLUDED.s_agent_name,
                    s_transport = EXCLUDED.s_transport""",
                (
                    session.id,
                    session.name,
                    session.started_at.isoformat(),
                    session.ended_at.isoformat() if session.ended_at else None,
                    session.status.value,
                    session.total_cost,
                    session.total_tokens,
                    session.total_prompt_tokens,
                    session.total_completion_tokens,
                    session.call_count,
                    session.cache_hits,
                    session.cache_savings,
                    session.pii_detections,
                    session.budget,
                    session.step_counter,
                    json.dumps(session.metadata) if session.metadata else None,
                    session.org_id,
                    session.team_id,
                    session.parent_session_id,
                    session.timeout,
                    session.idle_timeout,
                    session.last_heartbeat.isoformat() if session.last_heartbeat else None,
                    session.estimated_api_cost,
                    json.dumps(session.cost_by_model) if session.cost_by_model else None,
                    json.dumps(session.tokens_by_model) if session.tokens_by_model else None,
                    session.billing_mode,
                    1 if session.durable else 0,
                    session.agent_id,
                    session.agent_slug,
                    session.agent_version_id,
                    session.agent_version_number,
                    session.agent_name,
                    session.transport,
                ),
            )
            conn.commit()

    def get_session(self, session_id: str) -> Session | None:
        with self._pool.connection() as conn:
            row = conn.execute("SELECT * FROM sessions WHERE id = %s", (session_id,)).fetchone()
            if not row:
                return None
            return self._row_to_session(row)

    def list_sessions(
        self,
        limit: int = 100,
        offset: int = 0,
        status: str | None = None,
        org_id: str | None = None,
        team_id: str | None = None,
        end_user: str | None = None,
    ) -> list[Session]:
        with self._pool.connection() as conn:
            query = "SELECT * FROM sessions"
            conditions: list[str] = []
            params: list[Any] = []
            if status:
                conditions.append("status = %s")
                params.append(status)
            if org_id is not None:
                conditions.append("org_id = %s")
                params.append(org_id)
            if team_id is not None:
                conditions.append("team_id = %s")
                params.append(team_id)
            if end_user is not None:
                conditions.append("end_user = %s")
                params.append(end_user)
            if conditions:
                query += " WHERE " + " AND ".join(conditions)
            query += " ORDER BY started_at DESC LIMIT %s OFFSET %s"
            params.extend([limit, offset])
            rows = conn.execute(query, params).fetchall()
            return [self._row_to_session(r) for r in rows]

    def count_sessions(
        self,
        status: str | None = None,
        org_id: str | None = None,
        team_id: str | None = None,
        end_user: str | None = None,
    ) -> int:
        with self._pool.connection() as conn:
            query = "SELECT COUNT(*) FROM sessions"
            conditions: list[str] = []
            params: list[Any] = []
            if status:
                conditions.append("status = %s")
                params.append(status)
            if org_id is not None:
                conditions.append("org_id = %s")
                params.append(org_id)
            if team_id is not None:
                conditions.append("team_id = %s")
                params.append(team_id)
            if end_user is not None:
                conditions.append("end_user = %s")
                params.append(end_user)
            if conditions:
                query += " WHERE " + " AND ".join(conditions)
            row = conn.execute(query, params).fetchone()
            return row[0] if row else 0

    def list_child_sessions(
        self,
        parent_session_id: str,
        limit: int = 100,
    ) -> list[Session]:
        with self._pool.connection() as conn:
            rows = conn.execute(
                "SELECT * FROM sessions WHERE parent_session_id = %s "
                "ORDER BY started_at DESC LIMIT %s",
                (parent_session_id, limit),
            ).fetchall()
            return [self._row_to_session(r) for r in rows]

    def save_session_with_events(self, session: Session, events: list[Event]) -> None:
        with self._pool.connection() as conn:
            with conn.transaction():
                self._upsert_session(conn, session)
                for event in events:
                    self._insert_event(conn, event)

    def _upsert_session(self, conn: Any, session: Session) -> None:
        """Insert or update a session (used within transactions)."""
        conn.execute(
            """INSERT INTO sessions
               (id, name, started_at, ended_at, status, total_cost, total_tokens,
                total_prompt_tokens, total_completion_tokens, call_count,
                cache_hits, cache_savings, pii_detections, budget, step_counter,
                metadata, org_id, team_id,
                parent_session_id, timeout, idle_timeout, last_heartbeat,
                estimated_api_cost,
                cost_by_model_json, tokens_by_model_json,
                s_billing_mode, s_durable, s_agent_id, s_agent_slug,
                s_agent_version_id, s_agent_version_number, s_agent_name, s_transport)
               VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                       %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                       %s, %s, %s, %s, %s, %s, %s, %s)
               ON CONFLICT (id) DO UPDATE SET
                name = EXCLUDED.name,
                started_at = EXCLUDED.started_at,
                ended_at = EXCLUDED.ended_at,
                status = EXCLUDED.status,
                total_cost = EXCLUDED.total_cost,
                total_tokens = EXCLUDED.total_tokens,
                total_prompt_tokens = EXCLUDED.total_prompt_tokens,
                total_completion_tokens = EXCLUDED.total_completion_tokens,
                call_count = EXCLUDED.call_count,
                cache_hits = EXCLUDED.cache_hits,
                cache_savings = EXCLUDED.cache_savings,
                pii_detections = EXCLUDED.pii_detections,
                budget = EXCLUDED.budget,
                step_counter = EXCLUDED.step_counter,
                metadata = EXCLUDED.metadata,
                org_id = EXCLUDED.org_id,
                team_id = EXCLUDED.team_id,
                parent_session_id = EXCLUDED.parent_session_id,
                timeout = EXCLUDED.timeout,
                idle_timeout = EXCLUDED.idle_timeout,
                last_heartbeat = EXCLUDED.last_heartbeat,
                estimated_api_cost = EXCLUDED.estimated_api_cost,
                cost_by_model_json = EXCLUDED.cost_by_model_json,
                tokens_by_model_json = EXCLUDED.tokens_by_model_json,
                s_billing_mode = EXCLUDED.s_billing_mode,
                s_durable = EXCLUDED.s_durable,
                s_agent_id = EXCLUDED.s_agent_id,
                s_agent_slug = EXCLUDED.s_agent_slug,
                s_agent_version_id = EXCLUDED.s_agent_version_id,
                s_agent_version_number = EXCLUDED.s_agent_version_number,
                s_agent_name = EXCLUDED.s_agent_name,
                s_transport = EXCLUDED.s_transport""",
            (
                session.id,
                session.name,
                session.started_at.isoformat(),
                session.ended_at.isoformat() if session.ended_at else None,
                session.status.value,
                session.total_cost,
                session.total_tokens,
                session.total_prompt_tokens,
                session.total_completion_tokens,
                session.call_count,
                session.cache_hits,
                session.cache_savings,
                session.pii_detections,
                session.budget,
                session.step_counter,
                json.dumps(session.metadata) if session.metadata else None,
                session.org_id,
                session.team_id,
                session.parent_session_id,
                session.timeout,
                session.idle_timeout,
                session.last_heartbeat.isoformat() if session.last_heartbeat else None,
                session.estimated_api_cost,
                json.dumps(session.cost_by_model) if session.cost_by_model else None,
                json.dumps(session.tokens_by_model) if session.tokens_by_model else None,
                session.billing_mode,
                1 if session.durable else 0,
                session.agent_id,
                session.agent_slug,
                session.agent_version_id,
                session.agent_version_number,
                session.agent_name,
                session.transport,
            ),
        )

    # --- Event methods ---

    def save_event(self, event: Event) -> None:
        with self._pool.connection() as conn:
            self._insert_event(conn, event)
            conn.commit()

    def _insert_event(self, conn: Any, event: Event) -> None:
        """Insert or update an event."""
        row = self._event_to_row(event)
        conn.execute(
            """INSERT INTO events
               (id, session_id, step, event_type, timestamp, provider, model,
                prompt_tokens, completion_tokens, total_tokens, cost, latency_ms,
                is_streaming, request_hash, tool_name, mutates_state,
                pii_type, pii_mode, pii_field, action_taken,
                budget_limit, budget_spent, pattern_hash, repeat_count,
                original_model, saved_cost, cached_response_json, metadata,
                rating, score, comment, extra_json)
               VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                       %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                       %s, %s, %s, %s, %s, %s, %s, %s)
               ON CONFLICT (id) DO UPDATE SET
                session_id = EXCLUDED.session_id,
                step = EXCLUDED.step,
                event_type = EXCLUDED.event_type,
                timestamp = EXCLUDED.timestamp,
                provider = EXCLUDED.provider,
                model = EXCLUDED.model,
                prompt_tokens = EXCLUDED.prompt_tokens,
                completion_tokens = EXCLUDED.completion_tokens,
                total_tokens = EXCLUDED.total_tokens,
                cost = EXCLUDED.cost,
                latency_ms = EXCLUDED.latency_ms,
                is_streaming = EXCLUDED.is_streaming,
                request_hash = EXCLUDED.request_hash,
                tool_name = EXCLUDED.tool_name,
                mutates_state = EXCLUDED.mutates_state,
                pii_type = EXCLUDED.pii_type,
                pii_mode = EXCLUDED.pii_mode,
                pii_field = EXCLUDED.pii_field,
                action_taken = EXCLUDED.action_taken,
                budget_limit = EXCLUDED.budget_limit,
                budget_spent = EXCLUDED.budget_spent,
                pattern_hash = EXCLUDED.pattern_hash,
                repeat_count = EXCLUDED.repeat_count,
                original_model = EXCLUDED.original_model,
                saved_cost = EXCLUDED.saved_cost,
                cached_response_json = EXCLUDED.cached_response_json,
                metadata = EXCLUDED.metadata,
                rating = EXCLUDED.rating,
                score = EXCLUDED.score,
                comment = EXCLUDED.comment,
                extra_json = EXCLUDED.extra_json""",
            row,
        )

    def get_session_events(
        self,
        session_id: str,
        event_type: str | None = None,
        limit: int = 1000,
        offset: int = 0,
        desc: bool = False,
    ) -> list[Event]:
        with self._pool.connection() as conn:
            if session_id:
                query = "SELECT * FROM events WHERE session_id = %s"
                params: list[Any] = [session_id]
            else:
                query = "SELECT * FROM events WHERE 1=1"
                params = []
            if event_type:
                query += " AND event_type = %s"
                params.append(event_type)
            order = "DESC" if desc else "ASC"
            query += f" ORDER BY timestamp {order} LIMIT %s OFFSET %s"
            params.extend([limit, offset])
            rows = conn.execute(query, params).fetchall()
            return [self._row_to_event(r) for r in rows]

    def count_events(
        self,
        session_id: str = "",
        event_type: str | None = None,
    ) -> int:
        with self._pool.connection() as conn:
            if session_id:
                query = "SELECT COUNT(*) FROM events WHERE session_id = %s"
                params: list[Any] = [session_id]
            else:
                query = "SELECT COUNT(*) FROM events WHERE 1=1"
                params = []
            if event_type:
                query += " AND event_type = %s"
                params.append(event_type)
            row = conn.execute(query, params).fetchone()
            return row[0] if row else 0

    def get_pii_stats(self) -> dict[str, Any]:
        with self._pool.connection() as conn:
            # Sessions affected
            row = conn.execute(
                "SELECT COUNT(DISTINCT session_id) FROM events WHERE event_type = 'pii_detection'"
            ).fetchone()
            sessions_affected = row[0] if row else 0
            # By type
            by_type: dict[str, int] = {}
            for r in conn.execute(
                "SELECT pii_type, COUNT(*) FROM events"
                " WHERE event_type = 'pii_detection' GROUP BY pii_type"
            ).fetchall():
                if r[0]:
                    by_type[r[0]] = r[1]
            # By action
            by_action: dict[str, int] = {}
            for r in conn.execute(
                "SELECT action_taken, COUNT(*) FROM events"
                " WHERE event_type = 'pii_detection' GROUP BY action_taken"
            ).fetchall():
                if r[0]:
                    by_action[r[0]] = r[1]
            return {
                "sessions_affected": sessions_affected,
                "by_type": by_type,
                "by_action": by_action,
            }

    def get_global_stats(self) -> dict[str, Any]:
        with self._pool.connection() as conn:
            row = conn.execute(
                """SELECT
                    COUNT(*) as total_sessions,
                    SUM(CASE WHEN status = 'active' THEN 1 ELSE 0 END) as active_sessions,
                    COALESCE(SUM(total_cost), 0) as total_cost,
                    COALESCE(SUM(total_tokens), 0) as total_tokens,
                    COALESCE(SUM(call_count), 0) as total_calls,
                    COALESCE(SUM(cache_hits), 0) as total_cache_hits,
                    COALESCE(SUM(cache_savings), 0) as total_cache_savings,
                    COALESCE(SUM(pii_detections), 0) as total_pii_detections,
                    COALESCE(SUM(estimated_api_cost), 0) as total_estimated_api_cost
                FROM sessions"""
            ).fetchone()
            return dict(row) if row else {}

    def get_call_counts(self) -> dict[str, int]:
        with self._pool.connection() as conn:
            row = conn.execute(
                """SELECT
                    COUNT(*) as total,
                    SUM(CASE WHEN provider = 'local' THEN 1 ELSE 0 END) as local_calls,
                    SUM(CASE WHEN provider != 'local' THEN 1 ELSE 0 END) as cloud_calls
                FROM events
                WHERE event_type = 'llm_call'"""
            ).fetchone()
            return dict(row) if row else {"total": 0, "local_calls": 0, "cloud_calls": 0}

    def get_cost_by_model(self) -> dict[str, float]:
        with self._pool.connection() as conn:
            rows = conn.execute(
                """SELECT COALESCE(model, 'unknown') as model,
                          COALESCE(SUM(cost), 0) as total_cost
                FROM events
                WHERE event_type = 'llm_call'
                GROUP BY model"""
            ).fetchall()
            return {r["model"]: r["total_cost"] for r in rows}

    def cleanup(self, retention_days: int = 30) -> int:
        cutoff = (datetime.now(timezone.utc) - timedelta(days=retention_days)).isoformat()
        with self._pool.connection() as conn:
            cursor = conn.execute("DELETE FROM events WHERE timestamp < %s", (cutoff,))
            deleted = cast(int, cursor.rowcount)
            conn.execute(
                "DELETE FROM sessions WHERE ended_at IS NOT NULL AND ended_at < %s",
                (cutoff,),
            )
            conn.commit()
            return deleted

    def cleanup_durable_cache(self, session_id: str = "") -> int:
        """Clear cached_response_json blobs for completed durable sessions."""
        with self._pool.connection() as conn:
            if session_id:
                cursor = conn.execute(
                    "UPDATE events SET cached_response_json = NULL "
                    "WHERE session_id = %s AND cached_response_json IS NOT NULL",
                    (session_id,),
                )
            else:
                cursor = conn.execute(
                    "UPDATE events SET cached_response_json = NULL "
                    "WHERE cached_response_json IS NOT NULL "
                    "AND session_id IN ("
                    "  SELECT id FROM sessions "
                    "  WHERE status IN ('completed', 'error', 'budget_exceeded', 'loop_killed')"
                    ")",
                )
            updated = cast(int, cursor.rowcount)
            conn.commit()
            return updated

    def close(self) -> None:
        self._pool.close()

    # --- Experiment methods ---

    def save_experiment(self, experiment: Experiment) -> None:
        with self._pool.connection() as conn:
            conn.execute(
                """INSERT INTO experiments
                   (id, name, description, status, strategy, variants_json,
                    assignment_counts_json, metadata, created_at, updated_at)
                   VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                   ON CONFLICT (id) DO UPDATE SET
                    name = EXCLUDED.name,
                    description = EXCLUDED.description,
                    status = EXCLUDED.status,
                    strategy = EXCLUDED.strategy,
                    variants_json = EXCLUDED.variants_json,
                    assignment_counts_json = EXCLUDED.assignment_counts_json,
                    metadata = EXCLUDED.metadata,
                    created_at = EXCLUDED.created_at,
                    updated_at = EXCLUDED.updated_at""",
                (
                    experiment.id,
                    experiment.name,
                    experiment.description,
                    experiment.status.value,
                    experiment.strategy.value,
                    json.dumps([v.to_dict() for v in experiment.variants]),
                    json.dumps(experiment.assignment_counts),
                    json.dumps(experiment.metadata),
                    experiment.created_at.isoformat(),
                    experiment.updated_at.isoformat(),
                ),
            )
            conn.commit()

    def get_experiment(self, experiment_id: str) -> Experiment | None:
        with self._pool.connection() as conn:
            row = conn.execute(
                "SELECT * FROM experiments WHERE id = %s", (experiment_id,)
            ).fetchone()
            if not row:
                return None
            return self._row_to_experiment(row)

    def list_experiments(self, status: str | None = None) -> list[Experiment]:
        with self._pool.connection() as conn:
            query = "SELECT * FROM experiments"
            params: list[Any] = []
            if status:
                query += " WHERE status = %s"
                params.append(status)
            query += " ORDER BY created_at DESC"
            rows = conn.execute(query, params).fetchall()
            return [self._row_to_experiment(r) for r in rows]

    def save_assignment(self, assignment: ExperimentAssignment) -> None:
        with self._pool.connection() as conn:
            conn.execute(
                """INSERT INTO experiment_assignments
                   (session_id, experiment_id, variant_name, variant_config_json, assigned_at)
                   VALUES (%s, %s, %s, %s, %s)
                   ON CONFLICT (session_id) DO UPDATE SET
                    experiment_id = EXCLUDED.experiment_id,
                    variant_name = EXCLUDED.variant_name,
                    variant_config_json = EXCLUDED.variant_config_json,
                    assigned_at = EXCLUDED.assigned_at""",
                (
                    assignment.session_id,
                    assignment.experiment_id,
                    assignment.variant_name,
                    json.dumps(assignment.variant_config),
                    assignment.assigned_at.isoformat(),
                ),
            )
            conn.commit()

    def get_assignment(self, session_id: str) -> ExperimentAssignment | None:
        with self._pool.connection() as conn:
            row = conn.execute(
                "SELECT * FROM experiment_assignments WHERE session_id = %s",
                (session_id,),
            ).fetchone()
            if not row:
                return None
            return ExperimentAssignment(
                session_id=row["session_id"],
                experiment_id=row["experiment_id"],
                variant_name=row["variant_name"],
                variant_config=json.loads(row["variant_config_json"]),
                assigned_at=datetime.fromisoformat(row["assigned_at"]),
            )

    def list_assignments(self, experiment_id: str) -> list[ExperimentAssignment]:
        with self._pool.connection() as conn:
            rows = conn.execute(
                "SELECT * FROM experiment_assignments WHERE experiment_id = %s "
                "ORDER BY assigned_at",
                (experiment_id,),
            ).fetchall()
            return [
                ExperimentAssignment(
                    session_id=r["session_id"],
                    experiment_id=r["experiment_id"],
                    variant_name=r["variant_name"],
                    variant_config=json.loads(r["variant_config_json"]),
                    assigned_at=datetime.fromisoformat(r["assigned_at"]),
                )
                for r in rows
            ]

    def save_feedback(self, feedback: SessionFeedback) -> None:
        with self._pool.connection() as conn:
            conn.execute(
                """INSERT INTO session_feedback
                   (session_id, rating, score, comment, created_at)
                   VALUES (%s, %s, %s, %s, %s)
                   ON CONFLICT (session_id) DO UPDATE SET
                    rating = EXCLUDED.rating,
                    score = EXCLUDED.score,
                    comment = EXCLUDED.comment,
                    created_at = EXCLUDED.created_at""",
                (
                    feedback.session_id,
                    feedback.rating,
                    feedback.score,
                    feedback.comment,
                    feedback.created_at.isoformat(),
                ),
            )
            conn.commit()

    def get_feedback(self, session_id: str) -> SessionFeedback | None:
        with self._pool.connection() as conn:
            row = conn.execute(
                "SELECT * FROM session_feedback WHERE session_id = %s",
                (session_id,),
            ).fetchone()
            if not row:
                return None
            return SessionFeedback(
                session_id=row["session_id"],
                rating=row["rating"],
                score=row["score"],
                comment=row["comment"],
                created_at=datetime.fromisoformat(row["created_at"]),
            )

    def list_feedback(self, experiment_id: str | None = None) -> list[SessionFeedback]:
        with self._pool.connection() as conn:
            if experiment_id:
                rows = conn.execute(
                    """SELECT f.* FROM session_feedback f
                       JOIN experiment_assignments a ON f.session_id = a.session_id
                       WHERE a.experiment_id = %s
                       ORDER BY f.created_at""",
                    (experiment_id,),
                ).fetchall()
            else:
                rows = conn.execute("SELECT * FROM session_feedback ORDER BY created_at").fetchall()
            return [
                SessionFeedback(
                    session_id=r["session_id"],
                    rating=r["rating"],
                    score=r["score"],
                    comment=r["comment"],
                    created_at=datetime.fromisoformat(r["created_at"]),
                )
                for r in rows
            ]

    def get_experiment_metrics(self, experiment_id: str) -> dict[str, Any]:
        with self._pool.connection() as conn:
            rows = conn.execute(
                """SELECT
                    a.variant_name,
                    COUNT(a.session_id) as session_count,
                    COALESCE(AVG(s.total_cost), 0) as avg_cost,
                    COALESCE(SUM(s.total_cost), 0) as total_cost,
                    COALESCE(AVG(s.total_tokens), 0) as avg_tokens,
                    COALESCE(AVG(s.total_prompt_tokens), 0) as avg_prompt_tokens,
                    COALESCE(AVG(s.total_completion_tokens), 0) as avg_completion_tokens,
                    COALESCE(AVG(s.call_count), 0) as avg_call_count,
                    SUM(CASE WHEN f.rating = 'success' THEN 1 ELSE 0 END) as success_count,
                    SUM(CASE WHEN f.rating = 'failure' THEN 1 ELSE 0 END) as failure_count,
                    SUM(CASE WHEN f.rating = 'partial' THEN 1 ELSE 0 END) as partial_count,
                    SUM(CASE WHEN f.rating IS NULL THEN 1 ELSE 0 END) as unrated_count,
                    AVG(f.score) as avg_score
                FROM experiment_assignments a
                JOIN sessions s ON a.session_id = s.id
                LEFT JOIN session_feedback f ON a.session_id = f.session_id
                WHERE a.experiment_id = %s
                GROUP BY a.variant_name""",
                (experiment_id,),
            ).fetchall()

            # Compute latency metrics from events
            latency_rows = conn.execute(
                """SELECT a.variant_name, AVG(e.latency_ms) as avg_latency_ms
                FROM experiment_assignments a
                JOIN events e ON a.session_id = e.session_id AND e.event_type = 'llm_call'
                WHERE a.experiment_id = %s
                GROUP BY a.variant_name""",
                (experiment_id,),
            ).fetchall()
            latency_map = {r["variant_name"]: r["avg_latency_ms"] or 0.0 for r in latency_rows}

            # Compute median and p95 cost
            cost_rows = conn.execute(
                """SELECT a.variant_name, s.total_cost
                FROM experiment_assignments a
                JOIN sessions s ON a.session_id = s.id
                WHERE a.experiment_id = %s
                ORDER BY a.variant_name, s.total_cost""",
                (experiment_id,),
            ).fetchall()
            cost_by_variant: dict[str, list[float]] = {}
            for r in cost_rows:
                cost_by_variant.setdefault(r["variant_name"], []).append(r["total_cost"])

            variants: dict[str, dict[str, Any]] = {}
            for row in rows:
                vname = row["variant_name"]
                rated = row["success_count"] + row["failure_count"] + row["partial_count"]
                success_rate = row["success_count"] / rated if rated > 0 else 0.0

                costs = cost_by_variant.get(vname, [])
                median_cost = _median(costs)
                p95_cost = _percentile(costs, 95)

                variants[vname] = {
                    "session_count": row["session_count"],
                    "avg_cost": row["avg_cost"],
                    "total_cost": row["total_cost"],
                    "median_cost": median_cost,
                    "p95_cost": p95_cost,
                    "avg_tokens": row["avg_tokens"],
                    "avg_prompt_tokens": row["avg_prompt_tokens"],
                    "avg_completion_tokens": row["avg_completion_tokens"],
                    "avg_latency_ms": latency_map.get(vname, 0.0),
                    "avg_call_count": row["avg_call_count"],
                    "success_count": row["success_count"],
                    "failure_count": row["failure_count"],
                    "partial_count": row["partial_count"],
                    "unrated_count": row["unrated_count"],
                    "success_rate": success_rate,
                    "avg_score": row["avg_score"],
                }

            return {"experiment_id": experiment_id, "variants": variants}

    # --- Organization methods ---

    def save_organization(self, org: Organization) -> None:
        with self._pool.connection() as conn:
            compliance_json = ""
            if org.compliance_profile:
                compliance_json = org.compliance_profile.model_dump_json()
            conn.execute(
                """INSERT INTO organizations
                   (id, name, status, created_at, budget, total_cost, total_tokens,
                    pii_rules_json, metadata, compliance_profile_json)
                   VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                   ON CONFLICT (id) DO UPDATE SET
                    name = EXCLUDED.name,
                    status = EXCLUDED.status,
                    created_at = EXCLUDED.created_at,
                    budget = EXCLUDED.budget,
                    total_cost = EXCLUDED.total_cost,
                    total_tokens = EXCLUDED.total_tokens,
                    pii_rules_json = EXCLUDED.pii_rules_json,
                    metadata = EXCLUDED.metadata,
                    compliance_profile_json = EXCLUDED.compliance_profile_json""",
                (
                    org.id,
                    org.name,
                    org.status.value,
                    org.created_at.isoformat(),
                    org.budget,
                    org.total_cost,
                    org.total_tokens,
                    json.dumps([r.model_dump() for r in org.pii_rules]),
                    json.dumps(org.metadata),
                    compliance_json,
                ),
            )
            conn.commit()

    def get_organization(self, org_id: str) -> Organization | None:
        with self._pool.connection() as conn:
            row = conn.execute("SELECT * FROM organizations WHERE id = %s", (org_id,)).fetchone()
            if not row:
                return None
            return self._row_to_organization(row)

    def list_organizations(self) -> list[Organization]:
        with self._pool.connection() as conn:
            rows = conn.execute("SELECT * FROM organizations ORDER BY created_at DESC").fetchall()
            return [self._row_to_organization(r) for r in rows]

    # --- Team methods ---

    def save_team(self, team: Team) -> None:
        with self._pool.connection() as conn:
            compliance_json = ""
            if team.compliance_profile:
                compliance_json = team.compliance_profile.model_dump_json()
            conn.execute(
                """INSERT INTO teams
                   (id, org_id, name, status, created_at, budget, total_cost,
                    total_tokens, metadata, compliance_profile_json,
                    rate_limit_tps, rate_limit_priority,
                    rate_limit_max_queue, rate_limit_queue_timeout)
                   VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                   ON CONFLICT (id) DO UPDATE SET
                    org_id = EXCLUDED.org_id,
                    name = EXCLUDED.name,
                    status = EXCLUDED.status,
                    created_at = EXCLUDED.created_at,
                    budget = EXCLUDED.budget,
                    total_cost = EXCLUDED.total_cost,
                    total_tokens = EXCLUDED.total_tokens,
                    metadata = EXCLUDED.metadata,
                    compliance_profile_json = EXCLUDED.compliance_profile_json,
                    rate_limit_tps = EXCLUDED.rate_limit_tps,
                    rate_limit_priority = EXCLUDED.rate_limit_priority,
                    rate_limit_max_queue = EXCLUDED.rate_limit_max_queue,
                    rate_limit_queue_timeout = EXCLUDED.rate_limit_queue_timeout""",
                (
                    team.id,
                    team.org_id,
                    team.name,
                    team.status.value,
                    team.created_at.isoformat(),
                    team.budget,
                    team.total_cost,
                    team.total_tokens,
                    json.dumps(team.metadata),
                    compliance_json,
                    team.rate_limit_tps,
                    team.rate_limit_priority,
                    team.rate_limit_max_queue,
                    team.rate_limit_queue_timeout,
                ),
            )
            conn.commit()

    def get_team(self, team_id: str) -> Team | None:
        with self._pool.connection() as conn:
            row = conn.execute("SELECT * FROM teams WHERE id = %s", (team_id,)).fetchone()
            if not row:
                return None
            return self._row_to_team(row)

    def list_teams(self, org_id: str | None = None) -> list[Team]:
        with self._pool.connection() as conn:
            if org_id is not None:
                rows = conn.execute(
                    "SELECT * FROM teams WHERE org_id = %s ORDER BY created_at DESC",
                    (org_id,),
                ).fetchall()
            else:
                rows = conn.execute("SELECT * FROM teams ORDER BY created_at DESC").fetchall()
            return [self._row_to_team(r) for r in rows]

    # --- Hierarchy queries ---

    def get_org_stats(self, org_id: str) -> dict[str, Any]:
        with self._pool.connection() as conn:
            row = conn.execute(
                """SELECT
                    COUNT(*) as total_sessions,
                    SUM(CASE WHEN status = 'active' THEN 1 ELSE 0 END) as active_sessions,
                    COALESCE(SUM(total_cost), 0) as total_cost,
                    COALESCE(SUM(total_tokens), 0) as total_tokens,
                    COALESCE(SUM(call_count), 0) as total_calls,
                    COALESCE(SUM(cache_hits), 0) as total_cache_hits,
                    COALESCE(SUM(pii_detections), 0) as total_pii_detections
                FROM sessions WHERE org_id = %s""",
                (org_id,),
            ).fetchone()
            result = dict(row) if row else {}
            result["org_id"] = org_id
            return result

    def get_team_stats(self, team_id: str) -> dict[str, Any]:
        with self._pool.connection() as conn:
            row = conn.execute(
                """SELECT
                    COUNT(*) as total_sessions,
                    SUM(CASE WHEN status = 'active' THEN 1 ELSE 0 END) as active_sessions,
                    COALESCE(SUM(total_cost), 0) as total_cost,
                    COALESCE(SUM(total_tokens), 0) as total_tokens,
                    COALESCE(SUM(call_count), 0) as total_calls,
                    COALESCE(SUM(cache_hits), 0) as total_cache_hits,
                    COALESCE(SUM(pii_detections), 0) as total_pii_detections
                FROM sessions WHERE team_id = %s""",
                (team_id,),
            ).fetchone()
            result = dict(row) if row else {}
            result["team_id"] = team_id
            return result

    # --- Job methods ---

    def save_job(self, job: Job) -> None:
        with self._pool.connection() as conn:
            conn.execute(
                """INSERT INTO jobs
                   (id, session_id, org_id, team_id, status, provider, model,
                    messages_json, request_kwargs_json, webhook_url, webhook_secret,
                    result_json, error, error_code, created_at, started_at, completed_at,
                    retry_count, max_retries, ttl_seconds, metadata_json,
                    webhook_status, webhook_attempts, webhook_last_error)
                   VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                           %s, %s, %s, %s, %s, %s, %s, %s, %s)
                   ON CONFLICT (id) DO UPDATE SET
                    session_id = EXCLUDED.session_id,
                    org_id = EXCLUDED.org_id,
                    team_id = EXCLUDED.team_id,
                    status = EXCLUDED.status,
                    provider = EXCLUDED.provider,
                    model = EXCLUDED.model,
                    messages_json = EXCLUDED.messages_json,
                    request_kwargs_json = EXCLUDED.request_kwargs_json,
                    webhook_url = EXCLUDED.webhook_url,
                    webhook_secret = EXCLUDED.webhook_secret,
                    result_json = EXCLUDED.result_json,
                    error = EXCLUDED.error,
                    error_code = EXCLUDED.error_code,
                    created_at = EXCLUDED.created_at,
                    started_at = EXCLUDED.started_at,
                    completed_at = EXCLUDED.completed_at,
                    retry_count = EXCLUDED.retry_count,
                    max_retries = EXCLUDED.max_retries,
                    ttl_seconds = EXCLUDED.ttl_seconds,
                    metadata_json = EXCLUDED.metadata_json,
                    webhook_status = EXCLUDED.webhook_status,
                    webhook_attempts = EXCLUDED.webhook_attempts,
                    webhook_last_error = EXCLUDED.webhook_last_error""",
                (
                    job.id,
                    job.session_id,
                    job.org_id,
                    job.team_id,
                    job.status.value,
                    job.provider,
                    job.model,
                    json.dumps(job.messages),
                    json.dumps(job.request_kwargs),
                    job.webhook_url,
                    job.webhook_secret,
                    json.dumps(job.result) if job.result is not None else None,
                    job.error,
                    job.error_code,
                    job.created_at.isoformat(),
                    job.started_at.isoformat() if job.started_at else None,
                    job.completed_at.isoformat() if job.completed_at else None,
                    job.retry_count,
                    job.max_retries,
                    job.ttl_seconds,
                    json.dumps(job.metadata),
                    job.webhook_status,
                    job.webhook_attempts,
                    job.webhook_last_error,
                ),
            )
            conn.commit()

    def get_job(self, job_id: str) -> Job | None:
        with self._pool.connection() as conn:
            row = conn.execute("SELECT * FROM jobs WHERE id = %s", (job_id,)).fetchone()
            if not row:
                return None
            return self._row_to_job(row)

    def list_jobs(
        self,
        status: str | None = None,
        session_id: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[Job]:
        with self._pool.connection() as conn:
            query = "SELECT * FROM jobs"
            conditions: list[str] = []
            params: list[Any] = []
            if status:
                conditions.append("status = %s")
                params.append(status)
            if session_id:
                conditions.append("session_id = %s")
                params.append(session_id)
            if conditions:
                query += " WHERE " + " AND ".join(conditions)
            query += " ORDER BY created_at DESC LIMIT %s OFFSET %s"
            params.extend([limit, offset])
            rows = conn.execute(query, params).fetchall()
            return [self._row_to_job(r) for r in rows]

    def delete_job(self, job_id: str) -> bool:
        with self._pool.connection() as conn:
            cursor = conn.execute("DELETE FROM jobs WHERE id = %s", (job_id,))
            conn.commit()
            return cast(int, cursor.rowcount) > 0

    def get_job_stats(self) -> dict[str, Any]:
        with self._pool.connection() as conn:
            rows = conn.execute(
                "SELECT status, COUNT(*) as cnt FROM jobs GROUP BY status"
            ).fetchall()
            by_status = {r["status"]: r["cnt"] for r in rows}
            total = sum(by_status.values())

            row = conn.execute(
                """SELECT AVG(
                    EXTRACT(EPOCH FROM (
                        completed_at::timestamp - started_at::timestamp
                    )) * 1000
                ) as avg_ms
                FROM jobs WHERE started_at IS NOT NULL AND completed_at IS NOT NULL"""
            ).fetchone()
            avg_time = row["avg_ms"] or 0.0 if row else 0.0

            return {
                "total": total,
                "by_status": by_status,
                "avg_processing_time_ms": avg_time,
            }

    def _row_to_job(self, row: dict[str, Any]) -> Job:
        messages_raw = row["messages_json"]
        messages = json.loads(messages_raw) if messages_raw else []

        kwargs_raw = row["request_kwargs_json"]
        request_kwargs = json.loads(kwargs_raw) if kwargs_raw else {}

        result_raw = row["result_json"]
        result = json.loads(result_raw) if result_raw else None

        metadata_raw = row["metadata_json"]
        metadata = json.loads(metadata_raw) if metadata_raw else {}

        return Job(
            id=row["id"],
            session_id=row["session_id"],
            org_id=row["org_id"],
            team_id=row["team_id"],
            status=JobStatus(row["status"]),
            provider=row["provider"],
            model=row["model"],
            messages=messages,
            request_kwargs=request_kwargs,
            webhook_url=row["webhook_url"],
            webhook_secret=row["webhook_secret"],
            result=result,
            error=row["error"],
            error_code=row["error_code"],
            created_at=datetime.fromisoformat(row["created_at"]),
            started_at=datetime.fromisoformat(row["started_at"]) if row["started_at"] else None,
            completed_at=(
                datetime.fromisoformat(row["completed_at"]) if row["completed_at"] else None
            ),
            retry_count=row["retry_count"],
            max_retries=row["max_retries"],
            ttl_seconds=row["ttl_seconds"],
            metadata=metadata,
            webhook_status=row["webhook_status"],
            webhook_attempts=row["webhook_attempts"],
            webhook_last_error=row["webhook_last_error"],
        )

    # --- Purge methods (compliance) ---

    def purge_session(self, session_id: str) -> int:
        """Delete a session and all its events. Returns deleted event count."""
        with self._pool.connection() as conn:
            cursor = conn.execute("DELETE FROM events WHERE session_id = %s", (session_id,))
            deleted = cast(int, cursor.rowcount)
            conn.execute("DELETE FROM session_feedback WHERE session_id = %s", (session_id,))
            conn.execute("DELETE FROM experiment_assignments WHERE session_id = %s", (session_id,))
            conn.execute("DELETE FROM sessions WHERE id = %s", (session_id,))
            conn.commit()
            return deleted

    def purge_user_data(self, user_identifier: str) -> dict[str, int]:
        """Scan and delete all data matching a user identifier."""
        with self._pool.connection() as conn:
            # Find sessions where metadata contains the identifier
            rows = conn.execute(
                "SELECT id FROM sessions WHERE metadata LIKE %s",
                (f"%{user_identifier}%",),
            ).fetchall()
            session_ids = [r["id"] for r in rows]

            events_deleted = 0
            for sid in session_ids:
                cursor = conn.execute("DELETE FROM events WHERE session_id = %s", (sid,))
                events_deleted += cast(int, cursor.rowcount)
                conn.execute("DELETE FROM session_feedback WHERE session_id = %s", (sid,))
                conn.execute("DELETE FROM experiment_assignments WHERE session_id = %s", (sid,))

            sessions_deleted = 0
            for sid in session_ids:
                conn.execute("DELETE FROM sessions WHERE id = %s", (sid,))
                sessions_deleted += 1

            conn.commit()
            return {"sessions": sessions_deleted, "events": events_deleted}

    # --- Virtual key methods (proxy) ---

    def save_virtual_key(self, vk: VirtualKey) -> None:
        """Persist a virtual key."""
        with self._pool.connection() as conn:
            conn.execute(
                """INSERT INTO virtual_keys
                   (id, key_hash, key_preview, team_id, org_id, name,
                    created_at, revoked, scopes_json, metadata,
                    allowed_models_json, budget_limit, budget_spent,
                    rate_limit_tps, rate_limit_max_queue, rate_limit_queue_timeout,
                    agent_ids_json, billing_mode)
                   VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                   ON CONFLICT (id) DO UPDATE SET
                    key_hash = EXCLUDED.key_hash,
                    key_preview = EXCLUDED.key_preview,
                    team_id = EXCLUDED.team_id,
                    org_id = EXCLUDED.org_id,
                    name = EXCLUDED.name,
                    created_at = EXCLUDED.created_at,
                    revoked = EXCLUDED.revoked,
                    scopes_json = EXCLUDED.scopes_json,
                    metadata = EXCLUDED.metadata,
                    allowed_models_json = EXCLUDED.allowed_models_json,
                    budget_limit = EXCLUDED.budget_limit,
                    budget_spent = EXCLUDED.budget_spent,
                    rate_limit_tps = EXCLUDED.rate_limit_tps,
                    rate_limit_max_queue = EXCLUDED.rate_limit_max_queue,
                    rate_limit_queue_timeout = EXCLUDED.rate_limit_queue_timeout,
                    agent_ids_json = EXCLUDED.agent_ids_json,
                    billing_mode = EXCLUDED.billing_mode""",
                (
                    vk.id,
                    vk.key_hash,
                    vk.key_preview,
                    vk.team_id,
                    vk.org_id,
                    vk.name,
                    vk.created_at.isoformat(),
                    1 if vk.revoked else 0,
                    json.dumps(vk.scopes),
                    json.dumps(vk.metadata),
                    json.dumps(vk.allowed_models),
                    vk.budget_limit,
                    vk.budget_spent,
                    vk.rate_limit_tps,
                    vk.rate_limit_max_queue,
                    vk.rate_limit_queue_timeout,
                    json.dumps(vk.agent_ids),
                    vk.billing_mode,
                ),
            )
            conn.commit()

    def get_virtual_key_by_hash(self, key_hash: str) -> VirtualKey | None:
        """Look up a virtual key by its SHA256 hash."""
        with self._pool.connection() as conn:
            row = conn.execute(
                "SELECT * FROM virtual_keys WHERE key_hash = %s", (key_hash,)
            ).fetchone()
            if not row:
                return None
            return self._row_to_virtual_key(row)

    def get_virtual_key(self, key_id: str) -> VirtualKey | None:
        """Look up a virtual key by its ID."""
        with self._pool.connection() as conn:
            row = conn.execute("SELECT * FROM virtual_keys WHERE id = %s", (key_id,)).fetchone()
            if not row:
                return None
            return self._row_to_virtual_key(row)

    def list_virtual_keys(self, team_id: str | None = None) -> list[VirtualKey]:
        """List virtual keys, optionally filtered by team_id."""
        with self._pool.connection() as conn:
            if team_id is not None:
                rows = conn.execute(
                    "SELECT * FROM virtual_keys WHERE team_id = %s ORDER BY created_at DESC",
                    (team_id,),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM virtual_keys ORDER BY created_at DESC"
                ).fetchall()
            return [self._row_to_virtual_key(r) for r in rows]

    def revoke_virtual_key(self, key_id: str) -> bool:
        """Revoke a virtual key. Returns True if it existed."""
        with self._pool.connection() as conn:
            cursor = conn.execute(
                "UPDATE virtual_keys SET revoked = 1 WHERE id = %s AND revoked = 0",
                (key_id,),
            )
            conn.commit()
            return cast(int, cursor.rowcount) > 0

    def _row_to_virtual_key(self, row: dict[str, Any]) -> VirtualKey:
        """Convert a row to a VirtualKey."""
        from stateloom.proxy.virtual_key import VirtualKey

        scopes_raw = row["scopes_json"]
        scopes: list[str] = []
        if scopes_raw:
            try:
                scopes = json.loads(scopes_raw)
            except (json.JSONDecodeError, TypeError):
                pass

        metadata_raw = row["metadata"]
        metadata: dict[str, Any] = {}
        if metadata_raw:
            try:
                metadata = json.loads(metadata_raw)
            except (json.JSONDecodeError, TypeError):
                pass

        allowed_models: list[str] = []
        try:
            am_raw = row["allowed_models_json"]
            if am_raw:
                allowed_models = json.loads(am_raw)
        except (json.JSONDecodeError, TypeError, KeyError):
            pass

        budget_limit: float | None = None
        budget_spent: float = 0.0
        try:
            budget_limit = row["budget_limit"]
            budget_spent = row["budget_spent"] or 0.0
        except (KeyError, TypeError):
            pass

        rate_limit_tps: float | None = None
        rate_limit_max_queue: int = 100
        rate_limit_queue_timeout: float = 30.0
        try:
            rate_limit_tps = row["rate_limit_tps"]
            rate_limit_max_queue = row["rate_limit_max_queue"] or 100
            rate_limit_queue_timeout = row["rate_limit_queue_timeout"] or 30.0
        except (KeyError, TypeError):
            pass

        agent_ids: list[str] = []
        try:
            ai_raw = row["agent_ids_json"]
            if ai_raw:
                agent_ids = json.loads(ai_raw)
        except (json.JSONDecodeError, TypeError, KeyError):
            pass

        billing_mode = "api"
        try:
            billing_mode = row["billing_mode"] or "api"
        except (KeyError, TypeError):
            pass

        return VirtualKey(
            id=row["id"],
            key_hash=row["key_hash"],
            key_preview=row["key_preview"],
            team_id=row["team_id"],
            org_id=row["org_id"],
            name=row["name"],
            created_at=datetime.fromisoformat(row["created_at"]),
            revoked=bool(row["revoked"]),
            scopes=scopes,
            metadata=metadata,
            allowed_models=allowed_models,
            budget_limit=budget_limit,
            budget_spent=budget_spent,
            rate_limit_tps=rate_limit_tps,
            rate_limit_max_queue=int(rate_limit_max_queue),
            rate_limit_queue_timeout=float(rate_limit_queue_timeout),
            agent_ids=agent_ids,
            billing_mode=billing_mode,
        )

    # --- Secret methods ---

    def _get_fernet(self) -> Any | None:
        """Lazy-init Fernet cipher for encrypting persisted secrets."""
        if hasattr(self, "_fernet"):
            return self._fernet
        self._fernet: Any | None = None
        try:
            from cryptography.fernet import Fernet

            raw = os.environ.get("STATELOOM_SECRET_KEY", "")
            if not raw:
                key_path = Path.home() / ".stateloom" / "secret.key"
                if key_path.exists():
                    raw = key_path.read_text().strip()
                else:
                    raw = Fernet.generate_key().decode()
                    key_path.parent.mkdir(parents=True, exist_ok=True)
                    key_path.write_text(raw)
            dk = hashlib.sha256(raw.encode()).digest()
            self._fernet = Fernet(base64.urlsafe_b64encode(dk))
        except ImportError:
            logger.warning("cryptography not installed — secrets stored as base64 (not encrypted)")
        return self._fernet

    def save_secret(self, key: str, value: str) -> None:
        fernet = self._get_fernet()
        if fernet:
            encoded = fernet.encrypt(value.encode()).decode()
        else:
            encoded = base64.b64encode(value.encode()).decode()
        with self._pool.connection() as conn:
            conn.execute(
                """INSERT INTO secrets (key, value, updated_at) VALUES (%s, %s, %s)
                   ON CONFLICT (key) DO UPDATE SET
                    value = EXCLUDED.value,
                    updated_at = EXCLUDED.updated_at""",
                (key, encoded, datetime.now(timezone.utc).isoformat()),
            )
            conn.commit()

    def get_secret(self, key: str) -> str:
        with self._pool.connection() as conn:
            row = conn.execute("SELECT value FROM secrets WHERE key = %s", (key,)).fetchone()
            if not row:
                return ""
        fernet = self._get_fernet()
        try:
            if fernet:
                return cast(str, fernet.decrypt(row["value"].encode()).decode())
            return base64.b64decode(row["value"]).decode()
        except Exception:
            return ""

    def list_secrets(self) -> list[str]:
        """Return all secret key names (not values)."""
        with self._pool.connection() as conn:
            rows = conn.execute("SELECT key FROM secrets").fetchall()
            return [row["key"] for row in rows]

    def delete_secret(self, key: str) -> None:
        with self._pool.connection() as conn:
            conn.execute("DELETE FROM secrets WHERE key = %s", (key,))
            conn.commit()

    # --- Admin lock methods ---

    def save_admin_lock(
        self, setting: str, value: str, locked_by: str = "", reason: str = ""
    ) -> None:
        with self._pool.connection() as conn:
            conn.execute(
                """INSERT INTO admin_locks
                   (setting, value, locked_by, reason, locked_at)
                   VALUES (%s, %s, %s, %s, %s)
                   ON CONFLICT (setting) DO UPDATE SET
                    value = EXCLUDED.value,
                    locked_by = EXCLUDED.locked_by,
                    reason = EXCLUDED.reason,
                    locked_at = EXCLUDED.locked_at""",
                (setting, value, locked_by, reason, datetime.now(timezone.utc).isoformat()),
            )
            conn.commit()

    def get_admin_lock(self, setting: str) -> dict[str, Any] | None:
        with self._pool.connection() as conn:
            row = conn.execute(
                "SELECT setting, value, locked_by, reason, locked_at "
                "FROM admin_locks WHERE setting = %s",
                (setting,),
            ).fetchone()
            if not row:
                return None
            return {
                "setting": row["setting"],
                "value": row["value"],
                "locked_by": row["locked_by"],
                "reason": row["reason"],
                "locked_at": row["locked_at"],
            }

    def list_admin_locks(self) -> list[dict[str, Any]]:
        with self._pool.connection() as conn:
            rows = conn.execute(
                "SELECT setting, value, locked_by, reason, locked_at "
                "FROM admin_locks ORDER BY setting"
            ).fetchall()
            return [
                {
                    "setting": r["setting"],
                    "value": r["value"],
                    "locked_by": r["locked_by"],
                    "reason": r["reason"],
                    "locked_at": r["locked_at"],
                }
                for r in rows
            ]

    def delete_admin_lock(self, setting: str) -> None:
        with self._pool.connection() as conn:
            conn.execute("DELETE FROM admin_locks WHERE setting = %s", (setting,))
            conn.commit()

    # --- Agent methods ---

    def save_agent(self, agent: Agent) -> None:
        """Persist or update an agent."""
        with self._pool.connection() as conn:
            conn.execute(
                """INSERT INTO agents
                   (id, slug, team_id, org_id, name, description, status,
                    active_version_id, metadata, created_at, updated_at)
                   VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                   ON CONFLICT (id) DO UPDATE SET
                    slug = EXCLUDED.slug,
                    team_id = EXCLUDED.team_id,
                    org_id = EXCLUDED.org_id,
                    name = EXCLUDED.name,
                    description = EXCLUDED.description,
                    status = EXCLUDED.status,
                    active_version_id = EXCLUDED.active_version_id,
                    metadata = EXCLUDED.metadata,
                    created_at = EXCLUDED.created_at,
                    updated_at = EXCLUDED.updated_at""",
                (
                    agent.id,
                    agent.slug,
                    agent.team_id,
                    agent.org_id,
                    agent.name,
                    agent.description,
                    agent.status.value,
                    agent.active_version_id,
                    json.dumps(agent.metadata),
                    agent.created_at.isoformat(),
                    agent.updated_at.isoformat(),
                ),
            )
            conn.commit()

    def get_agent(self, agent_id: str) -> Agent | None:
        """Get an agent by ID."""
        with self._pool.connection() as conn:
            row = conn.execute("SELECT * FROM agents WHERE id = %s", (agent_id,)).fetchone()
            if not row:
                return None
            return self._row_to_agent(row)

    def get_agent_by_slug(self, slug: str, team_id: str) -> Agent | None:
        """Get an agent by slug within a team."""
        with self._pool.connection() as conn:
            row = conn.execute(
                "SELECT * FROM agents WHERE slug = %s AND team_id = %s",
                (slug, team_id),
            ).fetchone()
            if not row:
                return None
            return self._row_to_agent(row)

    def list_agents(
        self,
        team_id: str | None = None,
        org_id: str | None = None,
        status: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[Agent]:
        """List agents, optionally filtered."""
        with self._pool.connection() as conn:
            query = "SELECT * FROM agents"
            conditions: list[str] = []
            params: list[Any] = []
            if team_id is not None:
                conditions.append("team_id = %s")
                params.append(team_id)
            if org_id is not None:
                conditions.append("org_id = %s")
                params.append(org_id)
            if status is not None:
                conditions.append("status = %s")
                params.append(status)
            if conditions:
                query += " WHERE " + " AND ".join(conditions)
            query += " ORDER BY created_at DESC LIMIT %s OFFSET %s"
            params.extend([limit, offset])
            rows = conn.execute(query, params).fetchall()
            return [self._row_to_agent(r) for r in rows]

    def save_agent_version(self, version: AgentVersion) -> None:
        """Persist an agent version."""
        with self._pool.connection() as conn:
            conn.execute(
                """INSERT INTO agent_versions
                   (id, agent_id, version_number, model, system_prompt,
                    request_overrides_json, compliance_profile_json,
                    budget_per_session, metadata, created_at, created_by)
                   VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                   ON CONFLICT (id) DO UPDATE SET
                    agent_id = EXCLUDED.agent_id,
                    version_number = EXCLUDED.version_number,
                    model = EXCLUDED.model,
                    system_prompt = EXCLUDED.system_prompt,
                    request_overrides_json = EXCLUDED.request_overrides_json,
                    compliance_profile_json = EXCLUDED.compliance_profile_json,
                    budget_per_session = EXCLUDED.budget_per_session,
                    metadata = EXCLUDED.metadata,
                    created_at = EXCLUDED.created_at,
                    created_by = EXCLUDED.created_by""",
                (
                    version.id,
                    version.agent_id,
                    version.version_number,
                    version.model,
                    version.system_prompt,
                    json.dumps(version.request_overrides),
                    version.compliance_profile_json,
                    version.budget_per_session,
                    json.dumps(version.metadata),
                    version.created_at.isoformat(),
                    version.created_by,
                ),
            )
            conn.commit()

    def get_agent_version(self, version_id: str) -> AgentVersion | None:
        """Get an agent version by ID."""
        with self._pool.connection() as conn:
            row = conn.execute(
                "SELECT * FROM agent_versions WHERE id = %s", (version_id,)
            ).fetchone()
            if not row:
                return None
            return self._row_to_agent_version(row)

    def list_agent_versions(self, agent_id: str, limit: int = 100) -> list[AgentVersion]:
        """List versions for an agent (newest first)."""
        with self._pool.connection() as conn:
            rows = conn.execute(
                "SELECT * FROM agent_versions WHERE agent_id = %s "
                "ORDER BY version_number DESC LIMIT %s",
                (agent_id, limit),
            ).fetchall()
            return [self._row_to_agent_version(r) for r in rows]

    def get_next_version_number(self, agent_id: str) -> int:
        """Get the next version number for an agent."""
        with self._pool.connection() as conn:
            row = conn.execute(
                "SELECT MAX(version_number) as max_ver FROM agent_versions WHERE agent_id = %s",
                (agent_id,),
            ).fetchone()
            if row and row["max_ver"] is not None:
                return cast(int, row["max_ver"]) + 1
            return 1

    def _row_to_agent(self, row: dict[str, Any]) -> Agent:
        """Convert a row to an Agent."""
        from stateloom.agent.models import Agent
        from stateloom.core.types import AgentStatus

        metadata_raw = row["metadata"]
        metadata: dict[str, Any] = {}
        if metadata_raw:
            try:
                metadata = json.loads(metadata_raw)
            except (json.JSONDecodeError, TypeError):
                pass

        return Agent(
            id=row["id"],
            slug=row["slug"],
            team_id=row["team_id"],
            org_id=row["org_id"],
            name=row["name"],
            description=row["description"],
            status=AgentStatus(row["status"]),
            active_version_id=row["active_version_id"],
            metadata=metadata,
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
        )

    def _row_to_agent_version(self, row: dict[str, Any]) -> AgentVersion:
        """Convert a row to an AgentVersion."""
        from stateloom.agent.models import AgentVersion

        metadata_raw = row["metadata"]
        metadata: dict[str, Any] = {}
        if metadata_raw:
            try:
                metadata = json.loads(metadata_raw)
            except (json.JSONDecodeError, TypeError):
                pass

        overrides_raw = row["request_overrides_json"]
        overrides: dict[str, Any] = {}
        if overrides_raw:
            try:
                overrides = json.loads(overrides_raw)
            except (json.JSONDecodeError, TypeError):
                pass

        return AgentVersion(
            id=row["id"],
            agent_id=row["agent_id"],
            version_number=row["version_number"],
            model=row["model"],
            system_prompt=row["system_prompt"],
            request_overrides=overrides,
            compliance_profile_json=row["compliance_profile_json"],
            budget_per_session=row["budget_per_session"],
            metadata=metadata,
            created_at=datetime.fromisoformat(row["created_at"]),
            created_by=row["created_by"],
        )

    # --- Internal helpers ---

    def _row_to_organization(self, row: dict[str, Any]) -> Organization:
        pii_rules_raw = row["pii_rules_json"]
        pii_rules: list[PIIRule] = []
        if pii_rules_raw:
            try:
                for r in json.loads(pii_rules_raw):
                    pii_rules.append(PIIRule(**r))
            except (json.JSONDecodeError, TypeError):
                logger.warning(
                    "Corrupted pii_rules_json for org %s, using empty list",
                    row["id"],
                )

        metadata_raw = row["metadata"]
        metadata: dict[str, Any] = {}
        if metadata_raw:
            try:
                metadata = json.loads(metadata_raw)
            except (json.JSONDecodeError, TypeError):
                pass

        compliance_profile = None
        try:
            cp_raw = row["compliance_profile_json"]
            if cp_raw:
                compliance_profile = ComplianceProfile.model_validate_json(cp_raw)
        except (IndexError, KeyError):
            pass
        except Exception:
            logger.warning("Corrupted compliance_profile_json for org %s", row["id"])

        return Organization(
            id=row["id"],
            name=row["name"],
            status=OrgStatus(row["status"]),
            created_at=datetime.fromisoformat(row["created_at"]),
            budget=row["budget"],
            total_cost=row["total_cost"],
            total_tokens=row["total_tokens"],
            pii_rules=pii_rules,
            metadata=metadata,
            compliance_profile=compliance_profile,
        )

    def _row_to_team(self, row: dict[str, Any]) -> Team:
        metadata_raw = row["metadata"]
        metadata: dict[str, Any] = {}
        if metadata_raw:
            try:
                metadata = json.loads(metadata_raw)
            except (json.JSONDecodeError, TypeError):
                pass

        compliance_profile = None
        try:
            cp_raw = row.get("compliance_profile_json")
            if cp_raw:
                compliance_profile = ComplianceProfile.model_validate_json(cp_raw)
        except (IndexError, KeyError):
            pass
        except Exception:
            logger.warning("Corrupted compliance_profile_json for team %s", row["id"])

        rate_limit_tps = row.get("rate_limit_tps")
        rate_limit_priority = row.get("rate_limit_priority", 0)
        rate_limit_max_queue = row.get("rate_limit_max_queue", 100)
        rate_limit_queue_timeout = row.get("rate_limit_queue_timeout", 30.0)

        return Team(
            id=row["id"],
            org_id=row["org_id"],
            name=row["name"],
            status=TeamStatus(row["status"]),
            created_at=datetime.fromisoformat(row["created_at"]),
            budget=row["budget"],
            total_cost=row["total_cost"],
            total_tokens=row["total_tokens"],
            metadata=metadata,
            compliance_profile=compliance_profile,
            rate_limit_tps=rate_limit_tps,
            rate_limit_priority=rate_limit_priority or 0,
            rate_limit_max_queue=rate_limit_max_queue or 100,
            rate_limit_queue_timeout=rate_limit_queue_timeout or 30.0,
        )

    def _row_to_session(self, row: dict[str, Any]) -> Session:
        metadata_raw = row["metadata"]
        metadata: dict[str, Any] = {}
        if metadata_raw:
            try:
                metadata = json.loads(metadata_raw)
            except (json.JSONDecodeError, TypeError):
                logger.warning(
                    "Corrupted metadata JSON for session %s, using empty dict",
                    row["id"],
                )

        org_id = row.get("org_id") or ""
        team_id = row.get("team_id") or ""
        parent_session_id = row.get("parent_session_id") or None
        timeout = row.get("timeout")
        idle_timeout = row.get("idle_timeout")

        last_heartbeat = None
        last_heartbeat_raw = row.get("last_heartbeat")
        if last_heartbeat_raw:
            last_heartbeat = datetime.fromisoformat(last_heartbeat_raw)

        estimated_api_cost = row.get("estimated_api_cost") or 0.0

        cost_by_model: dict[str, float] = {}
        raw_cbm = row.get("cost_by_model_json")
        if raw_cbm:
            try:
                cost_by_model = json.loads(raw_cbm)
            except (json.JSONDecodeError, TypeError):
                pass

        tokens_by_model: dict[str, dict[str, int]] = {}
        raw_tbm = row.get("tokens_by_model_json")
        if raw_tbm:
            try:
                tokens_by_model = json.loads(raw_tbm)
            except (json.JSONDecodeError, TypeError):
                pass

        # Phase 2 typed fields — read from column, fallback to metadata for old rows
        s_billing_mode = row.get("s_billing_mode") or ""
        if not s_billing_mode and metadata.get("billing_mode"):
            s_billing_mode = metadata["billing_mode"]
        s_durable = bool(row.get("s_durable"))
        if not s_durable and metadata.get("durable"):
            s_durable = True
        s_agent_id = row.get("s_agent_id") or ""
        if not s_agent_id and metadata.get("agent_id"):
            s_agent_id = metadata["agent_id"]
        s_agent_slug = row.get("s_agent_slug") or ""
        if not s_agent_slug and metadata.get("agent_slug"):
            s_agent_slug = metadata["agent_slug"]
        s_agent_version_id = row.get("s_agent_version_id") or ""
        if not s_agent_version_id and metadata.get("agent_version_id"):
            s_agent_version_id = metadata["agent_version_id"]
        s_agent_version_number = row.get("s_agent_version_number") or 0
        if not s_agent_version_number and metadata.get("agent_version_number"):
            s_agent_version_number = int(metadata["agent_version_number"])
        s_agent_name = row.get("s_agent_name") or ""
        if not s_agent_name and metadata.get("agent_name"):
            s_agent_name = metadata["agent_name"]
        s_transport = row.get("s_transport") or ""
        if not s_transport and metadata.get("transport"):
            s_transport = metadata["transport"]

        return Session(
            id=row["id"],
            name=row["name"],
            org_id=org_id,
            team_id=team_id,
            started_at=datetime.fromisoformat(row["started_at"]),
            ended_at=datetime.fromisoformat(row["ended_at"]) if row["ended_at"] else None,
            status=SessionStatus(row["status"]),
            total_cost=row["total_cost"],
            estimated_api_cost=estimated_api_cost,
            cost_by_model=cost_by_model,
            tokens_by_model=tokens_by_model,
            total_tokens=row["total_tokens"],
            total_prompt_tokens=row["total_prompt_tokens"],
            total_completion_tokens=row["total_completion_tokens"],
            call_count=row["call_count"],
            cache_hits=row["cache_hits"],
            cache_savings=row["cache_savings"],
            pii_detections=row["pii_detections"],
            budget=row["budget"],
            step_counter=row["step_counter"],
            metadata=metadata,
            parent_session_id=parent_session_id,
            timeout=timeout,
            idle_timeout=idle_timeout,
            last_heartbeat=last_heartbeat,
            billing_mode=s_billing_mode,
            durable=s_durable,
            agent_id=s_agent_id,
            agent_slug=s_agent_slug,
            agent_version_id=s_agent_version_id,
            agent_version_number=s_agent_version_number,
            agent_name=s_agent_name,
            transport=s_transport,
        )

    def _row_to_experiment(self, row: dict[str, Any]) -> Experiment:
        return Experiment(
            id=row["id"],
            name=row["name"],
            description=row["description"],
            status=ExperimentStatus(row["status"]),
            strategy=AssignmentStrategy(row["strategy"]),
            variants=[VariantConfig.from_dict(v) for v in json.loads(row["variants_json"])],
            assignment_counts=json.loads(row["assignment_counts_json"]),
            metadata=json.loads(row["metadata"]),
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
        )

    # ── Generic event serialization (same pattern as SQLiteStore) ──────

    # Column alias maps: event model field name → SQL column name.
    # Needed when an event model uses a different field name than the SQL column
    # (e.g. BudgetEnforcementEvent.limit → SQL budget_limit).
    _COL_ALIASES: dict[EventType, dict[str, str]] = {
        EventType.BUDGET_ENFORCEMENT: {
            "limit": "budget_limit",
            "spent": "budget_spent",
            "action": "action_taken",
        },
        EventType.LOOP_DETECTION: {"action": "action_taken"},
        EventType.PII_DETECTION: {"mode": "pii_mode"},
        EventType.KILL_SWITCH: {
            "blocked_provider": "provider",
            "blocked_model": "model",
        },
    }

    # Fields stored in direct SQL columns (not extra_json)
    _DIRECT_COLS = {
        "provider",
        "model",
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
        "cost",
        "latency_ms",
        "is_streaming",
        "request_hash",
        "tool_name",
        "mutates_state",
        "pii_type",
        "pii_mode",
        "pii_field",
        "action_taken",
        "budget_limit",
        "budget_spent",
        "pattern_hash",
        "repeat_count",
        "original_model",
        "saved_cost",
        "cached_response_json",
        "rating",
        "score",
        "comment",
    }

    # Inverse alias map: SQL column name → model field name
    _INV_ALIASES: dict[EventType, dict[str, str]] = {
        EventType.BUDGET_ENFORCEMENT: {
            "budget_limit": "limit",
            "budget_spent": "spent",
            "action_taken": "action",
        },
        EventType.LOOP_DETECTION: {"action_taken": "action"},
        EventType.PII_DETECTION: {"pii_mode": "mode"},
        EventType.KILL_SWITCH: {
            "provider": "blocked_provider",
            "model": "blocked_model",
        },
    }

    def _event_to_row(self, event: Event) -> tuple[Any, ...]:
        d = event.model_dump(mode="python")
        et = event.event_type
        aliases = self._COL_ALIASES.get(et, {})

        # Base fields handled separately
        skip = {"id", "session_id", "step", "event_type", "timestamp", "metadata"}

        col_vals: dict[str, Any] = {}
        shadow: dict[str, Any] = {}

        for field_name, value in d.items():
            if field_name in skip:
                continue
            col_name = aliases.get(field_name, field_name)
            if col_name in self._DIRECT_COLS:
                col_vals[col_name] = value
            elif value not in (None, "", 0, 0.0, False, {}, []):
                shadow[field_name] = value

        # Bool → int for SQL
        if "is_streaming" in col_vals:
            col_vals["is_streaming"] = 1 if col_vals["is_streaming"] else 0
        if "mutates_state" in col_vals:
            col_vals["mutates_state"] = 1 if col_vals["mutates_state"] else 0

        extra_json = json.dumps(shadow, default=str) if shadow else None

        return (
            event.id,
            event.session_id,
            event.step,
            et.value,
            event.timestamp.isoformat(),
            col_vals.get("provider"),
            col_vals.get("model"),
            col_vals.get("prompt_tokens"),
            col_vals.get("completion_tokens"),
            col_vals.get("total_tokens"),
            col_vals.get("cost"),
            col_vals.get("latency_ms"),
            col_vals.get("is_streaming"),
            col_vals.get("request_hash"),
            col_vals.get("tool_name"),
            col_vals.get("mutates_state"),
            col_vals.get("pii_type"),
            col_vals.get("pii_mode"),
            col_vals.get("pii_field"),
            col_vals.get("action_taken"),
            col_vals.get("budget_limit"),
            col_vals.get("budget_spent"),
            col_vals.get("pattern_hash"),
            col_vals.get("repeat_count"),
            col_vals.get("original_model"),
            col_vals.get("saved_cost"),
            col_vals.get("cached_response_json"),
            json.dumps(event.metadata) if event.metadata else None,
            col_vals.get("rating"),
            col_vals.get("score"),
            col_vals.get("comment"),
            extra_json,
        )

    def _row_to_event(self, row: dict[str, Any]) -> Event:
        from pydantic import TypeAdapter

        et = EventType(row["event_type"])
        inv = self._INV_ALIASES.get(et, {})

        # Build kwargs from base fields
        kwargs: dict[str, Any] = {
            "id": row["id"],
            "session_id": row["session_id"],
            "step": row["step"],
            "event_type": et,
            "timestamp": datetime.fromisoformat(row["timestamp"]),
        }
        meta_raw = row["metadata"]
        if meta_raw:
            try:
                kwargs["metadata"] = json.loads(meta_raw)
            except (json.JSONDecodeError, TypeError):
                logger.warning(
                    "Corrupted metadata JSON for event %s, using empty dict",
                    row["id"],
                )
                kwargs["metadata"] = {}
        else:
            kwargs["metadata"] = {}

        # Merge direct columns (apply inverse aliases)
        for col in self._DIRECT_COLS:
            val = row.get(col)
            if val is not None:
                field_name = inv.get(col, col)
                kwargs[field_name] = val

        # Merge extra_json
        shadow_raw = row.get("extra_json")
        if shadow_raw:
            try:
                kwargs.update(json.loads(shadow_raw))
            except (json.JSONDecodeError, TypeError):
                logger.warning("Corrupted extra_json for event %s", row["id"])

        # Bool coercion for int columns
        if "is_streaming" in kwargs:
            kwargs["is_streaming"] = bool(kwargs["is_streaming"])
        if "mutates_state" in kwargs:
            kwargs["mutates_state"] = bool(kwargs["mutates_state"])

        _event_adapter: TypeAdapter[AnyEvent] = TypeAdapter(AnyEvent)
        try:
            return _event_adapter.validate_python(kwargs)
        except Exception:
            # Fallback for unknown event types
            logger.warning(
                "Failed to validate event %s (type=%s), returning base Event", row["id"], et
            )
            return Event(
                **{
                    k: v
                    for k, v in kwargs.items()
                    if k in {"id", "session_id", "step", "event_type", "timestamp", "metadata"}
                }
            )


def _median(values: list[float]) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    n = len(s)
    if n % 2 == 1:
        return s[n // 2]
    return (s[n // 2 - 1] + s[n // 2]) / 2


def _percentile(values: list[float], pct: int) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    idx = int(len(s) * pct / 100)
    return s[min(idx, len(s) - 1)]
