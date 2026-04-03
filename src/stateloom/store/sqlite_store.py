"""SQLite store with WAL mode for StateLoom persistence."""

from __future__ import annotations

import base64
import hashlib
import json
import logging
import os
import sqlite3
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from stateloom.core.config import ComplianceProfile, PIIRule
from stateloom.core.event import AnyEvent, Event
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

if TYPE_CHECKING:
    from stateloom.agent.models import Agent, AgentVersion
    from stateloom.auth.models import User, UserTeamRole
    from stateloom.auth.oidc_models import OIDCProvider
    from stateloom.proxy.virtual_key import VirtualKey

logger = logging.getLogger("stateloom.store.sqlite")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    id TEXT PRIMARY KEY,
    name TEXT,
    started_at TEXT NOT NULL,
    ended_at TEXT,
    status TEXT NOT NULL DEFAULT 'active',
    total_cost REAL NOT NULL DEFAULT 0.0,
    total_tokens INTEGER NOT NULL DEFAULT 0,
    total_prompt_tokens INTEGER NOT NULL DEFAULT 0,
    total_completion_tokens INTEGER NOT NULL DEFAULT 0,
    call_count INTEGER NOT NULL DEFAULT 0,
    cache_hits INTEGER NOT NULL DEFAULT 0,
    cache_savings REAL NOT NULL DEFAULT 0.0,
    pii_detections INTEGER NOT NULL DEFAULT 0,
    budget REAL,
    step_counter INTEGER NOT NULL DEFAULT 0,
    metadata TEXT
);

CREATE TABLE IF NOT EXISTS events (
    id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    step INTEGER NOT NULL,
    event_type TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    provider TEXT,
    model TEXT,
    prompt_tokens INTEGER,
    completion_tokens INTEGER,
    total_tokens INTEGER,
    cost REAL,
    latency_ms REAL,
    is_streaming INTEGER,
    request_hash TEXT,
    tool_name TEXT,
    mutates_state INTEGER,
    pii_type TEXT,
    pii_mode TEXT,
    pii_field TEXT,
    action_taken TEXT,
    budget_limit REAL,
    budget_spent REAL,
    pattern_hash TEXT,
    repeat_count INTEGER,
    original_model TEXT,
    saved_cost REAL,
    cached_response_json TEXT,
    metadata TEXT,
    rating TEXT,
    score REAL,
    comment TEXT,
    FOREIGN KEY (session_id) REFERENCES sessions(id)
);

CREATE INDEX IF NOT EXISTS idx_events_session ON events(session_id);
CREATE INDEX IF NOT EXISTS idx_events_type ON events(event_type);
CREATE INDEX IF NOT EXISTS idx_events_timestamp ON events(timestamp);
CREATE INDEX IF NOT EXISTS idx_events_type_ts ON events(event_type, timestamp);
CREATE INDEX IF NOT EXISTS idx_sessions_status ON sessions(status);
CREATE INDEX IF NOT EXISTS idx_sessions_started ON sessions(started_at);

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
);

CREATE INDEX IF NOT EXISTS idx_experiments_status ON experiments(status);

CREATE TABLE IF NOT EXISTS experiment_assignments (
    session_id TEXT PRIMARY KEY,
    experiment_id TEXT NOT NULL,
    variant_name TEXT NOT NULL,
    variant_config_json TEXT NOT NULL DEFAULT '{}',
    assigned_at TEXT NOT NULL,
    FOREIGN KEY (experiment_id) REFERENCES experiments(id),
    FOREIGN KEY (session_id) REFERENCES sessions(id)
);

CREATE INDEX IF NOT EXISTS idx_assignments_experiment ON experiment_assignments(experiment_id);

CREATE TABLE IF NOT EXISTS session_feedback (
    session_id TEXT PRIMARY KEY,
    rating TEXT NOT NULL,
    score REAL,
    comment TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL,
    FOREIGN KEY (session_id) REFERENCES sessions(id)
);

CREATE TABLE IF NOT EXISTS organizations (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL DEFAULT '',
    status TEXT NOT NULL DEFAULT 'active',
    created_at TEXT NOT NULL,
    budget REAL,
    total_cost REAL NOT NULL DEFAULT 0.0,
    total_tokens INTEGER NOT NULL DEFAULT 0,
    pii_rules_json TEXT NOT NULL DEFAULT '[]',
    metadata TEXT NOT NULL DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS teams (
    id TEXT PRIMARY KEY,
    org_id TEXT NOT NULL DEFAULT '',
    name TEXT NOT NULL DEFAULT '',
    status TEXT NOT NULL DEFAULT 'active',
    created_at TEXT NOT NULL,
    budget REAL,
    total_cost REAL NOT NULL DEFAULT 0.0,
    total_tokens INTEGER NOT NULL DEFAULT 0,
    metadata TEXT NOT NULL DEFAULT '{}',
    rate_limit_tps REAL,
    rate_limit_priority INTEGER NOT NULL DEFAULT 0,
    rate_limit_max_queue INTEGER NOT NULL DEFAULT 100,
    rate_limit_queue_timeout REAL NOT NULL DEFAULT 30.0,
    FOREIGN KEY (org_id) REFERENCES organizations(id)
);
CREATE INDEX IF NOT EXISTS idx_teams_org ON teams(org_id);

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
);
CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status);
CREATE INDEX IF NOT EXISTS idx_jobs_created ON jobs(created_at);

CREATE TABLE IF NOT EXISTS secrets (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

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
    budget_limit REAL,
    budget_spent REAL NOT NULL DEFAULT 0.0
);
CREATE INDEX IF NOT EXISTS idx_vk_hash ON virtual_keys(key_hash);
CREATE INDEX IF NOT EXISTS idx_vk_team ON virtual_keys(team_id);

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
);
CREATE INDEX IF NOT EXISTS idx_agents_slug_team ON agents(slug, team_id);
CREATE INDEX IF NOT EXISTS idx_agents_team ON agents(team_id);

CREATE TABLE IF NOT EXISTS agent_versions (
    id TEXT PRIMARY KEY,
    agent_id TEXT NOT NULL,
    version_number INTEGER NOT NULL,
    model TEXT NOT NULL DEFAULT '',
    system_prompt TEXT NOT NULL DEFAULT '',
    request_overrides_json TEXT NOT NULL DEFAULT '{}',
    compliance_profile_json TEXT NOT NULL DEFAULT '',
    budget_per_session REAL,
    metadata TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL,
    created_by TEXT NOT NULL DEFAULT '',
    FOREIGN KEY (agent_id) REFERENCES agents(id),
    UNIQUE(agent_id, version_number)
);
CREATE INDEX IF NOT EXISTS idx_agent_versions_agent ON agent_versions(agent_id);

CREATE TABLE IF NOT EXISTS admin_locks (
    setting TEXT PRIMARY KEY,
    value TEXT NOT NULL,
    locked_by TEXT NOT NULL DEFAULT '',
    reason TEXT NOT NULL DEFAULT '',
    locked_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS users (
    id TEXT PRIMARY KEY,
    email TEXT NOT NULL,
    display_name TEXT NOT NULL DEFAULT '',
    password_hash TEXT NOT NULL DEFAULT '',
    email_verified INTEGER NOT NULL DEFAULT 0,
    org_id TEXT NOT NULL DEFAULT '',
    org_role TEXT,
    oidc_provider_id TEXT NOT NULL DEFAULT '',
    oidc_subject TEXT NOT NULL DEFAULT '',
    is_active INTEGER NOT NULL DEFAULT 1,
    created_at TEXT NOT NULL,
    last_login TEXT,
    metadata TEXT NOT NULL DEFAULT '{}'
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_users_email ON users(email COLLATE NOCASE);
CREATE INDEX IF NOT EXISTS idx_users_org ON users(org_id);
CREATE UNIQUE INDEX IF NOT EXISTS idx_users_oidc ON users(oidc_provider_id, oidc_subject)
    WHERE oidc_provider_id != '' AND oidc_subject != '';

CREATE TABLE IF NOT EXISTS user_team_roles (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    team_id TEXT NOT NULL,
    role TEXT NOT NULL,
    granted_at TEXT NOT NULL,
    granted_by TEXT NOT NULL DEFAULT '',
    FOREIGN KEY (user_id) REFERENCES users(id),
    FOREIGN KEY (team_id) REFERENCES teams(id),
    UNIQUE(user_id, team_id)
);
CREATE INDEX IF NOT EXISTS idx_utr_user ON user_team_roles(user_id);
CREATE INDEX IF NOT EXISTS idx_utr_team ON user_team_roles(team_id);

CREATE TABLE IF NOT EXISTS refresh_tokens (
    token_hash TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    expires_at TEXT NOT NULL,
    family_id TEXT NOT NULL DEFAULT '',
    revoked INTEGER NOT NULL DEFAULT 0,
    FOREIGN KEY (user_id) REFERENCES users(id)
);
CREATE INDEX IF NOT EXISTS idx_rt_user ON refresh_tokens(user_id);

CREATE TABLE IF NOT EXISTS oidc_providers (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL DEFAULT '',
    issuer_url TEXT NOT NULL,
    client_id TEXT NOT NULL,
    client_secret_encrypted TEXT NOT NULL DEFAULT '',
    scopes TEXT NOT NULL DEFAULT 'openid email profile',
    group_claim TEXT NOT NULL DEFAULT '',
    group_role_mapping_json TEXT NOT NULL DEFAULT '{}',
    is_active INTEGER NOT NULL DEFAULT 1,
    created_at TEXT NOT NULL
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_oidc_issuer ON oidc_providers(issuer_url);
"""


class SQLiteStore:
    """SQLite-backed persistence with WAL mode for concurrent reads.

    Uses thread-local connections to avoid "database is locked" errors
    when accessed concurrently from middleware and dashboard threads.
    """

    def __init__(self, path: str = ".stateloom/data.db", auto_migrate: bool = True) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        self._path = path
        self._lock = threading.Lock()
        self._local = threading.local()
        # Initialize with a primary connection for schema setup
        self._conn = self._create_connection()

        if auto_migrate:
            alembic_ran = False
            try:
                from stateloom.store.migrator import run_migrations

                url = f"sqlite:///{os.path.abspath(path)}"
                alembic_ran = run_migrations(url)
            except Exception:
                logger.debug("Alembic migration failed, falling back to legacy", exc_info=True)

            if not alembic_ran:
                self._conn.executescript(_SCHEMA)
                self._migrate_schema()
                self._conn.commit()
        else:
            self._conn.executescript(_SCHEMA)
            self._migrate_schema()
            self._conn.commit()

    def _create_connection(self) -> sqlite3.Connection:
        """Create a new SQLite connection with standard pragmas."""
        conn = sqlite3.connect(self._path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA busy_timeout=5000")
        return conn

    def _get_conn(self) -> sqlite3.Connection:
        """Get a thread-local connection for read operations."""
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = self._create_connection()
            self._local.conn = conn
        return conn

    def _migrate_schema(self) -> None:
        """Add columns that may be missing on databases created before v0.2.

        Column names and types are hardcoded constants (never user input),
        so the f-string is safe from SQL injection. We use a whitelist to
        be explicit about what's allowed.
        """
        allowed_columns = frozenset(
            {
                "rating",
                "score",
                "comment",
                "extra_json",
                "org_id",
                "team_id",
                "compliance_profile_json",
                "rate_limit_tps",
                "rate_limit_priority",
                "rate_limit_max_queue",
                "rate_limit_queue_timeout",
                "parent_session_id",
                "timeout",
                "idle_timeout",
                "last_heartbeat",
                "allowed_models_json",
                "budget_limit",
                "budget_spent",
                "agent_ids_json",
                "billing_mode",
                "estimated_api_cost",
                "end_user",
                # Phase 2: promoted session metadata fields (on sessions table)
                "s_billing_mode",
                "s_durable",
                "s_agent_id",
                "s_agent_slug",
                "s_agent_version_id",
                "s_agent_version_number",
                "s_agent_name",
                "s_transport",
                # Session accumulator counters
                "guardrail_detections",
                "cost_by_model_json",
                "tokens_by_model_json",
            }
        )
        allowed_types = frozenset({"TEXT", "REAL", "INTEGER"})

        # Events table migrations
        existing_events = {
            row["name"] for row in self._conn.execute("PRAGMA table_info(events)").fetchall()
        }

        # Rename shadow_data_json → extra_json for existing databases
        if "shadow_data_json" in existing_events and "extra_json" not in existing_events:
            self._conn.execute("ALTER TABLE events RENAME COLUMN shadow_data_json TO extra_json")
            existing_events.discard("shadow_data_json")
            existing_events.add("extra_json")

        event_migrations = [
            ("rating", "TEXT"),
            ("score", "REAL"),
            ("comment", "TEXT"),
            ("extra_json", "TEXT"),
            ("request_messages_json", "TEXT"),
        ]
        for col, col_type in event_migrations:
            if col not in allowed_columns or col_type not in allowed_types:
                continue
            if col not in existing_events:
                self._conn.execute(f"ALTER TABLE events ADD COLUMN {col} {col_type}")

        # Sessions table migrations (org_id, team_id)
        existing_sessions = {
            row["name"] for row in self._conn.execute("PRAGMA table_info(sessions)").fetchall()
        }
        session_migrations = [
            ("org_id", "TEXT"),
            ("team_id", "TEXT"),
            ("parent_session_id", "TEXT"),
            ("timeout", "REAL"),
            ("idle_timeout", "REAL"),
            ("last_heartbeat", "TEXT"),
            ("estimated_api_cost", "REAL"),
            ("end_user", "TEXT"),
            ("cost_by_model_json", "TEXT"),
            ("tokens_by_model_json", "TEXT"),
            ("guardrail_detections", "INTEGER"),
            # Phase 2: promoted session metadata fields (prefixed s_ to avoid
            # collision with existing metadata/billing_mode columns on other tables)
            ("s_billing_mode", "TEXT"),
            ("s_durable", "INTEGER"),
            ("s_agent_id", "TEXT"),
            ("s_agent_slug", "TEXT"),
            ("s_agent_version_id", "TEXT"),
            ("s_agent_version_number", "INTEGER"),
            ("s_agent_name", "TEXT"),
            ("s_transport", "TEXT"),
        ]
        for col, col_type in session_migrations:
            if col not in allowed_columns or col_type not in allowed_types:
                continue
            if col not in existing_sessions:
                self._conn.execute(f"ALTER TABLE sessions ADD COLUMN {col} {col_type} DEFAULT ''")

        # Organizations table migrations
        existing_orgs = {
            row["name"] for row in self._conn.execute("PRAGMA table_info(organizations)").fetchall()
        }
        org_migrations = [
            ("compliance_profile_json", "TEXT"),
        ]
        for col, col_type in org_migrations:
            if col not in allowed_columns or col_type not in allowed_types:
                continue
            if col not in existing_orgs:
                self._conn.execute(
                    f"ALTER TABLE organizations ADD COLUMN {col} {col_type} DEFAULT ''"
                )

        # Teams table migrations
        existing_teams = {
            row["name"] for row in self._conn.execute("PRAGMA table_info(teams)").fetchall()
        }
        team_migrations = [
            ("compliance_profile_json", "TEXT"),
            ("rate_limit_tps", "REAL"),
            ("rate_limit_priority", "INTEGER"),
            ("rate_limit_max_queue", "INTEGER"),
            ("rate_limit_queue_timeout", "REAL"),
        ]
        for col, col_type in team_migrations:
            if col not in allowed_columns or col_type not in allowed_types:
                continue
            if col not in existing_teams:
                self._conn.execute(f"ALTER TABLE teams ADD COLUMN {col} {col_type} DEFAULT ''")

        # Virtual keys table migrations (allowed_models, budget)
        existing_vk = {
            row["name"] for row in self._conn.execute("PRAGMA table_info(virtual_keys)").fetchall()
        }
        vk_migrations = [
            ("allowed_models_json", "TEXT"),
            ("budget_limit", "REAL"),
            ("budget_spent", "REAL"),
            ("rate_limit_tps", "REAL"),
            ("rate_limit_max_queue", "INTEGER"),
            ("rate_limit_queue_timeout", "REAL"),
            ("agent_ids_json", "TEXT"),
            ("billing_mode", "TEXT"),
        ]
        for col, col_type in vk_migrations:
            if col not in allowed_columns or col_type not in allowed_types:
                continue
            if col not in existing_vk:
                default = "DEFAULT '[]'" if col == "allowed_models_json" else "DEFAULT 0.0"
                if col == "budget_limit":
                    default = ""
                elif col == "rate_limit_tps":
                    default = ""
                elif col == "rate_limit_max_queue":
                    default = "DEFAULT 100"
                elif col == "rate_limit_queue_timeout":
                    default = "DEFAULT 30.0"
                elif col == "agent_ids_json":
                    default = "DEFAULT '[]'"
                elif col == "billing_mode":
                    default = "DEFAULT 'api'"
                self._conn.execute(
                    f"ALTER TABLE virtual_keys ADD COLUMN {col} {col_type} {default}"
                )

        # Add indexes for org_id and team_id on sessions (idempotent)
        self._conn.execute("CREATE INDEX IF NOT EXISTS idx_sessions_org ON sessions(org_id)")
        self._conn.execute("CREATE INDEX IF NOT EXISTS idx_sessions_team ON sessions(team_id)")
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_sessions_parent ON sessions(parent_session_id)"
        )
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_sessions_end_user"
            " ON sessions(end_user) WHERE end_user != ''"
        )

        # Auth tables (for databases created before this version)
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS users (
                id TEXT PRIMARY KEY,
                email TEXT NOT NULL,
                display_name TEXT NOT NULL DEFAULT '',
                password_hash TEXT NOT NULL DEFAULT '',
                email_verified INTEGER NOT NULL DEFAULT 0,
                org_id TEXT NOT NULL DEFAULT '',
                org_role TEXT,
                oidc_provider_id TEXT NOT NULL DEFAULT '',
                oidc_subject TEXT NOT NULL DEFAULT '',
                is_active INTEGER NOT NULL DEFAULT 1,
                created_at TEXT NOT NULL,
                last_login TEXT,
                metadata TEXT NOT NULL DEFAULT '{}'
            );
            CREATE UNIQUE INDEX IF NOT EXISTS idx_users_email ON users(email COLLATE NOCASE);
            CREATE INDEX IF NOT EXISTS idx_users_org ON users(org_id);

            CREATE TABLE IF NOT EXISTS user_team_roles (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                team_id TEXT NOT NULL,
                role TEXT NOT NULL,
                granted_at TEXT NOT NULL,
                granted_by TEXT NOT NULL DEFAULT '',
                FOREIGN KEY (user_id) REFERENCES users(id),
                FOREIGN KEY (team_id) REFERENCES teams(id),
                UNIQUE(user_id, team_id)
            );
            CREATE INDEX IF NOT EXISTS idx_utr_user ON user_team_roles(user_id);
            CREATE INDEX IF NOT EXISTS idx_utr_team ON user_team_roles(team_id);

            CREATE TABLE IF NOT EXISTS refresh_tokens (
                token_hash TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                expires_at TEXT NOT NULL,
                family_id TEXT NOT NULL DEFAULT '',
                revoked INTEGER NOT NULL DEFAULT 0,
                FOREIGN KEY (user_id) REFERENCES users(id)
            );
            CREATE INDEX IF NOT EXISTS idx_rt_user ON refresh_tokens(user_id);

            CREATE TABLE IF NOT EXISTS oidc_providers (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL DEFAULT '',
                issuer_url TEXT NOT NULL,
                client_id TEXT NOT NULL,
                client_secret_encrypted TEXT NOT NULL DEFAULT '',
                scopes TEXT NOT NULL DEFAULT 'openid email profile',
                group_claim TEXT NOT NULL DEFAULT '',
                group_role_mapping_json TEXT NOT NULL DEFAULT '{}',
                is_active INTEGER NOT NULL DEFAULT 1,
                created_at TEXT NOT NULL
            );
            CREATE UNIQUE INDEX IF NOT EXISTS idx_oidc_issuer ON oidc_providers(issuer_url);
        """)

        # Agents and agent_versions tables (for databases created before this version)
        self._conn.executescript("""
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
            );
            CREATE INDEX IF NOT EXISTS idx_agents_slug_team ON agents(slug, team_id);
            CREATE INDEX IF NOT EXISTS idx_agents_team ON agents(team_id);

            CREATE TABLE IF NOT EXISTS agent_versions (
                id TEXT PRIMARY KEY,
                agent_id TEXT NOT NULL,
                version_number INTEGER NOT NULL,
                model TEXT NOT NULL DEFAULT '',
                system_prompt TEXT NOT NULL DEFAULT '',
                request_overrides_json TEXT NOT NULL DEFAULT '{}',
                compliance_profile_json TEXT NOT NULL DEFAULT '',
                budget_per_session REAL,
                metadata TEXT NOT NULL DEFAULT '{}',
                created_at TEXT NOT NULL,
                created_by TEXT NOT NULL DEFAULT '',
                FOREIGN KEY (agent_id) REFERENCES agents(id),
                UNIQUE(agent_id, version_number)
            );
            CREATE INDEX IF NOT EXISTS idx_agent_versions_agent ON agent_versions(agent_id);
        """)

        # Normalize legacy action_taken values (pre-ActionTaken enum)
        self._conn.execute(
            "UPDATE events SET action_taken = 'redacted' WHERE action_taken = 'redact'"
        )
        self._conn.execute(
            "UPDATE events SET action_taken = 'blocked' WHERE action_taken = 'block'"
        )

    def save_session(self, session: Session) -> None:
        # TODO(race-condition): INSERT OR REPLACE overwrites the entire session row.
        # When concurrent requests share the same session_id (e.g. Claude CLI sends
        # streaming + non-streaming in parallel), each reads accumulators at request
        # start and writes absolute values at request end — the later write can
        # overwrite the earlier one's token/cost updates.  Events are always saved
        # correctly so data is recoverable, but session totals (budget enforcement,
        # dashboard summaries) can be stale.  Fix: switch to incremental UPDATE
        # (SET total_cost = total_cost + delta) and track per-request deltas in
        # Session, or use a read-merge-write inside a single transaction.
        with self._lock:
            self._conn.execute(
                """INSERT OR REPLACE INTO sessions
                   (id, name, started_at, ended_at, status, total_cost, total_tokens,
                    total_prompt_tokens, total_completion_tokens, call_count,
                    cache_hits, cache_savings, pii_detections, budget, step_counter,
                    metadata, org_id, team_id,
                    parent_session_id, timeout, idle_timeout, last_heartbeat,
                    estimated_api_cost, end_user,
                    cost_by_model_json, tokens_by_model_json, guardrail_detections,
                    s_billing_mode, s_durable, s_agent_id, s_agent_slug,
                    s_agent_version_id, s_agent_version_number, s_agent_name,
                    s_transport)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                           ?, ?, ?, ?, ?, ?, ?, ?, ?,
                           ?, ?, ?, ?, ?, ?, ?, ?)""",
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
                    session.end_user,
                    json.dumps(session.cost_by_model) if session.cost_by_model else None,
                    json.dumps(session.tokens_by_model) if session.tokens_by_model else None,
                    session.guardrail_detections,
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
            self._conn.commit()

    def get_session(self, session_id: str) -> Session | None:
        conn = self._get_conn()
        row = conn.execute("SELECT * FROM sessions WHERE id = ?", (session_id,)).fetchone()
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
        conn = self._get_conn()
        query = "SELECT * FROM sessions"
        conditions: list[str] = []
        params: list[Any] = []
        if status:
            conditions.append("status = ?")
            params.append(status)
        if org_id is not None:
            conditions.append("org_id = ?")
            params.append(org_id)
        if team_id is not None:
            conditions.append("team_id = ?")
            params.append(team_id)
        if end_user is not None:
            conditions.append("end_user = ?")
            params.append(end_user)
        if conditions:
            query += " WHERE " + " AND ".join(conditions)
        query += " ORDER BY started_at DESC LIMIT ? OFFSET ?"
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
        conn = self._get_conn()
        query = "SELECT COUNT(*) FROM sessions"
        conditions: list[str] = []
        params: list[Any] = []
        if status:
            conditions.append("status = ?")
            params.append(status)
        if org_id is not None:
            conditions.append("org_id = ?")
            params.append(org_id)
        if team_id is not None:
            conditions.append("team_id = ?")
            params.append(team_id)
        if end_user is not None:
            conditions.append("end_user = ?")
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
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT * FROM sessions WHERE parent_session_id = ? ORDER BY started_at DESC LIMIT ?",
            (parent_session_id, limit),
        ).fetchall()
        return [self._row_to_session(r) for r in rows]

    def save_session_with_events(self, session: Session, events: list[Event]) -> None:
        with self._lock:
            for event in events:
                self._conn.execute(
                    """INSERT OR REPLACE INTO events
                       (id, session_id, step, event_type, timestamp, provider, model,
                        prompt_tokens, completion_tokens, total_tokens, cost, latency_ms,
                        is_streaming, request_hash, tool_name, mutates_state,
                        pii_type, pii_mode, pii_field, action_taken,
                        budget_limit, budget_spent, pattern_hash, repeat_count,
                        original_model, saved_cost, cached_response_json,
                        request_messages_json, metadata,
                        rating, score, comment, extra_json)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                               ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                               ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    self._event_to_row(event),
                )
            self._conn.execute(
                """INSERT OR REPLACE INTO sessions
                   (id, name, started_at, ended_at, status, total_cost, total_tokens,
                    total_prompt_tokens, total_completion_tokens, call_count,
                    cache_hits, cache_savings, pii_detections, budget, step_counter,
                    metadata, org_id, team_id,
                    parent_session_id, timeout, idle_timeout, last_heartbeat,
                    estimated_api_cost, end_user,
                    cost_by_model_json, tokens_by_model_json, guardrail_detections,
                    s_billing_mode, s_durable, s_agent_id, s_agent_slug,
                    s_agent_version_id, s_agent_version_number, s_agent_name,
                    s_transport)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                           ?, ?, ?, ?, ?, ?, ?, ?, ?,
                           ?, ?, ?, ?, ?, ?, ?, ?)""",
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
                    session.end_user,
                    json.dumps(session.cost_by_model) if session.cost_by_model else None,
                    json.dumps(session.tokens_by_model) if session.tokens_by_model else None,
                    session.guardrail_detections,
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
            self._conn.commit()

    def save_event(self, event: Event) -> None:
        with self._lock:
            self._conn.execute(
                """INSERT OR REPLACE INTO events
                   (id, session_id, step, event_type, timestamp, provider, model,
                    prompt_tokens, completion_tokens, total_tokens, cost, latency_ms,
                    is_streaming, request_hash, tool_name, mutates_state,
                    pii_type, pii_mode, pii_field, action_taken,
                    budget_limit, budget_spent, pattern_hash, repeat_count,
                    original_model, saved_cost, cached_response_json,
                    request_messages_json, metadata,
                    rating, score, comment, extra_json)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                           ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                           ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                self._event_to_row(event),
            )
            self._conn.commit()

    def get_session_events(
        self,
        session_id: str,
        event_type: str | None = None,
        limit: int = 1000,
        offset: int = 0,
        desc: bool = False,
    ) -> list[Event]:
        conn = self._get_conn()
        # Exclude the full request_messages_json payload from bulk queries —
        # use get_event_messages() for lazy-loading.  Include a lightweight
        # boolean so _event_details can set has_request_messages.
        cols = (
            "id, session_id, step, event_type, timestamp, provider, model, "
            "prompt_tokens, completion_tokens, total_tokens, cost, latency_ms, "
            "is_streaming, request_hash, tool_name, mutates_state, "
            "pii_type, pii_mode, pii_field, action_taken, "
            "budget_limit, budget_spent, pattern_hash, repeat_count, "
            "original_model, saved_cost, cached_response_json, metadata, "
            "rating, score, comment, extra_json, "
            "(CASE WHEN request_messages_json IS NOT NULL THEN '1' ELSE NULL END)"
            " AS request_messages_json"
        )
        if session_id:
            query = f"SELECT {cols} FROM events WHERE session_id = ?"
            params: list[Any] = [session_id]
        else:
            query = f"SELECT {cols} FROM events WHERE 1=1"
            params = []
        if event_type:
            query += " AND event_type = ?"
            params.append(event_type)
        order = "DESC" if desc else "ASC"
        query += f" ORDER BY timestamp {order} LIMIT ? OFFSET ?"
        params.extend([limit, offset])
        rows = conn.execute(query, params).fetchall()
        return [self._row_to_event(r) for r in rows]

    def count_events(
        self,
        session_id: str = "",
        event_type: str | None = None,
    ) -> int:
        conn = self._get_conn()
        if session_id:
            query = "SELECT COUNT(*) FROM events WHERE session_id = ?"
            params: list[Any] = [session_id]
        else:
            query = "SELECT COUNT(*) FROM events WHERE 1=1"
            params = []
        if event_type:
            query += " AND event_type = ?"
            params.append(event_type)
        row = conn.execute(query, params).fetchone()
        return row[0] if row else 0

    def get_pii_stats(self) -> dict[str, Any]:
        conn = self._get_conn()
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
        conn = self._get_conn()
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
                COALESCE(SUM(guardrail_detections), 0) as total_guardrail_detections,
                COALESCE(SUM(estimated_api_cost), 0) as total_estimated_api_cost
            FROM sessions"""
        ).fetchone()
        return dict(row) if row else {}

    def get_call_counts(self) -> dict[str, int]:
        conn = self._get_conn()
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
        conn = self._get_conn()
        rows = conn.execute(
            """SELECT COALESCE(model, 'unknown') as model,
                      COALESCE(SUM(cost), 0) as total_cost
            FROM events
            WHERE event_type = 'llm_call'
            GROUP BY model"""
        ).fetchall()
        return {r["model"]: r["total_cost"] for r in rows}

    # -- Observability aggregation (push computation to SQL) ----------------

    def aggregate_timeseries(
        self,
        start_iso: str,
        bucket_seconds: int,
        org_id: str = "",
        team_id: str = "",
    ) -> list[dict[str, Any]]:
        """Aggregate LLM call events into time buckets using SQL.

        Returns a list of dicts with keys: bucket_start, requests, cost,
        prompt_tokens, completion_tokens.
        """
        conn = self._get_conn()
        # Use SQLite's strftime to compute bucket start from the ISO timestamp.
        # cast((julianday(timestamp) - julianday(start)) * 86400 / bucket_seconds) as integer
        # gives the bucket index; multiply back to get the bucket start offset.
        query = """
            SELECT
                CAST((julianday(timestamp) - julianday(?)) * 86400 / ? AS INTEGER) AS bucket_idx,
                COUNT(*) AS requests,
                COALESCE(SUM(cost), 0) AS cost,
                COALESCE(SUM(prompt_tokens), 0) AS prompt_tokens,
                COALESCE(SUM(completion_tokens), 0) AS completion_tokens
            FROM events
            WHERE event_type = 'llm_call' AND timestamp >= ?
        """
        params: list[Any] = [start_iso, bucket_seconds, start_iso]
        if org_id or team_id:
            query += " AND session_id IN (SELECT id FROM sessions WHERE 1=1"
            if org_id:
                query += " AND org_id = ?"
                params.append(org_id)
            if team_id:
                query += " AND team_id = ?"
                params.append(team_id)
            query += ")"
        query += " GROUP BY bucket_idx ORDER BY bucket_idx"
        rows = conn.execute(query, params).fetchall()
        return [dict(r) for r in rows]

    def aggregate_latency(
        self,
        start_iso: str,
        org_id: str = "",
        team_id: str = "",
    ) -> list[float]:
        """Return sorted latency values (ms) for LLM calls since start_iso.

        Returns only non-zero latencies, already sorted ascending.
        """
        conn = self._get_conn()
        query = """
            SELECT latency_ms
            FROM events
            WHERE event_type = 'llm_call' AND timestamp >= ? AND latency_ms > 0
        """
        params: list[Any] = [start_iso]
        if org_id or team_id:
            query += " AND session_id IN (SELECT id FROM sessions WHERE 1=1"
            if org_id:
                query += " AND org_id = ?"
                params.append(org_id)
            if team_id:
                query += " AND team_id = ?"
                params.append(team_id)
            query += ")"
        query += " ORDER BY latency_ms ASC"
        rows = conn.execute(query, params).fetchall()
        return [r["latency_ms"] for r in rows]

    def aggregate_breakdown(
        self,
        start_iso: str,
        org_id: str = "",
        team_id: str = "",
    ) -> dict[str, Any]:
        """Aggregate model/provider breakdown and cache stats via SQL.

        Returns {"by_model": [...], "by_provider": [...], "cache_hits": int}.
        """
        conn = self._get_conn()

        tenant_filter = ""
        params: list[Any] = [start_iso]
        if org_id or team_id:
            tenant_filter = " AND session_id IN (SELECT id FROM sessions WHERE 1=1"
            if org_id:
                tenant_filter += " AND org_id = ?"
                params.append(org_id)
            if team_id:
                tenant_filter += " AND team_id = ?"
                params.append(team_id)
            tenant_filter += ")"

        # By model
        model_rows = conn.execute(
            f"""SELECT COALESCE(model, 'unknown') AS model,
                       COUNT(*) AS requests,
                       COALESCE(SUM(cost), 0) AS cost,
                       COALESCE(SUM(total_tokens), 0) AS tokens
                FROM events
                WHERE event_type = 'llm_call' AND timestamp >= ?{tenant_filter}
                GROUP BY model""",
            params,
        ).fetchall()

        # By provider
        provider_rows = conn.execute(
            f"""SELECT COALESCE(provider, 'unknown') AS provider,
                       COUNT(*) AS requests,
                       COALESCE(SUM(cost), 0) AS cost
                FROM events
                WHERE event_type = 'llm_call' AND timestamp >= ?{tenant_filter}
                GROUP BY provider""",
            params,
        ).fetchall()

        # Cache hits
        cache_params: list[Any] = [start_iso]
        cache_tenant = ""
        if org_id or team_id:
            cache_tenant = tenant_filter
            cache_params = list(params)
        cache_row = conn.execute(
            f"""SELECT COUNT(*) AS hits
                FROM events
                WHERE event_type = 'cache_hit' AND timestamp >= ?{cache_tenant}""",
            cache_params,
        ).fetchone()

        return {
            "by_model": [dict(r) for r in model_rows],
            "by_provider": [dict(r) for r in provider_rows],
            "cache_hits": cache_row["hits"] if cache_row else 0,
        }

    def cleanup(self, retention_days: int = 30) -> int:
        cutoff = (datetime.now(timezone.utc) - timedelta(days=retention_days)).isoformat()
        with self._lock:
            cursor = self._conn.execute("DELETE FROM events WHERE timestamp < ?", (cutoff,))
            deleted = cursor.rowcount
            self._conn.execute(
                "DELETE FROM sessions WHERE ended_at IS NOT NULL AND ended_at < ?",
                (cutoff,),
            )
            self._conn.commit()
            return deleted

    def cleanup_durable_cache(self, session_id: str = "") -> int:
        """Clear cached_response_json blobs for completed durable sessions.

        Args:
            session_id: If given, only clean this session. Otherwise clean all
                completed durable sessions.

        Returns:
            Number of events updated.
        """
        with self._lock:
            if session_id:
                cursor = self._conn.execute(
                    "UPDATE events SET cached_response_json = NULL "
                    "WHERE session_id = ? AND cached_response_json IS NOT NULL",
                    (session_id,),
                )
            else:
                cursor = self._conn.execute(
                    "UPDATE events SET cached_response_json = NULL "
                    "WHERE cached_response_json IS NOT NULL "
                    "AND session_id IN ("
                    "  SELECT id FROM sessions "
                    "  WHERE status IN ('completed', 'error', 'budget_exceeded', 'loop_killed')"
                    ")",
                )
            updated = cursor.rowcount
            self._conn.commit()
            return updated

    def get_event_messages(self, event_id: str) -> str | None:
        """Lazy-load request_messages_json for a single event."""
        conn = self._get_conn()
        row = conn.execute(
            "SELECT request_messages_json FROM events WHERE id = ?",
            (event_id,),
        ).fetchone()
        if row is None:
            return None
        val: str | None = row["request_messages_json"]
        return val

    def cleanup_request_messages(self, retention_hours: int = 24) -> int:
        """Null out request_messages_json older than retention_hours.

        Returns:
            Number of events updated.
        """
        cutoff = (datetime.now(timezone.utc) - timedelta(hours=retention_hours)).isoformat()
        with self._lock:
            cursor = self._conn.execute(
                "UPDATE events SET request_messages_json = NULL "
                "WHERE request_messages_json IS NOT NULL AND timestamp < ?",
                (cutoff,),
            )
            updated = cursor.rowcount
            self._conn.commit()
            return updated

    def close(self) -> None:
        self._conn.close()

    # --- Experiment methods ---

    def save_experiment(self, experiment: Experiment) -> None:
        with self._lock:
            self._conn.execute(
                """INSERT OR REPLACE INTO experiments
                   (id, name, description, status, strategy, variants_json,
                    assignment_counts_json, metadata, created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
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
            self._conn.commit()

    def get_experiment(self, experiment_id: str) -> Experiment | None:
        conn = self._get_conn()
        row = conn.execute("SELECT * FROM experiments WHERE id = ?", (experiment_id,)).fetchone()
        if not row:
            return None
        return self._row_to_experiment(row)

    def list_experiments(self, status: str | None = None) -> list[Experiment]:
        conn = self._get_conn()
        query = "SELECT * FROM experiments"
        params: list[Any] = []
        if status:
            query += " WHERE status = ?"
            params.append(status)
        query += " ORDER BY created_at DESC"
        rows = conn.execute(query, params).fetchall()
        return [self._row_to_experiment(r) for r in rows]

    def save_assignment(self, assignment: ExperimentAssignment) -> None:
        with self._lock:
            self._conn.execute(
                """INSERT OR REPLACE INTO experiment_assignments
                   (session_id, experiment_id, variant_name, variant_config_json, assigned_at)
                   VALUES (?, ?, ?, ?, ?)""",
                (
                    assignment.session_id,
                    assignment.experiment_id,
                    assignment.variant_name,
                    json.dumps(assignment.variant_config),
                    assignment.assigned_at.isoformat(),
                ),
            )
            self._conn.commit()

    def get_assignment(self, session_id: str) -> ExperimentAssignment | None:
        conn = self._get_conn()
        row = conn.execute(
            "SELECT * FROM experiment_assignments WHERE session_id = ?",
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
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT * FROM experiment_assignments WHERE experiment_id = ? ORDER BY assigned_at",
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
        with self._lock:
            self._conn.execute(
                """INSERT OR REPLACE INTO session_feedback
                   (session_id, rating, score, comment, created_at)
                   VALUES (?, ?, ?, ?, ?)""",
                (
                    feedback.session_id,
                    feedback.rating,
                    feedback.score,
                    feedback.comment,
                    feedback.created_at.isoformat(),
                ),
            )
            self._conn.commit()

    def get_feedback(self, session_id: str) -> SessionFeedback | None:
        conn = self._get_conn()
        row = conn.execute(
            "SELECT * FROM session_feedback WHERE session_id = ?",
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
        conn = self._get_conn()
        if experiment_id:
            rows = conn.execute(
                """SELECT f.* FROM session_feedback f
                   JOIN experiment_assignments a ON f.session_id = a.session_id
                   WHERE a.experiment_id = ?
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
        conn = self._get_conn()
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
            WHERE a.experiment_id = ?
            GROUP BY a.variant_name""",
            (experiment_id,),
        ).fetchall()

        # Compute latency metrics from events
        latency_rows = conn.execute(
            """SELECT a.variant_name, AVG(e.latency_ms) as avg_latency_ms
            FROM experiment_assignments a
            JOIN events e ON a.session_id = e.session_id AND e.event_type = 'llm_call'
            WHERE a.experiment_id = ?
            GROUP BY a.variant_name""",
            (experiment_id,),
        ).fetchall()
        latency_map = {r["variant_name"]: r["avg_latency_ms"] or 0.0 for r in latency_rows}

        # Compute median and p95 cost
        cost_rows = conn.execute(
            """SELECT a.variant_name, s.total_cost
            FROM experiment_assignments a
            JOIN sessions s ON a.session_id = s.id
            WHERE a.experiment_id = ?
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
        with self._lock:
            compliance_json = ""
            if org.compliance_profile:
                compliance_json = org.compliance_profile.model_dump_json()
            self._conn.execute(
                """INSERT OR REPLACE INTO organizations
                   (id, name, status, created_at, budget, total_cost, total_tokens,
                    pii_rules_json, metadata, compliance_profile_json)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
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
            self._conn.commit()

    def get_organization(self, org_id: str) -> Organization | None:
        conn = self._get_conn()
        row = conn.execute("SELECT * FROM organizations WHERE id = ?", (org_id,)).fetchone()
        if not row:
            return None
        return self._row_to_organization(row)

    def list_organizations(self) -> list[Organization]:
        conn = self._get_conn()
        rows = conn.execute("SELECT * FROM organizations ORDER BY created_at DESC").fetchall()
        return [self._row_to_organization(r) for r in rows]

    # --- Team methods ---

    def save_team(self, team: Team) -> None:
        with self._lock:
            compliance_json = ""
            if team.compliance_profile:
                compliance_json = team.compliance_profile.model_dump_json()
            self._conn.execute(
                """INSERT OR REPLACE INTO teams
                   (id, org_id, name, status, created_at, budget, total_cost,
                    total_tokens, metadata, compliance_profile_json,
                    rate_limit_tps, rate_limit_priority,
                    rate_limit_max_queue, rate_limit_queue_timeout)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
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
            self._conn.commit()

    def get_team(self, team_id: str) -> Team | None:
        conn = self._get_conn()
        row = conn.execute("SELECT * FROM teams WHERE id = ?", (team_id,)).fetchone()
        if not row:
            return None
        return self._row_to_team(row)

    def list_teams(self, org_id: str | None = None) -> list[Team]:
        conn = self._get_conn()
        if org_id is not None:
            rows = conn.execute(
                "SELECT * FROM teams WHERE org_id = ? ORDER BY created_at DESC",
                (org_id,),
            ).fetchall()
        else:
            rows = conn.execute("SELECT * FROM teams ORDER BY created_at DESC").fetchall()
        return [self._row_to_team(r) for r in rows]

    # --- Hierarchy queries ---

    def get_org_stats(self, org_id: str) -> dict[str, Any]:
        conn = self._get_conn()
        row = conn.execute(
            """SELECT
                COUNT(*) as total_sessions,
                SUM(CASE WHEN status = 'active' THEN 1 ELSE 0 END) as active_sessions,
                COALESCE(SUM(total_cost), 0) as total_cost,
                COALESCE(SUM(total_tokens), 0) as total_tokens,
                COALESCE(SUM(call_count), 0) as total_calls,
                COALESCE(SUM(cache_hits), 0) as total_cache_hits,
                COALESCE(SUM(pii_detections), 0) as total_pii_detections
            FROM sessions WHERE org_id = ?""",
            (org_id,),
        ).fetchone()
        result = dict(row) if row else {}
        result["org_id"] = org_id
        return result

    def get_team_stats(self, team_id: str) -> dict[str, Any]:
        conn = self._get_conn()
        row = conn.execute(
            """SELECT
                COUNT(*) as total_sessions,
                SUM(CASE WHEN status = 'active' THEN 1 ELSE 0 END) as active_sessions,
                COALESCE(SUM(total_cost), 0) as total_cost,
                COALESCE(SUM(total_tokens), 0) as total_tokens,
                COALESCE(SUM(call_count), 0) as total_calls,
                COALESCE(SUM(cache_hits), 0) as total_cache_hits,
                COALESCE(SUM(pii_detections), 0) as total_pii_detections
            FROM sessions WHERE team_id = ?""",
            (team_id,),
        ).fetchone()
        result = dict(row) if row else {}
        result["team_id"] = team_id
        return result

    # --- Job methods ---

    def save_job(self, job: Job) -> None:
        with self._lock:
            self._conn.execute(
                """INSERT OR REPLACE INTO jobs
                   (id, session_id, org_id, team_id, status, provider, model,
                    messages_json, request_kwargs_json, webhook_url, webhook_secret,
                    result_json, error, error_code, created_at, started_at, completed_at,
                    retry_count, max_retries, ttl_seconds, metadata_json,
                    webhook_status, webhook_attempts, webhook_last_error)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                           ?, ?, ?, ?, ?, ?, ?)""",
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
            self._conn.commit()

    def get_job(self, job_id: str) -> Job | None:
        conn = self._get_conn()
        row = conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
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
        conn = self._get_conn()
        query = "SELECT * FROM jobs"
        conditions: list[str] = []
        params: list[Any] = []
        if status:
            conditions.append("status = ?")
            params.append(status)
        if session_id:
            conditions.append("session_id = ?")
            params.append(session_id)
        if conditions:
            query += " WHERE " + " AND ".join(conditions)
        query += " ORDER BY created_at DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])
        rows = conn.execute(query, params).fetchall()
        return [self._row_to_job(r) for r in rows]

    def delete_job(self, job_id: str) -> bool:
        with self._lock:
            cursor = self._conn.execute("DELETE FROM jobs WHERE id = ?", (job_id,))
            self._conn.commit()
            return cursor.rowcount > 0

    def get_job_stats(self) -> dict[str, Any]:
        conn = self._get_conn()
        rows = conn.execute("SELECT status, COUNT(*) as cnt FROM jobs GROUP BY status").fetchall()
        by_status = {r["status"]: r["cnt"] for r in rows}
        total = sum(by_status.values())

        row = conn.execute(
            """SELECT AVG(
                (julianday(completed_at) - julianday(started_at)) * 86400000
            ) as avg_ms
            FROM jobs WHERE started_at IS NOT NULL AND completed_at IS NOT NULL"""
        ).fetchone()
        avg_time = row["avg_ms"] or 0.0 if row else 0.0

        return {
            "total": total,
            "by_status": by_status,
            "avg_processing_time_ms": avg_time,
        }

    def _row_to_job(self, row: sqlite3.Row) -> Job:
        d: dict[str, Any] = {
            "id": row["id"],
            "session_id": row["session_id"],
            "org_id": row["org_id"],
            "team_id": row["team_id"],
            "status": JobStatus(row["status"]),
            "provider": row["provider"],
            "model": row["model"],
            "messages": json.loads(row["messages_json"]) if row["messages_json"] else [],
            "request_kwargs": (
                json.loads(row["request_kwargs_json"]) if row["request_kwargs_json"] else {}
            ),
            "webhook_url": row["webhook_url"],
            "webhook_secret": row["webhook_secret"],
            "result": json.loads(row["result_json"]) if row["result_json"] else None,
            "error": row["error"],
            "error_code": row["error_code"],
            "created_at": datetime.fromisoformat(row["created_at"]),
            "started_at": (
                datetime.fromisoformat(row["started_at"]) if row["started_at"] else None
            ),
            "completed_at": (
                datetime.fromisoformat(row["completed_at"]) if row["completed_at"] else None
            ),
            "retry_count": row["retry_count"],
            "max_retries": row["max_retries"],
            "ttl_seconds": row["ttl_seconds"],
            "metadata": json.loads(row["metadata_json"]) if row["metadata_json"] else {},
            "webhook_status": row["webhook_status"],
            "webhook_attempts": row["webhook_attempts"],
            "webhook_last_error": row["webhook_last_error"],
        }
        return Job.model_validate(d)

    # --- Purge methods (compliance) ---

    def purge_session(self, session_id: str) -> int:
        """Delete a session and all its events. Returns deleted event count."""
        with self._lock:
            cursor = self._conn.execute("DELETE FROM events WHERE session_id = ?", (session_id,))
            deleted = cursor.rowcount
            self._conn.execute("DELETE FROM session_feedback WHERE session_id = ?", (session_id,))
            self._conn.execute(
                "DELETE FROM experiment_assignments WHERE session_id = ?", (session_id,)
            )
            self._conn.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
            self._conn.commit()
            return deleted

    def purge_user_data(self, user_identifier: str) -> dict[str, int]:
        """Scan and delete all data matching a user identifier."""
        with self._lock:
            # Find sessions where metadata contains the identifier
            rows = self._conn.execute(
                "SELECT id FROM sessions WHERE metadata LIKE ?",
                (f"%{user_identifier}%",),
            ).fetchall()
            session_ids = [r["id"] for r in rows]

            events_deleted = 0
            for sid in session_ids:
                cursor = self._conn.execute("DELETE FROM events WHERE session_id = ?", (sid,))
                events_deleted += cursor.rowcount
                self._conn.execute("DELETE FROM session_feedback WHERE session_id = ?", (sid,))
                self._conn.execute(
                    "DELETE FROM experiment_assignments WHERE session_id = ?", (sid,)
                )

            sessions_deleted = 0
            for sid in session_ids:
                self._conn.execute("DELETE FROM sessions WHERE id = ?", (sid,))
                sessions_deleted += 1

            self._conn.commit()
            return {"sessions": sessions_deleted, "events": events_deleted}

    # --- Virtual key methods (proxy) ---

    def save_virtual_key(self, vk: VirtualKey) -> None:
        """Persist a virtual key."""
        with self._lock:
            self._conn.execute(
                """INSERT OR REPLACE INTO virtual_keys
                   (id, key_hash, key_preview, team_id, org_id, name,
                    created_at, revoked, scopes_json, metadata,
                    allowed_models_json, budget_limit, budget_spent,
                    rate_limit_tps, rate_limit_max_queue, rate_limit_queue_timeout,
                    agent_ids_json, billing_mode)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
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
            self._conn.commit()

    def get_virtual_key_by_hash(self, key_hash: str) -> VirtualKey | None:
        """Look up a virtual key by its SHA256 hash."""
        conn = self._get_conn()
        row = conn.execute("SELECT * FROM virtual_keys WHERE key_hash = ?", (key_hash,)).fetchone()
        if not row:
            return None
        return self._row_to_virtual_key(row)

    def get_virtual_key(self, key_id: str) -> VirtualKey | None:
        """Look up a virtual key by its ID."""
        conn = self._get_conn()
        row = conn.execute("SELECT * FROM virtual_keys WHERE id = ?", (key_id,)).fetchone()
        if not row:
            return None
        return self._row_to_virtual_key(row)

    def list_virtual_keys(self, team_id: str | None = None) -> list[VirtualKey]:
        """List virtual keys, optionally filtered by team_id."""
        conn = self._get_conn()
        if team_id is not None:
            rows = conn.execute(
                "SELECT * FROM virtual_keys WHERE team_id = ? ORDER BY created_at DESC",
                (team_id,),
            ).fetchall()
        else:
            rows = conn.execute("SELECT * FROM virtual_keys ORDER BY created_at DESC").fetchall()
        return [self._row_to_virtual_key(r) for r in rows]

    def revoke_virtual_key(self, key_id: str) -> bool:
        """Revoke a virtual key. Returns True if it existed."""
        with self._lock:
            cursor = self._conn.execute(
                "UPDATE virtual_keys SET revoked = 1 WHERE id = ? AND revoked = 0",
                (key_id,),
            )
            self._conn.commit()
            return cursor.rowcount > 0

    def _row_to_virtual_key(self, row: sqlite3.Row) -> VirtualKey:
        """Convert a row to a VirtualKey."""
        from stateloom.proxy.virtual_key import VirtualKey

        def _safe_json(raw: Any, default: Any) -> Any:
            if not raw:
                return default
            try:
                return json.loads(raw)
            except (json.JSONDecodeError, TypeError):
                return default

        def _safe_col(col: str, default: Any = None) -> Any:
            try:
                return row[col]
            except (KeyError, IndexError):
                return default

        d: dict[str, Any] = {
            "id": row["id"],
            "key_hash": row["key_hash"],
            "key_preview": row["key_preview"],
            "team_id": row["team_id"],
            "org_id": row["org_id"],
            "name": row["name"],
            "created_at": datetime.fromisoformat(row["created_at"]),
            "revoked": bool(row["revoked"]),
            "scopes": _safe_json(row["scopes_json"], []),
            "metadata": _safe_json(row["metadata"], {}),
            "allowed_models": _safe_json(_safe_col("allowed_models_json"), []),
            "budget_limit": _safe_col("budget_limit"),
            "budget_spent": _safe_col("budget_spent", 0.0) or 0.0,
            "rate_limit_tps": _safe_col("rate_limit_tps"),
            "rate_limit_max_queue": int(_safe_col("rate_limit_max_queue", 100) or 100),
            "rate_limit_queue_timeout": float(_safe_col("rate_limit_queue_timeout", 30.0) or 30.0),
            "agent_ids": _safe_json(_safe_col("agent_ids_json"), []),
            "billing_mode": _safe_col("billing_mode", "api") or "api",
        }
        return VirtualKey.model_validate(d)

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
                key_path = Path(self._path).parent / "secret.key"
                if key_path.exists():
                    raw = key_path.read_text().strip()
                else:
                    raw = Fernet.generate_key().decode()
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
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO secrets (key, value, updated_at) VALUES (?, ?, ?)",
                (key, encoded, datetime.now(timezone.utc).isoformat()),
            )
            self._conn.commit()

    def get_secret(self, key: str) -> str:
        conn = self._get_conn()
        row = conn.execute("SELECT value FROM secrets WHERE key = ?", (key,)).fetchone()
        if not row:
            return ""
        fernet = self._get_fernet()
        try:
            if fernet:
                return cast(str, fernet.decrypt(row[0].encode()).decode())
            return base64.b64decode(row[0]).decode()
        except Exception:
            return ""

    def list_secrets(self) -> list[str]:
        """Return all secret key names (not values)."""
        conn = self._get_conn()
        return [row[0] for row in conn.execute("SELECT key FROM secrets").fetchall()]

    def delete_secret(self, key: str) -> None:
        with self._lock:
            self._conn.execute("DELETE FROM secrets WHERE key = ?", (key,))
            self._conn.commit()

    # --- Admin lock methods ---

    def save_admin_lock(
        self, setting: str, value: str, locked_by: str = "", reason: str = ""
    ) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO admin_locks"
                " (setting, value, locked_by, reason, locked_at)"
                " VALUES (?, ?, ?, ?, ?)",
                (setting, value, locked_by, reason, datetime.now(timezone.utc).isoformat()),
            )
            self._conn.commit()

    def get_admin_lock(self, setting: str) -> dict[str, Any] | None:
        conn = self._get_conn()
        row = conn.execute(
            "SELECT setting, value, locked_by, reason, locked_at"
            " FROM admin_locks WHERE setting = ?",
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
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT setting, value, locked_by, reason, locked_at FROM admin_locks ORDER BY setting"
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
        with self._lock:
            self._conn.execute("DELETE FROM admin_locks WHERE setting = ?", (setting,))
            self._conn.commit()

    # --- Agent methods ---

    def save_agent(self, agent: Agent) -> None:
        """Persist or update an agent."""
        with self._lock:
            self._conn.execute(
                """INSERT OR REPLACE INTO agents
                   (id, slug, team_id, org_id, name, description, status,
                    active_version_id, metadata, created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
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
            self._conn.commit()

    def get_agent(self, agent_id: str) -> Agent | None:
        """Get an agent by ID."""
        conn = self._get_conn()
        row = conn.execute("SELECT * FROM agents WHERE id = ?", (agent_id,)).fetchone()
        if not row:
            return None
        return self._row_to_agent(row)

    def get_agent_by_slug(self, slug: str, team_id: str) -> Agent | None:
        """Get an agent by slug within a team."""
        conn = self._get_conn()
        row = conn.execute(
            "SELECT * FROM agents WHERE slug = ? AND team_id = ?",
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
        conn = self._get_conn()
        query = "SELECT * FROM agents"
        conditions: list[str] = []
        params: list[Any] = []
        if team_id is not None:
            conditions.append("team_id = ?")
            params.append(team_id)
        if org_id is not None:
            conditions.append("org_id = ?")
            params.append(org_id)
        if status is not None:
            conditions.append("status = ?")
            params.append(status)
        if conditions:
            query += " WHERE " + " AND ".join(conditions)
        query += " ORDER BY created_at DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])
        rows = conn.execute(query, params).fetchall()
        return [self._row_to_agent(r) for r in rows]

    def save_agent_version(self, version: AgentVersion) -> None:
        """Persist an agent version."""
        with self._lock:
            self._conn.execute(
                """INSERT OR REPLACE INTO agent_versions
                   (id, agent_id, version_number, model, system_prompt,
                    request_overrides_json, compliance_profile_json,
                    budget_per_session, metadata, created_at, created_by)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
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
            self._conn.commit()

    def get_agent_version(self, version_id: str) -> AgentVersion | None:
        """Get an agent version by ID."""
        conn = self._get_conn()
        row = conn.execute("SELECT * FROM agent_versions WHERE id = ?", (version_id,)).fetchone()
        if not row:
            return None
        return self._row_to_agent_version(row)

    def list_agent_versions(self, agent_id: str, limit: int = 100) -> list[AgentVersion]:
        """List versions for an agent (newest first)."""
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT * FROM agent_versions WHERE agent_id = ? ORDER BY version_number DESC LIMIT ?",
            (agent_id, limit),
        ).fetchall()
        return [self._row_to_agent_version(r) for r in rows]

    def get_next_version_number(self, agent_id: str) -> int:
        """Get the next version number for an agent."""
        conn = self._get_conn()
        row = conn.execute(
            "SELECT MAX(version_number) as max_ver FROM agent_versions WHERE agent_id = ?",
            (agent_id,),
        ).fetchone()
        if row and row["max_ver"] is not None:
            return cast(int, row["max_ver"]) + 1
        return 1

    def _row_to_agent(self, row: sqlite3.Row) -> Agent:
        """Convert a row to an Agent."""
        from stateloom.agent.models import Agent
        from stateloom.core.types import AgentStatus

        def _safe_json(raw: Any, default: Any) -> Any:
            if not raw:
                return default
            try:
                return json.loads(raw)
            except (json.JSONDecodeError, TypeError):
                return default

        d: dict[str, Any] = {
            "id": row["id"],
            "slug": row["slug"],
            "team_id": row["team_id"],
            "org_id": row["org_id"],
            "name": row["name"],
            "description": row["description"],
            "status": AgentStatus(row["status"]),
            "active_version_id": row["active_version_id"],
            "metadata": _safe_json(row["metadata"], {}),
            "created_at": datetime.fromisoformat(row["created_at"]),
            "updated_at": datetime.fromisoformat(row["updated_at"]),
        }
        return Agent.model_validate(d)

    def _row_to_agent_version(self, row: sqlite3.Row) -> AgentVersion:
        """Convert a row to an AgentVersion."""
        from stateloom.agent.models import AgentVersion

        def _safe_json(raw: Any, default: Any) -> Any:
            if not raw:
                return default
            try:
                return json.loads(raw)
            except (json.JSONDecodeError, TypeError):
                return default

        d: dict[str, Any] = {
            "id": row["id"],
            "agent_id": row["agent_id"],
            "version_number": row["version_number"],
            "model": row["model"],
            "system_prompt": row["system_prompt"],
            "request_overrides": _safe_json(row["request_overrides_json"], {}),
            "compliance_profile_json": row["compliance_profile_json"],
            "budget_per_session": row["budget_per_session"],
            "metadata": _safe_json(row["metadata"], {}),
            "created_at": datetime.fromisoformat(row["created_at"]),
            "created_by": row["created_by"],
        }
        return AgentVersion.model_validate(d)

    # --- Internal helpers ---

    def _row_to_organization(self, row: sqlite3.Row) -> Organization:
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

    def _row_to_team(self, row: sqlite3.Row) -> Team:
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
            logger.warning("Corrupted compliance_profile_json for team %s", row["id"])

        # Rate limit fields with fallback for old rows
        row_keys = row.keys()
        rate_limit_tps = row["rate_limit_tps"] if "rate_limit_tps" in row_keys else None
        rate_limit_priority = row["rate_limit_priority"] if "rate_limit_priority" in row_keys else 0
        rate_limit_max_queue = (
            row["rate_limit_max_queue"] if "rate_limit_max_queue" in row_keys else 100
        )
        rate_limit_queue_timeout = (
            row["rate_limit_queue_timeout"] if "rate_limit_queue_timeout" in row_keys else 30.0
        )

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

    def _row_to_session(self, row: sqlite3.Row) -> Session:
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
        # Read org_id/team_id with fallback for old rows
        try:
            org_id = row["org_id"] or ""
        except (IndexError, KeyError):
            org_id = ""
        try:
            team_id = row["team_id"] or ""
        except (IndexError, KeyError):
            team_id = ""

        # Read parent-child and timeout fields with fallback for old rows
        try:
            parent_session_id = row["parent_session_id"] or None
        except (IndexError, KeyError):
            parent_session_id = None
        try:
            timeout = row["timeout"]
        except (IndexError, KeyError):
            timeout = None
        try:
            idle_timeout = row["idle_timeout"]
        except (IndexError, KeyError):
            idle_timeout = None
        try:
            last_heartbeat_raw = row["last_heartbeat"]
            last_heartbeat = (
                datetime.fromisoformat(last_heartbeat_raw) if last_heartbeat_raw else None
            )
        except (IndexError, KeyError):
            last_heartbeat = None

        try:
            estimated_api_cost = row["estimated_api_cost"] or 0.0
        except (IndexError, KeyError):
            estimated_api_cost = 0.0

        try:
            end_user = row["end_user"] or ""
        except (IndexError, KeyError):
            end_user = ""

        cost_by_model: dict[str, float] = {}
        try:
            raw = row["cost_by_model_json"]
            if raw:
                cost_by_model = json.loads(raw)
        except (IndexError, KeyError, json.JSONDecodeError, TypeError):
            pass

        tokens_by_model: dict[str, dict[str, int]] = {}
        try:
            raw = row["tokens_by_model_json"]
            if raw:
                tokens_by_model = json.loads(raw)
        except (IndexError, KeyError, json.JSONDecodeError, TypeError):
            pass

        # Read promoted metadata fields with fallback for old rows
        row_keys = row.keys()

        def _str_col(name: str) -> str:
            try:
                return row[name] or "" if name in row_keys else ""
            except (IndexError, KeyError):
                return ""

        def _int_col(name: str) -> int:
            try:
                return int(row[name] or 0) if name in row_keys else 0
            except (IndexError, KeyError, ValueError, TypeError):
                return 0

        s_billing_mode = _str_col("s_billing_mode")
        s_durable = bool(_int_col("s_durable"))
        s_agent_id = _str_col("s_agent_id")
        s_agent_slug = _str_col("s_agent_slug")
        s_agent_version_id = _str_col("s_agent_version_id")
        s_agent_version_number = _int_col("s_agent_version_number")
        s_agent_name = _str_col("s_agent_name")
        s_transport = _str_col("s_transport")

        # For old rows where typed columns are empty, fall back to metadata dict
        if not s_billing_mode and metadata.get("billing_mode"):
            s_billing_mode = metadata["billing_mode"]
        if not s_durable and metadata.get("durable"):
            s_durable = True
        if not s_agent_id and metadata.get("agent_id"):
            s_agent_id = metadata["agent_id"]
        if not s_agent_slug and metadata.get("agent_slug"):
            s_agent_slug = metadata["agent_slug"]
        if not s_agent_version_id and metadata.get("agent_version_id"):
            s_agent_version_id = metadata["agent_version_id"]
        if not s_agent_version_number and metadata.get("agent_version_number"):
            s_agent_version_number = int(metadata["agent_version_number"])
        if not s_agent_name and metadata.get("agent_name"):
            s_agent_name = metadata["agent_name"]
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
            guardrail_detections=row["guardrail_detections"]
            if "guardrail_detections" in row_keys
            else 0,
            budget=row["budget"],
            step_counter=row["step_counter"],
            metadata=metadata,
            parent_session_id=parent_session_id,
            timeout=timeout,
            idle_timeout=idle_timeout,
            last_heartbeat=last_heartbeat,
            end_user=end_user,
            billing_mode=s_billing_mode,
            durable=s_durable,
            agent_id=s_agent_id,
            agent_slug=s_agent_slug,
            agent_version_id=s_agent_version_id,
            agent_version_number=s_agent_version_number,
            agent_name=s_agent_name,
            transport=s_transport,
        )

    def _row_to_experiment(self, row: sqlite3.Row) -> Experiment:
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

    # --- User methods (auth) ---

    def save_user(self, user: User) -> None:
        with self._lock:
            self._conn.execute(
                """INSERT OR REPLACE INTO users
                   (id, email, display_name, password_hash, email_verified,
                    org_id, org_role, oidc_provider_id, oidc_subject,
                    is_active, created_at, last_login, metadata)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    user.id,
                    user.email,
                    user.display_name,
                    user.password_hash,
                    1 if user.email_verified else 0,
                    user.org_id,
                    user.org_role.value if user.org_role else None,
                    user.oidc_provider_id,
                    user.oidc_subject,
                    1 if user.is_active else 0,
                    user.created_at.isoformat(),
                    user.last_login.isoformat() if user.last_login else None,
                    json.dumps(user.metadata),
                ),
            )
            self._conn.commit()

    def get_user(self, user_id: str) -> User | None:
        conn = self._get_conn()
        row = conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
        if not row:
            return None
        return self._row_to_user(row)

    def get_user_by_email(self, email: str) -> User | None:
        conn = self._get_conn()
        row = conn.execute(
            "SELECT * FROM users WHERE email = ? COLLATE NOCASE", (email,)
        ).fetchone()
        if not row:
            return None
        return self._row_to_user(row)

    def get_user_by_oidc(self, provider_id: str, subject: str) -> User | None:
        conn = self._get_conn()
        row = conn.execute(
            "SELECT * FROM users WHERE oidc_provider_id = ? AND oidc_subject = ?",
            (provider_id, subject),
        ).fetchone()
        if not row:
            return None
        return self._row_to_user(row)

    def list_users(
        self,
        org_id: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[User]:
        conn = self._get_conn()
        query = "SELECT * FROM users"
        params: list[Any] = []
        if org_id is not None:
            query += " WHERE org_id = ?"
            params.append(org_id)
        query += " ORDER BY created_at DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])
        rows = conn.execute(query, params).fetchall()
        return [self._row_to_user(r) for r in rows]

    def delete_user(self, user_id: str) -> bool:
        with self._lock:
            cursor = self._conn.execute(
                "UPDATE users SET is_active = 0 WHERE id = ? AND is_active = 1",
                (user_id,),
            )
            self._conn.commit()
            return cursor.rowcount > 0

    def _row_to_user(self, row: sqlite3.Row) -> User:
        from stateloom.auth.models import User
        from stateloom.core.types import Role

        def _safe_json(raw: Any, default: Any) -> Any:
            if not raw:
                return default
            try:
                return json.loads(raw)
            except (json.JSONDecodeError, TypeError):
                return default

        org_role = None
        if row["org_role"]:
            try:
                org_role = Role(row["org_role"])
            except ValueError:
                pass

        d: dict[str, Any] = {
            "id": row["id"],
            "email": row["email"],
            "display_name": row["display_name"],
            "password_hash": row["password_hash"],
            "email_verified": bool(row["email_verified"]),
            "org_id": row["org_id"],
            "org_role": org_role,
            "oidc_provider_id": row["oidc_provider_id"],
            "oidc_subject": row["oidc_subject"],
            "is_active": bool(row["is_active"]),
            "created_at": datetime.fromisoformat(row["created_at"]),
            "last_login": (
                datetime.fromisoformat(row["last_login"]) if row["last_login"] else None
            ),
            "metadata": _safe_json(row["metadata"], {}),
        }
        return User.model_validate(d)

    # --- UserTeamRole methods (auth) ---

    def save_user_team_role(self, role: UserTeamRole) -> None:
        with self._lock:
            self._conn.execute(
                """INSERT OR REPLACE INTO user_team_roles
                   (id, user_id, team_id, role, granted_at, granted_by)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (
                    role.id,
                    role.user_id,
                    role.team_id,
                    role.role.value,
                    role.granted_at.isoformat(),
                    role.granted_by,
                ),
            )
            self._conn.commit()

    def get_user_team_roles(self, user_id: str) -> list[UserTeamRole]:
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT * FROM user_team_roles WHERE user_id = ?", (user_id,)
        ).fetchall()
        return [self._row_to_user_team_role(r) for r in rows]

    def get_team_members(self, team_id: str) -> list[UserTeamRole]:
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT * FROM user_team_roles WHERE team_id = ?", (team_id,)
        ).fetchall()
        return [self._row_to_user_team_role(r) for r in rows]

    def delete_user_team_role(self, role_id: str) -> bool:
        with self._lock:
            cursor = self._conn.execute("DELETE FROM user_team_roles WHERE id = ?", (role_id,))
            self._conn.commit()
            return cursor.rowcount > 0

    def _row_to_user_team_role(self, row: sqlite3.Row) -> Any:
        from stateloom.auth.models import UserTeamRole
        from stateloom.core.types import Role

        d: dict[str, Any] = {
            "id": row["id"],
            "user_id": row["user_id"],
            "team_id": row["team_id"],
            "role": Role(row["role"]),
            "granted_at": datetime.fromisoformat(row["granted_at"]),
            "granted_by": row["granted_by"],
        }
        return UserTeamRole.model_validate(d)

    # --- Refresh token methods (auth) ---

    def save_refresh_token(
        self,
        token_hash: str,
        user_id: str,
        expires_at: str,
        family_id: str = "",
    ) -> None:
        with self._lock:
            self._conn.execute(
                """INSERT OR REPLACE INTO refresh_tokens
                   (token_hash, user_id, expires_at, family_id, revoked)
                   VALUES (?, ?, ?, ?, 0)""",
                (token_hash, user_id, expires_at, family_id),
            )
            self._conn.commit()

    def get_refresh_token(self, token_hash: str) -> dict[str, Any] | None:
        conn = self._get_conn()
        row = conn.execute(
            "SELECT * FROM refresh_tokens WHERE token_hash = ?", (token_hash,)
        ).fetchone()
        if not row:
            return None
        return {
            "token_hash": row["token_hash"],
            "user_id": row["user_id"],
            "expires_at": row["expires_at"],
            "family_id": row["family_id"],
            "revoked": bool(row["revoked"]),
        }

    def revoke_refresh_token(self, token_hash: str) -> None:
        with self._lock:
            self._conn.execute(
                "UPDATE refresh_tokens SET revoked = 1 WHERE token_hash = ?",
                (token_hash,),
            )
            self._conn.commit()

    def revoke_all_refresh_tokens(self, user_id: str) -> int:
        with self._lock:
            cursor = self._conn.execute(
                "UPDATE refresh_tokens SET revoked = 1 WHERE user_id = ? AND revoked = 0",
                (user_id,),
            )
            self._conn.commit()
            return cursor.rowcount

    # --- OIDC provider methods ---

    def save_oidc_provider(self, provider: OIDCProvider) -> None:
        import json as _json

        with self._lock:
            self._conn.execute(
                """INSERT INTO oidc_providers (
                    id, name, issuer_url, client_id, client_secret_encrypted,
                    scopes, group_claim, group_role_mapping_json, is_active, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    name=excluded.name,
                    issuer_url=excluded.issuer_url,
                    client_id=excluded.client_id,
                    client_secret_encrypted=excluded.client_secret_encrypted,
                    scopes=excluded.scopes,
                    group_claim=excluded.group_claim,
                    group_role_mapping_json=excluded.group_role_mapping_json,
                    is_active=excluded.is_active
                """,
                (
                    provider.id,
                    provider.name,
                    provider.issuer_url,
                    provider.client_id,
                    provider.client_secret_encrypted,
                    provider.scopes,
                    provider.group_claim,
                    _json.dumps(provider.group_role_mapping),
                    1 if provider.is_active else 0,
                    provider.created_at.isoformat(),
                ),
            )
            self._conn.commit()

    def _row_to_oidc_provider(self, row: Any) -> OIDCProvider:
        import json as _json

        from stateloom.auth.oidc_models import OIDCProvider

        mapping_raw = row["group_role_mapping_json"]
        mapping = _json.loads(mapping_raw) if mapping_raw else {}

        return OIDCProvider(
            id=row["id"],
            name=row["name"],
            issuer_url=row["issuer_url"],
            client_id=row["client_id"],
            client_secret_encrypted=row["client_secret_encrypted"],
            scopes=row["scopes"],
            group_claim=row["group_claim"],
            group_role_mapping=mapping,
            is_active=bool(row["is_active"]),
            created_at=datetime.fromisoformat(row["created_at"]),
        )

    def get_oidc_provider(self, provider_id: str) -> OIDCProvider | None:
        conn = self._get_conn()
        row = conn.execute("SELECT * FROM oidc_providers WHERE id = ?", (provider_id,)).fetchone()
        if not row:
            return None
        return self._row_to_oidc_provider(row)

    def get_oidc_provider_by_issuer(self, issuer_url: str) -> OIDCProvider | None:
        conn = self._get_conn()
        row = conn.execute(
            "SELECT * FROM oidc_providers WHERE issuer_url = ?", (issuer_url,)
        ).fetchone()
        if not row:
            return None
        return self._row_to_oidc_provider(row)

    def list_oidc_providers(self) -> list[OIDCProvider]:
        conn = self._get_conn()
        rows = conn.execute("SELECT * FROM oidc_providers ORDER BY created_at").fetchall()
        return [self._row_to_oidc_provider(r) for r in rows]

    def delete_oidc_provider(self, provider_id: str) -> bool:
        with self._lock:
            cursor = self._conn.execute("DELETE FROM oidc_providers WHERE id = ?", (provider_id,))
            self._conn.commit()
            return cursor.rowcount > 0

    # --- Event serialization (generic, model_dump-based) ---

    # Column aliases: where a model field name differs from the SQL column name
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
        "request_messages_json",
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

        # Bool → int for SQLite
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
            col_vals.get("request_messages_json"),
            json.dumps(event.metadata) if event.metadata else None,
            col_vals.get("rating"),
            col_vals.get("score"),
            col_vals.get("comment"),
            extra_json,
        )

    def _row_to_event(self, row: sqlite3.Row) -> Event:
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
        row_keys = set(row.keys())
        for col in self._DIRECT_COLS:
            if col not in row_keys:
                continue
            val = row[col]
            if val is not None:
                field_name = inv.get(col, col)
                kwargs[field_name] = val

        # Merge extra_json
        shadow_raw = row["extra_json"]
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
    idx = min(idx, len(s) - 1)
    return s[idx]
