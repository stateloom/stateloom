"""Initial schema — all 13 tables + indexes.

Matches the _SCHEMA constant in sqlite_store.py at the time of Alembic adoption.
Uses IF NOT EXISTS for idempotency on both fresh and existing databases.

Revision ID: 001
Revises: None
"""

from alembic import op

revision = "001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("""
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
        )
    """)

    op.execute("""
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
            FOREIGN KEY (session_id) REFERENCES sessions(id)
        )
    """)

    op.execute("CREATE INDEX IF NOT EXISTS idx_events_session ON events(session_id)")
    op.execute("CREATE INDEX IF NOT EXISTS idx_events_type ON events(event_type)")
    op.execute("CREATE INDEX IF NOT EXISTS idx_events_timestamp ON events(timestamp)")
    op.execute("CREATE INDEX IF NOT EXISTS idx_sessions_status ON sessions(status)")
    op.execute("CREATE INDEX IF NOT EXISTS idx_sessions_started ON sessions(started_at)")

    op.execute("""
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
    """)
    op.execute("CREATE INDEX IF NOT EXISTS idx_experiments_status ON experiments(status)")

    op.execute("""
        CREATE TABLE IF NOT EXISTS experiment_assignments (
            session_id TEXT PRIMARY KEY,
            experiment_id TEXT NOT NULL,
            variant_name TEXT NOT NULL,
            variant_config_json TEXT NOT NULL DEFAULT '{}',
            assigned_at TEXT NOT NULL,
            FOREIGN KEY (experiment_id) REFERENCES experiments(id),
            FOREIGN KEY (session_id) REFERENCES sessions(id)
        )
    """)
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_assignments_experiment "
        "ON experiment_assignments(experiment_id)"
    )

    op.execute("""
        CREATE TABLE IF NOT EXISTS session_feedback (
            session_id TEXT PRIMARY KEY,
            rating TEXT NOT NULL,
            score REAL,
            comment TEXT NOT NULL DEFAULT '',
            created_at TEXT NOT NULL,
            FOREIGN KEY (session_id) REFERENCES sessions(id)
        )
    """)

    op.execute("""
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
        )
    """)

    op.execute("""
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
            FOREIGN KEY (org_id) REFERENCES organizations(id)
        )
    """)
    op.execute("CREATE INDEX IF NOT EXISTS idx_teams_org ON teams(org_id)")

    op.execute("""
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
    """)
    op.execute("CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status)")
    op.execute("CREATE INDEX IF NOT EXISTS idx_jobs_created ON jobs(created_at)")

    op.execute("""
        CREATE TABLE IF NOT EXISTS secrets (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
    """)

    op.execute("""
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
            metadata TEXT NOT NULL DEFAULT '{}'
        )
    """)
    op.execute("CREATE INDEX IF NOT EXISTS idx_vk_hash ON virtual_keys(key_hash)")
    op.execute("CREATE INDEX IF NOT EXISTS idx_vk_team ON virtual_keys(team_id)")

    op.execute("""
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
    """)
    op.execute("CREATE INDEX IF NOT EXISTS idx_agents_slug_team ON agents(slug, team_id)")
    op.execute("CREATE INDEX IF NOT EXISTS idx_agents_team ON agents(team_id)")

    op.execute("""
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
        )
    """)
    op.execute("CREATE INDEX IF NOT EXISTS idx_agent_versions_agent ON agent_versions(agent_id)")

    op.execute("""
        CREATE TABLE IF NOT EXISTS admin_locks (
            setting TEXT PRIMARY KEY,
            value TEXT NOT NULL,
            locked_by TEXT NOT NULL DEFAULT '',
            reason TEXT NOT NULL DEFAULT '',
            locked_at TEXT NOT NULL
        )
    """)


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS admin_locks")
    op.execute("DROP TABLE IF EXISTS agent_versions")
    op.execute("DROP TABLE IF EXISTS agents")
    op.execute("DROP TABLE IF EXISTS virtual_keys")
    op.execute("DROP TABLE IF EXISTS secrets")
    op.execute("DROP TABLE IF EXISTS jobs")
    op.execute("DROP TABLE IF EXISTS teams")
    op.execute("DROP TABLE IF EXISTS organizations")
    op.execute("DROP TABLE IF EXISTS session_feedback")
    op.execute("DROP TABLE IF EXISTS experiment_assignments")
    op.execute("DROP TABLE IF EXISTS experiments")
    op.execute("DROP TABLE IF EXISTS events")
    op.execute("DROP TABLE IF EXISTS sessions")
