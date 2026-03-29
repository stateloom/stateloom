"""Add columns introduced after initial schema.

Ports all ALTER TABLE ADD COLUMN operations from _migrate_schema() into a
proper Alembic migration.  Each ALTER is wrapped in try/except so this is
idempotent — safe to run on databases that already have these columns from
the legacy _SCHEMA constant.

Revision ID: 002
Revises: 001
"""

from alembic import op

revision = "002"
down_revision = "001"
branch_labels = None
depends_on = None


def _safe_add_column(table: str, column: str, col_type: str, default: str = "") -> None:
    """Add a column if it doesn't already exist (idempotent)."""
    ddl = f"ALTER TABLE {table} ADD COLUMN {column} {col_type}"
    if default:
        ddl += f" {default}"
    try:
        op.execute(ddl)
    except Exception:
        # Column already exists — expected for databases upgraded from _SCHEMA
        pass


def upgrade() -> None:
    # --- Events table ---
    _safe_add_column("events", "rating", "TEXT")
    _safe_add_column("events", "score", "REAL")
    _safe_add_column("events", "comment", "TEXT")
    _safe_add_column("events", "shadow_data_json", "TEXT")

    # --- Sessions table ---
    _safe_add_column("sessions", "org_id", "TEXT", "DEFAULT ''")
    _safe_add_column("sessions", "team_id", "TEXT", "DEFAULT ''")
    _safe_add_column("sessions", "parent_session_id", "TEXT")
    _safe_add_column("sessions", "timeout", "REAL")
    _safe_add_column("sessions", "idle_timeout", "REAL")
    _safe_add_column("sessions", "last_heartbeat", "TEXT")
    _safe_add_column("sessions", "estimated_api_cost", "REAL")

    # --- Organizations table ---
    _safe_add_column("organizations", "compliance_profile_json", "TEXT", "DEFAULT ''")

    # --- Teams table ---
    _safe_add_column("teams", "compliance_profile_json", "TEXT", "DEFAULT ''")
    _safe_add_column("teams", "rate_limit_tps", "REAL")
    _safe_add_column("teams", "rate_limit_priority", "INTEGER", "DEFAULT 0")
    _safe_add_column("teams", "rate_limit_max_queue", "INTEGER", "DEFAULT 100")
    _safe_add_column("teams", "rate_limit_queue_timeout", "REAL", "DEFAULT 30.0")

    # --- Virtual keys table ---
    _safe_add_column("virtual_keys", "allowed_models_json", "TEXT", "DEFAULT '[]'")
    _safe_add_column("virtual_keys", "budget_limit", "REAL")
    _safe_add_column("virtual_keys", "budget_spent", "REAL", "DEFAULT 0.0")
    _safe_add_column("virtual_keys", "rate_limit_tps", "REAL")
    _safe_add_column("virtual_keys", "rate_limit_max_queue", "INTEGER", "DEFAULT 100")
    _safe_add_column("virtual_keys", "rate_limit_queue_timeout", "REAL", "DEFAULT 30.0")
    _safe_add_column("virtual_keys", "agent_ids_json", "TEXT", "DEFAULT '[]'")
    _safe_add_column("virtual_keys", "billing_mode", "TEXT", "DEFAULT 'api'")

    # --- Session indexes ---
    op.execute("CREATE INDEX IF NOT EXISTS idx_sessions_org ON sessions(org_id)")
    op.execute("CREATE INDEX IF NOT EXISTS idx_sessions_team ON sessions(team_id)")
    op.execute("CREATE INDEX IF NOT EXISTS idx_sessions_parent ON sessions(parent_session_id)")


def downgrade() -> None:
    # Additive columns are not dropped to avoid data loss.
    pass
