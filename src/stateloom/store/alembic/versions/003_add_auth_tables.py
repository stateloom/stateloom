"""Add auth tables (users, user_team_roles, refresh_tokens, oidc_providers)
and end_user column on sessions.

Revision ID: 003
Revises: 002
"""

from alembic import op

revision = "003"
down_revision = "002"
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
        pass


def upgrade() -> None:
    # --- Sessions: end_user column ---
    _safe_add_column("sessions", "end_user", "TEXT", "DEFAULT ''")
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_sessions_end_user "
        "ON sessions(end_user) WHERE end_user != ''"
    )

    # --- Users table ---
    op.execute("""
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
        )
    """)
    op.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_users_email ON users(email COLLATE NOCASE)")
    op.execute("CREATE INDEX IF NOT EXISTS idx_users_org ON users(org_id)")
    op.execute(
        "CREATE UNIQUE INDEX IF NOT EXISTS idx_users_oidc "
        "ON users(oidc_provider_id, oidc_subject) "
        "WHERE oidc_provider_id != '' AND oidc_subject != ''"
    )

    # --- User team roles table ---
    op.execute("""
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
        )
    """)
    op.execute("CREATE INDEX IF NOT EXISTS idx_utr_user ON user_team_roles(user_id)")
    op.execute("CREATE INDEX IF NOT EXISTS idx_utr_team ON user_team_roles(team_id)")

    # --- Refresh tokens table ---
    op.execute("""
        CREATE TABLE IF NOT EXISTS refresh_tokens (
            token_hash TEXT PRIMARY KEY,
            user_id TEXT NOT NULL,
            expires_at TEXT NOT NULL,
            family_id TEXT NOT NULL DEFAULT '',
            revoked INTEGER NOT NULL DEFAULT 0,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
    """)
    op.execute("CREATE INDEX IF NOT EXISTS idx_rt_user ON refresh_tokens(user_id)")

    # --- OIDC providers table ---
    op.execute("""
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
        )
    """)
    op.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_oidc_issuer ON oidc_providers(issuer_url)")


def downgrade() -> None:
    # Additive tables/columns are not dropped to avoid data loss.
    pass
