"""Add overwrite_count column to sessions table.

Tracks how many times a session ID has been reused (overwritten).

Revision ID: 008
Revises: 007
"""

from alembic import op

revision = "008"
down_revision = "007"
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
        # Column already exists — expected for databases upgraded from _migrate_schema
        pass


def upgrade() -> None:
    _safe_add_column("sessions", "overwrite_count", "INTEGER", "DEFAULT 0")


def downgrade() -> None:
    # Additive columns are not dropped to avoid data loss.
    pass
