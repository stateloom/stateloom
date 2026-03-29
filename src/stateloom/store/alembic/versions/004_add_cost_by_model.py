"""Add per-model cost breakdown columns to sessions table.

Adds cost_by_model_json and tokens_by_model_json TEXT columns for tracking
per-model cost and token breakdowns within a session.

Revision ID: 004
Revises: 003
"""

from alembic import op

revision = "004"
down_revision = "003"
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
    _safe_add_column("sessions", "cost_by_model_json", "TEXT")
    _safe_add_column("sessions", "tokens_by_model_json", "TEXT")


def downgrade() -> None:
    # Additive columns are not dropped to avoid data loss.
    pass
