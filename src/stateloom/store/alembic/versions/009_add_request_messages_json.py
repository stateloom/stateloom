"""Add request_messages_json column to events table.

Stores full request messages for lazy-load inspection in the dashboard.

Revision ID: 009
Revises: 008
"""

from alembic import op

revision = "009"
down_revision = "008"
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
    _safe_add_column("events", "request_messages_json", "TEXT")


def downgrade() -> None:
    # Additive columns are not dropped to avoid data loss.
    pass
