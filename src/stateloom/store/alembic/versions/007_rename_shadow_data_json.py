"""Rename shadow_data_json to extra_json on events table.

The column holds overflow fields from all event types (not just shadow data).
The new name better reflects its purpose.

Revision ID: 007
Revises: 006
"""

from alembic import op

revision = "007"
down_revision = "006"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # SQLite 3.25+ supports RENAME COLUMN.  Guard with try/except for older
    # versions or databases that already have the new name.
    try:
        op.execute("ALTER TABLE events RENAME COLUMN shadow_data_json TO extra_json")
    except Exception:
        # Column may already be named extra_json (legacy _migrate_schema ran first)
        pass


def downgrade() -> None:
    try:
        op.execute("ALTER TABLE events RENAME COLUMN extra_json TO shadow_data_json")
    except Exception:
        pass
