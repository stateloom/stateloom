"""Normalize legacy action_taken values in events table.

Old events stored present-tense values ("redact", "block") before the
ActionTaken enum was introduced.  This migration updates them to the
canonical past-tense form ("redacted", "blocked") so dashboard queries
don't need read-time normalization.

Revision ID: 010
Revises: 009
"""

from alembic import op

revision = "010"
down_revision = "009"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("UPDATE events SET action_taken = 'redacted' WHERE action_taken = 'redact'")
    op.execute("UPDATE events SET action_taken = 'blocked' WHERE action_taken = 'block'")


def downgrade() -> None:
    op.execute("UPDATE events SET action_taken = 'redact' WHERE action_taken = 'redacted'")
    op.execute("UPDATE events SET action_taken = 'block' WHERE action_taken = 'blocked'")
