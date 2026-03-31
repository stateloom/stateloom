"""Add typed session fields (billing_mode, durable, agent_*, transport).

Promotes 8 commonly-accessed metadata keys to dedicated columns on the
sessions table for type safety, IDE support, and direct-column persistence.
Column names are prefixed with ``s_`` to disambiguate from identically-named
columns on other tables (e.g. ``billing_mode`` on ``virtual_keys``).

Revision ID: 006
Revises: 005
"""

from alembic import op

revision = "006"
down_revision = "005"
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
    _safe_add_column("sessions", "s_billing_mode", "TEXT", "DEFAULT ''")
    _safe_add_column("sessions", "s_durable", "INTEGER", "DEFAULT 0")
    _safe_add_column("sessions", "s_agent_id", "TEXT", "DEFAULT ''")
    _safe_add_column("sessions", "s_agent_slug", "TEXT", "DEFAULT ''")
    _safe_add_column("sessions", "s_agent_version_id", "TEXT", "DEFAULT ''")
    _safe_add_column("sessions", "s_agent_version_number", "INTEGER", "DEFAULT 0")
    _safe_add_column("sessions", "s_agent_name", "TEXT", "DEFAULT ''")
    _safe_add_column("sessions", "s_transport", "TEXT", "DEFAULT ''")

    # Backfill from existing metadata JSON for historical sessions
    col_meta_map = {
        "s_billing_mode": "billing_mode",
        "s_agent_id": "agent_id",
        "s_agent_slug": "agent_slug",
        "s_agent_version_id": "agent_version_id",
        "s_agent_name": "agent_name",
        "s_transport": "transport",
    }
    for col, meta_key in col_meta_map.items():
        try:
            op.execute(
                f"UPDATE sessions SET {col} = json_extract(metadata, '$.{meta_key}') "
                f"WHERE json_extract(metadata, '$.{meta_key}') IS NOT NULL AND {col} = ''"
            )
        except Exception:
            pass

    # Integer fields need cast
    try:
        op.execute(
            "UPDATE sessions SET s_durable = 1 "
            "WHERE json_extract(metadata, '$.durable') = 1 AND s_durable = 0"
        )
    except Exception:
        pass
    try:
        op.execute(
            "UPDATE sessions SET s_agent_version_number = "
            "CAST(json_extract(metadata, '$.agent_version_number') AS INTEGER) "
            "WHERE json_extract(metadata, '$.agent_version_number') IS NOT NULL "
            "AND s_agent_version_number = 0"
        )
    except Exception:
        pass


def downgrade() -> None:
    # Additive columns are not dropped to avoid data loss.
    pass
