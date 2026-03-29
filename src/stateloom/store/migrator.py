"""Programmatic Alembic migration runner.

Requires: pip install stateloom[migrations]
Falls back gracefully when alembic is not installed.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger("stateloom.store.migrator")

_ALEMBIC_DIR = Path(__file__).parent / "alembic"


def run_migrations(url: str) -> bool:
    """Run pending Alembic migrations.

    Args:
        url: SQLAlchemy-style database URL (e.g. ``sqlite:///path/to/db``
             or ``postgresql://...``).

    Returns:
        True if Alembic ran successfully, False if alembic is not installed.
    """
    try:
        from alembic import command
    except ImportError:
        logger.debug("alembic not installed, skipping managed migrations")
        return False

    config = get_alembic_config(url)

    # Detect existing database: if sessions table exists but no alembic_version,
    # stamp at head without running DDL (safe adoption of existing schema).
    if _is_existing_db(url):
        logger.info("Existing StateLoom database detected — stamping at head revision")
        command.stamp(config, "head")
        return True

    command.upgrade(config, "head")
    return True


def get_alembic_config(url: str) -> Any:
    """Build an alembic Config pointing to our bundled migrations."""
    from alembic.config import Config

    config = Config()
    config.set_main_option("script_location", str(_ALEMBIC_DIR))
    config.set_main_option("sqlalchemy.url", url)
    return config


def _is_existing_db(url: str) -> bool:
    """Check if this is an existing StateLoom DB (has sessions table, no alembic_version)."""
    from sqlalchemy import create_engine, inspect

    engine = create_engine(url)
    try:
        inspector = inspect(engine)
        tables = inspector.get_table_names()
        has_sessions = "sessions" in tables
        has_alembic = "alembic_version" in tables
        return has_sessions and not has_alembic
    finally:
        engine.dispose()
