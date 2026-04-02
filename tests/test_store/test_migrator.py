"""Tests for Alembic migration runner."""

import os
import sqlite3
import tempfile
from unittest.mock import patch

import pytest

# These tests require alembic + sqlalchemy to be installed
pytest.importorskip("alembic")
pytest.importorskip("sqlalchemy")

from stateloom.store.migrator import _is_existing_db, run_migrations  # noqa: E402
from stateloom.store.sqlite_store import _SCHEMA, SQLiteStore  # noqa: E402


def _make_temp_db():
    """Create a temp file and return (fd, path, url)."""
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    url = f"sqlite:///{os.path.abspath(path)}"
    return fd, path, url


def _get_tables(path: str) -> set[str]:
    """Return the set of table names in a SQLite database."""
    conn = sqlite3.connect(path)
    rows = conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    conn.close()
    return {r[0] for r in rows}


def _get_columns(path: str, table: str) -> set[str]:
    """Return the set of column names for a table."""
    conn = sqlite3.connect(path)
    rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
    conn.close()
    return {r[1] for r in rows}


def _get_alembic_version(path: str) -> str | None:
    """Return the current alembic version stamp, or None if table doesn't exist."""
    conn = sqlite3.connect(path)
    try:
        row = conn.execute("SELECT version_num FROM alembic_version").fetchone()
        return row[0] if row else None
    except sqlite3.OperationalError:
        return None
    finally:
        conn.close()


class TestFreshDatabase:
    def test_creates_all_tables(self):
        """run_migrations on an empty DB should create all 13 tables."""
        _, path, url = _make_temp_db()
        try:
            result = run_migrations(url)
            assert result is True

            tables = _get_tables(path)
            expected = {
                "sessions",
                "events",
                "experiments",
                "experiment_assignments",
                "session_feedback",
                "organizations",
                "teams",
                "jobs",
                "secrets",
                "virtual_keys",
                "agents",
                "agent_versions",
                "admin_locks",
                "alembic_version",
            }
            assert expected.issubset(tables)
        finally:
            os.unlink(path)

    def test_alembic_version_at_head(self):
        """After fresh migration, alembic_version should point to 002."""
        _, path, url = _make_temp_db()
        try:
            run_migrations(url)
            version = _get_alembic_version(path)
            assert version == "010"
        finally:
            os.unlink(path)

    def test_migration_002_adds_columns(self):
        """Migration 002 adds columns; migration 007 renames shadow_data_json → extra_json."""
        _, path, url = _make_temp_db()
        try:
            run_migrations(url)
            event_cols = _get_columns(path, "events")
            assert "extra_json" in event_cols
            assert "rating" in event_cols
            assert "score" in event_cols
            assert "comment" in event_cols

            session_cols = _get_columns(path, "sessions")
            assert "org_id" in session_cols
            assert "team_id" in session_cols
            assert "parent_session_id" in session_cols
            assert "estimated_api_cost" in session_cols

            vk_cols = _get_columns(path, "virtual_keys")
            assert "allowed_models_json" in vk_cols
            assert "billing_mode" in vk_cols
            assert "agent_ids_json" in vk_cols
        finally:
            os.unlink(path)


class TestExistingDatabase:
    def test_stamps_without_ddl(self):
        """Existing DB (has sessions, no alembic_version) should be stamped at head."""
        _, path, url = _make_temp_db()
        try:
            # Create a legacy database with _SCHEMA
            conn = sqlite3.connect(path)
            conn.executescript(_SCHEMA)
            conn.commit()
            conn.close()

            # Verify it's an existing DB
            assert _is_existing_db(url) is True

            # Run migrations — should stamp, not run DDL
            result = run_migrations(url)
            assert result is True

            # Verify stamped at head
            version = _get_alembic_version(path)
            assert version == "010"

            # Verify data intact (tables still exist)
            tables = _get_tables(path)
            assert "sessions" in tables
            assert "events" in tables
        finally:
            os.unlink(path)

    def test_existing_data_preserved(self):
        """Stamping an existing DB should not lose data."""
        _, path, url = _make_temp_db()
        try:
            # Create legacy DB with some data
            conn = sqlite3.connect(path)
            conn.executescript(_SCHEMA)
            conn.execute(
                "INSERT INTO sessions (id, started_at, status) VALUES (?, ?, ?)",
                ("test-sess", "2024-01-01T00:00:00", "active"),
            )
            conn.commit()
            conn.close()

            # Run migrations
            run_migrations(url)

            # Verify data preserved
            conn = sqlite3.connect(path)
            row = conn.execute("SELECT id FROM sessions WHERE id = ?", ("test-sess",)).fetchone()
            assert row is not None
            assert row[0] == "test-sess"
            conn.close()
        finally:
            os.unlink(path)


class TestIdempotency:
    def test_rerun_is_safe(self):
        """Running migrations twice on the same DB should not error."""
        _, path, url = _make_temp_db()
        try:
            run_migrations(url)
            # Second run should be a no-op
            run_migrations(url)
            version = _get_alembic_version(path)
            assert version == "010"
        finally:
            os.unlink(path)


class TestFallback:
    def test_returns_false_when_alembic_missing(self):
        """When alembic is not importable, run_migrations returns False."""
        with patch.dict("sys.modules", {"alembic": None, "alembic.command": None}):
            # Re-import to pick up the mock
            import importlib

            from stateloom.store import migrator

            importlib.reload(migrator)
            result = migrator.run_migrations("sqlite:///dummy.db")
            assert result is False
            # Restore
            importlib.reload(migrator)


class TestSQLiteStoreIntegration:
    def test_uses_alembic_when_available(self):
        """SQLiteStore should use Alembic and have alembic_version table."""
        _, path, url = _make_temp_db()
        try:
            store = SQLiteStore(path=path, auto_migrate=True)
            tables = _get_tables(path)
            assert "alembic_version" in tables
            assert "sessions" in tables
            store.close()
        finally:
            os.unlink(path)

    def test_fallback_when_alembic_unavailable(self):
        """SQLiteStore with mocked-away alembic should fall back to legacy."""
        _, path, url = _make_temp_db()
        try:
            with patch(
                "stateloom.store.migrator.run_migrations",
                side_effect=ImportError("mocked"),
            ):
                # This will hit the except branch and fall back to legacy
                store = SQLiteStore(path=path, auto_migrate=True)
                tables = _get_tables(path)
                # Legacy path creates tables but no alembic_version
                assert "sessions" in tables
                assert "events" in tables
                store.close()
        finally:
            os.unlink(path)

    def test_auto_migrate_false_skips_alembic(self):
        """SQLiteStore(auto_migrate=False) should only use legacy path."""
        _, path, url = _make_temp_db()
        try:
            store = SQLiteStore(path=path, auto_migrate=False)
            tables = _get_tables(path)
            # Legacy path — no alembic_version table
            assert "alembic_version" not in tables
            assert "sessions" in tables
            store.close()
        finally:
            os.unlink(path)


class TestIsExistingDb:
    def test_empty_db(self):
        """Empty database is not an existing DB."""
        _, path, url = _make_temp_db()
        try:
            assert _is_existing_db(url) is False
        finally:
            os.unlink(path)

    def test_db_with_sessions_no_alembic(self):
        """DB with sessions table but no alembic_version is existing."""
        _, path, url = _make_temp_db()
        try:
            conn = sqlite3.connect(path)
            conn.execute("CREATE TABLE sessions (id TEXT PRIMARY KEY)")
            conn.commit()
            conn.close()
            assert _is_existing_db(url) is True
        finally:
            os.unlink(path)

    def test_db_with_both_tables(self):
        """DB with both sessions and alembic_version is NOT existing (already migrated)."""
        _, path, url = _make_temp_db()
        try:
            conn = sqlite3.connect(path)
            conn.execute("CREATE TABLE sessions (id TEXT PRIMARY KEY)")
            conn.execute("CREATE TABLE alembic_version (version_num TEXT)")
            conn.commit()
            conn.close()
            assert _is_existing_db(url) is False
        finally:
            os.unlink(path)
