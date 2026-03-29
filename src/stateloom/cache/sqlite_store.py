"""SQLite-backed cache store for StateLoom (separate DB from event store)."""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import threading
import time

from stateloom.cache.base import CacheEntry

logger = logging.getLogger("stateloom.cache.sqlite")

_SQLITE_BUSY_TIMEOUT_MS = 5000

_CACHE_SCHEMA = """
CREATE TABLE IF NOT EXISTS cache_entries (
    request_hash TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    response_json TEXT NOT NULL,
    model TEXT NOT NULL DEFAULT '',
    provider TEXT NOT NULL DEFAULT '',
    cost REAL NOT NULL DEFAULT 0.0,
    created_at REAL NOT NULL,
    embedding_json TEXT
);

CREATE INDEX IF NOT EXISTS idx_cache_session ON cache_entries(session_id);
CREATE INDEX IF NOT EXISTS idx_cache_created ON cache_entries(created_at);
"""


class SQLiteCacheStore:
    """WAL-mode SQLite cache store in a separate database file.

    Uses time.time() for timestamps (epoch seconds, survives restarts).
    Embeddings are stored as JSON arrays.
    """

    def __init__(
        self,
        path: str = ".stateloom/cache.db",
        max_size: int = 1000,
    ) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        self._path = path
        self._max_size = max_size
        self._lock = threading.Lock()
        self._conn = self._create_connection()
        self._conn.executescript(_CACHE_SCHEMA)
        self._conn.commit()

    def _create_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute(f"PRAGMA busy_timeout={_SQLITE_BUSY_TIMEOUT_MS}")
        return conn

    def get(self, request_hash: str, session_id: str | None = None) -> CacheEntry | None:
        with self._lock:
            if session_id is not None:
                row = self._conn.execute(
                    "SELECT * FROM cache_entries WHERE request_hash = ? AND session_id = ?",
                    (request_hash, session_id),
                ).fetchone()
            else:
                row = self._conn.execute(
                    "SELECT * FROM cache_entries WHERE request_hash = ?",
                    (request_hash,),
                ).fetchone()
            if row is None:
                return None
            return self._row_to_entry(row)

    def put(self, entry: CacheEntry) -> None:
        embedding_json = json.dumps(entry.embedding) if entry.embedding else None
        with self._lock:
            self._conn.execute(
                """INSERT OR REPLACE INTO cache_entries
                   (request_hash, session_id, response_json, model, provider,
                    cost, created_at, embedding_json)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    entry.request_hash,
                    entry.session_id,
                    entry.response_json,
                    entry.model,
                    entry.provider,
                    entry.cost,
                    entry.created_at,
                    embedding_json,
                ),
            )
            # LRU-style eviction: keep only max_size most recent entries
            count = self._conn.execute("SELECT COUNT(*) FROM cache_entries").fetchone()[0]
            if count > self._max_size:
                excess = count - self._max_size
                self._conn.execute(
                    """DELETE FROM cache_entries WHERE request_hash IN (
                        SELECT request_hash FROM cache_entries
                        ORDER BY created_at ASC LIMIT ?
                    )""",
                    (excess,),
                )
            self._conn.commit()

    def get_all_entries(self, session_id: str | None = None) -> list[CacheEntry]:
        with self._lock:
            if session_id is not None:
                rows = self._conn.execute(
                    "SELECT * FROM cache_entries WHERE session_id = ? ORDER BY created_at ASC",
                    (session_id,),
                ).fetchall()
            else:
                rows = self._conn.execute(
                    "SELECT * FROM cache_entries ORDER BY created_at ASC"
                ).fetchall()
            return [self._row_to_entry(r) for r in rows]

    def evict_expired(self, ttl: float) -> int:
        if ttl <= 0:
            return 0
        cutoff = time.time() - ttl
        with self._lock:
            cursor = self._conn.execute(
                "DELETE FROM cache_entries WHERE created_at < ?",
                (cutoff,),
            )
            self._conn.commit()
            return cursor.rowcount

    def size(self) -> int:
        with self._lock:
            row = self._conn.execute("SELECT COUNT(*) FROM cache_entries").fetchone()
            return row[0] if row else 0

    def clear(self) -> None:
        with self._lock:
            self._conn.execute("DELETE FROM cache_entries")
            self._conn.commit()

    def purge_by_content(self, identifier: str) -> int:
        """Delete cache entries containing the identifier in response_json."""
        with self._lock:
            cursor = self._conn.execute(
                "DELETE FROM cache_entries WHERE response_json LIKE ?",
                (f"%{identifier}%",),
            )
            self._conn.commit()
            return cursor.rowcount

    def close(self) -> None:
        self._conn.close()

    @staticmethod
    def _row_to_entry(row: sqlite3.Row) -> CacheEntry:
        embedding = None
        if row["embedding_json"]:
            try:
                embedding = json.loads(row["embedding_json"])
            except (json.JSONDecodeError, TypeError):
                logger.warning("Corrupted embedding JSON for hash %s", row["request_hash"])
        return CacheEntry(
            request_hash=row["request_hash"],
            session_id=row["session_id"],
            response_json=row["response_json"],
            model=row["model"],
            provider=row["provider"],
            cost=row["cost"],
            created_at=row["created_at"],
            embedding=embedding,
        )
