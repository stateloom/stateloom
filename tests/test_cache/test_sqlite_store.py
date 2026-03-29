"""Tests for the SQLite cache store."""

from __future__ import annotations

import os
import tempfile
import time

import pytest

from stateloom.cache.base import CacheEntry
from stateloom.cache.sqlite_store import SQLiteCacheStore


def _make_entry(
    request_hash: str = "abc123",
    session_id: str = "sess-1",
    model: str = "gpt-4",
    cost: float = 0.01,
    embedding: list[float] | None = None,
) -> CacheEntry:
    return CacheEntry(
        request_hash=request_hash,
        session_id=session_id,
        response_json='{"text":"hello"}',
        model=model,
        provider="openai",
        cost=cost,
        created_at=time.time(),
        embedding=embedding,
    )


@pytest.fixture
def store(tmp_path):
    db_path = str(tmp_path / "cache.db")
    s = SQLiteCacheStore(path=db_path, max_size=1000)
    yield s
    s.close()


class TestSQLiteCacheStore:
    def test_put_and_get(self, store):
        entry = _make_entry()
        store.put(entry)
        result = store.get("abc123")
        assert result is not None
        assert result.request_hash == "abc123"
        assert result.model == "gpt-4"
        assert result.response_json == '{"text":"hello"}'

    def test_get_miss(self, store):
        assert store.get("nonexistent") is None

    def test_session_scoped_get(self, store):
        entry = _make_entry(session_id="sess-1")
        store.put(entry)
        assert store.get("abc123", session_id="sess-1") is not None
        assert store.get("abc123", session_id="sess-2") is None

    def test_lru_eviction(self, tmp_path):
        store = SQLiteCacheStore(path=str(tmp_path / "cache.db"), max_size=3)
        for i in range(5):
            e = _make_entry(request_hash=f"hash-{i}")
            e.created_at = time.time() + i * 0.001  # Ensure order
            store.put(e)

        assert store.size() == 3
        # Oldest evicted
        assert store.get("hash-0") is None
        assert store.get("hash-1") is None
        # Newest remain
        assert store.get("hash-4") is not None
        store.close()

    def test_ttl_eviction(self, store):
        entry = _make_entry()
        entry.created_at = time.time() - 100
        store.put(entry)

        assert store.size() == 1
        evicted = store.evict_expired(ttl=50)
        assert evicted == 1
        assert store.size() == 0

    def test_ttl_zero_no_eviction(self, store):
        store.put(_make_entry())
        evicted = store.evict_expired(ttl=0)
        assert evicted == 0
        assert store.size() == 1

    def test_get_all_entries(self, store):
        store.put(_make_entry(request_hash="h1", session_id="s1"))
        store.put(_make_entry(request_hash="h2", session_id="s2"))
        store.put(_make_entry(request_hash="h3", session_id="s1"))

        all_entries = store.get_all_entries()
        assert len(all_entries) == 3

        s1_entries = store.get_all_entries(session_id="s1")
        assert len(s1_entries) == 2

    def test_clear(self, store):
        store.put(_make_entry())
        assert store.size() == 1
        store.clear()
        assert store.size() == 0

    def test_embedding_roundtrip(self, store):
        embedding = [0.1, 0.2, 0.3, 0.4, 0.5]
        entry = _make_entry(embedding=embedding)
        store.put(entry)

        result = store.get("abc123")
        assert result is not None
        assert result.embedding is not None
        assert len(result.embedding) == 5
        assert abs(result.embedding[0] - 0.1) < 1e-6

    def test_persistence_across_connections(self, tmp_path):
        db_path = str(tmp_path / "cache.db")
        store1 = SQLiteCacheStore(path=db_path)
        store1.put(_make_entry(request_hash="persistent"))
        store1.close()

        store2 = SQLiteCacheStore(path=db_path)
        result = store2.get("persistent")
        assert result is not None
        assert result.request_hash == "persistent"
        store2.close()

    def test_overwrite_existing(self, store):
        store.put(_make_entry(request_hash="h1", cost=0.01))
        store.put(_make_entry(request_hash="h1", cost=0.05))
        assert store.size() == 1
        result = store.get("h1")
        assert result is not None
        assert result.cost == 0.05
