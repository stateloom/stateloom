"""Tests for the in-memory cache store."""

from __future__ import annotations

import time

import pytest

from stateloom.cache.base import CacheEntry
from stateloom.cache.memory_store import MemoryCacheStore


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


class TestMemoryCacheStore:
    def test_put_and_get_global(self):
        store = MemoryCacheStore()
        entry = _make_entry()
        store.put(entry)

        # Global get (no session filter)
        result = store.get("abc123")
        assert result is not None
        assert result.request_hash == "abc123"
        assert result.model == "gpt-4"

    def test_get_miss(self):
        store = MemoryCacheStore()
        assert store.get("nonexistent") is None

    def test_put_and_get_session_scoped(self):
        store = MemoryCacheStore()
        entry = _make_entry(session_id="sess-1")
        store.put(entry)

        # Same session → hit
        assert store.get("abc123", session_id="sess-1") is not None
        # Different session → miss
        assert store.get("abc123", session_id="sess-2") is None

    def test_lru_eviction(self):
        store = MemoryCacheStore(max_size=3)
        for i in range(5):
            store.put(_make_entry(request_hash=f"hash-{i}"))

        assert store.size() == 3
        # Oldest entries evicted
        assert store.get("hash-0") is None
        assert store.get("hash-1") is None
        # Newest entries remain
        assert store.get("hash-2") is not None
        assert store.get("hash-3") is not None
        assert store.get("hash-4") is not None

    def test_ttl_eviction(self):
        store = MemoryCacheStore()
        # Create an entry with a past timestamp
        entry = _make_entry()
        entry.created_at = time.time() - 100
        store.put(entry)

        assert store.size() == 1
        evicted = store.evict_expired(ttl=50)
        assert evicted == 1
        assert store.size() == 0

    def test_ttl_zero_no_eviction(self):
        store = MemoryCacheStore()
        store.put(_make_entry())
        evicted = store.evict_expired(ttl=0)
        assert evicted == 0
        assert store.size() == 1

    def test_get_all_entries(self):
        store = MemoryCacheStore()
        store.put(_make_entry(request_hash="h1", session_id="s1"))
        store.put(_make_entry(request_hash="h2", session_id="s2"))
        store.put(_make_entry(request_hash="h3", session_id="s1"))

        all_entries = store.get_all_entries()
        assert len(all_entries) == 3

        s1_entries = store.get_all_entries(session_id="s1")
        assert len(s1_entries) == 2

        s2_entries = store.get_all_entries(session_id="s2")
        assert len(s2_entries) == 1

    def test_clear(self):
        store = MemoryCacheStore()
        store.put(_make_entry())
        assert store.size() == 1
        store.clear()
        assert store.size() == 0

    def test_close(self):
        store = MemoryCacheStore()
        store.put(_make_entry())
        store.close()
        assert store.size() == 0

    def test_size(self):
        store = MemoryCacheStore()
        assert store.size() == 0
        store.put(_make_entry(request_hash="h1"))
        assert store.size() == 1
        store.put(_make_entry(request_hash="h2"))
        assert store.size() == 2

    def test_overwrite_existing_entry(self):
        store = MemoryCacheStore()
        store.put(_make_entry(request_hash="h1", cost=0.01))
        store.put(_make_entry(request_hash="h1", cost=0.05))
        assert store.size() == 1
        result = store.get("h1")
        assert result is not None
        assert result.cost == 0.05

    def test_lru_moves_to_end_on_get(self):
        store = MemoryCacheStore(max_size=3)
        store.put(_make_entry(request_hash="h0"))
        store.put(_make_entry(request_hash="h1"))
        store.put(_make_entry(request_hash="h2"))

        # Access h0 → moves it to end (most recent)
        store.get("h0")
        # Insert new → should evict h1 (least recently used)
        store.put(_make_entry(request_hash="h3"))

        assert store.get("h0") is not None
        assert store.get("h1") is None
        assert store.get("h2") is not None
        assert store.get("h3") is not None
