"""Tests for Redis cache store.

Requires a running Redis instance. Set STATELOOM_TEST_REDIS_URL to enable.
Example: STATELOOM_TEST_REDIS_URL=redis://localhost:6379
"""

import os
import time

import pytest

from stateloom.cache.base import CacheEntry

pytestmark = pytest.mark.skipif(
    not os.environ.get("STATELOOM_TEST_REDIS_URL"),
    reason="Redis not configured (set STATELOOM_TEST_REDIS_URL)",
)


@pytest.fixture
def redis_cache():
    """Create a RedisCacheStore connected to the test Redis."""
    from stateloom.cache.redis_store import RedisCacheStore

    url = os.environ["STATELOOM_TEST_REDIS_URL"]
    store = RedisCacheStore(url=url, max_size=100)
    store.clear()
    yield store
    store.clear()
    store.close()


def _make_entry(
    request_hash: str = "abc123",
    session_id: str = "session-1",
    model: str = "gpt-4o",
    cost: float = 0.01,
) -> CacheEntry:
    return CacheEntry(
        request_hash=request_hash,
        session_id=session_id,
        response_json='{"choices": []}',
        model=model,
        provider="openai",
        cost=cost,
        created_at=time.time(),
    )


def test_put_and_get(redis_cache):
    entry = _make_entry()
    redis_cache.put(entry)

    retrieved = redis_cache.get("abc123")
    assert retrieved is not None
    assert retrieved.request_hash == "abc123"
    assert retrieved.session_id == "session-1"
    assert retrieved.model == "gpt-4o"


def test_get_nonexistent(redis_cache):
    assert redis_cache.get("nonexistent") is None


def test_get_with_session_scope(redis_cache):
    entry = _make_entry(session_id="session-1")
    redis_cache.put(entry)

    # Same session should find it
    assert redis_cache.get("abc123", session_id="session-1") is not None

    # Different session should not find it
    assert redis_cache.get("abc123", session_id="session-2") is None


def test_size(redis_cache):
    assert redis_cache.size() == 0

    redis_cache.put(_make_entry(request_hash="h1"))
    assert redis_cache.size() == 1

    redis_cache.put(_make_entry(request_hash="h2"))
    assert redis_cache.size() == 2


def test_get_all_entries(redis_cache):
    redis_cache.put(_make_entry(request_hash="h1", session_id="s1"))
    redis_cache.put(_make_entry(request_hash="h2", session_id="s1"))
    redis_cache.put(_make_entry(request_hash="h3", session_id="s2"))

    all_entries = redis_cache.get_all_entries()
    assert len(all_entries) == 3

    s1_entries = redis_cache.get_all_entries(session_id="s1")
    assert len(s1_entries) == 2


def test_clear(redis_cache):
    redis_cache.put(_make_entry(request_hash="h1"))
    redis_cache.put(_make_entry(request_hash="h2"))
    assert redis_cache.size() == 2

    redis_cache.clear()
    assert redis_cache.size() == 0


def test_evict_expired(redis_cache):
    old_entry = CacheEntry(
        request_hash="old",
        session_id="s1",
        response_json="{}",
        model="gpt-4o",
        provider="openai",
        cost=0.0,
        created_at=time.time() - 3600,  # 1 hour ago
    )
    redis_cache.put(old_entry)
    redis_cache.put(_make_entry(request_hash="new"))

    assert redis_cache.size() == 2

    evicted = redis_cache.evict_expired(ttl=1800)  # 30 min TTL
    assert evicted == 1
    assert redis_cache.size() == 1


def test_purge_by_content(redis_cache):
    entry1 = CacheEntry(
        request_hash="h1",
        session_id="s1",
        response_json='{"text": "contains SECRET_DATA here"}',
        model="gpt-4o",
        provider="openai",
        cost=0.0,
        created_at=time.time(),
    )
    entry2 = _make_entry(request_hash="h2")
    redis_cache.put(entry1)
    redis_cache.put(entry2)

    deleted = redis_cache.purge_by_content("SECRET_DATA")
    assert deleted == 1
    assert redis_cache.size() == 1
    assert redis_cache.get("h1") is None
    assert redis_cache.get("h2") is not None


def test_lru_eviction(redis_cache):
    """Test that max_size is enforced."""
    from stateloom.cache.redis_store import RedisCacheStore

    url = os.environ["STATELOOM_TEST_REDIS_URL"]
    small_store = RedisCacheStore(url=url, max_size=3)
    small_store.clear()

    for i in range(5):
        small_store.put(_make_entry(request_hash=f"h{i}"))

    assert small_store.size() == 3
    small_store.clear()
    small_store.close()


def test_embedding_roundtrip(redis_cache):
    """Test that embeddings survive serialization."""
    embedding = [0.1, 0.2, 0.3, 0.4]
    entry = CacheEntry(
        request_hash="emb1",
        session_id="s1",
        response_json="{}",
        model="gpt-4o",
        provider="openai",
        cost=0.0,
        created_at=time.time(),
        embedding=embedding,
    )
    redis_cache.put(entry)

    retrieved = redis_cache.get("emb1")
    assert retrieved is not None
    assert retrieved.embedding == embedding
