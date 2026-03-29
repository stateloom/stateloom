"""Redis-backed cache store for StateLoom.

Uses Redis hashes for per-entry storage, sorted sets for TTL eviction,
and per-session sets for scoped lookups. Install with: pip install stateloom[redis]
"""

from __future__ import annotations

import json
import logging
import time

from stateloom.cache.base import CacheEntry

logger = logging.getLogger("stateloom.cache.redis_store")

try:
    import redis as redis_lib

    _REDIS_AVAILABLE = True
except ImportError:
    _REDIS_AVAILABLE = False

# Key prefixes
_ENTRY_PREFIX = "agcache:"
_INDEX_KEY = "agcache:index"
_SESSION_PREFIX = "agcache:session:"


class RedisCacheStore:
    """Redis-backed cache store implementing the CacheStore protocol.

    Data model:
      - Hash per entry: agcache:{request_hash} → session_id, response_json,
        model, provider, cost, created_at, embedding_json
      - Sorted set: agcache:index → request_hashes scored by created_at
      - Set per session: agcache:session:{session_id} → request_hashes
    """

    def __init__(
        self,
        url: str = "redis://localhost:6379",
        max_size: int = 1000,
    ) -> None:
        if not _REDIS_AVAILABLE:
            raise ImportError(
                "Redis cache store requires the redis package. "
                "Install with: pip install stateloom[redis]"
            )
        self._url = url
        self._max_size = max_size
        self._redis = redis_lib.Redis.from_url(url, decode_responses=True)
        # Verify connectivity
        self._redis.ping()

    def get(self, request_hash: str, session_id: str | None = None) -> CacheEntry | None:
        key = f"{_ENTRY_PREFIX}{request_hash}"
        data = self._redis.hgetall(key)
        if not data:
            return None

        # Session scoping: if session_id given, only return if it matches
        if session_id and data.get("session_id") != session_id:
            return None

        return self._data_to_entry(request_hash, data)

    def put(self, entry: CacheEntry) -> None:
        key = f"{_ENTRY_PREFIX}{entry.request_hash}"

        embedding_json = json.dumps(entry.embedding) if entry.embedding else ""

        pipe = self._redis.pipeline()
        pipe.hset(
            key,
            mapping={
                "session_id": entry.session_id,
                "response_json": entry.response_json,
                "model": entry.model,
                "provider": entry.provider,
                "cost": str(entry.cost),
                "created_at": str(entry.created_at),
                "embedding_json": embedding_json,
            },
        )
        pipe.zadd(_INDEX_KEY, {entry.request_hash: entry.created_at})
        if entry.session_id:
            pipe.sadd(f"{_SESSION_PREFIX}{entry.session_id}", entry.request_hash)
        pipe.execute()

        # LRU eviction if over max_size
        self._evict_lru()

    def get_all_entries(self, session_id: str | None = None) -> list[CacheEntry]:
        if session_id:
            hashes = self._redis.smembers(f"{_SESSION_PREFIX}{session_id}")
        else:
            hashes = self._redis.zrange(_INDEX_KEY, 0, -1)

        entries: list[CacheEntry] = []
        for request_hash in hashes:
            data = self._redis.hgetall(f"{_ENTRY_PREFIX}{request_hash}")
            if data:
                entries.append(self._data_to_entry(request_hash, data))
        return entries

    def evict_expired(self, ttl: float) -> int:
        """Remove entries older than ttl seconds. Returns count evicted."""
        cutoff = time.time() - ttl
        # Get all hashes with score (created_at) below the cutoff
        expired_hashes = self._redis.zrangebyscore(_INDEX_KEY, "-inf", cutoff)
        if not expired_hashes:
            return 0

        pipe = self._redis.pipeline()
        for request_hash in expired_hashes:
            entry_key = f"{_ENTRY_PREFIX}{request_hash}"
            # Get session_id before deleting so we can clean up session set
            session_id = self._redis.hget(entry_key, "session_id")
            pipe.delete(entry_key)
            pipe.zrem(_INDEX_KEY, request_hash)
            if session_id:
                pipe.srem(f"{_SESSION_PREFIX}{session_id}", request_hash)
        pipe.execute()
        return len(expired_hashes)

    def size(self) -> int:
        return self._redis.zcard(_INDEX_KEY)

    def clear(self) -> None:
        # Get all hashes in the index
        all_hashes = self._redis.zrange(_INDEX_KEY, 0, -1)
        if not all_hashes:
            # Still delete the index key itself
            self._redis.delete(_INDEX_KEY)
            return

        # Collect session keys to delete
        session_keys: set[str] = set()
        pipe = self._redis.pipeline()
        for request_hash in all_hashes:
            entry_key = f"{_ENTRY_PREFIX}{request_hash}"
            session_id = self._redis.hget(entry_key, "session_id")
            if session_id:
                session_keys.add(f"{_SESSION_PREFIX}{session_id}")
            pipe.delete(entry_key)
        pipe.delete(_INDEX_KEY)
        for sk in session_keys:
            pipe.delete(sk)
        pipe.execute()

    def purge_by_content(self, identifier: str) -> int:
        """Delete cache entries containing the identifier in response_json."""
        all_hashes = self._redis.zrange(_INDEX_KEY, 0, -1)
        deleted = 0

        for request_hash in all_hashes:
            entry_key = f"{_ENTRY_PREFIX}{request_hash}"
            response_json = self._redis.hget(entry_key, "response_json")
            if response_json and identifier in response_json:
                session_id = self._redis.hget(entry_key, "session_id")
                pipe = self._redis.pipeline()
                pipe.delete(entry_key)
                pipe.zrem(_INDEX_KEY, request_hash)
                if session_id:
                    pipe.srem(f"{_SESSION_PREFIX}{session_id}", request_hash)
                pipe.execute()
                deleted += 1

        return deleted

    def close(self) -> None:
        self._redis.close()

    # --- Internal helpers ---

    def _data_to_entry(self, request_hash: str, data: dict) -> CacheEntry:
        """Convert a Redis hash dict to a CacheEntry."""
        embedding = None
        embedding_raw = data.get("embedding_json", "")
        if embedding_raw:
            try:
                embedding = json.loads(embedding_raw)
            except (json.JSONDecodeError, TypeError):
                pass

        return CacheEntry(
            request_hash=request_hash,
            session_id=data.get("session_id", ""),
            response_json=data.get("response_json", ""),
            model=data.get("model", ""),
            provider=data.get("provider", ""),
            cost=float(data.get("cost", 0.0)),
            created_at=float(data.get("created_at", 0.0)),
            embedding=embedding,
        )

    def _evict_lru(self) -> None:
        """Evict oldest entries if over max_size."""
        current_size = self._redis.zcard(_INDEX_KEY)
        if current_size <= self._max_size:
            return

        # Remove the oldest entries (lowest scores)
        excess = current_size - self._max_size
        oldest = self._redis.zrange(_INDEX_KEY, 0, excess - 1)

        pipe = self._redis.pipeline()
        for request_hash in oldest:
            entry_key = f"{_ENTRY_PREFIX}{request_hash}"
            session_id = self._redis.hget(entry_key, "session_id")
            pipe.delete(entry_key)
            pipe.zrem(_INDEX_KEY, request_hash)
            if session_id:
                pipe.srem(f"{_SESSION_PREFIX}{session_id}", request_hash)
        pipe.execute()
