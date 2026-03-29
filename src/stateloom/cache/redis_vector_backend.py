"""Redis Vector Search backend for distributed semantic cache.

Uses Redis with the RediSearch module for vector similarity search,
enabling shared semantic cache across multiple nodes in a cluster.
"""

from __future__ import annotations

import logging

logger = logging.getLogger("stateloom.cache.redis_vector_backend")

# Guard optional dependencies
try:
    import numpy as np
    import redis
    from redis.commands.search.field import TagField, VectorField
    from redis.commands.search.indexDefinition import IndexDefinition, IndexType
    from redis.commands.search.query import Query

    _REDIS_VECTOR_AVAILABLE = True
except ImportError:
    _REDIS_VECTOR_AVAILABLE = False


class RedisVectorBackend:
    """Redis Vector Search backend using RediSearch.

    Stores vectors as Redis hashes with a RediSearch vector index for
    KNN similarity search. Supports session_id filtering via TagField.
    """

    def __init__(
        self,
        url: str = "redis://localhost:6379",
        index_name: str = "stateloom_vectors",
        dimension: int = 384,
    ) -> None:
        if not _REDIS_VECTOR_AVAILABLE:
            raise ImportError(
                "RedisVectorBackend requires redis[search] and numpy. "
                "Install with: pip install stateloom[redis]"
            )
        self._client = redis.from_url(url)
        self._index_name = index_name
        self._dim = dimension
        self._prefix = "agvec:"
        self._ensure_index()

    def _ensure_index(self) -> None:
        """Create the RediSearch index if it doesn't exist."""
        try:
            self._client.ft(self._index_name).info()
        except redis.ResponseError:
            schema = (
                VectorField(
                    "embedding",
                    "FLAT",
                    {
                        "TYPE": "FLOAT32",
                        "DIM": self._dim,
                        "DISTANCE_METRIC": "COSINE",
                    },
                ),
                TagField("session_id"),
            )
            definition = IndexDefinition(
                prefix=[self._prefix],
                index_type=IndexType.HASH,
            )
            self._client.ft(self._index_name).create_index(schema, definition=definition)

    def add(self, entry_id: str, embedding: list[float], metadata: dict) -> None:
        key = f"{self._prefix}{entry_id}"
        vec_bytes = np.array(embedding, dtype=np.float32).tobytes()
        self._client.hset(
            key,
            mapping={
                "embedding": vec_bytes,
                "session_id": metadata.get("session_id", ""),
            },
        )

    def search(
        self,
        embedding: list[float],
        top_k: int = 10,
        session_id: str | None = None,
    ) -> list[tuple[str, float]]:
        vec_bytes = np.array(embedding, dtype=np.float32).tobytes()

        # Build query with optional session_id filter
        if session_id:
            filter_expr = f"@session_id:{{{session_id}}}"
        else:
            filter_expr = "*"

        q = (
            Query(f"({filter_expr})=>[KNN {top_k} @embedding $vec AS score]")
            .sort_by("score")
            .return_fields("score")
            .dialect(2)
        )

        results = self._client.ft(self._index_name).search(q, query_params={"vec": vec_bytes})

        entries: list[tuple[str, float]] = []
        for doc in results.docs:
            entry_id = doc.id.removeprefix(self._prefix)
            # RediSearch COSINE returns distance (0=identical, 2=opposite)
            # Convert to similarity: similarity = 1 - distance
            distance = float(doc.score)
            similarity = 1.0 - distance
            entries.append((entry_id, similarity))
        return entries

    def remove(self, entry_id: str) -> None:
        self._client.delete(f"{self._prefix}{entry_id}")

    def reset(self) -> None:
        cursor = 0
        while True:
            cursor, keys = self._client.scan(cursor=cursor, match=f"{self._prefix}*", count=100)
            if keys:
                self._client.delete(*keys)
            if cursor == 0:
                break

    def rebuild(self, entries: list[tuple[str, list[float], dict]]) -> None:
        self.reset()
        if not entries:
            return
        pipe = self._client.pipeline(transaction=False)
        for entry_id, embedding, metadata in entries:
            key = f"{self._prefix}{entry_id}"
            vec_bytes = np.array(embedding, dtype=np.float32).tobytes()
            pipe.hset(
                key,
                mapping={
                    "embedding": vec_bytes,
                    "session_id": metadata.get("session_id", ""),
                },
            )
        pipe.execute()

    @property
    def size(self) -> int:
        count = 0
        cursor = 0
        while True:
            cursor, keys = self._client.scan(cursor=cursor, match=f"{self._prefix}*", count=100)
            count += len(keys)
            if cursor == 0:
                break
        return count

    def close(self) -> None:
        self._client.close()
