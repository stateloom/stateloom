"""Vector storage backend abstraction for semantic cache.

Separates vector storage/search from embedding generation so that
SemanticMatcher can delegate to pluggable backends (FAISS, Redis, etc.).
"""

from __future__ import annotations

import logging
import threading
from typing import Any, Protocol, cast, runtime_checkable

logger = logging.getLogger("stateloom.cache.vector_backend")

# Guard optional dependencies
try:
    import faiss  # type: ignore[import-untyped]
    import numpy as np

    _FAISS_AVAILABLE = True
except ImportError:
    _FAISS_AVAILABLE = False


@runtime_checkable
class VectorBackend(Protocol):
    """Protocol for vector storage backends."""

    def add(self, entry_id: str, embedding: list[float], metadata: dict[str, Any]) -> None:
        """Add a vector to the index."""
        ...

    def search(
        self,
        embedding: list[float],
        top_k: int = 10,
        session_id: str | None = None,
    ) -> list[tuple[str, float]]:
        """Search for similar vectors. Returns list of (entry_id, score)."""
        ...

    def remove(self, entry_id: str) -> None:
        """Remove a vector by entry_id."""
        ...

    def reset(self) -> None:
        """Clear all vectors from the index."""
        ...

    def rebuild(self, entries: list[tuple[str, list[float], dict[str, Any]]]) -> None:
        """Bulk rebuild the index from (entry_id, embedding, metadata) tuples."""
        ...

    @property
    def size(self) -> int:
        """Number of vectors in the index."""
        ...

    def close(self) -> None:
        """Release resources."""
        ...


class FaissBackend:
    """FAISS IndexFlatIP vector backend (in-memory, single-node).

    Uses inner product on L2-normalized vectors for cosine similarity.
    """

    def __init__(self, dimension: int) -> None:
        if not _FAISS_AVAILABLE:
            raise ImportError(
                "FaissBackend requires faiss-cpu and numpy. "
                "Install with: pip install stateloom[semantic]"
            )
        self._dim = dimension
        self._index = faiss.IndexFlatIP(dimension)
        self._entry_ids: list[str] = []
        self._session_ids: list[str] = []
        self._lock = threading.Lock()

    def add(self, entry_id: str, embedding: list[float], metadata: dict[str, Any]) -> None:
        vec = np.array([embedding], dtype=np.float32)
        with self._lock:
            self._index.add(vec)
            self._entry_ids.append(entry_id)
            self._session_ids.append(metadata.get("session_id", ""))

    def search(
        self,
        embedding: list[float],
        top_k: int = 10,
        session_id: str | None = None,
    ) -> list[tuple[str, float]]:
        with self._lock:
            if self._index.ntotal == 0:
                return []
            vec = np.array([embedding], dtype=np.float32)
            k = min(top_k, self._index.ntotal)
            scores, indices = self._index.search(vec, k)
            results: list[tuple[str, float]] = []
            for score, idx in zip(scores[0], indices[0]):
                if idx < 0:
                    continue
                if session_id and self._session_ids[idx] != session_id:
                    continue
                results.append((self._entry_ids[idx], float(score)))
            return results

    def remove(self, entry_id: str) -> None:
        with self._lock:
            if entry_id not in self._entry_ids:
                return
            idx = self._entry_ids.index(entry_id)
            self._entry_ids.pop(idx)
            self._session_ids.pop(idx)
            # FAISS IndexFlatIP doesn't support single-vector removal — rebuild
            if self._index.ntotal > 0:
                all_vecs = (
                    faiss.rev_swig_ptr(self._index.get_xb(), self._index.ntotal * self._index.d)
                    .reshape(self._index.ntotal, self._index.d)
                    .copy()
                )
                keep_vecs = np.delete(all_vecs, idx, axis=0)
                self._index.reset()
                if len(keep_vecs) > 0:
                    self._index.add(keep_vecs)

    def reset(self) -> None:
        with self._lock:
            self._index.reset()
            self._entry_ids.clear()
            self._session_ids.clear()

    def rebuild(self, entries: list[tuple[str, list[float], dict[str, Any]]]) -> None:
        if not entries:
            self.reset()
            return
        vectors = np.array([emb for _, emb, _ in entries], dtype=np.float32)
        with self._lock:
            self._index.reset()
            self._entry_ids = [eid for eid, _, _ in entries]
            self._session_ids = [m.get("session_id", "") for _, _, m in entries]
            self._index.add(vectors)

    @property
    def size(self) -> int:
        return cast(int, self._index.ntotal)

    def close(self) -> None:
        pass
