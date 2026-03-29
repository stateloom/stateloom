"""Semantic similarity matching for the StateLoom cache using sentence-transformers.

Embedding generation is handled here; vector storage/search is delegated
to a pluggable ``VectorBackend`` (defaults to ``FaissBackend``).
"""

from __future__ import annotations

import logging
import threading
from typing import TYPE_CHECKING, Any

from stateloom.cache.base import CacheEntry

logger = logging.getLogger("stateloom.cache.semantic")

# Guard optional dependencies
try:
    from sentence_transformers import SentenceTransformer

    _SEMANTIC_AVAILABLE = True
except ImportError:
    _SEMANTIC_AVAILABLE = False

if TYPE_CHECKING:
    from stateloom.cache.vector_backend import VectorBackend


def is_semantic_available() -> bool:
    """Check if sentence-transformers is installed."""
    return _SEMANTIC_AVAILABLE


class SemanticMatcher:
    """Semantic cache matcher with pluggable vector backend.

    Embeds request content using sentence-transformers and delegates
    vector storage/search to a ``VectorBackend`` instance.  When no
    backend is provided, defaults to ``FaissBackend`` (preserving the
    original in-memory FAISS behavior).
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        vector_backend: Any | None = None,
    ) -> None:
        if not _SEMANTIC_AVAILABLE:
            raise ImportError(
                "Semantic cache requires sentence-transformers. "
                "Install with: pip install stateloom[semantic]"
            )

        self._model = SentenceTransformer(model_name)
        self._dim = self._model.get_sentence_embedding_dimension()

        # Pluggable vector backend
        if vector_backend is None:
            from stateloom.cache.vector_backend import FaissBackend

            vector_backend = FaissBackend(dimension=self._dim)
        self._backend: VectorBackend = vector_backend

        # Local entry lookup: entry_id (request_hash) -> CacheEntry
        self._entries: dict[str, CacheEntry] = {}
        self._lock = threading.Lock()

    def embed_request(self, request_kwargs: dict) -> list[float]:
        """Extract text from request kwargs and return a normalized embedding vector."""
        text = self._extract_text(request_kwargs)
        embedding = self._model.encode(text, normalize_embeddings=True)
        return embedding.tolist()

    def add(self, entry: CacheEntry) -> None:
        """Add a cache entry to the vector index."""
        if entry.embedding is None:
            return
        with self._lock:
            self._entries[entry.request_hash] = entry
        self._backend.add(
            entry.request_hash,
            entry.embedding,
            {"session_id": entry.session_id},
        )

    def search(
        self,
        embedding: list[float],
        session_id: str | None = None,
    ) -> tuple[CacheEntry | None, float]:
        """Search for the most similar cached entry.

        Returns (entry, score) or (None, 0.0) if no match.
        """
        results = self._backend.search(embedding, top_k=10, session_id=session_id)
        with self._lock:
            for entry_id, score in results:
                entry = self._entries.get(entry_id)
                if entry is not None:
                    return entry, score
        return None, 0.0

    def rebuild_from_entries(self, entries: list[CacheEntry]) -> None:
        """Bulk rebuild the vector index from persisted entries (startup path)."""
        tuples = [
            (e.request_hash, e.embedding, {"session_id": e.session_id})
            for e in entries
            if e.embedding is not None
        ]
        if tuples:
            self._backend.rebuild(tuples)
        with self._lock:
            self._entries = {e.request_hash: e for e in entries if e.embedding is not None}

    @staticmethod
    def _extract_text(request_kwargs: dict) -> str:
        """Extract message content from request kwargs for embedding.

        Includes model name as a prefix to differentiate requests
        targeting different models.
        """
        parts: list[str] = []

        # Include model as context prefix
        model = request_kwargs.get("model", "")
        if model:
            parts.append(f"model:{model}")

        # Extract messages (OpenAI/Anthropic format)
        messages = request_kwargs.get("messages", [])
        for msg in messages:
            if isinstance(msg, dict):
                content = msg.get("content", "")
                if isinstance(content, str):
                    parts.append(content)
                elif isinstance(content, list):
                    # Handle structured content blocks
                    for block in content:
                        if isinstance(block, dict) and "text" in block:
                            parts.append(block["text"])

        # Extract contents (Gemini format)
        contents = request_kwargs.get("contents", [])
        if isinstance(contents, str):
            parts.append(contents)
        elif isinstance(contents, list):
            for item in contents:
                if isinstance(item, str):
                    parts.append(item)
                elif isinstance(item, dict):
                    text = item.get("text", "")
                    if text:
                        parts.append(text)

        return " ".join(parts) if parts else ""
