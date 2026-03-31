"""Tests for the semantic matcher (requires faiss-cpu + sentence-transformers)."""

from __future__ import annotations

import time

import pytest

faiss = pytest.importorskip("faiss")
pytest.importorskip("sentence_transformers")

from stateloom.cache.base import CacheEntry
from stateloom.cache.semantic import SemanticMatcher, is_semantic_available


def _make_entry(
    request_hash: str = "abc123",
    session_id: str = "sess-1",
    embedding: list[float] | None = None,
) -> CacheEntry:
    return CacheEntry(
        request_hash=request_hash,
        session_id=session_id,
        response_json='{"text":"hello"}',
        model="gpt-4",
        provider="openai",
        cost=0.01,
        created_at=time.time(),
        embedding=embedding,
    )


class TestSemanticMatcher:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.matcher = SemanticMatcher(model_name="all-MiniLM-L6-v2")

    def test_is_available(self):
        assert is_semantic_available() is True

    def test_embed_request(self):
        kwargs = {
            "model": "gpt-4",
            "messages": [{"role": "user", "content": "What is the capital of France?"}],
        }
        embedding = self.matcher.embed_request(kwargs)
        assert isinstance(embedding, list)
        assert len(embedding) > 0
        # All-MiniLM-L6-v2 has 384 dims
        assert len(embedding) == 384

    def test_add_and_search(self):
        kwargs = {
            "model": "gpt-4",
            "messages": [{"role": "user", "content": "What is the capital of France?"}],
        }
        embedding = self.matcher.embed_request(kwargs)
        entry = _make_entry(embedding=embedding)
        self.matcher.add(entry)

        # Exact same query → high similarity
        result, score = self.matcher.search(embedding)
        assert result is not None
        assert score > 0.99  # Should be almost 1.0 for exact match

    def test_similar_queries_match(self):
        kwargs1 = {
            "model": "gpt-4",
            "messages": [{"role": "user", "content": "What is the capital of France?"}],
        }
        kwargs2 = {
            "model": "gpt-4",
            "messages": [{"role": "user", "content": "Tell me the capital city of France"}],
        }

        emb1 = self.matcher.embed_request(kwargs1)
        entry = _make_entry(request_hash="h1", embedding=emb1)
        self.matcher.add(entry)

        emb2 = self.matcher.embed_request(kwargs2)
        result, score = self.matcher.search(emb2)
        assert result is not None
        # Paraphrased questions should have high similarity
        assert score > 0.7

    def test_dissimilar_queries_low_score(self):
        kwargs1 = {
            "model": "gpt-4",
            "messages": [{"role": "user", "content": "What is the capital of France?"}],
        }
        kwargs2 = {
            "model": "gpt-4",
            "messages": [{"role": "user", "content": "Write a Python function to sort a list"}],
        }

        emb1 = self.matcher.embed_request(kwargs1)
        entry = _make_entry(request_hash="h1", embedding=emb1)
        self.matcher.add(entry)

        emb2 = self.matcher.embed_request(kwargs2)
        result, score = self.matcher.search(emb2)
        # Different topics → lower similarity
        assert score < 0.7

    def test_session_filter(self):
        kwargs = {
            "model": "gpt-4",
            "messages": [{"role": "user", "content": "Hello world"}],
        }
        emb = self.matcher.embed_request(kwargs)
        entry = _make_entry(session_id="sess-1", embedding=emb)
        self.matcher.add(entry)

        # Search with matching session → hit
        result, score = self.matcher.search(emb, session_id="sess-1")
        assert result is not None

        # Search with different session → miss
        result, score = self.matcher.search(emb, session_id="sess-2")
        assert result is None

    def test_empty_index_search(self):
        emb = self.matcher.embed_request({"messages": [{"role": "user", "content": "test"}]})
        result, score = self.matcher.search(emb)
        assert result is None
        assert score == 0.0

    def test_rebuild_from_entries(self):
        kwargs1 = {
            "messages": [{"role": "user", "content": "What is AI?"}],
        }
        kwargs2 = {
            "messages": [{"role": "user", "content": "How does machine learning work?"}],
        }

        emb1 = self.matcher.embed_request(kwargs1)
        emb2 = self.matcher.embed_request(kwargs2)

        entries = [
            _make_entry(request_hash="h1", embedding=emb1),
            _make_entry(request_hash="h2", embedding=emb2),
        ]
        self.matcher.rebuild_from_entries(entries)

        # Should find entries
        result, score = self.matcher.search(emb1)
        assert result is not None
        assert result.request_hash == "h1"

    def test_rebuild_skips_entries_without_embeddings(self):
        entries = [
            _make_entry(request_hash="h1", embedding=None),
            _make_entry(request_hash="h2", embedding=[0.1] * 384),
        ]
        self.matcher.rebuild_from_entries(entries)
        # Only one entry with embedding added
        assert len(self.matcher._entries) == 1

    def test_add_skips_no_embedding(self):
        entry = _make_entry(embedding=None)
        self.matcher.add(entry)
        assert len(self.matcher._entries) == 0

    def test_extract_text_gemini_format(self):
        kwargs = {
            "model": "gemini-pro",
            "contents": [{"text": "Hello from Gemini"}],
        }
        text = self.matcher._extract_text(kwargs)
        assert "Hello from Gemini" in text
        assert "gemini-pro" in text
