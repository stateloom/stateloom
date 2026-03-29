"""Tests for the VectorBackend protocol and FaissBackend implementation."""

from __future__ import annotations

import pytest

faiss = pytest.importorskip("faiss")
np = pytest.importorskip("numpy")

from stateloom.cache.vector_backend import FaissBackend, VectorBackend


class TestFaissBackend:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.dim = 4
        self.backend = FaissBackend(dimension=self.dim)

    def test_implements_protocol(self):
        assert isinstance(self.backend, VectorBackend)

    def test_add_and_search(self):
        vec = [1.0, 0.0, 0.0, 0.0]
        self.backend.add("entry-1", vec, {"session_id": "s1"})
        assert self.backend.size == 1

        results = self.backend.search(vec, top_k=5)
        assert len(results) == 1
        assert results[0][0] == "entry-1"
        assert results[0][1] > 0.99

    def test_search_empty_index(self):
        results = self.backend.search([1.0, 0.0, 0.0, 0.0])
        assert results == []

    def test_search_session_filter(self):
        vec = [1.0, 0.0, 0.0, 0.0]
        self.backend.add("entry-1", vec, {"session_id": "s1"})
        self.backend.add("entry-2", vec, {"session_id": "s2"})

        results = self.backend.search(vec, session_id="s1")
        assert len(results) == 1
        assert results[0][0] == "entry-1"

        results = self.backend.search(vec, session_id="s2")
        assert len(results) == 1
        assert results[0][0] == "entry-2"

        results = self.backend.search(vec, session_id="s3")
        assert len(results) == 0

    def test_remove(self):
        self.backend.add("entry-1", [1.0, 0.0, 0.0, 0.0], {})
        self.backend.add("entry-2", [0.0, 1.0, 0.0, 0.0], {})
        assert self.backend.size == 2

        self.backend.remove("entry-1")
        assert self.backend.size == 1

        results = self.backend.search([1.0, 0.0, 0.0, 0.0])
        assert len(results) == 1
        assert results[0][0] == "entry-2"

    def test_remove_nonexistent(self):
        self.backend.add("entry-1", [1.0, 0.0, 0.0, 0.0], {})
        self.backend.remove("nonexistent")
        assert self.backend.size == 1

    def test_reset(self):
        self.backend.add("entry-1", [1.0, 0.0, 0.0, 0.0], {})
        self.backend.add("entry-2", [0.0, 1.0, 0.0, 0.0], {})
        assert self.backend.size == 2

        self.backend.reset()
        assert self.backend.size == 0
        assert self.backend.search([1.0, 0.0, 0.0, 0.0]) == []

    def test_rebuild(self):
        entries = [
            ("e1", [1.0, 0.0, 0.0, 0.0], {"session_id": "s1"}),
            ("e2", [0.0, 1.0, 0.0, 0.0], {"session_id": "s2"}),
            ("e3", [0.0, 0.0, 1.0, 0.0], {"session_id": "s1"}),
        ]
        self.backend.rebuild(entries)
        assert self.backend.size == 3

        results = self.backend.search([1.0, 0.0, 0.0, 0.0], session_id="s1")
        assert len(results) >= 1
        assert results[0][0] == "e1"

    def test_rebuild_empty(self):
        self.backend.add("entry-1", [1.0, 0.0, 0.0, 0.0], {})
        self.backend.rebuild([])
        assert self.backend.size == 0

    def test_close(self):
        # Should not raise
        self.backend.close()

    def test_multiple_results_ordered_by_score(self):
        self.backend.add("e1", [1.0, 0.0, 0.0, 0.0], {})
        self.backend.add("e2", [0.9, 0.1, 0.0, 0.0], {})
        self.backend.add("e3", [0.0, 0.0, 0.0, 1.0], {})

        results = self.backend.search([1.0, 0.0, 0.0, 0.0], top_k=10)
        assert len(results) == 3
        # First result should be closest match
        assert results[0][0] == "e1"
        assert results[0][1] >= results[1][1]
