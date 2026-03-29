"""Tests for RedisVectorBackend (mocked redis)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

np = pytest.importorskip("numpy")


class TestRedisVectorBackend:
    """Test RedisVectorBackend with mocked redis client."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Set up with mocked redis imports."""
        self.mock_redis = MagicMock()
        self.mock_client = MagicMock()
        self.mock_redis.from_url.return_value = self.mock_client

        # Mock the ft() search interface
        self.mock_ft = MagicMock()
        self.mock_client.ft.return_value = self.mock_ft

        # Simulate index already exists (no exception on info())
        self.mock_ft.info.return_value = {}

        with patch.dict(
            "sys.modules",
            {
                "redis": self.mock_redis,
                "redis.commands": MagicMock(),
                "redis.commands.search": MagicMock(),
                "redis.commands.search.field": MagicMock(),
                "redis.commands.search.indexDefinition": MagicMock(),
                "redis.commands.search.query": MagicMock(),
            },
        ):
            # Need to reload or import fresh
            import importlib

            import stateloom.cache.redis_vector_backend as rvb

            importlib.reload(rvb)
            self.rvb = rvb
            self.backend = rvb.RedisVectorBackend.__new__(rvb.RedisVectorBackend)
            self.backend._client = self.mock_client
            self.backend._index_name = "test_vectors"
            self.backend._dim = 4
            self.backend._prefix = "agvec:"

    def test_add_stores_hash(self):
        self.backend.add("entry-1", [1.0, 0.0, 0.0, 0.0], {"session_id": "s1"})
        self.mock_client.hset.assert_called_once()
        call_args = self.mock_client.hset.call_args
        assert call_args[0][0] == "agvec:entry-1"
        mapping = call_args[1]["mapping"]
        assert mapping["session_id"] == "s1"
        assert "embedding" in mapping

    def test_remove_deletes_key(self):
        self.backend.remove("entry-1")
        self.mock_client.delete.assert_called_once_with("agvec:entry-1")

    def test_reset_scans_and_deletes(self):
        self.mock_client.scan.return_value = (0, [b"agvec:e1", b"agvec:e2"])
        self.backend.reset()
        self.mock_client.scan.assert_called()
        self.mock_client.delete.assert_called()

    def test_close(self):
        self.backend.close()
        self.mock_client.close.assert_called_once()

    def test_rebuild_resets_and_adds(self):
        self.mock_client.scan.return_value = (0, [])
        mock_pipe = MagicMock()
        self.mock_client.pipeline.return_value = mock_pipe

        entries = [
            ("e1", [1.0, 0.0, 0.0, 0.0], {"session_id": "s1"}),
            ("e2", [0.0, 1.0, 0.0, 0.0], {"session_id": "s2"}),
        ]
        self.backend.rebuild(entries)
        assert mock_pipe.hset.call_count == 2
        mock_pipe.execute.assert_called_once()

    def test_search_builds_query(self):
        mock_result = MagicMock()
        mock_doc = MagicMock()
        mock_doc.id = "agvec:entry-1"
        mock_doc.score = 0.1  # distance
        mock_result.docs = [mock_doc]
        self.mock_ft.search.return_value = mock_result

        results = self.backend.search([1.0, 0.0, 0.0, 0.0])
        assert len(results) == 1
        assert results[0][0] == "entry-1"
        # similarity = 1 - distance = 0.9
        assert abs(results[0][1] - 0.9) < 0.01

    def test_size_counts_keys(self):
        self.mock_client.scan.return_value = (0, [b"agvec:e1", b"agvec:e2", b"agvec:e3"])
        assert self.backend.size == 3
