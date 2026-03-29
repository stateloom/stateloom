"""Tests for the semantic complexity classifier."""

from __future__ import annotations

import threading
from unittest.mock import MagicMock, patch

import pytest

from stateloom.middleware.semantic_router import (
    SemanticComplexityClassifier,
)


class TestSemanticComplexityClassifier:
    """Tests for SemanticComplexityClassifier."""

    def test_classify_returns_none_when_unavailable(self):
        """Without sentence-transformers, classify returns None."""
        with patch("stateloom.middleware.semantic_router._ST_AVAILABLE", False):
            classifier = SemanticComplexityClassifier()
            assert classifier.classify("What is 2 + 2?") is None

    def test_is_available_false_when_no_backends(self):
        """is_available is False when sentence-transformers not installed."""
        with patch("stateloom.middleware.semantic_router._ST_AVAILABLE", False):
            classifier = SemanticComplexityClassifier()
            assert classifier.is_available is False

    def test_is_available_true_with_st(self):
        """is_available is True when sentence-transformers is available."""
        with patch("stateloom.middleware.semantic_router._ST_AVAILABLE", True):
            classifier = SemanticComplexityClassifier()
            assert classifier.is_available is True

    def test_lazy_initialization(self):
        """Classifier is not initialized until first classify() call."""
        with patch("stateloom.middleware.semantic_router._ST_AVAILABLE", False):
            classifier = SemanticComplexityClassifier()
            assert classifier._initialized is False
            classifier.classify("test")
            assert classifier._initialized is True

    def test_predict_failure_returns_none(self):
        """If the model raises during predict, classify returns None."""
        mock_model = MagicMock()
        mock_model.predict.side_effect = RuntimeError("model broken")

        with patch("stateloom.middleware.semantic_router._ST_AVAILABLE", True):
            classifier = SemanticComplexityClassifier()
            # Inject the mock model directly
            classifier._model = mock_model
            classifier._backend = "sentence_transformers"
            classifier._initialized = True
            result = classifier.classify("test prompt")
            assert result is None

    def test_thread_safety(self):
        """Concurrent classify calls don't crash."""
        with patch("stateloom.middleware.semantic_router._ST_AVAILABLE", False):
            classifier = SemanticComplexityClassifier()
            results = []
            errors = []

            def worker():
                try:
                    r = classifier.classify("What is 2 + 2?")
                    results.append(r)
                except Exception as e:
                    errors.append(e)

            threads = [threading.Thread(target=worker) for _ in range(10)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

            assert len(errors) == 0
            assert len(results) == 10
            # All None since no backend available
            assert all(r is None for r in results)

    def test_backend_set_to_empty_when_all_fail(self):
        """Backend stays empty when all backends fail."""
        with patch("stateloom.middleware.semantic_router._ST_AVAILABLE", False):
            classifier = SemanticComplexityClassifier()
            classifier._lazy_init()
            assert classifier._backend == ""
            assert classifier._initialized is True

    def test_custom_hypotheses(self):
        """Custom hypothesis text is accepted and stored."""
        classifier = SemanticComplexityClassifier(
            complex_hypothesis="This is a hard task",
            simple_hypothesis="This is an easy task",
        )
        assert classifier._complex_hypothesis == "This is a hard task"
        assert classifier._simple_hypothesis == "This is an easy task"

    def test_default_hypotheses(self):
        """Default hypotheses are used when none provided."""
        classifier = SemanticComplexityClassifier()
        assert "advanced reasoning" in classifier._complex_hypothesis
        assert "simple" in classifier._simple_hypothesis

    def test_classify_with_mock_model(self):
        """Verify classify computes correct score from model predictions."""
        mock_model = MagicMock()
        # complex_score=0.2, simple_score=0.8 → complexity = 0.2/1.0 = 0.2
        mock_model.predict.return_value = [0.2, 0.8]

        with patch("stateloom.middleware.semantic_router._ST_AVAILABLE", True):
            classifier = SemanticComplexityClassifier()
            classifier._model = mock_model
            classifier._backend = "sentence_transformers"
            classifier._initialized = True
            score = classifier.classify("What is 2 + 2?")
            assert score is not None
            assert abs(score - 0.2) < 1e-6

    def test_classify_with_equal_scores(self):
        """Equal scores should return 0.5."""
        mock_model = MagicMock()
        mock_model.predict.return_value = [0.5, 0.5]

        with patch("stateloom.middleware.semantic_router._ST_AVAILABLE", True):
            classifier = SemanticComplexityClassifier()
            classifier._model = mock_model
            classifier._backend = "sentence_transformers"
            classifier._initialized = True
            score = classifier.classify("ambiguous prompt")
            assert score is not None
            assert abs(score - 0.5) < 1e-6

    def test_classify_with_zero_scores(self):
        """Both scores zero should return 0.5 fallback."""
        mock_model = MagicMock()
        mock_model.predict.return_value = [0.0, 0.0]

        with patch("stateloom.middleware.semantic_router._ST_AVAILABLE", True):
            classifier = SemanticComplexityClassifier()
            classifier._model = mock_model
            classifier._backend = "sentence_transformers"
            classifier._initialized = True
            score = classifier.classify("test")
            assert score == 0.5


class TestSemanticComplexityWithSentenceTransformers:
    """Tests requiring sentence-transformers (skipped if not installed)."""

    @pytest.fixture(autouse=True)
    def _require_st(self):
        pytest.importorskip("sentence_transformers")

    def test_classify_simple_prompt_low_score(self):
        """Simple prompts should produce low complexity scores."""
        classifier = SemanticComplexityClassifier()
        score = classifier.classify("What is 2 + 2?")
        assert score is not None
        assert score < 0.55  # Should lean toward simple

    def test_classify_complex_prompt_high_score(self):
        """Complex prompts should produce high complexity scores."""
        classifier = SemanticComplexityClassifier()
        score = classifier.classify(
            "Design a distributed consensus algorithm that handles Byzantine faults "
            "and prove its correctness using formal verification methods."
        )
        assert score is not None
        assert score > 0.45  # Should lean toward complex

    def test_score_range(self):
        """All scores should be in [0.0, 1.0] range."""
        classifier = SemanticComplexityClassifier()
        prompts = [
            "Hi",
            "What is 1+1?",
            "Explain the theory of relativity in detail",
            "Write a compiler for a functional programming language",
        ]
        for prompt in prompts:
            score = classifier.classify(prompt)
            assert score is not None
            assert 0.0 <= score <= 1.0


class TestFallbackIntegration:
    """Integration test: semantic unavailable falls back to heuristic."""

    def test_fallback_to_heuristic_when_semantic_unavailable(self):
        """When classifier returns None, auto-router should use heuristic."""
        from stateloom.core.config import StateLoomConfig
        from stateloom.core.session import Session
        from stateloom.middleware.auto_router import AutoRouterMiddleware
        from stateloom.middleware.base import MiddlewareContext
        from stateloom.store.memory_store import MemoryStore

        config = StateLoomConfig(
            auto_route_enabled=True,
            auto_route_model="llama3.2",
            local_model_enabled=True,
            local_model_default="llama3.2",
            console_output=False,
            auto_route_semantic_enabled=False,
        )
        store = MemoryStore()

        with patch("stateloom.middleware.auto_router.OllamaClient") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.is_available.return_value = True
            mock_client_cls.return_value = mock_client

            router = AutoRouterMiddleware(config, store)

            # Semantic should be disabled
            assert router._semantic_classifier is None

            session = Session(id="test")
            ctx = MiddlewareContext(
                session=session,
                config=config,
                provider="openai",
                model="gpt-4",
                request_kwargs={"messages": [{"role": "user", "content": "hello"}]},
            )

            decision = router._should_route_local(ctx)
            # Should still get a complexity score from heuristics
            assert decision.semantic_complexity is None
