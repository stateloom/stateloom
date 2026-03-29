"""Semantic complexity classifier for the auto-router.

Uses zero-shot NLI (Natural Language Inference) via a CrossEncoder model
to classify prompt complexity. Tests each prompt against two hypotheses
("requires advanced reasoning" vs "simple task") and returns a normalized
complexity score. Falls back to None (triggering heuristic fallback)
when sentence-transformers is not available.
"""

from __future__ import annotations

import logging
import threading
from typing import Any

logger = logging.getLogger("stateloom.middleware.semantic_router")

# Guard optional dependency
try:
    from sentence_transformers import CrossEncoder

    _ST_AVAILABLE = True
except ImportError:
    _ST_AVAILABLE = False

# Default NLI hypotheses
_COMPLEX_HYPOTHESIS = "This requires advanced reasoning and expertise"
_SIMPLE_HYPOTHESIS = "This is a simple, straightforward task"


class SemanticComplexityClassifier:
    """Classifies prompt complexity using zero-shot NLI.

    Uses a CrossEncoder model to test each prompt against two hypotheses:
    one for complex tasks and one for simple tasks. Returns a 0.0-1.0
    score where:
    - 0.0 = very simple (strongly entails the simple hypothesis)
    - 1.0 = very complex (strongly entails the complex hypothesis)

    Returns None when sentence-transformers is not available or on any error.
    """

    def __init__(
        self,
        *,
        model_name: str = "cross-encoder/nli-MiniLM2-L6-H768",
        complex_hypothesis: str | None = None,
        simple_hypothesis: str | None = None,
    ) -> None:
        self._model_name = model_name
        self._complex_hypothesis = complex_hypothesis or _COMPLEX_HYPOTHESIS
        self._simple_hypothesis = simple_hypothesis or _SIMPLE_HYPOTHESIS

        # Lazy-initialized state
        self._initialized = False
        self._init_lock = threading.Lock()
        self._model: Any = None
        self._backend: str = ""  # "sentence_transformers" or ""

    @property
    def is_available(self) -> bool:
        """Check if the NLI backend is available without initializing."""
        return _ST_AVAILABLE

    def _lazy_init(self) -> bool:
        """Initialize the CrossEncoder on first use. Returns True if successful."""
        if self._initialized:
            return self._backend != ""

        with self._init_lock:
            if self._initialized:
                return self._backend != ""

            try:
                if _ST_AVAILABLE:
                    self._model = CrossEncoder(self._model_name)
                    self._backend = "sentence_transformers"
                    self._initialized = True
                    return True
            except Exception:
                logger.debug("CrossEncoder init failed", exc_info=True)

            if not _ST_AVAILABLE:
                logger.info(
                    "sentence-transformers not installed — semantic complexity "
                    "scoring disabled, using heuristic fallback. "
                    "Install with: pip install stateloom[semantic]"
                )
            self._initialized = True
            return False

    def classify(self, text: str) -> float | None:
        """Classify text complexity on a 0.0-1.0 scale.

        Returns None if no NLI backend is available or on any error.
        """
        try:
            if not self._lazy_init():
                return None

            scores = self._model.predict(
                [
                    (text, self._complex_hypothesis),
                    (text, self._simple_hypothesis),
                ]
            )

            complex_score = float(scores[0])
            simple_score = float(scores[1])

            # Avoid division by zero
            total = complex_score + simple_score
            if total <= 0:
                return 0.5

            return complex_score / total
        except Exception:
            logger.debug("Semantic classification failed", exc_info=True)
            return None
