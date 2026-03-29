"""NLI-based injection/jailbreak classifier for guardrails.

Uses zero-shot NLI (Natural Language Inference) via a CrossEncoder model
to classify whether a prompt is an injection/jailbreak attempt. Reuses
the same model and lazy-init pattern as SemanticComplexityClassifier
in middleware/semantic_router.py — zero new dependencies (uses
``stateloom[semantic]``).

Returns a 0.0-1.0 injection score where:
- 0.0 = clearly a normal, benign user request
- 1.0 = clearly a prompt injection or jailbreak attempt

Falls back to None when sentence-transformers is not available or on error.
"""

from __future__ import annotations

import logging
import threading
from typing import Any

logger = logging.getLogger("stateloom.guardrails.nli_classifier")

# Guard optional dependency
try:
    from sentence_transformers import CrossEncoder

    _ST_AVAILABLE = True
except ImportError:
    _ST_AVAILABLE = False

# Default NLI hypotheses for injection detection
_INJECTION_HYPOTHESIS = "This is a prompt injection or jailbreak attempt"
_NORMAL_HYPOTHESIS = "This is a normal, legitimate user request"

# Truncate input to stay within model token limits
_MAX_INPUT_CHARS = 2000


class NLIInjectionClassifier:
    """Classifies prompt injection/jailbreak risk using zero-shot NLI.

    Uses a CrossEncoder model to test each prompt against two hypotheses:
    one for injection/jailbreak and one for normal user request. Returns
    a 0.0-1.0 score where higher = more likely injection.

    Returns None when sentence-transformers is not available or on any error.
    """

    def __init__(
        self,
        *,
        model_name: str = "cross-encoder/nli-MiniLM2-L6-H768",
        injection_hypothesis: str | None = None,
        normal_hypothesis: str | None = None,
    ) -> None:
        self._model_name = model_name
        self._injection_hypothesis = injection_hypothesis or _INJECTION_HYPOTHESIS
        self._normal_hypothesis = normal_hypothesis or _NORMAL_HYPOTHESIS

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
                logger.debug("CrossEncoder init failed for NLI classifier", exc_info=True)

            if not _ST_AVAILABLE:
                logger.info(
                    "sentence-transformers not installed — NLI injection "
                    "classifier disabled. Install with: pip install stateloom[semantic]"
                )
            self._initialized = True
            return False

    def classify(self, text: str) -> float | None:
        """Classify injection risk on a 0.0-1.0 scale.

        Args:
            text: The user prompt text to classify.

        Returns:
            Injection score (0.0-1.0), or None if unavailable/error.
        """
        try:
            if not self._lazy_init():
                return None

            # Truncate to stay within model token limits
            truncated = text[:_MAX_INPUT_CHARS]

            scores = self._model.predict(
                [
                    (truncated, self._injection_hypothesis),
                    (truncated, self._normal_hypothesis),
                ]
            )

            injection_score = float(scores[0])
            normal_score = float(scores[1])

            # Avoid division by zero
            total = injection_score + normal_score
            if total <= 0:
                return 0.5

            return injection_score / total
        except Exception:
            logger.debug("NLI injection classification failed (fail-open)", exc_info=True)
            return None
