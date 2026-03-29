"""NER-based PII detector using GLiNER for zero-shot entity recognition."""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger("stateloom.pii.ner")

try:
    from gliner import GLiNER

    _GLINER_AVAILABLE = True
except ImportError:
    _GLINER_AVAILABLE = False


@dataclass
class NERMatch:
    """A single NER entity detection."""

    entity_type: str  # "person", "location", etc.
    text: str
    start: int
    end: int
    score: float


class NERDetector:
    """GLiNER-based NER for detecting unstructured PII.

    Lazy-initializes the model on first call. Thread-safe.
    Gracefully returns empty results when gliner is not installed.
    """

    def __init__(
        self,
        model_name: str = "urchade/gliner_small-v2.1",
        labels: list[str] | None = None,
        threshold: float = 0.5,
    ) -> None:
        self._model_name = model_name
        self._labels = labels or ["person", "location", "organization", "address"]
        self._threshold = threshold
        self._model: Any = None
        self._lock = threading.Lock()
        self._initialized = False
        self._failed = False

    @property
    def is_available(self) -> bool:
        """Whether GLiNER is installed and model loading hasn't failed."""
        return _GLINER_AVAILABLE and not self._failed

    def _lazy_init(self) -> bool:
        """Load the model on first use. Returns True if model is ready."""
        if self._initialized:
            return self._model is not None
        with self._lock:
            if self._initialized:
                return self._model is not None
            if not _GLINER_AVAILABLE:
                logger.info(
                    "GLiNER not installed — NER-based PII detection disabled. "
                    "Install with: pip install stateloom[ner]"
                )
                self._initialized = True
                return False
            try:
                self._model = GLiNER.from_pretrained(self._model_name)
                self._initialized = True
                logger.info("NER model loaded: %s", self._model_name)
                return True
            except Exception:
                logger.warning("Failed to load NER model", exc_info=True)
                self._failed = True
                self._initialized = True
                return False

    def detect(self, text: str) -> list[NERMatch]:
        """Detect named entities in text. Returns empty list on failure."""
        if not self._lazy_init():
            return []
        try:
            entities = self._model.predict_entities(text, self._labels, threshold=self._threshold)
            return [
                NERMatch(
                    entity_type=e["label"],
                    text=e["text"],
                    start=e["start"],
                    end=e["end"],
                    score=e.get("score", 0.0),
                )
                for e in entities
            ]
        except Exception:
            logger.warning("NER detection failed", exc_info=True)
            return []
