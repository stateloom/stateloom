"""Similarity scoring for shadow draft comparison (cloud vs local)."""

from __future__ import annotations

import difflib
import logging
import threading
from dataclasses import dataclass
from typing import Any

try:
    from sentence_transformers import SentenceTransformer

    _SEMANTIC_AVAILABLE = True
except ImportError:
    _SEMANTIC_AVAILABLE = False

logger = logging.getLogger("stateloom.middleware.similarity")


@dataclass
class SimilarityResult:
    """Result of comparing cloud and local model responses."""

    score: float  # 0.0–1.0
    method: str  # "difflib"
    cloud_preview: str  # first 200 chars
    local_preview: str  # first 200 chars
    cloud_length: int
    local_length: int
    length_ratio: float  # min(len)/max(len)


def extract_response_text(response: Any, provider: str) -> str:
    """Extract plain text from an LLM response object.

    Handles ``ChatResponse`` (from ``stateloom.chat.Client``), native SDK
    objects, and Ollama responses. Falls back through registered adapters.
    Returns empty string on failure.
    """
    if response is None:
        return ""

    # Fast path: ChatResponse from stateloom.chat.Client
    try:
        from stateloom.chat import ChatResponse

        if isinstance(response, ChatResponse):
            return response.content or ""
    except ImportError:
        pass

    from stateloom.intercept.provider_registry import get_adapter, get_all_adapters

    # Try the specified provider's adapter first
    adapter = get_adapter(provider)
    if adapter is not None:
        text = adapter.extract_content(response)
        if text:
            return text

    # Fallback: try all registered adapters
    all_adapters = get_all_adapters()
    for name, fallback in all_adapters.items():
        if name == provider:
            continue
        text = fallback.extract_content(response)
        if text:
            return text

    # Structural fallback when registry is empty (tests, standalone use)
    if not all_adapters:
        text = _structural_fallback(response)
        if text:
            return text

    logger.debug(
        "extract_response_text: could not extract text from %s response (type=%s)",
        provider,
        type(response).__name__,
    )
    return ""


def _structural_fallback(response: Any) -> str:
    """Try builtin adapter classes directly when the registry is empty."""
    try:
        from stateloom.intercept.adapters.anthropic_adapter import AnthropicAdapter
        from stateloom.intercept.adapters.gemini_adapter import GeminiAdapter
        from stateloom.intercept.adapters.openai_adapter import OpenAIAdapter
        from stateloom.local.adapter import OllamaAdapter

        for cls in (OllamaAdapter, OpenAIAdapter, AnthropicAdapter, GeminiAdapter):
            adapter = cls()
            text = adapter.extract_content(response)
            if text:
                return text
    except Exception:
        pass
    return ""


def compute_similarity(
    cloud_text: str,
    local_text: str,
    preview_length: int = 200,
) -> SimilarityResult | None:
    """Compute similarity between cloud and local response text.

    Returns None if either text is empty.
    Uses difflib.SequenceMatcher on lowercased text.
    """
    if not cloud_text or not local_text:
        return None

    score = difflib.SequenceMatcher(None, cloud_text.lower(), local_text.lower()).ratio()

    cloud_len = len(cloud_text)
    local_len = len(local_text)
    max_len = max(cloud_len, local_len)
    min_len = min(cloud_len, local_len)
    length_ratio = min_len / max_len if max_len > 0 else 0.0

    return SimilarityResult(
        score=score,
        method="difflib",
        cloud_preview=cloud_text[:preview_length],
        local_preview=local_text[:preview_length],
        cloud_length=cloud_len,
        local_length=local_len,
        length_ratio=length_ratio,
    )


class SemanticSimilarityScorer:
    """Embedding-based cosine similarity scorer using sentence-transformers.

    Follows the same lazy-init + thread-safety pattern as
    ``SemanticComplexityClassifier`` in ``semantic_router.py``.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        self._model_name = model_name
        self._model: Any = None
        self._initialized = False
        self._init_lock = threading.Lock()

    @property
    def is_available(self) -> bool:
        return _SEMANTIC_AVAILABLE

    def _lazy_init(self) -> bool:
        if self._initialized:
            return self._model is not None
        with self._init_lock:
            if self._initialized:
                return self._model is not None
            if _SEMANTIC_AVAILABLE:
                try:
                    self._model = SentenceTransformer(self._model_name)
                except Exception:
                    logger.debug("SentenceTransformer init failed", exc_info=True)
            self._initialized = True
            return self._model is not None

    def score(self, text_a: str, text_b: str) -> float | None:
        """Compute cosine similarity between two texts via embeddings.

        Returns a float in [0, 1] or None if the model is unavailable.
        """
        if not self._lazy_init():
            return None
        try:
            embeddings = self._model.encode([text_a, text_b], normalize_embeddings=True)
            cos_sim = float(embeddings[0] @ embeddings[1])
            return max(0.0, min(1.0, cos_sim))
        except Exception:
            logger.debug("Semantic similarity scoring failed", exc_info=True)
            return None


def compute_semantic_similarity(
    cloud_text: str,
    local_text: str,
    scorer: SemanticSimilarityScorer,
    preview_length: int = 200,
) -> SimilarityResult | None:
    """Compute semantic (embedding-based) similarity between cloud and local text.

    Returns None if either text is empty or the scorer fails.
    """
    if not cloud_text or not local_text:
        return None

    sim_score = scorer.score(cloud_text, local_text)
    if sim_score is None:
        return None

    cloud_len = len(cloud_text)
    local_len = len(local_text)
    max_len = max(cloud_len, local_len)
    min_len = min(cloud_len, local_len)
    length_ratio = min_len / max_len if max_len > 0 else 0.0

    return SimilarityResult(
        score=sim_score,
        method="semantic",
        cloud_preview=cloud_text[:preview_length],
        local_preview=local_text[:preview_length],
        cloud_length=cloud_len,
        local_length=local_len,
        length_ratio=length_ratio,
    )


def compute_similarity_auto(
    cloud_text: str,
    local_text: str,
    *,
    method: str = "difflib",
    scorer: SemanticSimilarityScorer | None = None,
    preview_length: int = 200,
) -> SimilarityResult | None:
    """Dispatch similarity computation based on method.

    Args:
        method: ``"difflib"``, ``"semantic"``, or ``"auto"``.
        scorer: Required for ``"semantic"`` and ``"auto"`` methods.

    ``"auto"`` tries semantic first, falls back to difflib.
    """
    if method == "semantic":
        if scorer is None:
            return None
        return compute_semantic_similarity(cloud_text, local_text, scorer, preview_length)

    if method == "auto":
        if scorer is not None:
            result = compute_semantic_similarity(cloud_text, local_text, scorer, preview_length)
            if result is not None:
                return result
        return compute_similarity(cloud_text, local_text, preview_length)

    # Default: difflib
    return compute_similarity(cloud_text, local_text, preview_length)
