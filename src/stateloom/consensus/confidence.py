"""Adapter-aware confidence extraction for the consensus framework."""

from __future__ import annotations

import logging
import re

from stateloom.consensus.prompts import DEFAULT_CONFIDENCE_INSTRUCTION

logger = logging.getLogger("stateloom.consensus.confidence")

_CONFIDENCE_RE = re.compile(r"\[Confidence:\s*([\d.]+)\]", re.IGNORECASE)


def get_confidence_instruction(provider: str = "") -> str:
    """Return the provider-preferred confidence instruction.

    Looks up the adapter via provider_registry; falls back to the default
    instruction if no adapter is found or the adapter has no override.
    """
    if provider:
        try:
            from stateloom.intercept.provider_registry import get_adapter

            adapter = get_adapter(provider)
            if adapter is not None:
                instruction = adapter.confidence_instruction()
                if instruction:
                    return instruction
        except Exception:
            logger.debug("Failed to get adapter for confidence instruction", exc_info=True)
    return DEFAULT_CONFIDENCE_INSTRUCTION


def extract_confidence(text: str, provider: str = "") -> float:
    """Extract confidence score from response text.

    Tries adapter-specific extraction first, then falls back to regex.
    Returns 0.5 if no confidence found.
    """
    if provider:
        try:
            from stateloom.intercept.provider_registry import get_adapter

            adapter = get_adapter(provider)
            if adapter is not None:
                result = adapter.extract_confidence(text)
                if result is not None:
                    return max(0.0, min(1.0, result))
        except Exception:
            logger.debug("Adapter confidence extraction failed", exc_info=True)

    # Default regex fallback
    match = _CONFIDENCE_RE.search(text)
    if match:
        try:
            val = float(match.group(1))
            return max(0.0, min(1.0, val))
        except (ValueError, IndexError):
            pass
    return 0.5
