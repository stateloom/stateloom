"""Request normalization for cache key hashing.

Strips dynamic elements (UUIDs, timestamps, hex IDs, dates) from message
content before hashing so that semantically identical requests with
different dynamic values produce the same cache key.

Always-on — no config toggle. The LLM still sees the original request;
normalization only affects the cache key computation.
"""

from __future__ import annotations

import copy
import re
from typing import Any

_DEFAULT_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    (
        "uuid",
        re.compile(
            r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}" r"-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b"
        ),
    ),
    (
        "iso_ts",
        re.compile(r"\b\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}" r"(?:\.\d+)?(?:Z|[+-]\d{2}:?\d{2})?\b"),
    ),
    ("unix_ts", re.compile(r"\b1[6-9]\d{8}(?:\d{3})?\b")),
    (
        "date",
        re.compile(r"\b(?:\d{4}[-/]\d{2}[-/]\d{2}|\d{2}[-/.]\d{2}[-/.]\d{4})\b"),
    ),
    ("hex_id", re.compile(r"\b[0-9a-fA-F]{24,}\b")),
]

_PLACEHOLDER = "<DYN>"


class RequestNormalizer:
    """Normalizes request kwargs for cache key computation.

    Strips dynamic elements from message content using regex patterns.
    Returns a deep copy — the original request is never mutated.
    """

    def __init__(self, custom_patterns: list[str] | None = None) -> None:
        self._patterns = list(_DEFAULT_PATTERNS)
        if custom_patterns:
            for p in custom_patterns:
                self._patterns.append(("custom", re.compile(p)))

    def normalize_kwargs(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        """Return a deep copy with message content normalized.

        Only touches content fields — model, role, temperature, etc.
        are untouched.
        """
        normalized = copy.deepcopy(kwargs)

        # OpenAI/Anthropic messages
        for msg in normalized.get("messages", []):
            if isinstance(msg, dict):
                content = msg.get("content")
                if isinstance(content, str):
                    msg["content"] = self._strip(content)
                elif isinstance(content, list):
                    for block in content:
                        if isinstance(block, dict) and "text" in block:
                            block["text"] = self._strip(block["text"])

        # Gemini contents
        contents = normalized.get("contents", [])
        if isinstance(contents, str):
            normalized["contents"] = self._strip(contents)
        elif isinstance(contents, list):
            for i, item in enumerate(contents):
                if isinstance(item, str):
                    contents[i] = self._strip(item)
                elif isinstance(item, dict) and "text" in item:
                    item["text"] = self._strip(item["text"])

        # System prompt (Anthropic style)
        system = normalized.get("system")
        if isinstance(system, str):
            normalized["system"] = self._strip(system)

        return normalized

    def _strip(self, text: str) -> str:
        """Replace all dynamic patterns with the placeholder."""
        for _, pattern in self._patterns:
            text = pattern.sub(_PLACEHOLDER, text)
        return text
