"""PII rehydrator — mask outbound PII, restore on response."""

from __future__ import annotations

from stateloom.pii.scanner import PIIMatch


class PIIRehydrator:
    """Session-scoped PII masking and rehydration.

    Maintains a mapping of placeholder -> original value so that
    PII can be masked before reaching the LLM and restored in responses.
    """

    def __init__(self) -> None:
        self._map: dict[str, str] = {}  # placeholder -> original
        self._counter: int = 0

    def redact(self, text: str, matches: list[PIIMatch]) -> str:
        """Replace PII matches with placeholders. Returns redacted text."""
        if not matches:
            return text

        # Sort by position in reverse so indices stay valid
        sorted_matches = sorted(matches, key=lambda m: m.start, reverse=True)

        for match in sorted_matches:
            placeholder = f"[{match.pattern_name.upper()}_{self._counter}]"
            self._map[placeholder] = match.matched_text
            text = text[: match.start] + placeholder + text[match.end :]
            self._counter += 1

        return text

    def rehydrate(self, text: str) -> str:
        """Replace placeholders with original values in LLM response."""
        for placeholder, original in self._map.items():
            text = text.replace(placeholder, original)
        return text

    @property
    def redaction_count(self) -> int:
        """Number of PII items currently masked."""
        return len(self._map)
