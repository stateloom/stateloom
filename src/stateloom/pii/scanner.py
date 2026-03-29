"""PII scanner — fast regex-based detection with optional NER second pass."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

from stateloom.core.config import StateLoomConfig
from stateloom.pii.patterns import PII_PATTERNS, PIIPattern, luhn_check, resolve_pattern_names

if TYPE_CHECKING:
    from stateloom.pii.ner_detector import NERDetector

logger = logging.getLogger("stateloom.pii.scanner")


@dataclass
class PIIMatch:
    """A single PII match found during scanning."""

    pattern_name: str
    matched_text: str
    start: int
    end: int
    field: str = ""  # e.g. "messages[0].content"


class PIIScanner:
    """Compiled regex scanner for PII detection, with optional NER second pass.

    Designed for <2ms scan time on typical prompts (regex only).
    NER adds ~10-50ms depending on text length and model.
    """

    def __init__(
        self,
        config: StateLoomConfig | None = None,
        ner_detector: NERDetector | None = None,
    ) -> None:
        # If specific patterns are configured via rules, only use those
        if config and config.pii_rules:
            rule_names = [r.pattern for r in config.pii_rules]
            active_keys = resolve_pattern_names(rule_names)
        else:
            active_keys = list(PII_PATTERNS.keys())

        self._patterns: dict[str, PIIPattern] = {
            k: v for k, v in PII_PATTERNS.items() if k in active_keys
        }
        self._ner = ner_detector

    def scan(self, text: str, field: str = "") -> list[PIIMatch]:
        """Scan text for PII. Returns list of matches."""
        matches: list[PIIMatch] = []

        # 1. Regex pass
        for name, pattern in self._patterns.items():
            for m in pattern.regex.finditer(text):
                # Extra validation for credit cards
                if pattern.validator == "luhn" and not luhn_check(m.group()):
                    continue

                matches.append(
                    PIIMatch(
                        pattern_name=name,
                        matched_text=m.group(),
                        start=m.start(),
                        end=m.end(),
                        field=field,
                    )
                )

        # 2. NER pass (merge + deduplicate)
        if self._ner is not None:
            matches = self._merge_ner_matches(text, field, matches)

        return matches

    def _merge_ner_matches(
        self, text: str, field: str, regex_matches: list[PIIMatch]
    ) -> list[PIIMatch]:
        """Run NER detector and merge with regex matches, deduplicating overlaps."""
        try:
            assert self._ner is not None
            ner_results = self._ner.detect(text)
        except Exception:
            logger.warning("NER detection failed during merge", exc_info=True)
            return regex_matches

        if not ner_results:
            return regex_matches

        # Build existing span set for overlap check
        existing_spans = [(m.start, m.end) for m in regex_matches]

        for nr in ner_results:
            # Skip if overlaps with any regex match
            overlaps = any(
                nr.start < re_end and nr.end > re_start for re_start, re_end in existing_spans
            )
            if not overlaps:
                regex_matches.append(
                    PIIMatch(
                        pattern_name=f"ner_{nr.entity_type}",
                        matched_text=nr.text,
                        start=nr.start,
                        end=nr.end,
                        field=field,
                    )
                )
                existing_spans.append((nr.start, nr.end))

        return regex_matches

    def scan_messages(
        self,
        messages: list[dict],
        field_offset: int = 0,
    ) -> list[PIIMatch]:
        """Scan a list of chat messages for PII."""
        all_matches: list[PIIMatch] = []
        for i, msg in enumerate(messages):
            idx = i + field_offset
            content = msg.get("content", "")
            if isinstance(content, str):
                field = f"messages[{idx}].content"
                all_matches.extend(self.scan(content, field=field))
            elif isinstance(content, list):
                # Handle multi-part content (e.g. OpenAI vision)
                for j, part in enumerate(content):
                    if isinstance(part, dict) and part.get("type") == "text":
                        text = part.get("text", "")
                        field = f"messages[{idx}].content[{j}].text"
                        all_matches.extend(self.scan(text, field=field))
        return all_matches
