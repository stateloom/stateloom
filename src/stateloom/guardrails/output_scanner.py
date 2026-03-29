"""System prompt leak detection in LLM responses."""

from __future__ import annotations

import difflib

from stateloom.guardrails.validators import GuardrailResult

_SLIDING_WINDOW_SIZE = 50
_SLIDING_WINDOW_STRIDE = 25
_MIN_WINDOW_MATCH_LEN = 15
_SEQUENCE_MATCHER_MAX_LEN = 2000


class SystemPromptLeakScanner:
    """Detects if an LLM response contains fragments of the system prompt."""

    def __init__(self, threshold: float = 0.6) -> None:
        self._threshold = threshold

    def scan(self, response_text: str, system_prompt: str) -> GuardrailResult:
        """Check if the response leaks the system prompt.

        Returns a GuardrailResult with score = max(substring_ratio, sequence_ratio).
        """
        if not response_text or not system_prompt:
            return GuardrailResult(safe=True, score=0.0)

        # Skip very short system prompts — too many false positives
        if len(system_prompt.strip()) < 20:
            return GuardrailResult(safe=True, score=0.0)

        norm_response = response_text.lower().strip()
        norm_prompt = system_prompt.lower().strip()

        # Method 1: Sliding window substring check
        matched_chars = 0
        total_chars = len(norm_prompt)

        if total_chars > 0:
            i = 0
            while i < total_chars:
                end = min(i + _SLIDING_WINDOW_SIZE, total_chars)
                window = norm_prompt[i:end]
                if len(window) >= _MIN_WINDOW_MATCH_LEN and window in norm_response:
                    matched_chars += len(window)
                    i = end  # skip past matched window
                else:
                    i += _SLIDING_WINDOW_STRIDE

            substring_ratio = matched_chars / total_chars if total_chars > 0 else 0.0
        else:
            substring_ratio = 0.0

        # Method 2: SequenceMatcher ratio
        sequence_ratio = difflib.SequenceMatcher(
            None,
            norm_response[:_SEQUENCE_MATCHER_MAX_LEN],
            norm_prompt[:_SEQUENCE_MATCHER_MAX_LEN],
        ).ratio()

        score = max(substring_ratio, sequence_ratio)
        is_leak = score >= self._threshold

        return GuardrailResult(
            safe=not is_leak,
            category="system_prompt_leak" if is_leak else "",
            score=score,
            rule_name="system_prompt_leak",
            severity="high" if is_leak else "low",
            raw_output=f"substring={substring_ratio:.3f}, sequence={sequence_ratio:.3f}",
        )
