"""Stream PII buffer — holds back chunks for PII scanning before release."""

from __future__ import annotations

import logging

from stateloom.core.types import PIIMode
from stateloom.pii.rehydrator import PIIRehydrator
from stateloom.pii.scanner import PIIMatch, PIIScanner

logger = logging.getLogger("stateloom.pii.stream_buffer")

OVERLAP_CHARS = 20  # overlap window to catch PII spanning chunk boundaries


class StreamPIIBuffer:
    """Accumulates streaming text, scans for PII, yields clean text.

    Uses regex-only scanning (NER is too slow for per-chunk use).

    Usage::

        buffer = StreamPIIBuffer(scanner, mode, buffer_size=3)
        for chunk in stream:
            text_delta = extract_text(chunk)
            clean = buffer.feed(text_delta)
            if clean is not None:
                yield modified_chunk(chunk, clean)
        # Flush remaining text
        final = buffer.flush()
        if final:
            yield final_chunk(final)
    """

    def __init__(
        self,
        scanner: PIIScanner,
        mode: PIIMode,
        buffer_size: int = 3,
        rehydrator: PIIRehydrator | None = None,
    ) -> None:
        self._scanner = scanner
        self._mode = mode
        self._buffer_size = max(1, buffer_size)
        self._rehydrator = rehydrator or PIIRehydrator()
        self._pending_chunks: list[str] = []
        self._released_text: str = ""  # text already released (for overlap scanning)
        self._pii_detected: list[PIIMatch] = []

    def feed(self, text_delta: str) -> str | None:
        """Feed a chunk's text delta. Returns clean text to release, or None if buffering."""
        if not text_delta:
            return None

        self._pending_chunks.append(text_delta)

        if len(self._pending_chunks) < self._buffer_size:
            return None  # Still buffering

        return self._scan_and_release()

    def flush(self) -> str:
        """Flush all remaining buffered text. Call after stream ends."""
        if not self._pending_chunks:
            return ""
        return self._scan_and_release(force_all=True)

    def _scan_and_release(self, force_all: bool = False) -> str:
        """Scan buffered text and release clean portion."""
        full_text = "".join(self._pending_chunks)

        # Include overlap from previously released text for boundary detection
        overlap = self._released_text[-OVERLAP_CHARS:] if self._released_text else ""
        scan_text = overlap + full_text

        matches = self._scanner.scan(scan_text, field="stream_response")

        # Adjust match positions to account for overlap prefix
        offset = len(overlap)
        relevant_matches = [
            PIIMatch(
                pattern_name=m.pattern_name,
                matched_text=m.matched_text,
                start=m.start - offset,
                end=m.end - offset,
                field=m.field,
            )
            for m in matches
            if m.end > offset  # only matches that touch the new text
        ]

        if force_all:
            release_text = full_text
            self._pending_chunks = []
        else:
            # Release all but last chunk (keep one chunk back for boundary safety)
            release_text = "".join(self._pending_chunks[:-1])
            self._pending_chunks = [self._pending_chunks[-1]]

        if not relevant_matches:
            self._released_text += release_text
            return release_text

        self._pii_detected.extend(relevant_matches)

        # Redact PII in the text being released
        release_matches = [m for m in relevant_matches if 0 <= m.start < len(release_text)]
        if release_matches:
            release_text = self._rehydrator.redact(release_text, release_matches)

        self._released_text += release_text
        return release_text

    @property
    def detected_pii(self) -> list[PIIMatch]:
        """All PII matches detected so far."""
        return list(self._pii_detected)
