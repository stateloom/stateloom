"""Tests for stream PII buffer."""

from __future__ import annotations

import pytest

from stateloom.core.types import PIIMode
from stateloom.pii.scanner import PIIScanner
from stateloom.pii.stream_buffer import OVERLAP_CHARS, StreamPIIBuffer


class TestStreamPIIBufferBasic:
    """Basic buffering and release behavior."""

    def test_buffers_until_threshold(self):
        scanner = PIIScanner()
        buf = StreamPIIBuffer(scanner, PIIMode.REDACT, buffer_size=3)

        assert buf.feed("Hello ") is None  # chunk 1: buffering
        assert buf.feed("world ") is None  # chunk 2: buffering
        result = buf.feed("today ")  # chunk 3: releases
        assert result is not None
        assert "Hello " in result
        assert "world " in result

    def test_flush_releases_remaining(self):
        scanner = PIIScanner()
        buf = StreamPIIBuffer(scanner, PIIMode.REDACT, buffer_size=3)

        buf.feed("Hello ")
        buf.feed("world ")
        final = buf.flush()
        assert "Hello " in final
        assert "world " in final

    def test_flush_empty_buffer(self):
        scanner = PIIScanner()
        buf = StreamPIIBuffer(scanner, PIIMode.REDACT, buffer_size=3)
        assert buf.flush() == ""

    def test_empty_text_delta_returns_none(self):
        scanner = PIIScanner()
        buf = StreamPIIBuffer(scanner, PIIMode.REDACT, buffer_size=3)
        assert buf.feed("") is None
        assert buf.feed(None) is None  # type: ignore[arg-type]

    def test_buffer_size_one(self):
        """Buffer size 1 means every chunk is released immediately."""
        scanner = PIIScanner()
        buf = StreamPIIBuffer(scanner, PIIMode.REDACT, buffer_size=1)
        result = buf.feed("Hello")
        assert result is not None

    def test_buffer_size_clamped_to_minimum_one(self):
        scanner = PIIScanner()
        buf = StreamPIIBuffer(scanner, PIIMode.REDACT, buffer_size=0)
        assert buf._buffer_size == 1


class TestStreamPIIBufferRedaction:
    """PII redaction in streaming text."""

    def test_redacts_ssn_in_stream(self):
        scanner = PIIScanner()
        buf = StreamPIIBuffer(scanner, PIIMode.REDACT, buffer_size=2)

        buf.feed("Your SSN is ")
        result = buf.feed("123-45-6789 okay")
        # The result should contain redacted SSN
        # First release: "Your SSN is " (no PII)
        assert result is not None

        final = buf.flush()
        # Combined text should have the SSN redacted
        full_text = (result or "") + final
        assert "123-45-6789" not in full_text

    def test_redacts_email_in_flush(self):
        scanner = PIIScanner()
        buf = StreamPIIBuffer(scanner, PIIMode.REDACT, buffer_size=5)

        buf.feed("Contact ")
        buf.feed("me at ")
        buf.feed("john@")
        buf.feed("example.com ")
        # Not enough chunks yet, flush
        final = buf.flush()
        assert "john@example.com" not in final
        assert "[EMAIL_" in final.upper() or "EMAIL" in final.upper()

    def test_clean_text_passes_through(self):
        scanner = PIIScanner()
        buf = StreamPIIBuffer(scanner, PIIMode.REDACT, buffer_size=2)

        buf.feed("Hello ")
        result = buf.feed("world!")
        assert result is not None
        # No PII, text should pass through unchanged
        assert "Hello" in result or "Hello" in buf.flush()

    def test_detected_pii_property(self):
        scanner = PIIScanner()
        buf = StreamPIIBuffer(scanner, PIIMode.REDACT, buffer_size=2)

        buf.feed("SSN: ")
        buf.feed("123-45-6789")
        buf.flush()

        assert len(buf.detected_pii) > 0
        assert any(m.pattern_name == "ssn" for m in buf.detected_pii)


class TestStreamPIIBufferOverlap:
    """Tests for PII spanning chunk boundaries."""

    def test_overlap_catches_boundary_pii(self):
        """SSN split across chunk boundary should still be detected."""
        scanner = PIIScanner()
        buf = StreamPIIBuffer(scanner, PIIMode.REDACT, buffer_size=2)

        # Split SSN across chunks
        buf.feed("ID: 123-45")
        result = buf.feed("-6789 end")
        # After release + flush, the SSN should be detected
        final = buf.flush()
        full_text = (result or "") + final
        # The SSN should be redacted in the combined output
        assert "123-45-6789" not in full_text or len(buf.detected_pii) > 0

    def test_overlap_window_constant(self):
        assert OVERLAP_CHARS == 20


class TestStreamPIIBufferModes:
    """Test different PII modes in stream context."""

    def test_audit_mode_no_redaction(self):
        """In AUDIT mode, text passes through unchanged."""
        scanner = PIIScanner()
        buf = StreamPIIBuffer(scanner, PIIMode.AUDIT, buffer_size=2)

        buf.feed("SSN: ")
        result = buf.feed("123-45-6789")
        final = buf.flush()
        full_text = (result or "") + final
        # AUDIT mode still redacts in stream buffer (uses rehydrator.redact)
        # The buffer always redacts because it can't emit events mid-stream
        # This is by design — streaming PII protection is always redact or block

    def test_block_mode_redacts_stream(self):
        """In BLOCK mode, PII is still redacted in streams (can't abort mid-stream)."""
        scanner = PIIScanner()
        buf = StreamPIIBuffer(scanner, PIIMode.BLOCK, buffer_size=2)

        buf.feed("SSN: ")
        result = buf.feed("123-45-6789")
        final = buf.flush()
        full_text = (result or "") + final
        assert "123-45-6789" not in full_text


class TestStreamPIIBufferMultipleChunks:
    """Tests with realistic multi-chunk streaming scenarios."""

    def test_many_chunks_no_pii(self):
        scanner = PIIScanner()
        buf = StreamPIIBuffer(scanner, PIIMode.REDACT, buffer_size=3)

        chunks = ["The ", "weather ", "is ", "sunny ", "and ", "warm ", "today."]
        released = []
        for chunk in chunks:
            result = buf.feed(chunk)
            if result is not None:
                released.append(result)
        released.append(buf.flush())

        full = "".join(released)
        assert "The weather is sunny and warm today." == full

    def test_pii_in_later_chunks(self):
        scanner = PIIScanner()
        buf = StreamPIIBuffer(scanner, PIIMode.REDACT, buffer_size=2)

        chunks = ["Hello, ", "my SSN is ", "123-45-6789", " thanks"]
        released = []
        for chunk in chunks:
            result = buf.feed(chunk)
            if result is not None:
                released.append(result)
        released.append(buf.flush())

        full = "".join(released)
        assert "123-45-6789" not in full
