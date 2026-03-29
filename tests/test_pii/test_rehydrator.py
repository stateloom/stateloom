"""Tests for PII rehydrator."""

from stateloom.pii.rehydrator import PIIRehydrator
from stateloom.pii.scanner import PIIMatch


def test_redact_single():
    rehydrator = PIIRehydrator()
    matches = [PIIMatch(pattern_name="email", matched_text="test@example.com", start=10, end=26)]
    text = "Contact: test@example.com please"
    redacted = rehydrator.redact(text, matches)
    assert "test@example.com" not in redacted
    assert "[EMAIL_" in redacted


def test_redact_multiple():
    rehydrator = PIIRehydrator()
    matches = [
        PIIMatch(pattern_name="email", matched_text="a@b.com", start=0, end=7),
        PIIMatch(pattern_name="ssn", matched_text="123-45-6789", start=12, end=23),
    ]
    text = "a@b.com and 123-45-6789"
    redacted = rehydrator.redact(text, matches)
    assert "a@b.com" not in redacted
    assert "123-45-6789" not in redacted


def test_rehydrate():
    rehydrator = PIIRehydrator()
    matches = [PIIMatch(pattern_name="email", matched_text="test@example.com", start=5, end=21)]
    text = "Hi, test@example.com here"
    redacted = rehydrator.redact(text, matches)

    # Simulate LLM response that echoes the placeholder
    response = f"I see your email is {redacted.split('Hi, ')[1].split(' here')[0]}"
    rehydrated = rehydrator.rehydrate(response)
    assert "test@example.com" in rehydrated


def test_redaction_count():
    rehydrator = PIIRehydrator()
    assert rehydrator.redaction_count == 0
    matches = [
        PIIMatch(pattern_name="email", matched_text="a@b.com", start=0, end=7),
        PIIMatch(pattern_name="email", matched_text="c@d.com", start=10, end=17),
    ]
    rehydrator.redact("a@b.com / c@d.com", matches)
    assert rehydrator.redaction_count == 2
