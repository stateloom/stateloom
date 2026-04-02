"""Tests for PII scanner."""

import time

from stateloom.pii.scanner import PIIScanner


def test_detect_email():
    scanner = PIIScanner()
    matches = scanner.scan("Contact john@example.com for details")
    assert len(matches) == 1
    assert matches[0].pattern_name == "email"
    assert matches[0].matched_text == "john@example.com"


def test_detect_multiple_emails():
    scanner = PIIScanner()
    matches = scanner.scan("Email alice@test.com or bob@test.org")
    assert len(matches) == 2


def test_detect_credit_card_valid():
    scanner = PIIScanner()
    # 4111 1111 1111 1111 is a valid test card number (passes Luhn)
    matches = scanner.scan("Card: 4111 1111 1111 1111")
    assert len(matches) == 1
    assert matches[0].pattern_name == "credit_card"


def test_reject_invalid_credit_card():
    scanner = PIIScanner()
    # Random numbers that fail Luhn
    matches = scanner.scan("ID: 1234 5678 9012 3456")
    assert len(matches) == 0


def test_detect_ssn():
    scanner = PIIScanner()
    matches = scanner.scan("SSN: 123-45-6789")
    assert len(matches) == 1
    assert matches[0].pattern_name == "ssn"


def test_detect_openai_api_key():
    scanner = PIIScanner()
    matches = scanner.scan("Key: sk-abcdefghijklmnopqrstuvwxyz1234567890")
    assert len(matches) >= 1
    api_matches = [m for m in matches if "api_key" in m.pattern_name]
    assert len(api_matches) >= 1


def test_detect_openai_api_key_proj_format():
    scanner = PIIScanner()
    matches = scanner.scan("Key: sk-proj-abc123def456ghi789jklmno")
    assert len(matches) >= 1
    api_matches = [m for m in matches if "api_key" in m.pattern_name]
    assert len(api_matches) >= 1


def test_detect_openai_api_key_svcacct_format():
    scanner = PIIScanner()
    matches = scanner.scan("Key: sk-svcacct-abc123def456ghi789jklmno")
    assert len(matches) >= 1
    api_matches = [m for m in matches if "api_key" in m.pattern_name]
    assert len(api_matches) >= 1


def test_detect_aws_key():
    scanner = PIIScanner()
    matches = scanner.scan("AWS key: AKIAIOSFODNN7EXAMPLE")
    assert len(matches) == 1
    assert matches[0].pattern_name == "api_key_aws"


def test_no_false_positives_on_normal_text():
    scanner = PIIScanner()
    matches = scanner.scan("The weather today is sunny and warm in California.")
    assert len(matches) == 0


def test_scan_messages():
    scanner = PIIScanner()
    messages = [
        {"role": "user", "content": "My email is test@example.com"},
        {"role": "assistant", "content": "Got it, no PII here."},
    ]
    matches = scanner.scan_messages(messages)
    assert len(matches) == 1
    assert matches[0].field == "messages[0].content"


def test_scan_performance():
    """PII scan must complete in <5ms for typical prompt length."""
    scanner = PIIScanner()
    text = "Hello, this is a normal message without any PII. " * 100  # ~5KB
    start = time.perf_counter()
    scanner.scan(text)
    elapsed_ms = (time.perf_counter() - start) * 1000
    assert elapsed_ms < 5.0, f"Scan took {elapsed_ms:.1f}ms, expected <5ms"


def test_credit_card_regex_no_redos():
    """Adversarial input must not cause catastrophic backtracking (ReDoS)."""
    scanner = PIIScanner()
    # Near-miss: 30 digits with spaces that fail Luhn — triggers backtracking
    # in the old regex (?:\d[ -]*?){13,19} but is linear with \d(?:[ -]?\d){12,18}
    adversarial = "1234 5678 9012 3456 7890 1234 5678 9012"
    start = time.perf_counter()
    scanner.scan(adversarial)
    elapsed_ms = (time.perf_counter() - start) * 1000
    assert elapsed_ms < 50.0, f"ReDoS regression: scan took {elapsed_ms:.1f}ms, expected <50ms"
