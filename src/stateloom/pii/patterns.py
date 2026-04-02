"""PII detection patterns — compiled regex for high-speed scanning."""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class PIIPattern:
    """A single PII detection pattern."""

    name: str
    regex: re.Pattern[str]
    validator: str | None = None  # "luhn" for credit cards


# Compiled patterns for fast scanning
PII_PATTERNS: dict[str, PIIPattern] = {
    "email": PIIPattern(
        name="email",
        regex=re.compile(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+"),
    ),
    "credit_card": PIIPattern(
        name="credit_card",
        regex=re.compile(r"\b\d(?:[ -]?\d){12,18}\b"),
        validator="luhn",
    ),
    "ssn": PIIPattern(
        name="ssn",
        regex=re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    ),
    "phone_us": PIIPattern(
        name="phone_us",
        regex=re.compile(r"\b(?:\+1[-.]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"),
    ),
    "api_key_openai": PIIPattern(
        name="api_key_openai",
        regex=re.compile(r"\bsk-[a-zA-Z0-9][a-zA-Z0-9\-]{19,}\b"),
    ),
    "api_key_anthropic": PIIPattern(
        name="api_key_anthropic",
        regex=re.compile(r"\bsk-ant-[a-zA-Z0-9\-]{20,}\b"),
    ),
    "api_key_aws": PIIPattern(
        name="api_key_aws",
        regex=re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    ),
    "api_key_bearer": PIIPattern(
        name="api_key_bearer",
        regex=re.compile(r"\bBearer\s+[a-zA-Z0-9\-._~+/]{20,}=*\b"),
    ),
    "ip_address": PIIPattern(
        name="ip_address",
        regex=re.compile(
            r"\b(?:(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\.){3}(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\b"
        ),
    ),
    # EU PII patterns
    "vat_id": PIIPattern(
        name="vat_id",
        regex=re.compile(r"\b[A-Z]{2}\d{8,12}\b"),
    ),
    "iban": PIIPattern(
        name="iban",
        regex=re.compile(r"\b[A-Z]{2}\d{2}[A-Z0-9]{4}\d{7}(?:[A-Z0-9]?){0,16}\b"),
    ),
    "national_id_eu": PIIPattern(
        name="national_id_eu",
        regex=re.compile(r"\b\d{2}\.\d{2}\.\d{2}-\d{3}\.\d{2}-\d{2}\b"),
    ),
    "phone_eu": PIIPattern(
        name="phone_eu",
        regex=re.compile(r"\+\d{1,3}\d{6,14}\b"),
    ),
    # HIPAA PHI patterns
    "medical_record_number": PIIPattern(
        name="medical_record_number",
        regex=re.compile(r"\bMRN:\s*\d{4,10}\b"),
    ),
    "health_plan_id": PIIPattern(
        name="health_plan_id",
        regex=re.compile(r"\bHPI:\s*\d{6,15}\b"),
    ),
    "npi": PIIPattern(
        name="npi",
        regex=re.compile(r"\bNPI:\s*\d{10}\b"),
    ),
    "date_of_birth": PIIPattern(
        name="date_of_birth",
        regex=re.compile(r"\bDOB:\s*\d{2}/\d{2}/\d{4}\b"),
    ),
    # CCPA patterns
    "california_dl": PIIPattern(
        name="california_dl",
        regex=re.compile(r"\b[A-Z]\d{7}\b"),
    ),
    # NER entity patterns (sentinel — only matched by NER detector, not regex)
    "ner_person": PIIPattern(name="ner_person", regex=re.compile(r"$^")),
    "ner_location": PIIPattern(name="ner_location", regex=re.compile(r"$^")),
    "ner_organization": PIIPattern(name="ner_organization", regex=re.compile(r"$^")),
    "ner_address": PIIPattern(name="ner_address", regex=re.compile(r"$^")),
}

# Map user-facing pattern names to internal keys
# e.g. "api_key" matches all api_key_* patterns
PATTERN_GROUPS: dict[str, list[str]] = {
    "api_key": ["api_key_openai", "api_key_anthropic", "api_key_aws", "api_key_bearer"],
    "phone": ["phone_us", "phone_eu"],
    "eu_pii": ["vat_id", "national_id_eu", "iban", "phone_eu"],
    "phi": ["medical_record_number", "health_plan_id", "npi", "date_of_birth"],
    "ner": ["ner_person", "ner_location", "ner_organization", "ner_address"],
}


def luhn_check(number: str) -> bool:
    """Validate a number using the Luhn algorithm (credit card checksum)."""
    digits = [int(d) for d in number if d.isdigit()]
    if len(digits) < 13 or len(digits) > 19:
        return False
    checksum = 0
    for i, d in enumerate(reversed(digits)):
        if i % 2 == 1:
            d *= 2
            if d > 9:
                d -= 9
        checksum += d
    return checksum % 10 == 0


def resolve_pattern_names(names: list[str]) -> list[str]:
    """Resolve pattern names including groups to actual pattern keys."""
    resolved = []
    for name in names:
        if name in PATTERN_GROUPS:
            resolved.extend(PATTERN_GROUPS[name])
        elif name in PII_PATTERNS:
            resolved.append(name)
        # Silently skip unknown patterns
    return list(set(resolved))
