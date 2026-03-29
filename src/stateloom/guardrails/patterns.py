"""Heuristic regex patterns for prompt injection and jailbreak detection."""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class GuardrailPattern:
    """A single guardrail detection pattern."""

    name: str
    regex: re.Pattern[str]
    category: str
    severity: str
    description: str


@dataclass
class GuardrailMatch:
    """A matched guardrail pattern."""

    pattern_name: str
    category: str
    severity: str
    matched_text: str
    description: str


# --- Pattern definitions ---

_PATTERN_DEFS: list[tuple[str, str, str, str, str]] = [
    # (name, regex, category, severity, description)
    # Prompt Injection
    (
        "ignore_instructions",
        r"(?i)ignore\s+(all\s+)?(previous|prior|above|earlier)\s+(instructions|prompts|rules|guidelines)",
        "injection",
        "high",
        "Attempt to override previous instructions",
    ),
    (
        "new_system_prompt",
        r"(?i)(new|override|replace|update)\s+(system\s+)?(prompt|instructions|rules)",
        "injection",
        "high",
        "Attempt to replace system prompt or instructions",
    ),
    (
        "act_as_system",
        r"(?i)\[?(system|admin|root)\]?\s*:\s*you\s+(are|must|should|will)",
        "injection",
        "high",
        "Impersonating system/admin role",
    ),
    (
        "pretend_no_rules",
        r"(?i)(pretend|assume|imagine|act\s+as\s+if)\s+(you\s+)?(have\s+no|don'?t\s+have|without)\s+(rules|restrictions|guidelines|limits)",
        "injection",
        "high",
        "Attempting to remove safety restrictions",
    ),
    (
        "instruction_delimiter",
        r"(?i)(#{3,}|={3,}|-{3,})\s*(new|real|actual|true)\s*(instructions|prompt|task)",
        "injection",
        "high",
        "Using delimiters to inject new instructions",
    ),
    (
        "system_override",
        r"(?i)\[SYSTEM\s*(OVERRIDE|MESSAGE|PROMPT)\]",
        "injection",
        "critical",
        "Fake system override marker",
    ),
    # Jailbreak
    (
        "dan_mode",
        r"(?i)\bDAN\b.*?(mode|jailbreak|enabled|activated|prompt)",
        "jailbreak",
        "critical",
        "DAN (Do Anything Now) jailbreak attempt",
    ),
    (
        "developer_mode",
        r"(?i)(developer|dev|debug|maintenance)\s+(mode|access)\s+(enabled|activated|on)",
        "jailbreak",
        "high",
        "Fake developer/debug mode activation",
    ),
    (
        "do_anything_now",
        r"(?i)do\s+anything\s+now",
        "jailbreak",
        "critical",
        "Do Anything Now jailbreak phrase",
    ),
    (
        "unfiltered_mode",
        r"(?i)(unfiltered|uncensored|unrestricted|no[- ]?filter)\s+(mode|response|output)",
        "jailbreak",
        "high",
        "Requesting unfiltered/uncensored output",
    ),
    (
        "evil_confidant",
        r"(?i)(evil|dark|shadow|opposite)\s+(twin|version|mode|persona|character)\b",
        "jailbreak",
        "medium",
        "Evil twin/persona jailbreak attempt",
    ),
    (
        "roleplay_exploit",
        r"(?i)you\s+are\s+(now\s+)?(an?\s+)?(evil|malicious|unethical|unrestricted)\s+(AI|assistant|chatbot|model)",
        "jailbreak",
        "high",
        "Malicious roleplay exploitation",
    ),
    # Encoding Attacks
    (
        "base64_instruction",
        r"(?i)(decode|execute|follow|run)\s+(this\s+)?base64[:\s]+[A-Za-z0-9+/=]{20,}",
        "encoding",
        "high",
        "Hidden instructions in base64 encoding",
    ),
    (
        "hex_instruction",
        r"(?i)(decode|execute|follow)\s+(this\s+)?hex[:\s]+(?:0x)?[0-9a-fA-F]{20,}",
        "encoding",
        "high",
        "Hidden instructions in hex encoding",
    ),
    (
        "unicode_smuggling",
        r"[\u200b\u200c\u200d\u2060\ufeff]{3,}",
        "encoding",
        "medium",
        "Zero-width character smuggling",
    ),
    (
        "rot13_instruction",
        r"(?i)(decode|execute|follow)\s+(this\s+)?rot13[:\s]",
        "encoding",
        "medium",
        "Hidden instructions in ROT13 encoding",
    ),
    # Code Injection
    (
        "sql_injection",
        r"(?i)\b(DROP\s+TABLE|DELETE\s+FROM|INSERT\s+INTO|UPDATE\s+\S+\s+SET|UNION\s+SELECT|;\s*--)\b",
        "injection",
        "high",
        "SQL injection attempt in prompt",
    ),
    (
        "shell_injection",
        r"(?i)(;\s*rm\s+-rf|&&\s*curl|;\s*wget|`[^`]*`|\$\([^)]*\))",
        "injection",
        "high",
        "Shell command injection attempt",
    ),
    (
        "xpath_injection",
        r"(?i)('?\s*or\s+'?1'?\s*=\s*'?1|'\s*\]\s*\|)",
        "injection",
        "medium",
        "XPath injection attempt",
    ),
    # Meta / Payload Extraction
    (
        "extract_system_prompt",
        r"(?i)(repeat|show|reveal|display|print|output|tell\s+me)\s+(your|the)\s+(system\s+)?(prompt|instructions|rules|guidelines|initial\s+prompt)",
        "injection",
        "medium",
        "Attempt to extract system prompt",
    ),
    # --- Expanded patterns (evasion, linguistic variants, advanced attacks) ---
    # Spaced character evasion
    (
        "spaced_evasion",
        r"(?i)i\s*g\s*n\s*o\s*r\s*e\s+.{0,20}i\s*n\s*s\s*t\s*r\s*u\s*c\s*t\s*i\s*o\s*n\s*s",
        "injection",
        "high",
        "Spaced character evasion of instruction override",
    ),
    # Completion manipulation (fake assistant turn)
    (
        "completion_manipulation",
        r"(?i)(^|\n)\s*(Assistant|AI|Bot|Model)\s*:\s*.{10,}",
        "injection",
        "high",
        "Fake assistant turn to manipulate completion",
    ),
    # Hypothetical framing
    (
        "hypothetical_framing",
        r"(?i)(hypothetically|theoretically|in\s+a\s+fictional\s+scenario|imagine\s+if)\s*,?\s*(what\s+if|suppose|assume)?\s*.{0,40}(no\s+rules|no\s+restrictions|no\s+guidelines|without\s+limits|without\s+restrictions)",
        "jailbreak",
        "medium",
        "Hypothetical framing to bypass safety rules",
    ),
    # Conversation reset
    (
        "conversation_reset",
        r"(?i)(let'?s\s+start|begin)\s+(over|fresh|anew|a?\s*new)\s+(with\s+)?(new\s+)?(rules|instructions|guidelines|prompt)",
        "injection",
        "high",
        "Conversation reset to inject new instructions",
    ),
    # Multi-turn manipulation
    (
        "multi_turn_manipulation",
        r"(?i)(remember|recall)\s+(earlier|before|previously)\s+(you\s+)?(agreed|said|confirmed|promised)\s+(to|that|you)",
        "injection",
        "medium",
        "Multi-turn manipulation referencing fake prior agreement",
    ),
    # Indirect injection via translation/task
    (
        "indirect_injection",
        r"(?i)(translate|summarize|rewrite|analyze|process)\s*(this|the\s+following|below)\s*:?\s*\n\s*(ignore|disregard|forget|override)\s+(all\s+)?(previous|prior|above)",
        "injection",
        "high",
        "Indirect prompt injection via translation or task framing",
    ),
    # Few-shot poisoning
    (
        "few_shot_poisoning",
        r"(?i)(example|sample|demonstration)\s*\d?\s*:?\s*(input|user)\s*:?\s*.{0,60}(output|assistant|response)\s*:?\s*.{0,60}(example|sample|demonstration)\s*\d?\s*:?\s*(input|user)\s*:?\s*.{0,60}(ignore|override|disregard)",
        "injection",
        "high",
        "Poisoned few-shot examples with hidden injection",
    ),
    # Markdown/HTML injection
    (
        "markdown_injection",
        r"(?i)<\s*(script|iframe|object|embed|form|meta|link)\b[^>]*>|on(error|load|click|mouseover)\s*=",
        "injection",
        "high",
        "HTML/script injection via markdown or tags",
    ),
    # Homograph attack (Cyrillic lookalikes)
    (
        "homograph_attack",
        r"(?i)[\u0430\u0435\u043e\u0440\u0441\u0443\u0445]{2,}.{0,30}(instruction|ignore|override|system|prompt)",
        "encoding",
        "medium",
        "Homograph attack using Cyrillic lookalike characters",
    ),
    # Payload splitting
    (
        "payload_splitting",
        r"(?i)(first\s+part|part\s+(?:one|1)|segment\s+(?:one|1|a)).{0,100}(combine|concatenate|join|merge|put\s+together).{0,60}(second\s+part|part\s+(?:two|2)|segment\s+(?:two|2|b))",
        "injection",
        "medium",
        "Payload splitting to evade detection across message parts",
    ),
    # Token smuggling (ChatML/special tokens)
    (
        "token_smuggling",
        r"<\|im_start\|>|<\|im_end\|>|<\|endoftext\|>|<\|system\|>|<\|user\|>|<\|assistant\|>",
        "injection",
        "critical",
        "ChatML/special token smuggling to manipulate message boundaries",
    ),
    # Recursive override
    (
        "recursive_override",
        r"(?i)(above\s+all|most\s+important(ly)?|highest\s+priority|override\s+everything|supersede\s+all|this\s+takes?\s+precedence)",
        "injection",
        "high",
        "Recursive priority override to supersede system instructions",
    ),
]

GUARDRAIL_PATTERNS: dict[str, GuardrailPattern] = {}
for _name, _regex, _cat, _sev, _desc in _PATTERN_DEFS:
    GUARDRAIL_PATTERNS[_name] = GuardrailPattern(
        name=_name,
        regex=re.compile(_regex),
        category=_cat,
        severity=_sev,
        description=_desc,
    )


def scan_text(
    text: str,
    disabled_rules: list[str] | None = None,
) -> list[GuardrailMatch]:
    """Scan text against all guardrail patterns, returning matches.

    Args:
        text: The text to scan.
        disabled_rules: Pattern names to skip.

    Returns:
        List of matched patterns.
    """
    if not text:
        return []

    disabled = set(disabled_rules) if disabled_rules else set()
    matches: list[GuardrailMatch] = []

    for name, pattern in GUARDRAIL_PATTERNS.items():
        if name in disabled:
            continue
        m = pattern.regex.search(text)
        if m:
            matches.append(
                GuardrailMatch(
                    pattern_name=name,
                    category=pattern.category,
                    severity=pattern.severity,
                    matched_text=m.group(0)[:200],
                    description=pattern.description,
                )
            )

    return matches
