"""
StateLoom PII Detection — Audit, Redact, Block Modes

Demonstrates:
  - Three PII modes: audit (log only), redact (replace with placeholders), block (reject)
  - Per-pattern rules (e.g. block credit cards, redact emails, audit phone numbers)
  - Monkey-patching: PII scanning works on native SDK calls (openai, anthropic, gemini) too
  - Enabling GDPR compliance via the dashboard API (auto-configures strict PII rules)
  - PII detection events visible in the dashboard waterfall

Run with one or more provider keys:

    export OPENAI_API_KEY=sk-...
    export ANTHROPIC_API_KEY=sk-ant-...
    export GOOGLE_API_KEY=AIza...
    python examples/04_pii_detection.py
"""

import os

import stateloom
from stateloom.core.config import PIIRule
from stateloom.core.types import PIIMode

# ── Detect available providers ────────────────────────────────────────

providers = {}

if os.environ.get("OPENAI_API_KEY"):
    try:
        import openai
        providers["openai"] = "gpt-4o-mini"
    except ImportError:
        pass

if os.environ.get("ANTHROPIC_API_KEY"):
    try:
        import anthropic
        providers["anthropic"] = "claude-haiku-4-5-20251001"
    except ImportError:
        pass

if os.environ.get("GOOGLE_API_KEY"):
    try:
        import google.generativeai as genai
        providers["gemini"] = "gemini-2.5-flash"
    except ImportError:
        pass

if not providers:
    print("Set at least one API key: OPENAI_API_KEY, ANTHROPIC_API_KEY, or GOOGLE_API_KEY")
    raise SystemExit(1)

# Pick a model for stateloom.chat() calls
CHAT_MODEL = next(iter(providers.values()))

print(f"Providers available: {', '.join(providers)}\n")


def call_native(prompt: str, provider: str | None = None) -> str:
    """Make a call using the native provider SDK (not stateloom.chat).

    StateLoom's monkey-patching intercepts these calls too — PII scanning,
    cost tracking, and all middleware apply transparently.

    If *provider* is None, uses the first available provider.
    """
    target = provider or next(iter(providers))

    if target == "openai":
        client = openai.OpenAI()
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
        )
        return resp.choices[0].message.content or ""

    if target == "anthropic":
        client = anthropic.Anthropic()
        resp = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=256,
            messages=[{"role": "user", "content": prompt}],
        )
        return resp.content[0].text

    if target == "gemini":
        genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
        model = genai.GenerativeModel("gemini-2.5-flash")
        resp = model.generate_content(prompt)
        return resp.text or ""

    return ""


# ── 1. Audit mode (default) ──────────────────────────────────────────
# PII is detected and logged, but the request passes through unchanged.
# Check the dashboard to see PIIDetectionEvent entries.

print("=" * 60)
print("1. Audit mode — detect and log, don't modify")
print("=" * 60)

stateloom.init(pii=True)  # audit mode by default

# Use unique emails per provider so PII dedup creates events for each call.
_audit_prompts = {
    "openai": "Send the report to jane.doe@example.com and cc bob@corp.io",
    "anthropic": "Email admin@internal.dev about the deploy status",
    "gemini": "Forward the docs to ops-team@startup.io please",
}

with stateloom.session("pii-audit-demo") as s:
    # Native SDK calls — monkey-patched, PII scanning applies
    for name in providers:
        text = call_native(_audit_prompts.get(name, _audit_prompts["openai"]), name)
        print(f"  {name} (native SDK): {text[:100]}")

    # stateloom.chat() call — same PII scanning
    response = stateloom.chat(
        model=CHAT_MODEL,
        messages=[{
            "role": "user",
            "content": "Also notify sarah@company.org about the update",
        }],
    )
    print(f"  stateloom.chat: {response.content[:100]}")
    print(f"  PII detections: {s.pii_detections} (emails detected, request sent as-is)\n")

stateloom.shutdown()


# ── 2. Redact mode — replace PII with placeholders ───────────────────
# Emails and phone numbers are replaced before reaching the LLM.
# The LLM never sees the actual PII — only [EMAIL_0] etc.

print("=" * 60)
print("2. Redact mode — PII replaced with [EMAIL_0] etc.")
print("=" * 60)

stateloom.init(
    pii=True,
    pii_rules=[
        PIIRule(pattern="email", mode=PIIMode.REDACT),
        PIIRule(pattern="phone", mode=PIIMode.REDACT),
        PIIRule(pattern="ssn", mode=PIIMode.AUDIT),  # SSNs just logged
    ],
)

# Unique PII per provider to get individual dashboard events.
_redact_prompts = {
    "openai": "Contact Jane at jane@example.com or 555-867-5309. Her SSN is 123-45-6789.",
    "anthropic": "Reach out to bob@corp.io at 212-555-0100. His SSN is 987-65-4321.",
    "gemini": "Notify dev-ops@techfirm.com at 415-555-0199. Reference SSN 456-78-9012.",
}

with stateloom.session("pii-redact-demo") as s:
    # Native SDK — email and phone redacted before the LLM sees them
    for name in providers:
        text = call_native(_redact_prompts.get(name, _redact_prompts["openai"]), name)
        print(f"  {name} (native SDK): {text[:120]}")

    # stateloom.chat() — same redaction applies
    response = stateloom.chat(
        model=CHAT_MODEL,
        messages=[{
            "role": "user",
            "content": "Also reach out to hr@acme.co at 310-555-0142",
        }],
    )
    print(f"  stateloom.chat: {response.content[:120]}")
    print(f"  PII detections: {s.pii_detections}")
    print("  (Email and phone redacted; SSN audited only)\n")

stateloom.shutdown()


# ── 3. Block mode — reject requests containing PII ───────────────────
# Credit card numbers and SSNs cause the request to be rejected entirely.
# Works the same whether you use native SDKs or stateloom.chat().

print("=" * 60)
print("3. Block mode — request rejected if credit card detected")
print("=" * 60)

stateloom.init(
    pii=True,
    pii_rules=[
        PIIRule(pattern="credit_card", mode=PIIMode.BLOCK),
        PIIRule(pattern="ssn", mode=PIIMode.BLOCK),
        PIIRule(pattern="email", mode=PIIMode.REDACT),
    ],
)

with stateloom.session("pii-block-demo") as s:
    # Email gets redacted, not blocked — call succeeds
    for name in providers:
        text = call_native("Email me at user@example.com with the summary.", name)
        print(f"  {name} email (redacted, allowed): {text[:80]}")

    # Credit card via native SDK — blocked by monkey-patch
    try:
        call_native("Charge my card 4111-1111-1111-1111 for the order.")
    except stateloom.StateLoomPIIBlockedError as e:
        print(f"  Credit card via native SDK (blocked): {e}")

    # SSN via stateloom.chat() — same block
    try:
        stateloom.chat(
            model=CHAT_MODEL,
            messages=[{
                "role": "user",
                "content": "My SSN is 078-05-1120, use it for verification.",
            }],
        )
    except stateloom.StateLoomPIIBlockedError as e:
        print(f"  SSN via stateloom.chat (blocked): {e}")

    print(f"  PII detections: {s.pii_detections}\n")

stateloom.shutdown()


# ── 4. GDPR compliance via dashboard ──────────────────────────────────
# Enable GDPR from the dashboard: Compliance → Set Global Profile → GDPR
# This auto-configures strict PII rules:
#   - Block: SSN, national ID, IBAN, VAT ID
#   - Redact: email, phone
#   - Block local routing (data stays with approved providers)
#
# Once enabled in the dashboard, re-run this example — the GDPR rules
# will apply to all sessions automatically.

print("=" * 60)
print("4. GDPR compliance — enable in the dashboard")
print("=" * 60)

input("  Enable GDPR in the dashboard (Compliance → Set Global Profile → GDPR),\n"
      "  then press Enter to continue...")

stateloom.init(pii=True)

with stateloom.session("gdpr-demo") as s:
    # Email — redacted under GDPR (not blocked)
    text = call_native("Send the report to jane@example.com please.")
    print(f"  Email via native SDK (redacted, allowed): {text[:80]}")

    # IBAN — blocked under GDPR
    try:
        call_native("Wire funds to IBAN DE89370400440532013000 please.")
    except stateloom.StateLoomPIIBlockedError as e:
        print(f"  IBAN via native SDK (blocked): {e}")

    # SSN via stateloom.chat() — blocked under GDPR
    try:
        stateloom.chat(
            model=CHAT_MODEL,
            messages=[{
                "role": "user",
                "content": "My SSN is 078-05-1120, verify it.",
            }],
        )
    except stateloom.StateLoomPIIBlockedError as e:
        print(f"  SSN via stateloom.chat (blocked): {e}")

    print(f"  PII detections: {s.pii_detections}")

stateloom.shutdown()

print(f"\nDashboard: http://localhost:4782")
print("Check the PII Detections column in each session for detection counts.")
