"""
StateLoom Compliance Enforcement — GDPR / HIPAA / CCPA

Demonstrates:
  1. Global GDPR compliance — no org needed, just init(compliance="gdpr")
  2. HIPAA profile — zero-retention mode, strict PII blocking
  3. CCPA profile — right to deletion, consumer privacy
  4. Tamper-proof audit trail with SHA-256 integrity hashes
  5. Right to Be Forgotten — purge all user data on request
  6. Org-level compliance — per-org/team profile overrides

Why Compliance?
  Regulated industries (healthcare, finance, government) need provable
  compliance when using LLMs. StateLoom provides declarative compliance
  profiles that automatically enforce data handling policies:
  - GDPR: EU data residency, 30-day retention, PII blocking
  - HIPAA: zero-retention logs, no payload storage, BAA-ready
  - CCPA: consumer deletion rights, 90-day retention
  Each action produces a tamper-proof audit event with legal rule
  citations (e.g., "GDPR-Art-17 — Right to erasure").

Requires:

    pip install stateloom
    export OPENAI_API_KEY=sk-...
    # or
    export ANTHROPIC_API_KEY=sk-ant-...
    export GOOGLE_API_KEY=AIza...

    python examples/18_compliance.py
"""

import os

import stateloom
from stateloom.compliance.profiles import ccpa_profile, gdpr_profile, hipaa_profile
from stateloom.core.config import ComplianceProfile

# ── Detect available provider ────────────────────────────────────────

if os.environ.get("OPENAI_API_KEY"):
    MODEL = "gpt-4o-mini"
elif os.environ.get("ANTHROPIC_API_KEY"):
    MODEL = "claude-haiku-4-5-20251001"
elif os.environ.get("GOOGLE_API_KEY"):
    MODEL = "gemini-2.5-flash"
else:
    print("Set at least one API key: OPENAI_API_KEY, ANTHROPIC_API_KEY, or GOOGLE_API_KEY")
    raise SystemExit(1)

print(f"Using model: {MODEL}\n")


# =====================================================================
# 1. Global GDPR Compliance — No Org Needed
# =====================================================================
# Just pass compliance="gdpr" to init(). Every session automatically
# gets GDPR enforcement: PII rules, streaming control, audit events.

print("=" * 60)
print("1. Global GDPR compliance — init(compliance='gdpr')")
print("=" * 60)

# Inspect the built-in GDPR profile
profile = gdpr_profile()
print(f"  Standard: {profile.standard}")
print(f"  Region: {profile.region}")
print(f"  Session TTL: {profile.session_ttl_days} days")
print(f"  Cache TTL: {profile.cache_ttl_seconds // 86400} days")
print(f"  Block local routing: {profile.block_local_routing}")
print(f"  Block shadow/model testing: {profile.block_shadow}")
print(f"  PII rules: {len(profile.pii_rules)} rules")
for rule in profile.pii_rules[:4]:
    print(f"    {rule.pattern}: {rule.mode}")

print()

# Initialize with global GDPR — applies to ALL sessions
stateloom.init(
    budget=5.0,
    console_output=True,
    compliance="gdpr",
)

# LLM call under GDPR — PII scanning is automatically enforced
with stateloom.session("gdpr-demo", budget=1.0) as s:
    text = stateloom.chat(
        model=MODEL,
        messages=[{"role": "user", "content": "Summarize GDPR Article 17 in one sentence."}],
    ).content
    print(f"  Response: {text[:120]}")
    print(f"  PII detections: {s.pii_detections}")

print()


# =====================================================================
# 2. HIPAA Profile — Healthcare Data Protection
# =====================================================================

print("=" * 60)
print("2. HIPAA compliance profile")
print("=" * 60)

profile = hipaa_profile()
print(f"  Standard: {profile.standard}")
print(f"  Zero-retention logs: {profile.zero_retention_logs}")
print(f"  Cache TTL: {profile.cache_ttl_seconds} (disabled)")
print(f"  PII rules:")
for rule in profile.pii_rules[:4]:
    print(f"    {rule.pattern}: {rule.mode}")

print()


# =====================================================================
# 3. CCPA Profile — California Consumer Privacy
# =====================================================================

print("=" * 60)
print("3. CCPA compliance profile")
print("=" * 60)

profile = ccpa_profile()
print(f"  Standard: {profile.standard}")
print(f"  Region: {profile.region}")
print(f"  Session TTL: {profile.session_ttl_days} days")
print(f"  PII rules:")
for rule in profile.pii_rules:
    print(f"    {rule.pattern}: {rule.mode}")

print()


# =====================================================================
# 4. Tamper-Proof Audit Trail
# =====================================================================
# Every compliance action produces an audit event with a SHA-256
# integrity hash. Tampering with the event changes the hash.

print("=" * 60)
print("4. Audit trail — tamper-proof compliance events")
print("=" * 60)

from stateloom.compliance.audit import compute_audit_hash
from stateloom.core.event import ComplianceAuditEvent

# Create a sample audit event (normally the middleware does this)
event = ComplianceAuditEvent(
    session_id="demo-session",
    step=1,
    compliance_standard="gdpr",
    action="pii_blocked",
    legal_rule="GDPR-Art-32 — Security of processing",
    justification="SSN detected in user message, blocked per GDPR profile",
    target_type="session",
    target_id="demo-session",
)

# Compute integrity hash
integrity_hash = compute_audit_hash(event)
event.integrity_hash = integrity_hash

print(f"  Action: {event.action}")
print(f"  Legal rule: {event.legal_rule}")
print(f"  Justification: {event.justification}")
print(f"  Integrity hash: {integrity_hash[:40]}...")
print(f"  Timestamp: {event.timestamp}")

# Verify: changing any field changes the hash
event_tampered = event.model_copy(update={"action": "pii_redacted"})
tampered_hash = compute_audit_hash(event_tampered)
print(f"  Tampered hash:  {tampered_hash[:40]}...")
print(f"  Hashes match: {integrity_hash == tampered_hash} (should be False)")

print()


# =====================================================================
# 5. Right to Be Forgotten — Purge User Data
# =====================================================================
# GDPR Article 17 requires the ability to delete all data associated
# with a user on request. StateLoom's purge engine handles sessions,
# events, cache entries, jobs, and virtual keys — with an audit trail.

print("=" * 60)
print("5. Right to Be Forgotten — GDPR Article 17")
print("=" * 60)

# First, create some sessions with user data
with stateloom.session("gdpr-user-001", budget=1.0) as s:
    stateloom.chat(
        model=MODEL,
        messages=[{"role": "user", "content": "Schedule a meeting for tomorrow."}],
    )

with stateloom.session("gdpr-user-002", budget=1.0) as s:
    stateloom.chat(
        model=MODEL,
        messages=[{"role": "user", "content": "What are the latest project updates?"}],
    )

print("  Created 2 sessions with user data.")

# User requests data deletion
result = stateloom.purge_user_data(
    user_identifier="gdpr-user",
    standard="gdpr",
)

print(f"  Purge result:")
print(f"    Sessions deleted: {result.get('sessions_deleted', 0)}")
print(f"    Events deleted: {result.get('events_deleted', 0)}")
print(f"    Audit event ID: {result.get('audit_event_id', 'N/A')}")

print()


# ── 6. Org-Level Compliance Override ─────────────────────────────────

print("=" * 60)
print("6. Org-level compliance — per-org profile overrides")
print("=" * 60)

# For multi-tenant setups, compliance can be set per org or team.
# Org/team profiles override the global compliance from init().
org = stateloom.create_organization(
    name="EU Healthcare Corp",
    compliance_profile="hipaa",  # HIPAA overrides the global GDPR
)
team = stateloom.create_team(org_id=org.id, name="Research Team")
cp = org.compliance_profile
print(f"  Org: {org.name} (compliance={cp.standard if cp else 'none'})")
print(f"  Resolution order: team > org > global (init)")
print(f"  This org uses HIPAA even though global is GDPR")

print()


# ── 7. Compliance Cleanup — Auto-Purge Expired Sessions ──────────────

print("=" * 60)
print("7. Compliance cleanup — auto-purge expired sessions")
print("=" * 60)

# GDPR profile has session_ttl_days=30. Sessions older than this are
# auto-purged when compliance_cleanup() runs (e.g., via cron or startup).
purged = stateloom.compliance_cleanup()
print(f"  Sessions purged by TTL: {purged}")

print()


# ── 8. Custom Compliance Profile ──────────────────────────────────────

print("=" * 60)
print("8. Custom compliance profile — PCI-DSS example")
print("=" * 60)

from stateloom.core.config import PIIRule

custom_profile = ComplianceProfile(
    standard="pci-dss",
    region="us",
    session_ttl_days=7,
    zero_retention_logs=False,
    block_streaming=True,
    pii_rules=[
        PIIRule(pattern="credit_card", mode="block"),
        PIIRule(pattern="ssn", mode="block"),
    ],
)
print(f"  Standard: {custom_profile.standard}")
print(f"  Session TTL: {custom_profile.session_ttl_days} days")
print(f"  Block streaming: {custom_profile.block_streaming}")
print(f"  PII rules: {len(custom_profile.pii_rules)}")
for rule in custom_profile.pii_rules:
    print(f"    {rule.pattern}: {rule.mode}")

print()


# ── Summary ──────────────────────────────────────────────────────────

print("=" * 60)
print("Summary")
print("=" * 60)
print("""
  Compliance features:
    Profiles:
      - GDPR: EU data residency, 30-day TTL, PII block/redact, consent checks
      - HIPAA: zero-retention logs, no caching, strict PII blocking
      - CCPA: 90-day TTL, right to deletion, consumer privacy

    Setup:
      - Global: init(compliance="gdpr") — applies to all sessions
      - Per-org: create_organization(compliance_profile="hipaa")
      - Per-team: create_team(compliance_profile="ccpa")
      - Resolution: team > org > global

    Audit Trail:
      - SHA-256 integrity hash on every compliance event
      - Legal rule citations (GDPR-Art-17, HIPAA-164.502, CCPA-1798.105)
      - Tamper-proof: changing any field invalidates the hash

    Data Rights:
      - Right to Be Forgotten: purge sessions, events, cache, jobs, keys
      - TTL enforcement: auto-purge expired sessions
      - Audit events record every deletion

    Enforcement:
      - Data residency: restrict which provider endpoints are allowed
      - Streaming control: disable streaming for full audit capture
      - PII rules: block/redact sensitive data per regulation
""")

print("Dashboard: http://localhost:4782")
print("Check the Compliance tab for audit trail, profiles, and purge history.")
