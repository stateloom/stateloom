"""
StateLoom Kill Switch — Global & Granular Emergency Traffic Control

Demonstrates:
  - Global kill switch: block all LLM traffic instantly
  - Granular rules: block by model pattern, provider, or environment
  - Error mode: raise StateLoomKillSwitchError (default)
  - Response mode: return a static dict instead of raising
  - Runtime toggling: activate / deactivate without restarting
  - Dashboard API integration: view kill switch events in the waterfall

Use cases:
  - Provider outage → block all traffic to the failing provider
  - Cost runaway → kill switch as an emergency brake
  - Compliance incident → instantly halt all LLM calls
  - Model deprecation → block specific models by glob pattern

Requires at least one provider API key:

    export OPENAI_API_KEY=sk-...
    # or
    export ANTHROPIC_API_KEY=sk-ant-...
    export GOOGLE_API_KEY=AIza...

    python examples/13_kill_switch.py
"""

import os

import stateloom

# ── Init ──────────────────────────────────────────────────────────────

stateloom.init(budget=5.0, console_output=True)

# ── Detect available provider and pick a model ────────────────────────

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


def make_call(prompt: str) -> str:
    """Make an LLM call via stateloom.chat()."""
    response = stateloom.chat(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content


# =====================================================================
# 1. Global Kill Switch — Block ALL Traffic
# =====================================================================
# Flip one switch and every LLM call in every session is blocked.
# Use this during a provider outage or compliance incident.

print("=" * 60)
print("1. Global kill switch — block all LLM traffic")
print("=" * 60)

# Normal call works fine
with stateloom.session("ks-normal-1", budget=1.0):
    text = make_call("What is a circuit breaker? One sentence.")
    print(f"  Before kill switch: {text[:100]}")

# Activate the global kill switch
stateloom.kill_switch(active=True, message="Provider outage — all traffic paused")
print("  Kill switch activated!")

# Now all calls are blocked
with stateloom.session("ks-blocked-1", budget=1.0):
    try:
        make_call("This should never reach the LLM.")
    except stateloom.StateLoomKillSwitchError as e:
        print(f"  Blocked: {e}")

# Deactivate — traffic resumes
stateloom.kill_switch(active=False)
print("  Kill switch deactivated.")

with stateloom.session("ks-resumed-1", budget=1.0):
    text = make_call("What is rate limiting? One sentence.")
    print(f"  After deactivation: {text[:100]}")

print()


# =====================================================================
# 2. Granular Rules — Block by Model Pattern
# =====================================================================
# Block specific models using glob patterns while others stay live.
# Useful for deprecating models or blocking expensive ones in dev.

print("=" * 60)
print("2. Granular rules — block by model glob pattern")
print("=" * 60)

# Add a rule that blocks models matching "gpt-4*"
stateloom.add_kill_switch_rule(
    model="gpt-4*",
    message="GPT-4 models blocked — use gpt-4o-mini instead",
    reason="cost_control",
)

# Show current rules
rules = stateloom.kill_switch_rules()
print(f"  Active rules: {len(rules)}")
for r in rules:
    print(f"    model={r.get('model')}, reason={r.get('reason')}")

# If we're using a gpt-4* model, this would be blocked.
# If using Anthropic or Gemini, the rule doesn't match — call goes through.
with stateloom.session("ks-rules-1", budget=1.0):
    try:
        text = make_call("What is a load balancer? One sentence.")
        print(f"  Call succeeded (model '{MODEL}' didn't match 'gpt-4*'): {text[:80]}")
    except stateloom.StateLoomKillSwitchError as e:
        print(f"  Blocked by rule: {e}")

# Clean up rules
stateloom.clear_kill_switch_rules()
print(f"  Rules cleared: {len(stateloom.kill_switch_rules())} remaining")
print()


# =====================================================================
# 3. Provider-Level Block
# =====================================================================
# Block an entire provider while others remain operational.
# Useful during a provider-specific outage.

print("=" * 60)
print("3. Provider-level block — simulate provider outage")
print("=" * 60)

# Block OpenAI as a provider
stateloom.add_kill_switch_rule(
    provider="openai",
    message="OpenAI experiencing elevated error rates",
    reason="provider_outage",
)

print("  Rule added: block provider=openai")

with stateloom.session("ks-provider-1", budget=1.0):
    try:
        text = make_call("What is DNS? One sentence.")
        print(f"  Call succeeded (provider is not openai): {text[:80]}")
    except stateloom.StateLoomKillSwitchError as e:
        print(f"  Blocked: {e}")

stateloom.clear_kill_switch_rules()
print("  Rules cleared.")
print()


# =====================================================================
# 4. Response Mode — Graceful Degradation
# =====================================================================
# Instead of raising an exception, return a static response dict.
# The caller can check for the kill_switch key and degrade gracefully
# (show a fallback message, retry later, etc.)

print("=" * 60)
print("4. Response mode — graceful degradation instead of error")
print("=" * 60)

# Switch to response mode
gate = stateloom.get_gate()
gate.config.kill_switch_response_mode = "response"

stateloom.kill_switch(active=True, message="Scheduled maintenance window")
print("  Kill switch active in response mode.")

with stateloom.session("ks-response-mode-1", budget=1.0):
    # In response mode, the call doesn't raise — it returns a dict
    response = stateloom.chat(
        model=MODEL,
        messages=[{"role": "user", "content": "This won't reach the LLM."}],
    )
    # The response content will be empty (static dict has no real content)
    print(f"  Response content: '{response.content}' (empty — kill switch response)")
    print(f"  Raw response: {response.raw}")

# Restore defaults
stateloom.kill_switch(active=False)
gate.config.kill_switch_response_mode = "error"
print("  Restored to error mode.")
print()


# =====================================================================
# 5. Multiple Rules — First Match Wins
# =====================================================================
# You can stack multiple rules. The first matching rule determines
# the error message and reason. Rules are checked in order.

print("=" * 60)
print("5. Multiple rules — first match wins")
print("=" * 60)

stateloom.add_kill_switch_rule(
    model="gpt-4o",
    message="gpt-4o deprecated — migrate to gpt-4o-mini",
    reason="model_deprecated",
)
stateloom.add_kill_switch_rule(
    model="claude-3-opus*",
    message="claude-3-opus temporarily disabled",
    reason="cost_control",
)
stateloom.add_kill_switch_rule(
    provider="gemini",
    message="Gemini API under maintenance",
    reason="maintenance",
)

rules = stateloom.kill_switch_rules()
print(f"  {len(rules)} rules active:")
for r in rules:
    model_str = r.get("model") or r.get("provider") or "?"
    print(f"    {model_str}: {r.get('message')}")

# Only matching rules block — non-matching models pass through
with stateloom.session("ks-multi-rule-1", budget=1.0):
    try:
        text = make_call("What is HTTPS? One sentence.")
        print(f"  Call result: {text[:80]}")
    except stateloom.StateLoomKillSwitchError as e:
        print(f"  Blocked: {e}")

stateloom.clear_kill_switch_rules()
print()


# =====================================================================
# 6. Kill Switch + Budget — Defense in Depth
# =====================================================================
# Combine kill switch with budget enforcement for layered protection.
# Kill switch blocks instantly (before any processing);
# budget enforcement catches cost overruns within allowed traffic.

print("=" * 60)
print("6. Kill switch + budget — layered defense")
print("=" * 60)

with stateloom.session("ks-layered-1", budget=0.50) as s:
    # First call — everything normal
    text = make_call("What is TLS? One sentence.")
    print(f"  Call 1 (normal): {text[:80]}")
    print(f"  Cost: ${s.total_cost:.4f}")

    # Simulate: ops team activates kill switch mid-session
    stateloom.kill_switch(active=True, message="Emergency: cost anomaly detected")
    print("  Kill switch activated mid-session!")

    # Second call — blocked by kill switch (never reaches budget check)
    try:
        make_call("This is blocked before budget is even checked.")
    except stateloom.StateLoomKillSwitchError as e:
        print(f"  Call 2 blocked: {e}")

    stateloom.kill_switch(active=False)

print()


# ── Summary ──────────────────────────────────────────────────────────

print("=" * 60)
print("Summary")
print("=" * 60)
print("""
  Kill switch features:
    - Global toggle: stateloom.kill_switch(active=True/False)
    - Granular rules: stateloom.add_kill_switch_rule(model="gpt-4*")
    - Error mode (default): raises StateLoomKillSwitchError
    - Response mode: returns a static dict for graceful degradation
    - Glob patterns: "gpt-4*", "claude-*", etc.
    - Filters: model, provider, environment, agent_version
    - Dashboard: kill switch events appear in the waterfall trace
    - Position: runs first in the middleware chain (before everything)
""")

print("Dashboard: http://localhost:4782")
print("Check the Kill Switch events in session waterfalls and the Kill Switch tab.")
