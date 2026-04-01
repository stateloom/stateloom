"""
StateLoom Circuit Breaker — Automatic Provider Failover

Demonstrates:
  1. Circuit state monitoring (closed / open / half-open)
  2. Failure accumulation and automatic circuit trip
  3. Tier-based fallback model suggestions
  4. Manual circuit reset
  5. Circuit breaker + budget enforcement (layered protection)

Why Circuit Breaker?
  LLM providers go down. When OpenAI returns 500s, you don't want every
  request in your system to wait 30 seconds and fail. The circuit breaker
  detects provider failures via a sliding window, "opens" the circuit to
  block traffic immediately, and suggests a same-tier fallback model from
  a healthy provider. After a recovery timeout, it sends a lightweight
  probe to check if the provider is back — and automatically closes the
  circuit when it recovers.

Requires:

    pip install stateloom
    export OPENAI_API_KEY=sk-...
    # and ideally a second provider for fallback:
    export ANTHROPIC_API_KEY=sk-ant-...
    # or
    export GOOGLE_API_KEY=AIza...

    python examples/circuit_breaker.py
"""

import os

import stateloom

# ── Init with circuit breaker enabled ────────────────────────────────

stateloom.init(
    budget=5.0,
    console_output=True,
    circuit_breaker=True,
)

# ── Detect available providers ──────────────────────────────────────

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
# 1. Circuit Status — All Providers Healthy
# =====================================================================

print("=" * 60)
print("1. Circuit status — all providers healthy")
print("=" * 60)

status = stateloom.circuit_breaker_status()
print(f"  Enabled: {status['enabled']}")
for provider, info in status.get("providers", {}).items():
    state = info.get("state", "unknown")
    failures = info.get("failure_count", 0)
    threshold = info.get("failure_threshold", 0)
    print(f"  {provider}: {state} ({failures}/{threshold} failures)")

print()


# =====================================================================
# 2. Normal Operation — Calls Succeed Through Closed Circuit
# =====================================================================

print("=" * 60)
print("2. Normal operation — calls flow through closed circuit")
print("=" * 60)

with stateloom.session("cb-normal-demo", budget=1.0) as s:
    text = stateloom.chat(
        model=MODEL,
        messages=[{"role": "user", "content": "What is a circuit breaker pattern? One sentence."}],
    ).content
    print(f"  Response: {text[:120]}")
    print(f"  Cost: ${s.total_cost:.4f}")

# Circuit still closed after successful call
status = stateloom.circuit_breaker_status()
for provider, info in status.get("providers", {}).items():
    if info.get("failure_count", 0) > 0:
        print(f"  {provider}: {info['state']} ({info['failure_count']} failures)")
    else:
        print("  All circuits: closed (healthy)")
        break

print()


# =====================================================================
# 3. Tier-Based Fallback Mapping
# =====================================================================
# When a provider's circuit opens, StateLoom suggests a same-tier
# fallback from a healthy provider. Tier mapping:
#   tier-1-flagship: gpt-4o, claude-sonnet, gemini-1.5-pro
#   tier-2-fast:     gpt-4o-mini, claude-haiku, gemini-flash
#   tier-3-legacy:   gpt-3.5-turbo, claude-3-haiku

print("=" * 60)
print("3. Tier-based fallback — same quality, different provider")
print("=" * 60)

print("  Built-in tier mapping:")
print("    Tier 1 (flagship): gpt-4o <-> claude-sonnet <-> gemini-1.5-pro")
print("    Tier 2 (fast):     gpt-4o-mini <-> claude-haiku <-> gemini-flash")
print()

# You can also set explicit fallbacks
stateloom.init(
    budget=5.0,
    console_output=True,
    circuit_breaker=True,
    circuit_breaker_fallback_map={
        "gpt-4o": "claude-sonnet-4-20250514",
        "gpt-4o-mini": "claude-haiku-4-5-20251001",
    },
)

print("  Custom fallback map configured:")
print("    gpt-4o -> claude-sonnet-4-20250514")
print("    gpt-4o-mini -> claude-haiku-4-5-20251001")

print()


# =====================================================================
# 4. Manual Circuit Reset
# =====================================================================
# Ops teams can manually reset a circuit after verifying a provider
# has recovered, without waiting for the automatic recovery timeout.

print("=" * 60)
print("4. Manual circuit reset — ops team intervention")
print("=" * 60)

# Reset a specific provider
result = stateloom.reset_circuit_breaker("openai")
print(f"  Reset openai circuit: {'success' if result else 'not tracked'}")

# Check status after reset
status = stateloom.circuit_breaker_status()
for provider, info in status.get("providers", {}).items():
    print(f"  {provider}: {info.get('state', 'unknown')}")

print()


# =====================================================================
# 5. Circuit Breaker + Budget — Layered Protection
# =====================================================================
# Circuit breaker protects against provider outages.
# Budget enforcement protects against cost overruns.
# Together they provide layered operational safety.

print("=" * 60)
print("5. Layered protection — circuit breaker + budget")
print("=" * 60)

with stateloom.session("cb-layered-demo", budget=0.50) as s:
    text = stateloom.chat(
        model=MODEL,
        messages=[{"role": "user", "content": "What is observability? One sentence."}],
    ).content
    print(f"  Call succeeded: {text[:100]}")
    print(f"  Cost: ${s.total_cost:.4f} / ${s.budget:.4f} budget")

print()


# ── Summary ──────────────────────────────────────────────────────────

print("=" * 60)
print("Summary")
print("=" * 60)
print("""
  Circuit breaker features:
    - Per-provider health tracking (sliding window failure count)
    - Three states: closed (healthy) -> open (failing) -> half-open (probing)
    - Automatic recovery: synthetic probe after timeout, auto-close on success
    - Tier-based fallback: same-quality model from a healthy provider
    - Custom fallback maps: explicit model-to-model overrides
    - Manual reset: ops teams can force-close a circuit
    - Non-retryable: circuit breaker errors don't trigger blast radius

  State machine:
    CLOSED  -- N failures in window --> OPEN
    OPEN    -- recovery timeout     --> HALF-OPEN
    HALF-OPEN -- probe succeeds     --> CLOSED (recovered!)
    HALF-OPEN -- probe fails        --> OPEN (still down)
""")

print("Dashboard: http://localhost:4782")
print("Check circuit breaker status and provider health in the Safety tab.")
