"""
StateLoom Caching — Exact-Match & Cost Savings

Demonstrates:
  - Automatic request caching (exact-match, enabled by default)
  - Cache hits return instant responses with zero LLM cost
  - Session-level cache hit tracking
  - Global vs session-scoped caching
  - TTL-based cache expiration
  - Cache savings accumulation

Run:

    export OPENAI_API_KEY=sk-...
    python examples/06_caching.py

    # or with Anthropic / Gemini
    export ANTHROPIC_API_KEY=sk-ant-...
    python examples/06_caching.py
"""

import os
import stateloom

# ── Detect provider ──────────────────────────────────────────────────

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


# ── 1. Basic caching (exact match) ──────────────────────────────────
# Caching is enabled by default. Identical requests return cached
# responses instantly — no LLM call, no cost.

print("=" * 60)
print("1. Exact-match caching (default, global scope)")
print("=" * 60)

stateloom.init(cache=True)  # cache is on by default, shown explicitly here

PROMPT = "What is a hash table? One sentence."

with stateloom.session("cache-demo-1", budget=1.0) as s:
    # First call — cache miss, hits the LLM
    response = stateloom.chat(
        model=MODEL,
        messages=[{"role": "user", "content": PROMPT}],
    )
    cost_after_first = s.total_cost
    print(f"  Call 1 (miss): {response.content[:100]}")
    print(f"    Cost: ${cost_after_first:.6f} | Cache hits: {s.cache_hits}")

    # Second call — identical request, served from cache
    response = stateloom.chat(
        model=MODEL,
        messages=[{"role": "user", "content": PROMPT}],
    )
    cost_after_second = s.total_cost
    print(f"  Call 2 (hit):  {response.content[:100]}")
    print(f"    Cost: ${cost_after_second:.6f} | Cache hits: {s.cache_hits}")
    print(f"    Saved: ${cost_after_first - (cost_after_second - cost_after_first):.6f}")

    # Third call — still cached
    response = stateloom.chat(
        model=MODEL,
        messages=[{"role": "user", "content": PROMPT}],
    )
    print(f"  Call 3 (hit):  Cache hits: {s.cache_hits}")
    print(f"    Total cost: ${s.total_cost:.6f} (only 1 LLM call charged)")

print()


# ── 2. Global scope — cache shared across sessions ──────────────────
# With cache_scope="global" (default), a response cached in one session
# is available to all subsequent sessions.

print("=" * 60)
print("2. Global cache — shared across sessions")
print("=" * 60)

SHARED_PROMPT = "Define 'idempotency' in one sentence."

# Session A — populates the cache
with stateloom.session("cache-session-a", budget=1.0) as sa:
    response = stateloom.chat(
        model=MODEL,
        messages=[{"role": "user", "content": SHARED_PROMPT}],
    )
    print(f"  Session A: cost=${sa.total_cost:.6f}, cache hits={sa.cache_hits}")

# Session B — same prompt, different session, gets cache hit
with stateloom.session("cache-session-b", budget=1.0) as sb:
    response = stateloom.chat(
        model=MODEL,
        messages=[{"role": "user", "content": SHARED_PROMPT}],
    )
    print(f"  Session B: cost=${sb.total_cost:.6f}, cache hits={sb.cache_hits}")
    print(f"    Session B paid nothing — served from global cache!")

print()


# ── 3. Session-scoped cache ─────────────────────────────────────────
# With cache_scope="session", caches are isolated per session.
# Useful when responses depend on session-specific context.

print("=" * 60)
print("3. Session-scoped cache (isolated per session)")
print("=" * 60)

stateloom.init(cache=True, cache_scope="session")

SCOPED_PROMPT = "What is a mutex? One sentence."

# Session X — cache miss, then hit
with stateloom.session("scoped-session-x", budget=1.0) as sx:
    stateloom.chat(model=MODEL, messages=[{"role": "user", "content": SCOPED_PROMPT}])
    stateloom.chat(model=MODEL, messages=[{"role": "user", "content": SCOPED_PROMPT}])
    print(f"  Session X: {sx.call_count} calls, {sx.cache_hits} cache hit(s), cost=${sx.total_cost:.6f}")

# Session Y — same prompt, but session-scoped so it's a miss
with stateloom.session("scoped-session-y", budget=1.0) as sy:
    stateloom.chat(model=MODEL, messages=[{"role": "user", "content": SCOPED_PROMPT}])
    print(f"  Session Y: {sy.call_count} calls, {sy.cache_hits} cache hit(s), cost=${sy.total_cost:.6f}")
    print(f"    Session Y had a cache miss — scoped caches don't share!")

print()


# ── 4. Multiple unique prompts vs repeated prompts ──────────────────
# Shows the cost difference between varied and repeated requests.

print("=" * 60)
print("4. Cost comparison: unique vs repeated prompts")
print("=" * 60)

stateloom.init(cache=True, cache_scope="global")

QUESTIONS = [
    "What is a binary tree? One sentence.",
    "What is a linked list? One sentence.",
    "What is a stack? One sentence.",
]

# All unique prompts
with stateloom.session("unique-prompts", budget=1.0) as s_unique:
    for q in QUESTIONS:
        stateloom.chat(model=MODEL, messages=[{"role": "user", "content": q}])
    print(f"  3 unique prompts:   cost=${s_unique.total_cost:.6f}, cache hits={s_unique.cache_hits}")

# Same 3 prompts repeated (all cached from previous session via global scope)
with stateloom.session("repeated-prompts", budget=1.0) as s_repeat:
    for q in QUESTIONS:
        stateloom.chat(model=MODEL, messages=[{"role": "user", "content": q}])
    print(f"  3 repeated prompts: cost=${s_repeat.total_cost:.6f}, cache hits={s_repeat.cache_hits}")

savings = s_unique.total_cost - s_repeat.total_cost
print(f"  Savings from cache: ${savings:.6f} ({s_repeat.cache_hits} hits)")

print()


# ── 5. Loop detection (opt-in) ──────────────────────────────────────
# When enabled, loop detection blocks requests that repeat more than
# `loop_threshold` times in a session. This protects against runaway
# agent loops that burn budget on the same failing prompt.

print("=" * 60)
print("5. Loop detection (opt-in, threshold=5)")
print("=" * 60)

stateloom.init(cache=False, loop_detection=True, loop_threshold=5)

LOOP_PROMPT = "What is a semaphore? One sentence."

with stateloom.session("loop-demo", budget=1.0) as s:
    for i in range(7):
        response = stateloom.chat(
            model=MODEL,
            messages=[{"role": "user", "content": LOOP_PROMPT}],
        )
        text = response.content[:80] if hasattr(response, "content") else str(response)[:80]
        blocked = "[BLOCKED]" if "loop detected" in text.lower() else ""
        print(f"  Call {i + 1}: {text}  {blocked}")

    print(f"  Calls: {s.call_count} | Cost: ${s.total_cost:.6f}")
    print(f"  Calls 1-4 succeeded, call 5+ blocked by loop detector")

print(f"\nDashboard: http://localhost:4782")
