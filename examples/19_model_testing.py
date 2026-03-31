"""
StateLoom Model Testing — Dark Launch Candidates Against Production

Demonstrates:
  1. Cloud-to-cloud model testing: run a candidate alongside every call
  2. Similarity scoring: automated quality comparison (difflib + semantic)
  3. Sampling: test a percentage of traffic (cost control)
  4. Smart skip logic: what gets tested vs skipped
  5. Migration readiness: use dashboard to assess when to switch models

Why Model Testing?
  You want to switch from gpt-4o to claude-haiku to save money.
  But how do you know the quality is good enough? Model testing runs the
  candidate model in parallel with every production call, compares responses
  via similarity scoring, and shows migration readiness in the dashboard —
  all without affecting your users. Candidate calls are fire-and-forget:
  the primary response is never delayed.

  Works with cloud-to-cloud (gpt-4o vs claude) and local models (Ollama).

Requires:

    pip install stateloom
    export OPENAI_API_KEY=sk-...
    export ANTHROPIC_API_KEY=sk-ant-...
    # or
    export GOOGLE_API_KEY=AIza...

    python examples/19_model_testing.py
"""

import os
import time

import stateloom

# ── Detect available providers ─────────────────────────────────────

available = []
if os.environ.get("OPENAI_API_KEY"):
    available.append("gpt-4o-mini")
if os.environ.get("ANTHROPIC_API_KEY"):
    available.append("claude-haiku-4-5-20251001")
if os.environ.get("GOOGLE_API_KEY"):
    available.append("gemini-2.5-flash")

if len(available) < 2:
    if len(available) == 1:
        print(f"Only one provider detected ({available[0]}).")
        print("Set a second API key for cross-provider model testing:")
    else:
        print("No API keys found.")
        print("Set at least two API keys for cross-provider model testing:")
    print("  export OPENAI_API_KEY=sk-...")
    print("  export ANTHROPIC_API_KEY=sk-ant-...")
    print("  export GOOGLE_API_KEY=AIza...")
    raise SystemExit(1)

MODEL = available[0]           # Primary (production)
CANDIDATE = available[1]       # Candidate (dark launch)

print(f"Primary model:   {MODEL}")
print(f"Candidate model: {CANDIDATE}")
print()


# =====================================================================
# 1. Basic Model Testing — Candidate Runs Alongside Production
# =====================================================================
# Every cloud LLM call automatically launches the candidate model in
# a background thread. The primary response is returned immediately;
# the candidate result is compared and persisted for dashboard review.

print("=" * 60)
print("1. Cloud-to-cloud model testing — candidate alongside production")
print("=" * 60)

stateloom.init(
    budget=5.0,
    console_output=True,
    shadow=True,
    shadow_model=CANDIDATE,
)

with stateloom.session("model-test-basic", budget=2.0) as s:
    prompts = [
        "What is dependency injection? One sentence.",
        "Explain the CAP theorem briefly.",
        "What is eventual consistency? One sentence.",
    ]
    for prompt in prompts:
        response = stateloom.chat(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
        )
        print(f"  Primary: {response.content[:80]}")

    # Give background threads time to complete similarity scoring
    time.sleep(3)

    print(f"\n  Total: ${s.total_cost:.4f} | {s.call_count} calls")

# Check model testing status
status = stateloom.shadow_status()
print(f"  Model testing: {'active' if status.get('enabled') else 'off'}")
print(f"  Candidate: {status.get('model', 'none')}")

print()


# =====================================================================
# 2. Similarity Scoring — Automated Quality Comparison
# =====================================================================
# StateLoom computes response similarity using difflib (default) or
# semantic embeddings (requires sentence-transformers). Scores range
# from 0.0 (completely different) to 1.0 (identical).

print("=" * 60)
print("2. Similarity scoring — how it works")
print("=" * 60)

from stateloom.middleware.similarity import compute_similarity

# Manual similarity comparison example
response_a = (
    "Dependency injection is a design pattern where objects receive their "
    "dependencies from external sources rather than creating them internally."
)
response_b = (
    "Dependency injection is a software design pattern in which an object's "
    "dependencies are provided by an external entity instead of being created "
    "by the object itself."
)

result = compute_similarity(response_a, response_b)
assert result is not None  # Both texts are non-empty
print(f"  Method: {result.method}")
print(f"  Score: {result.score:.2%}")
print(f"  Primary preview: {result.cloud_preview[:60]}...")
print(f"  Candidate preview: {result.local_preview[:60]}...")
print(f"  Length ratio: {result.length_ratio:.2f}")

print()
print("  Score interpretation:")
print("    0.7 - 1.0  High similarity (green)  — safe to migrate")
print("    0.4 - 0.7  Medium similarity (yellow) — review needed")
print("    0.0 - 0.4  Low similarity (red) — not ready")

print()


# =====================================================================
# 3. Sampling — Test a Percentage of Traffic
# =====================================================================
# In production, you might not want to test every single call.
# Set sample_rate to control what fraction of eligible calls get tested.

print("=" * 60)
print("3. Sampling — test 50% of production traffic")
print("=" * 60)

stateloom.init(
    budget=5.0,
    console_output=True,
    shadow=True,
    shadow_model=CANDIDATE,
    shadow_sample_rate=0.5,  # Test only 50% of calls
)

with stateloom.session("model-test-sampling", budget=2.0) as s:
    for i in range(6):
        stateloom.chat(
            model=MODEL,
            messages=[{"role": "user", "content": f"Define term #{i + 1}: microservices."}],
        )

    time.sleep(3)
    print(f"  Made {s.call_count} primary calls (roughly half tested by candidate)")

status = stateloom.shadow_status()
print(f"  Sample rate: {status.get('sample_rate', 1.0):.0%}")

print()


# =====================================================================
# 4. Smart Skip Logic — What Gets Tested
# =====================================================================
# Not all calls are eligible for model testing. StateLoom automatically
# skips calls that aren't suitable for comparison:

print("=" * 60)
print("4. Smart skip logic — automatic eligibility filtering")
print("=" * 60)

print("""  Skipped automatically:
    - Cached responses (no meaningful comparison)
    - PII-containing requests (safety: don't send PII to candidate)
    - Streaming calls (need full response for comparison)
    - Compliance-blocked (respects block_shadow in profiles)

  Also skipped for local candidates only:
    - Tool continuations (role="tool" messages)
    - Requests with tools/functions defined
    - Image content (vision models)
    - Requests needing realtime data (weather, stocks, news)
    - Oversized context (> max_context_tokens)

  Tested:
    - Standard text completions
    - Multi-turn conversations
    - System prompt + user message combos
""")

status = stateloom.shadow_status()
skip_stats = status.get("skip_stats", {})
if skip_stats:
    print("  Skip stats from this run:")
    for reason, count in skip_stats.items():
        if count > 0:
            print(f"    {reason}: {count}")

print()


# =====================================================================
# 5. Dashboard Integration — Migration Readiness
# =====================================================================

print("=" * 60)
print("5. Dashboard — migration readiness reports")
print("=" * 60)

print("""  The dashboard at http://localhost:4782 shows:
    - Model Test tab in each session detail view
    - Similarity score (%) for each tested call
    - Side-by-side response previews (primary vs candidate)
    - Latency ratio (is the candidate fast enough?)
    - Cost comparison per call
    - Aggregate migration readiness score across all tests

  Workflow:
    1. Enable shadow testing with sample_rate=0.1 (10%)
    2. Run for a few days on production traffic
    3. Check dashboard: "85% of calls score >0.7"
    4. Increase sample_rate to 0.5, then 1.0
    5. When confident, switch primary model
""")

print()


# ── Summary ──────────────────────────────────────────────────────────

print("=" * 60)
print("Summary")
print("=" * 60)
print(f"""
  This run tested: {MODEL} (primary) vs {CANDIDATE} (candidate)

  Model testing features:
    - Fire-and-forget: candidate never delays the primary response
    - Cloud-to-cloud: compare any two providers (gpt-4o vs claude)
    - Local models: also works with Ollama for cost-saving migration
    - Difflib similarity: fast text comparison (no deps)
    - Semantic similarity: embedding-based scoring (pip install stateloom[semantic])
    - Sampling: test 10-100% of traffic for cost control
    - Smart skipping: auto-skip streaming, PII, cached responses
    - PII safety: pre-scan requests, skip candidates if PII detected
    - Compliance-aware: respects block_shadow in compliance profiles

  Use cases:
    - Compare cloud models before switching providers
    - Validate local model quality before migration
    - Estimate cost savings from switching
    - Dark-launch a new model on live traffic safely
""")

print("Dashboard: http://localhost:4782")
print("Check the Model Testing tab in session details for similarity scores.")
