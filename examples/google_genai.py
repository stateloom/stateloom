"""
StateLoom + google-genai — New Google GenAI SDK Integration

Demonstrates:
  - Auto-patching: google.genai.Client() calls flow through StateLoom middleware
  - Both generate_content (non-streaming) and generate_content_stream (streaming)
  - Cost tracking, budget enforcement, and session grouping
  - Multi-turn conversation tracking

The new google-genai SDK uses per-client API keys and keyword-only arguments.
StateLoom auto-patches both Models (sync) and AsyncModels (async) classes.

Requires:

    pip install stateloom google-genai
    export GOOGLE_API_KEY=AIza...

    python examples/google_genai.py
"""

import os
import sys

import stateloom

stateloom.init(
    budget=5.0,
    console_output=True,
)

try:
    from google import genai
except ImportError:
    print("google-genai not installed. Install with: pip install google-genai")
    sys.exit(1)

if not os.environ.get("GOOGLE_API_KEY"):
    print("Set GOOGLE_API_KEY environment variable.")
    print("Get a key at: https://aistudio.google.com/apikey")
    sys.exit(1)

MODEL = "gemini-2.5-flash"
client = genai.Client()


# =====================================================================
# PART A — Basic generate_content with cost tracking
# =====================================================================

print("=" * 60)
print("A. Basic generate_content — auto-patched, zero config")
print("=" * 60)

with stateloom.session("genai-basic", budget=2.0) as s:
    response = client.models.generate_content(
        model=MODEL,
        contents="What is the google-genai Python SDK? Answer in 2 sentences.",
    )
    print(f"  Response: {response.text[:150]}")
    print(f"  Cost: ${s.total_cost:.6f} | Tokens: {s.total_tokens} | Calls: {s.call_count}")

print()


# =====================================================================
# PART B — Multi-turn conversation
# =====================================================================

print("=" * 60)
print("B. Multi-turn conversation — costs accumulate in session")
print("=" * 60)

with stateloom.session("genai-multi-turn", budget=2.0) as s:
    r1 = client.models.generate_content(
        model=MODEL,
        contents="Explain what RAG is in one sentence.",
    )
    print(f"  Turn 1: {r1.text[:120]}")
    print(f"  Cost so far: ${s.total_cost:.6f}")

    # Multi-turn using Content dicts
    r2 = client.models.generate_content(
        model=MODEL,
        contents=[
            {"role": "user", "parts": [{"text": "Explain what RAG is in one sentence."}]},
            {"role": "model", "parts": [{"text": r1.text}]},
            {"role": "user", "parts": [{"text": "Now give one concrete use case."}]},
        ],
    )
    print(f"  Turn 2: {r2.text[:120]}")
    print(f"  Total: ${s.total_cost:.6f} | {s.total_tokens} tokens | {s.call_count} calls")

print()


# =====================================================================
# PART C — Streaming via generate_content_stream
# =====================================================================

print("=" * 60)
print("C. Streaming — generate_content_stream auto-patched")
print("=" * 60)

with stateloom.session("genai-streaming", budget=2.0) as s:
    print("  Streaming: ", end="", flush=True)
    stream = client.models.generate_content_stream(
        model=MODEL,
        contents="Write a haiku about cloud computing.",
    )
    for chunk in stream:
        if chunk.text:
            print(chunk.text, end="", flush=True)
    print()
    print(f"  Cost: ${s.total_cost:.6f} | Tokens: {s.total_tokens} | Calls: {s.call_count}")

print()


# =====================================================================
# PART D — Budget enforcement
# =====================================================================

print("=" * 60)
print("D. Budget enforcement — works transparently")
print("=" * 60)

try:
    with stateloom.session("genai-budget", budget=0.001) as s:
        for i in range(10):
            client.models.generate_content(
                model=MODEL,
                contents=f"Write a haiku about the number {i + 1}.",
            )
            print(f"  Call {i + 1}: ${s.total_cost:.6f}")
except stateloom.StateLoomBudgetError as e:
    print(f"  Budget enforced: {e}")

print()


# ── Summary ──────────────────────────────────────────────────────────

print("=" * 60)
print("Summary")
print("=" * 60)
print(f"""
  google-genai + StateLoom:
    - stateloom.init() auto-patches google.genai.Client methods
    - generate_content AND generate_content_stream both patched
    - Sync and async (Models + AsyncModels) all covered
    - Cost tracking, budgets, PII, guardrails work transparently
    - No code changes needed in your existing google-genai calls
    - Tested with model: {MODEL}
""")

print("Dashboard: http://localhost:4782")
print("Check per-session cost breakdown and the waterfall timeline.")
