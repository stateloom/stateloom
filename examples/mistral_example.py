"""
StateLoom + Mistral — SDK Integration

Demonstrates:
  - Auto-patching: Mistral().chat.complete() flows through StateLoom middleware
  - Cost tracking with Mistral's token-based pricing
  - Streaming via chat.stream()
  - Budget enforcement

StateLoom auto-patches both Chat.complete (sync) and Chat.complete_async (async).

Requires:

    pip install stateloom mistralai
    export MISTRAL_API_KEY=...

    python examples/mistral.py
"""

import os
import sys

import stateloom

stateloom.init(
    budget=5.0,
    console_output=True,
)

try:
    from mistralai.client import Mistral
except ImportError:
    try:
        from mistralai import Mistral  # type: ignore[attr-defined]
    except ImportError:
        print("Mistral not installed. Install with: pip install mistralai")
        sys.exit(1)

if not os.environ.get("MISTRAL_API_KEY"):
    print("Set MISTRAL_API_KEY environment variable.")
    print("Get a key at: https://console.mistral.ai/api-keys")
    sys.exit(1)

MODEL = "mistral-medium-2505"
client = Mistral(api_key=os.environ["MISTRAL_API_KEY"])


# =====================================================================
# PART A — Basic chat with cost tracking
# =====================================================================

print("=" * 60)
print("A. Basic chat.complete — auto-patched, zero config")
print("=" * 60)

with stateloom.session("mistral-basic", budget=2.0) as s:
    response = client.chat.complete(
        model=MODEL,
        messages=[{"role": "user", "content": "What is Mistral AI? Answer in 2 sentences."}],
    )
    print(f"  Response: {response.choices[0].message.content[:150]}")
    print(f"  Cost: ${s.total_cost:.6f} | Tokens: {s.total_tokens} | Calls: {s.call_count}")

print()


# =====================================================================
# PART B — Multi-turn conversation
# =====================================================================

print("=" * 60)
print("B. Multi-turn conversation — costs accumulate")
print("=" * 60)

with stateloom.session("mistral-multi-turn", budget=2.0) as s:
    r1 = client.chat.complete(
        model=MODEL,
        messages=[{"role": "user", "content": "What is mixture of experts? One sentence."}],
    )
    text1 = r1.choices[0].message.content
    print(f"  Turn 1: {text1[:120]}")

    r2 = client.chat.complete(
        model=MODEL,
        messages=[
            {"role": "user", "content": "What is mixture of experts? One sentence."},
            {"role": "assistant", "content": text1},
            {"role": "user", "content": "Give one concrete benefit."},
        ],
    )
    text2 = r2.choices[0].message.content
    print(f"  Turn 2: {text2[:120]}")
    print(f"  Total: ${s.total_cost:.6f} | {s.total_tokens} tokens | {s.call_count} calls")

print()


# =====================================================================
# PART C — Streaming
# =====================================================================

print("=" * 60)
print("C. Streaming via chat.stream()")
print("=" * 60)

with stateloom.session("mistral-streaming", budget=2.0) as s:
    print("  Streaming: ", end="", flush=True)
    stream = client.chat.stream(
        model=MODEL,
        messages=[{"role": "user", "content": "Write a haiku about open-source AI."}],
    )
    for event in stream:
        delta = event.data.choices[0].delta.content
        if delta:
            print(delta, end="", flush=True)
    print()
    print(f"  Cost: ${s.total_cost:.6f} | Tokens: {s.total_tokens}")

print()


# =====================================================================
# PART D — Budget enforcement
# =====================================================================

print("=" * 60)
print("D. Budget enforcement")
print("=" * 60)

try:
    with stateloom.session("mistral-budget", budget=0.0001) as s:
        for i in range(10):
            client.chat.complete(
                model=MODEL,
                messages=[{"role": "user", "content": f"Write a haiku about {i + 1}."}],
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
  Mistral + StateLoom:
    - stateloom.init() auto-patches Mistral().chat.complete()
    - Both sync and async are patched
    - Streaming via chat.stream() fully supported
    - Cost tracking, budgets, PII, guardrails work transparently
    - Tested with model: {MODEL}
""")

print("Dashboard: http://localhost:4782")
