"""
StateLoom + Cohere — V2 SDK Integration

Demonstrates:
  - Auto-patching: cohere.ClientV2().chat() flows through StateLoom middleware
  - Cost tracking with Cohere's token-based pricing
  - Streaming via stream=True
  - Budget enforcement

StateLoom auto-patches both V2Client.chat (sync) and AsyncV2Client.chat (async).

Requires:

    pip install stateloom cohere
    export CO_API_KEY=...

    python examples/cohere.py
"""

import os
import sys

import stateloom

stateloom.init(
    budget=5.0,
    console_output=True,
)

try:
    import cohere
except ImportError:
    print("Cohere not installed. Install with: pip install cohere")
    sys.exit(1)

if not os.environ.get("CO_API_KEY"):
    print("Set CO_API_KEY environment variable.")
    print("Get a key at: https://dashboard.cohere.com/api-keys")
    sys.exit(1)

MODEL = "command-a-03-2025"
client = cohere.ClientV2()


# =====================================================================
# PART A — Basic chat with cost tracking
# =====================================================================

print("=" * 60)
print("A. Basic V2Client.chat — auto-patched, zero config")
print("=" * 60)

with stateloom.session("cohere-basic", budget=2.0) as s:
    response = client.chat(
        model=MODEL,
        messages=[{"role": "user", "content": "What is Cohere? Answer in 2 sentences."}],
    )
    print(f"  Response: {response.message.content[0].text[:150]}")
    print(f"  Cost: ${s.total_cost:.6f} | Tokens: {s.total_tokens} | Calls: {s.call_count}")

print()


# =====================================================================
# PART B — Multi-turn conversation
# =====================================================================

print("=" * 60)
print("B. Multi-turn conversation — costs accumulate")
print("=" * 60)

with stateloom.session("cohere-multi-turn", budget=2.0) as s:
    r1 = client.chat(
        model=MODEL,
        messages=[{"role": "user", "content": "What is RAG? One sentence."}],
    )
    text1 = r1.message.content[0].text
    print(f"  Turn 1: {text1[:120]}")

    r2 = client.chat(
        model=MODEL,
        messages=[
            {"role": "user", "content": "What is RAG? One sentence."},
            {"role": "assistant", "content": text1},
            {"role": "user", "content": "Give one concrete use case."},
        ],
    )
    text2 = r2.message.content[0].text
    print(f"  Turn 2: {text2[:120]}")
    print(f"  Total: ${s.total_cost:.6f} | {s.total_tokens} tokens | {s.call_count} calls")

print()


# =====================================================================
# PART C — Streaming
# =====================================================================

print("=" * 60)
print("C. Streaming via chat_stream()")
print("=" * 60)

with stateloom.session("cohere-streaming", budget=2.0) as s:
    print("  Streaming: ", end="", flush=True)
    stream = client.chat_stream(
        model=MODEL,
        messages=[{"role": "user", "content": "Write a haiku about embeddings."}],
    )
    for event in stream:
        if hasattr(event, "delta") and hasattr(event.delta, "message"):
            if event.delta.message and event.delta.message.content:
                text = event.delta.message.content.text
                if text:
                    print(text, end="", flush=True)
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
    with stateloom.session("cohere-budget", budget=0.001) as s:
        for i in range(10):
            client.chat(
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
  Cohere + StateLoom:
    - stateloom.init() auto-patches cohere.ClientV2.chat()
    - Both sync and async clients are patched
    - Streaming via chat_stream() fully supported
    - Cost tracking, budgets, PII, guardrails work transparently
    - Tested with model: {MODEL}
""")

print("Dashboard: http://localhost:4782")
