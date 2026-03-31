"""
StateLoom Quick Start — Multi-Provider Session

Demonstrates:
  - Auto-patching OpenAI, Anthropic, and Gemini SDKs
  - Production session with budget enforcement and PII scanning
  - Per-model cost and token tracking across providers
  - The unified stateloom.Client as an alternative

Run with one provider at a time or all at once:

    export OPENAI_API_KEY=sk-...
    python examples/01_quick_start.py

    export ANTHROPIC_API_KEY=sk-ant-...
    python examples/01_quick_start.py

    export GOOGLE_API_KEY=AIza...
    python examples/01_quick_start.py
"""

import os
import stateloom

# ── Init ──────────────────────────────────────────────────────────────

stateloom.init(
    budget=2.0,    # default per-session budget (USD)
    pii=True,      # scan every request for PII
)

# ── Detect which SDKs + keys are available ────────────────────────────

providers_available = []

if os.environ.get("OPENAI_API_KEY"):
    try:
        import openai
        providers_available.append("openai")
    except ImportError:
        print("  openai package not installed — pip install openai")

if os.environ.get("ANTHROPIC_API_KEY"):
    try:
        import anthropic
        providers_available.append("anthropic")
    except ImportError:
        print("  anthropic package not installed — pip install anthropic")

if os.environ.get("GOOGLE_API_KEY"):
    try:
        import google.generativeai as genai
        providers_available.append("gemini")
    except ImportError:
        print("  google-generativeai package not installed — pip install google-generativeai")

if not providers_available:
    print("No provider API keys found. Set at least one of:")
    print("  export OPENAI_API_KEY=sk-...")
    print("  export ANTHROPIC_API_KEY=sk-ant-...")
    print("  export GOOGLE_API_KEY=AIza...")
    raise SystemExit(1)

print(f"Providers available: {', '.join(providers_available)}\n")

# ── Run a production session ──────────────────────────────────────────
# Everything inside this block is tracked as one session: cost, tokens,
# PII detections, cache hits — across all providers.

PROMPT = "Explain what a circuit breaker pattern is in distributed systems, in 2-3 sentences."

with stateloom.session("quick-start-demo", budget=2.0) as s:

    # ── OpenAI ────────────────────────────────────────────────────────
    if "openai" in providers_available:
        print("── OpenAI ──")
        client = openai.OpenAI()
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": PROMPT}],
        )
        print(f"   {response.choices[0].message.content[:120]}...")
        print()

    # ── Anthropic ─────────────────────────────────────────────────────
    if "anthropic" in providers_available:
        print("── Anthropic ──")
        client = anthropic.Anthropic()
        message = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=256,
            messages=[{"role": "user", "content": PROMPT}],
        )
        print(f"   {message.content[0].text[:120]}...")
        print()

    # ── Gemini ────────────────────────────────────────────────────────
    if "gemini" in providers_available:
        print("── Gemini ──")
        genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
        model = genai.GenerativeModel("gemini-2.5-flash")
        response = model.generate_content(PROMPT)
        print(f"   {response.text[:120]}...")
        print()

    # ── Session summary ───────────────────────────────────────────────
    print("=" * 60)
    print("SESSION SUMMARY")
    print("=" * 60)
    print(f"  Session ID:    {s.id}")
    print(f"  Total cost:    ${s.total_cost:.4f}")
    print(f"  Total tokens:  {s.total_tokens}")
    print(f"    Prompt:      {s.total_prompt_tokens}")
    print(f"    Completion:  {s.total_completion_tokens}")
    print(f"  LLM calls:     {s.call_count}")
    print(f"  Cache hits:    {s.cache_hits}")
    print(f"  PII detected:  {s.pii_detections}")
    print()
    if s.cost_by_model:
        print("  Cost by model:")
        for model_name, cost in s.cost_by_model.items():
            tokens = s.tokens_by_model.get(model_name, {})
            print(f"    {model_name}: ${cost:.4f} ({tokens.get('total', 0)} tokens)")
    print()
    print(f"  Budget remaining: ${s.budget - s.total_cost:.4f} of ${s.budget:.2f}")
    print(f"  Dashboard: http://localhost:4782")


# ── Bonus: unified Client (no SDK imports needed) ────────────────────
# stateloom.Client dispatches to the right provider based on model name.
# Uncomment to try:
#
# with stateloom.session("unified-client-demo", budget=1.0) as s:
#     response = stateloom.chat(model="gpt-4o-mini", messages=[{"role": "user", "content": "What is 2 + 2?"}])
#     print(response.content)
#     print(f"Cost: ${s.total_cost:.4f}")
