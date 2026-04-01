"""
StateLoom Durable Resumption — Crash Recovery & Semantic Retries

Demonstrates:
  - Durable sessions (Temporal-like checkpointing)
  - Crash recovery — cached LLM responses replay on restart
  - Semantic retries with retry_loop (self-healing bad output)
  - durable_task decorator (session + retries combined)
  - Named checkpoints for progress tracking

Run:

    export OPENAI_API_KEY=sk-...
    python examples/durable_resumption.py

    # Run it AGAIN to see durable replay in action:
    python examples/durable_resumption.py

    # or with Anthropic / Gemini
    export ANTHROPIC_API_KEY=sk-ant-...
    python examples/durable_resumption.py
"""

import json
import os

import stateloom

# ── Init ──────────────────────────────────────────────────────────────

stateloom.init(budget=5.0)

# Detect provider and set up both stateloom.chat() and native SDK client
PROVIDER = None
MODEL = None
sdk_client = None  # Native SDK client (auto-patched by StateLoom)

if os.environ.get("OPENAI_API_KEY"):
    PROVIDER = "openai"
    MODEL = "gpt-4o-mini"
    import openai

    sdk_client = openai.OpenAI()
elif os.environ.get("ANTHROPIC_API_KEY"):
    PROVIDER = "anthropic"
    MODEL = "claude-haiku-4-5-20251001"
    import anthropic

    sdk_client = anthropic.Anthropic()
elif os.environ.get("GOOGLE_API_KEY"):
    PROVIDER = "gemini"
    MODEL = "gemini-2.5-flash"
    import google.generativeai as genai

    sdk_client = genai.GenerativeModel(MODEL)
else:
    print("Set at least one API key: OPENAI_API_KEY, ANTHROPIC_API_KEY, or GOOGLE_API_KEY")
    raise SystemExit(1)

print(f"Using model: {MODEL} (provider: {PROVIDER})\n")


def sdk_chat(prompt: str, system: str | None = None) -> str:
    """Call the native SDK — auto-patched by StateLoom, so durable/cost tracking works."""
    if PROVIDER == "openai":
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        resp = sdk_client.chat.completions.create(model=MODEL, messages=messages)
        return resp.choices[0].message.content
    elif PROVIDER == "anthropic":
        kwargs = {
            "model": MODEL,
            "max_tokens": 256,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system:
            kwargs["system"] = system
        resp = sdk_client.messages.create(**kwargs)
        return resp.content[0].text
    else:  # gemini
        resp = sdk_client.generate_content(prompt)
        return resp.text


# ── 1. Durable session (crash recovery) ──────────────────────────────
# With durable=True, every LLM response is checkpointed to the store.
# If the process crashes and restarts, completed steps replay from cache
# instantly — no duplicate API calls, no duplicate cost.

print("=" * 60)
print("1. Durable session (run twice to see replay)")
print("=" * 60)

with stateloom.session("durable-demo-1", budget=2.0, durable=True) as s:
    # Step 1 — on first run: live LLM call. On second run: instant replay.
    response = stateloom.chat(
        model=MODEL,
        messages=[{"role": "user", "content": "What is eventual consistency? One sentence."}],
    )
    print(f"  Step 1: {response.content[:100]}")
    print(f"    Cost: ${s.total_cost:.6f} | Step: {s.step_counter}")

    # Step 2 — same pattern
    response = stateloom.chat(
        model=MODEL,
        messages=[{"role": "user", "content": "What is the CAP theorem? One sentence."}],
    )
    print(f"  Step 2: {response.content[:100]}")
    print(f"    Cost: ${s.total_cost:.6f} | Step: {s.step_counter}")

    # Step 3
    response = stateloom.chat(
        model=MODEL,
        messages=[{"role": "user", "content": "What is CRDT? One sentence."}],
    )
    print(f"  Step 3: {response.content[:100]}")
    print(f"    Cost: ${s.total_cost:.6f} | Step: {s.step_counter}")

print(f"  Total: ${s.total_cost:.6f} | {s.call_count} calls")
print("  (Run again — all 3 steps will replay from cache at zero cost)")

print()


# ── 2. Named checkpoints (native SDK) ────────────────────────────────
# Mark progress milestones within a session. Checkpoints appear as
# labeled dividers in the dashboard waterfall timeline.
# Uses native SDK calls (auto-patched) instead of stateloom.chat().

print("=" * 60)
print("2. Named checkpoints (native SDK calls)")
print("=" * 60)

with stateloom.session("checkpoint-demo", budget=2.0, durable=True) as s:
    stateloom.checkpoint("data-loaded", "Input data validated and ready")
    print("  Checkpoint: data-loaded")

    text = sdk_chat("Summarize MapReduce in one sentence.")
    print(f"  LLM call (SDK): {text[:80]}")

    stateloom.checkpoint("analysis-complete", "LLM analysis finished")
    print("  Checkpoint: analysis-complete")

    text = sdk_chat("Summarize Spark vs MapReduce in one sentence.")
    print(f"  LLM call (SDK): {text[:80]}")

    stateloom.checkpoint("pipeline-done", "All steps completed")
    print("  Checkpoint: pipeline-done")

print("  Dashboard shows checkpoints as dividers in the waterfall timeline")

print()


# ── 3. Semantic retries with native SDK ───────────────────────────────
# When an LLM returns bad output (invalid JSON, hallucinated fields),
# retry_loop automatically retries. Uses native SDK calls (auto-patched).

print("=" * 60)
print("3. Semantic retries — retry_loop (native SDK)")
print("=" * 60)

with stateloom.session("retry-demo", budget=2.0) as s:
    result = None
    for attempt in stateloom.retry_loop(retries=3):
        with attempt:
            text = sdk_chat(
                'Return a JSON object with keys "language" and "year_created" for Python.',
                system="You are a JSON API. Return ONLY valid JSON, no markdown.",
            )
            # Parse — if this fails, the retry loop catches and retries
            text = text.strip()
            if text.startswith("```"):
                text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
            result = json.loads(text)
            assert "language" in result and "year_created" in result

    print(f"  Result: {result}")
    print(f"  Calls: {s.call_count} (includes retries if any)")
    print(f"  Cost: ${s.total_cost:.6f}")

print()


# ── 4. durable_task decorator ─────────────────────────────────────────
# Combines durable session + automatic retries in a single decorator.
# One session spans all retry attempts — cached responses replay
# correctly on restart.

print("=" * 60)
print("4. durable_task decorator (session + retries)")
print("=" * 60)


@stateloom.durable_task(retries=3, session_id="durable-task-demo", budget=2.0)
def extract_facts(topic: str) -> dict:
    """Extract structured facts — retries automatically on bad JSON."""
    response = stateloom.chat(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": "You are a JSON API. Return ONLY valid JSON, no markdown.",
            },
            {
                "role": "user",
                "content": f'Return a JSON object with "topic", "summary" (one sentence), '
                f'and "key_concepts" (list of 3 strings) for: {topic}',
            },
        ],
    )
    text = response.content.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    result = json.loads(text)
    assert "topic" in result and "key_concepts" in result
    return result


try:
    facts = extract_facts("distributed systems")
    print(f"  Topic: {facts.get('topic', 'N/A')}")
    print(f"  Summary: {facts.get('summary', 'N/A')[:80]}")
    print(f"  Concepts: {', '.join(facts.get('key_concepts', []))}")
    print("  (Durable — run again to replay from cache)")
except stateloom.StateLoomRetryError as e:
    print(f"  All retries exhausted: {e}")

print("\nDashboard: http://localhost:4782")
