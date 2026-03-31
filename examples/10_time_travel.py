"""
StateLoom Time-Travel Debugging — Record, Replay & Export

Demonstrates:
  - Record a 6-step durable session (mix of stateloom.chat + native SDK)
  - Replay from any step — mock steps 1..N, live from N+1
  - Export / import sessions as portable JSON bundles
  - VCR-cassette mock — record once, replay forever (zero-cost testing)
  - Parent-child session export with include_children

Run:

    export OPENAI_API_KEY=sk-...
    python examples/10_time_travel.py

    # or with Anthropic / Gemini
    export ANTHROPIC_API_KEY=sk-ant-...
    python examples/10_time_travel.py
"""

import os
import tempfile
import time
import stateloom

# ── Init ──────────────────────────────────────────────────────────────

stateloom.init(budget=5.0)

# Detect provider
PROVIDER = None
MODEL = None
sdk_client = None

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
    """Call the native SDK — auto-patched by StateLoom."""
    if PROVIDER == "openai":
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        resp = sdk_client.chat.completions.create(model=MODEL, messages=messages)
        return resp.choices[0].message.content
    elif PROVIDER == "anthropic":
        kwargs = {"model": MODEL, "max_tokens": 256, "messages": [{"role": "user", "content": prompt}]}
        if system:
            kwargs["system"] = system
        resp = sdk_client.messages.create(**kwargs)
        return resp.content[0].text
    else:  # gemini
        resp = sdk_client.generate_content(prompt)
        return resp.text


# ── The pipeline ──────────────────────────────────────────────────────
# A 6-step analysis pipeline. Steps 1-3 use stateloom.chat(),
# steps 4-6 use native SDK calls (auto-patched by StateLoom).

QUESTIONS = [
    "What is a hash table? One sentence.",                  # Step 1 — stateloom.chat
    "What is a binary search tree? One sentence.",          # Step 2 — stateloom.chat
    "What is a bloom filter? One sentence.",                # Step 3 — stateloom.chat
    "What is a skip list? One sentence.",                   # Step 4 — native SDK
    "What is a trie? One sentence.",                        # Step 5 — native SDK
    "Compare all five data structures in one sentence.",    # Step 6 — native SDK
]


def run_pipeline():
    """Run the 6-step pipeline. Steps 1-3 use stateloom.chat, 4-6 use native SDK."""
    results = []
    for i, question in enumerate(QUESTIONS):
        t0 = time.perf_counter()
        if i < 3:
            # Steps 1-3: stateloom.chat()
            response = stateloom.chat(
                model=MODEL,
                messages=[{"role": "user", "content": question}],
            )
            text = response.content
        else:
            # Steps 4-6: native SDK (auto-patched)
            text = sdk_chat(question)
        elapsed = (time.perf_counter() - t0) * 1000
        results.append((text, elapsed))
        label = "stateloom.chat" if i < 3 else "native SDK"
        print(f"    Step {i+1} ({label}): {text[:70]}  [{elapsed:.0f}ms]")
    return results


# ── 1. Record a 6-step durable session ───────────────────────────────
# With durable=True, every LLM response is checkpointed to the store.
# This enables both crash recovery and time-travel replay.

print("=" * 60)
print("1. Record a 6-step durable session")
print("=" * 60)

with stateloom.session("replay-pipeline", budget=2.0, durable=True) as s:
    stateloom.checkpoint("pipeline-start", "Starting data structure analysis")
    run_pipeline()
    stateloom.checkpoint("pipeline-done", "Analysis complete")

print(f"  Total: ${s.total_cost:.6f} | {s.call_count} calls | {s.step_counter} steps")

print()


# ── 2. Replay from step 4 — mock 1-3, live from 4 ───────────────────
# stateloom.replay() mocks steps 1..N with cached responses (instant,
# zero cost), then lets steps N+1 onwards execute live against the API.
# This is "time-travel" — jump to any point in the session and re-run.
# Use as a context manager for clean ContextVar cleanup.

print("=" * 60)
print("2. Replay from step 4 (mock steps 1-3, live 4-6)")
print("=" * 60)

with stateloom.replay(session="replay-pipeline", mock_until_step=3, strict=False):
    print("  Steps 1-3 are cached (instant), steps 4-6 are live API calls:")
    run_pipeline()

print()


# ── 3. Replay from step 6 — mock 1-5, live only step 6 ──────────────
# Mock almost everything — only the final comparison step runs live.

print("=" * 60)
print("3. Replay from step 6 (mock steps 1-5, live only step 6)")
print("=" * 60)

with stateloom.replay(session="replay-pipeline", mock_until_step=5, strict=False):
    print("  Steps 1-5 are cached (instant), only step 6 is live:")
    run_pipeline()

print()


# ── 4. Full durable replay — all steps from cache ────────────────────
# Re-open the same durable session — all 6 steps replay from cache.
# Zero API cost, instant execution. Useful for crash recovery.

print("=" * 60)
print("4. Full durable replay (all 6 steps from cache)")
print("=" * 60)

with stateloom.session("replay-pipeline", budget=2.0, durable=True) as s:
    print("  All steps replay from cache (zero cost):")
    run_pipeline()

print(f"  Cost this run: ${s.total_cost:.6f} (should be ~$0)")

print()


# ── 5. Export session as JSON bundle ──────────────────────────────────
# Export the recorded session as a portable JSON bundle. Contains
# full session metadata, all events, and cached responses.

print("=" * 60)
print("5. Export session as JSON bundle")
print("=" * 60)

bundle = stateloom.export_session("replay-pipeline")

print(f"  Schema version: {bundle.get('schema_version', 'N/A')}")
print(f"  Session ID: {bundle['session']['id']}")
print(f"  Events: {len(bundle['events'])}")
print(f"  Event types: {', '.join(sorted(set(e['event_type'] for e in bundle['events'])))}")

for i, event in enumerate(bundle["events"]):
    etype = event.get("event_type", "unknown")
    model = event.get("model", "")
    cost = event.get("cost", 0)
    label = event.get("label", "")
    cached = "yes" if event.get("cached_response_json") else "no"
    if etype == "checkpoint":
        print(f"    [{i+1}] {etype}: {label}")
    elif etype == "llm_call":
        print(f"    [{i+1}] {etype}: model={model}, cost=${cost:.6f}, cached_response={cached}")
    else:
        print(f"    [{i+1}] {etype}")

print()


# ── 6. Export to file → import with new ID ────────────────────────────
# Write the bundle to a JSON file, then import it back with a new
# session ID. Useful for sharing sessions across environments.

print("=" * 60)
print("6. Export to file → Import with new ID")
print("=" * 60)

with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
    export_path = f.name

stateloom.export_session("replay-pipeline", path=export_path)
file_size = os.path.getsize(export_path)
print(f"  Exported to: {export_path}")
print(f"  File size: {file_size:,} bytes")

imported = stateloom.import_session(export_path, session_id_override="imported-pipeline")
print(f"  Imported as: {imported.id}")
print(f"  Cost: ${imported.total_cost:.6f}")
print(f"  Steps: {imported.step_counter}")

imported_bundle = stateloom.export_session("imported-pipeline")
print(f"  Events in imported session: {len(imported_bundle['events'])}")

os.unlink(export_path)

print()


# ── 7. VCR-cassette mock — record once, replay forever ───────────────
# stateloom.mock() records LLM responses on first run, then replays
# from cache on subsequent runs — zero API cost after first recording.

print("=" * 60)
print("7. VCR-cassette mock (native SDK)")
print("=" * 60)

with stateloom.mock("vcr-demo") as m:
    text1 = sdk_chat("What is Redis? One sentence.")
    text2 = sdk_chat("What is Memcached? One sentence.")

print(f"  Is replay: {m.is_replay}")
print(f"  Response 1: {text1[:80]}")
print(f"  Response 2: {text2[:80]}")

# Run the SAME mock again — replays from cache
with stateloom.mock("vcr-demo") as m2:
    text1_replay = sdk_chat("What is Redis? One sentence.")
    text2_replay = sdk_chat("What is Memcached? One sentence.")

print(f"  Replay run — is_replay: {m2.is_replay}")
print(f"  Replay 1: {text1_replay[:80]}")
print(f"  Replay 2: {text2_replay[:80]}")
print(f"  (Second run was instant — zero API cost)")

print()


# ── 8. Parent-child session export ───────────────────────────────────
# Sessions support parent-child hierarchies. Export with
# include_children=True to capture the full tree.

print("=" * 60)
print("8. Parent-child session export (stateloom.chat)")
print("=" * 60)

with stateloom.session("parent-demo", budget=2.0) as parent:
    r = stateloom.chat(
        model=MODEL,
        messages=[{"role": "user", "content": "Name 3 database types in one line."}],
    )
    print(f"  Parent: {r.content[:80]}")

    with stateloom.session("child-demo-1", budget=1.0, parent=parent.id) as child:
        r = stateloom.chat(
            model=MODEL,
            messages=[{"role": "user", "content": "What is a relational database? One sentence."}],
        )
        print(f"  Child 1: {r.content[:80]}")

full_bundle = stateloom.export_session("parent-demo", include_children=True)
children_count = len(full_bundle.get("children", []))
print(f"  Parent events: {len(full_bundle['events'])}")
print(f"  Children exported: {children_count}")
if children_count > 0:
    child_bundle = full_bundle["children"][0]
    print(f"  Child ID: {child_bundle['session']['id']}")
    print(f"  Child events: {len(child_bundle['events'])}")

print()
print(f"Dashboard: http://localhost:4782")
