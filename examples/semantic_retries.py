"""
StateLoom Semantic Retries — Self-Healing LLM Output

Demonstrates:
  - retry_loop — iterable retry helper for bad JSON / missing fields
  - on_retry callback — observe each failed attempt
  - validate callback — custom validation triggers retry on False
  - durable_task decorator — durable session + retries in one
  - Non-retryable errors — budget / PII / guardrail errors propagate immediately
  - Native SDK calls — retries work with auto-patched provider SDKs

Run:

    export OPENAI_API_KEY=sk-...
    python examples/semantic_retries.py

    # or with Anthropic / Gemini
    export ANTHROPIC_API_KEY=sk-ant-...
    python examples/semantic_retries.py
"""

import json
import os

import stateloom

# ── Init ──────────────────────────────────────────────────────────────

stateloom.init(budget=5.0)

# Detect provider and set up both stateloom.chat() and native SDK client
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

# Session IDs include the provider so switching API keys creates fresh sessions
# instead of replaying cached responses from a different provider.
SID_PREFIX = f"{PROVIDER}-"


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
        kwargs = {
            "model": MODEL,
            "max_tokens": 512,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system:
            kwargs["system"] = system
        resp = sdk_client.messages.create(**kwargs)
        return resp.content[0].text
    else:  # gemini
        resp = sdk_client.generate_content(prompt)
        return resp.text


def strip_markdown_json(text: str) -> str:
    """Strip markdown code fences from JSON output."""
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    return text


# ── 1. Basic retry_loop — JSON parsing ────────────────────────────────
# The most common use case: LLM returns invalid JSON or missing fields.
# retry_loop catches the exception and retries transparently.

print("=" * 60)
print("1. Basic retry_loop — JSON parsing (stateloom.chat)")
print("=" * 60)

with stateloom.session("retry-json-demo", budget=2.0) as s:
    result = None
    for attempt in stateloom.retry_loop(retries=3):
        with attempt:
            response = stateloom.chat(
                model=MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a JSON API. Return ONLY valid JSON, no markdown.",
                    },
                    {
                        "role": "user",
                        "content": (
                            'Return {"name": "Python", "paradigm": "multi-paradigm", "year": 1991}'
                        ),
                    },
                ],
            )
            text = strip_markdown_json(response.content)
            result = json.loads(text)
            assert "name" in result and "paradigm" in result and "year" in result

    print(f"  Result: {result}")
    print(f"  Calls: {s.call_count} | Cost: ${s.total_cost:.6f}")

print()


# ── 2. retry_loop with on_retry callback ──────────────────────────────
# Observe each failed attempt — useful for logging or metrics.
# Uses native SDK calls (auto-patched by StateLoom).

print("=" * 60)
print("2. retry_loop with on_retry callback (native SDK)")
print("=" * 60)

retry_log = []


def on_retry(attempt_num: int, error: Exception) -> None:
    retry_log.append(f"Attempt {attempt_num} failed: {type(error).__name__}")
    print(f"    [on_retry] Attempt {attempt_num} failed: {type(error).__name__}: {error}")


with stateloom.session("retry-callback-demo", budget=2.0) as s:
    result = None
    for attempt in stateloom.retry_loop(retries=3, on_retry=on_retry):
        with attempt:
            text = sdk_chat(
                "Return a JSON array of exactly 3 programming languages with "
                'keys "name" and "creator". Example: [{"name": "C", "creator": "Ritchie"}]',
                system="You are a JSON API. Return ONLY valid JSON, no markdown.",
            )
            text = strip_markdown_json(text)
            result = json.loads(text)
            assert isinstance(result, list) and len(result) == 3
            for item in result:
                assert "name" in item and "creator" in item

    print(f"  Result: {json.dumps(result, indent=2)}")
    print(f"  Calls: {s.call_count} | Retries logged: {len(retry_log)}")

print()


# ── 3. retry_loop with validate callback ──────────────────────────────
# Instead of raising exceptions, use a validate function.
# If validate returns False, the retry loop triggers a retry.

print("=" * 60)
print("3. retry_loop with validate callback (stateloom.chat)")
print("=" * 60)


def validate_has_three_items(data: dict) -> bool:
    """Validate that the response has exactly 3 key_concepts."""
    concepts = data.get("key_concepts", [])
    ok = isinstance(concepts, list) and len(concepts) == 3
    if not ok:
        count = len(concepts) if isinstance(concepts, list) else "non-list"
        print(f"    [validate] Expected 3 concepts, got {count}")
    return ok


with stateloom.session("retry-validate-demo", budget=2.0) as s:
    result = None
    for attempt in stateloom.retry_loop(retries=3):
        with attempt:
            response = stateloom.chat(
                model=MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a JSON API. Return ONLY valid JSON, no markdown.",
                    },
                    {
                        "role": "user",
                        "content": (
                            'Return {"topic": "databases",'
                            ' "key_concepts": ["indexing",'
                            ' "normalization", "ACID"]}'
                        ),
                    },
                ],
            )
            text = strip_markdown_json(response.content)
            result = json.loads(text)
            assert "topic" in result
            if not validate_has_three_items(result):
                raise ValueError("Validation failed: expected 3 concepts")

    print(f"  Result: {result}")
    print(f"  Calls: {s.call_count} | Cost: ${s.total_cost:.6f}")

print()


# ── 4. durable_task decorator ─────────────────────────────────────────
# Combines a durable session + automatic retries in one decorator.
# One session spans all retry attempts — run again to see durable replay.

print("=" * 60)
print("4. durable_task decorator (stateloom.chat)")
print("=" * 60)


@stateloom.durable_task(retries=3, session_id=f"{SID_PREFIX}durable-retry-demo", budget=2.0)
def extract_tech_stack(company: str) -> dict:
    """Extract structured tech facts — retries on bad JSON."""
    response = stateloom.chat(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": "You are a JSON API. Return ONLY valid JSON, no markdown.",
            },
            {
                "role": "user",
                "content": f'Return a JSON object with "company", "languages" (list of 3), '
                f'and "cloud_provider" for: {company}',
            },
        ],
    )
    text = strip_markdown_json(response.content)
    result = json.loads(text)
    assert "company" in result and "languages" in result and "cloud_provider" in result
    assert isinstance(result["languages"], list) and len(result["languages"]) >= 1
    return result


try:
    facts = extract_tech_stack("Netflix")
    print(f"  Company: {facts.get('company', 'N/A')}")
    print(f"  Languages: {', '.join(facts.get('languages', []))}")
    print(f"  Cloud: {facts.get('cloud_provider', 'N/A')}")
    print("  (Durable — run again to see instant replay)")
except stateloom.StateLoomRetryError as e:
    print(f"  All retries exhausted: {e}")

print()


# ── 5. durable_task with native SDK + on_retry ───────────────────────
# Same decorator, but using native SDK calls and a retry observer.

print("=" * 60)
print("5. durable_task with native SDK + on_retry callback")
print("=" * 60)

sdk_retry_count = 0


def count_retries(attempt: int, error: Exception) -> None:
    global sdk_retry_count
    sdk_retry_count += 1
    print(f"    [on_retry] SDK attempt {attempt} failed: {error}")


@stateloom.durable_task(
    retries=3,
    session_id=f"{SID_PREFIX}durable-sdk-retry-demo",
    budget=2.0,
    on_retry=count_retries,
)
def classify_language(lang: str) -> dict:
    """Classify a programming language using native SDK."""
    text = sdk_chat(
        f'Return a JSON object with "language", "type" (compiled/interpreted/both), '
        f'and "typing" (static/dynamic) for: {lang}',
        system="You are a JSON API. Return ONLY valid JSON, no markdown.",
    )
    text = strip_markdown_json(text)
    result = json.loads(text)
    assert "language" in result and "type" in result and "typing" in result
    return result


try:
    info = classify_language("Rust")
    print(f"  Language: {info.get('language', 'N/A')}")
    print(f"  Type: {info.get('type', 'N/A')}")
    print(f"  Typing: {info.get('typing', 'N/A')}")
    print(f"  SDK retries: {sdk_retry_count}")
except stateloom.StateLoomRetryError as e:
    print(f"  All retries exhausted: {e}")

print()


# ── 6. Non-retryable errors ──────────────────────────────────────────
# Budget, PII, guardrail, and kill-switch errors propagate immediately.
# They are NEVER retried — retrying would waste resources or violate policy.

print("=" * 60)
print("6. Non-retryable errors — budget exhaustion")
print("=" * 60)

with stateloom.session("retry-budget-demo", budget=0.0001) as s:
    try:
        for attempt in stateloom.retry_loop(retries=5):
            with attempt:
                response = stateloom.chat(
                    model=MODEL,
                    messages=[{"role": "user", "content": "Say hello."}],
                )
                print(f"  Response: {response.content[:50]}")
    except stateloom.StateLoomBudgetError as e:
        print(f"  Budget error propagated immediately (not retried): {e}")
    except Exception:
        # Budget may not trigger if the call is cheap enough
        print(f"  Call succeeded (budget not exceeded): cost ${s.total_cost:.6f}")

print()
print("Dashboard: http://localhost:4782")
