# API Reference

Complete reference for all public functions, classes, and error types exported by `stateloom`.

All symbols are importable from the top-level package:

```python
import stateloom
from stateloom import init, session, chat, Client, StateLoomError
```

## Table of Contents

- [Initialization & Lifecycle](#initialization--lifecycle) — `init()`, `get_gate()`, `shutdown()`, `register_provider()`, `wrap()`
- [Sessions](#sessions) — `session()`, `async_session()`, `cancel_session()`, `checkpoint()`, `export_session()`, `import_session()`
- [Unified Chat](#unified-chat) — `chat()`, `achat()`, `Client`, `ChatResponse`
- [Tool Tracking](#tool-tracking) — `@tool()`
- [Experiments](#experiments) — `create_experiment()`, `start_experiment()`, `feedback()`, `leaderboard()`, `backtest()`
- [Consensus](#consensus-multi-agent-debate) — `consensus()`, `consensus_sync()`, `ConsensusResult`
- [Replay & Durable Sessions](#replay--durable-sessions) — `replay()`, `durable_task()`, `retry_loop()`, `mock()`
- [Local Models](#local-models) — `pull_model()`, `list_local_models()`, `recommend_models()`, `hot_swap_model()`
- [Auto-Routing](#auto-routing) — `set_auto_route_scorer()`, `RoutingContext`
- [Kill Switch](#kill-switch) — `kill_switch()`, `add_kill_switch_rule()`
- [Blast Radius](#blast-radius) — `blast_radius_status()`, `unpause_session()`, `unpause_agent()`
- [Rate Limiting](#rate-limiting) — `rate_limiter_status()`
- [Circuit Breaker](#circuit-breaker) — `circuit_breaker_status()`, `reset_circuit_breaker()`
- [Session Suspension](#session-suspension) — `suspend_session()`, `signal_session()`, `suspend()`, `async_suspend()`
- [Compliance](#compliance) — `purge_user_data()`, `compliance_cleanup()`
- [Organizations & Teams](#organizations--teams) — `create_organization()`, `create_team()`
- [Managed Agents](#managed-agents) — `create_agent()`, `create_agent_version()`, `activate_agent_version()`
- [Virtual Keys](#virtual-keys) — `create_virtual_key()`, `list_virtual_keys()`, `revoke_virtual_key()`
- [Async Jobs](#async-jobs) — `submit_job()`, `get_job()`, `list_jobs()`
- [Config Locking](#config-locking) — `lock_setting()`, `unlock_setting()`
- [Prompt File Watcher](#prompt-file-watcher) — `prompt_watcher_status()`, `rescan_prompts()`
- [Guardrails](#guardrails) — `guardrails_status()`, `configure_guardrails()`
- [Security](#security) — `security_status()`, `vault_store()`, `vault_retrieve()`
- [Threading & Integrations](#threading--integrations) — `patch_threading()`, `langchain_callback()`
- [Error Reference](#error-reference) — all error types and non-retryable errors
- [Type Reference](#type-reference) — key types and enums

---

## Initialization & Lifecycle

### `init(**kwargs) -> Gate`

Initialize StateLoom. This is the main entry point — call once at application startup. See [Configuration](../README.md#configuration) for all available parameters.

```python
import stateloom

# Zero-config — auto-patches OpenAI/Anthropic/Gemini, starts dashboard
stateloom.init()

# With options
stateloom.init(
    budget=10.0,
    pii=True,
    dashboard_port=8080,
    local_model="llama3.2",
    default_model="gpt-4o",
)
```

**Parameters** (all keyword-only):

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `auto_patch` | `bool` | `True` | Auto-detect and patch installed LLM clients |
| `default_model` | `str` | `""` | Default model for `stateloom.chat()` |
| `budget` | `float \| None` | `None` | Per-session budget in USD |
| `pii` | `bool` | `False` | Enable PII detection |
| `pii_rules` | `list[PIIRule]` | `None` | Per-pattern PII rules |
| `dashboard` | `bool` | `True` | Start dashboard at localhost:4781 |
| `dashboard_port` | `int` | `4781` | Dashboard port |
| `console_output` | `bool` | `True` | Print per-call summary to terminal |
| `fail_open` | `bool` | `True` | Middleware errors never break LLM calls |
| `store_backend` | `str` | `"sqlite"` | `"sqlite"`, `"memory"`, or `"postgres"` |
| `cache` | `bool` | `True` | Enable request caching |
| `local_model` | `str \| None` | auto-detect | Ollama model name (e.g. `"llama3.2"`) |
| `shadow` | `bool` | auto | Enable model testing |
| `auto_route` | `bool \| None` | `None` | Auto-routing (`None` = auto when local_model set) |
| `circuit_breaker` | `bool` | `False` | Enable provider failover |
| `metrics_enabled` | `bool` | `False` | Enable Prometheus metrics |
| `async_jobs_enabled` | `bool` | `False` | Enable background job processing |
| `proxy` | `bool` | `False` | Enable proxy mode |
| `prompts_dir` | `str` | `""` | Directory for file-based prompt versioning |

**Returns:** The `Gate` singleton instance.

---

### `get_gate() -> Gate`

Get the current Gate instance. Raises `StateLoomError` if `init()` hasn't been called.

```python
gate = stateloom.get_gate()
print(gate.config.budget_per_session)
```

---

### `shutdown()`

Shut down StateLoom, clean up resources, unpatch all providers, and stop the dashboard server.

```python
stateloom.shutdown()
```

---

### `register_provider(adapter, *, pricing=None)`

Register a custom LLM provider adapter. Call **before** `init()`.

```python
from stateloom.intercept.provider_adapter import BaseProviderAdapter

class MyAdapter(BaseProviderAdapter):
    @property
    def name(self) -> str:
        return "my-provider"
    # ... implement extract_tokens, get_patch_targets, etc.

stateloom.register_provider(
    MyAdapter(),
    pricing={"my-model-large": (0.000002, 0.000006)},
)
stateloom.init()
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `adapter` | `ProviderAdapter` | A provider adapter instance |
| `pricing` | `dict[str, tuple[float, float]] \| None` | Model name → `(input_cost_per_token, output_cost_per_token)` |

---

### `wrap(client) -> client`

Wrap an LLM client for explicit interception without monkey-patching.

```python
import openai

gate = stateloom.init(auto_patch=False)
client = stateloom.wrap(openai.OpenAI())

# This call now flows through the middleware pipeline
response = client.chat.completions.create(model="gpt-4o", messages=[...])
```

---

## Sessions

### `session(...) -> Session`

Context manager that scopes LLM calls into a tracked session with shared cost, tokens, and metadata.

```python
with stateloom.session("task-123", name="Summarize docs", budget=5.0) as s:
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": "Summarize this..."}],
    )
    print(s.total_cost)    # 0.0012
    print(s.total_tokens)  # 156
    print(s.step_counter)  # 1
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `session_id` | `str \| None` | `None` | Explicit session ID (auto-generated if omitted) |
| `name` | `str \| None` | `None` | Human-readable session name |
| `budget` | `float \| None` | `None` | Per-session budget in USD |
| `experiment` | `str \| None` | `None` | Experiment ID for A/B testing |
| `variant` | `str \| None` | `None` | Manual variant assignment |
| `org_id` | `str` | `""` | Organization scope |
| `team_id` | `str` | `""` | Team scope |
| `durable` | `bool` | `False` | Enable crash-recovery checkpointing |
| `parent` | `str \| None` | `None` | Parent session ID — **auto-derived** from the enclosing `session()` context when omitted; only needed for out-of-context linking |
| `timeout` | `float \| None` | `None` | Max session duration in seconds |
| `idle_timeout` | `float \| None` | `None` | Max idle time between calls in seconds |

**Returns:** `Session` object with accumulators for cost, tokens, steps, and metadata.

---

### `async_session(...) -> Session`

Async version of `session()` — same parameters.

```python
async with stateloom.async_session("task-456", budget=2.0) as s:
    response = await client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": "Hello!"}],
    )
    print(s.total_cost)
```

---

### `set_session_id(session_id)`

Set the current session ID for distributed context propagation (e.g., across microservices).

```python
# In a request handler that receives a session ID from another service
stateloom.set_session_id("session-from-upstream")

# Subsequent LLM calls are associated with this session
response = client.chat.completions.create(...)
```

---

### `get_session_id() -> str | None`

Get the current session ID from the ContextVar.

```python
sid = stateloom.get_session_id()
print(sid)  # "task-123" or None if no session is active
```

---

### `cancel_session(session_id) -> bool`

Cancel an active session. The next LLM call in the session raises `StateLoomCancellationError`.

```python
# From another thread or process
stateloom.cancel_session("long-running-task")
```

**Returns:** `True` if the session was found and cancelled.

---

### `checkpoint(label, description="")`

Create a named checkpoint in the current session. Appears as a labeled divider in the dashboard waterfall timeline.

```python
with stateloom.session("task-123") as s:
    response1 = client.chat.completions.create(...)
    stateloom.checkpoint("data-loaded", description="Raw data fetched from API")

    response2 = client.chat.completions.create(...)
    stateloom.checkpoint("analysis-complete")
```

---

### `export_session(session_id, path=None, *, include_children=False, scrub_pii=False) -> dict`

Export a session as a portable JSON bundle.

```python
# Export to dict
bundle = stateloom.export_session("session-123")

# Export to file with PII scrubbing and child sessions
bundle = stateloom.export_session(
    "session-123",
    path="session.json.gz",
    include_children=True,
    scrub_pii=True,
)
```

**Returns:** The bundle dict (also written to file if `path` is given).

---

### `import_session(source, *, session_id_override=None) -> Session`

Import a session bundle into the store.

```python
# From a file
session = stateloom.import_session("session.json")

# From a dict with ID override to avoid collisions
session = stateloom.import_session(bundle, session_id_override="imported-123")
```

**Returns:** The imported `Session` object.

---

### `pin(session, name)`

Pin a session as a regression test baseline.

```python
stateloom.pin(session="session-123", name="baseline-v1")
```

---

### `share(session) -> str`

Share a session for collaborative debugging. Returns a shareable URL.

```python
url = stateloom.share(session="session-123")
print(url)  # https://app.stateloom.io/shared/abc123
```

---

## Unified Chat

### `chat(*, model=None, messages, **kwargs) -> ChatResponse`

One-liner sync chat through the full middleware pipeline. Uses a default `Client` with auto-created session.

```python
stateloom.init(default_model="gpt-4o")

# Simple call
response = stateloom.chat(
    messages=[{"role": "user", "content": "What is the capital of France?"}],
)
print(response.content)  # "The capital of France is Paris."

# With explicit model (bypasses auto-routing)
response = stateloom.chat(
    model="claude-sonnet-4-20250514",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello!"},
    ],
    temperature=0.3,
)

# Access metadata
print(response._stateloom["cost"])        # 0.0003
print(response._stateloom["latency_ms"])  # 312
print(response._stateloom["routed_local"])  # True/False
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | `str \| None` | `None` | Model to use. When omitted, uses `default_model` and enables auto-routing |
| `messages` | `list[dict]` | required | Chat messages in OpenAI format |
| `**kwargs` | | | Provider-specific parameters (`temperature`, `max_tokens`, etc.) |

**Returns:** `ChatResponse`

---

### `achat(*, model=None, messages, **kwargs) -> ChatResponse`

Async version of `chat()`.

```python
response = await stateloom.achat(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Hello!"}],
)
```

---

### `Client`

Unified chat client with session ownership. Supports BYOK (Bring Your Own Key).

```python
# As context manager — session scoped to block
with stateloom.Client(session_id="task-1", budget=5.0) as client:
    r1 = client.chat(model="gpt-4o", messages=[...])
    r2 = client.chat(model="claude-sonnet-4-20250514", messages=[...])
    print(client.session.total_cost)

# Standalone
client = stateloom.Client(session_id="agent-run")
r = client.chat(model="gpt-4o", messages=[...])
client.close()

# With BYOK
client = stateloom.Client(provider_keys={"openai": "sk-my-key"})
r = client.chat(model="gpt-4o", messages=[...])

# Async
async with stateloom.Client(budget=2.0) as client:
    r = await client.achat(model="gpt-4o", messages=[...])
```

**Constructor parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `session_id` | `str \| None` | `None` | Explicit session ID |
| `budget` | `float \| None` | `None` | Per-session budget in USD |
| `org_id` | `str` | `""` | Organization scope |
| `team_id` | `str` | `""` | Team scope |
| `provider_keys` | `dict[str, str] \| None` | `None` | BYOK keys (e.g. `{"openai": "sk-..."}`) |
| `billing_mode` | `str` | `"api"` | `"api"` or `"subscription"` |

---

### `ChatResponse`

Response wrapper returned by `chat()`, `achat()`, `Client.chat()`, and `Client.achat()`.

| Field | Type | Description |
|-------|------|-------------|
| `content` | `str` | Extracted text content from the response |
| `model` | `str` | Requested model name |
| `provider` | `str` | Requested provider name |
| `raw` | `Any` | Raw provider SDK response object |
| `_stateloom` | `dict` | Metadata dict (see below) |

**`_stateloom` metadata keys:**

| Key | Type | Description |
|-----|------|-------------|
| `actual_model` | `str` | Model that actually served the request (may differ if auto-routed) |
| `actual_provider` | `str` | Provider that served the request |
| `routed_local` | `bool` | Whether the request was routed to a local model |
| `cached` | `bool` | Whether the response came from cache |
| `cost` | `float` | Total cost in USD |
| `latency_ms` | `float` | End-to-end latency in milliseconds |
| `prompt_tokens` | `int` | Input token count |
| `completion_tokens` | `int` | Output token count |
| `session_id` | `str` | Session ID |
| `events` | `list[str]` | Event types recorded (e.g. `["llm_call", "cost_tracking"]`) |

---

## Tool Tracking

### `tool(*, mutates_state=False, name=None, session_root=False)`

Decorator for tool functions. Enables tool execution visibility in the dashboard, safe replay handling, and loop detection.

```python
@stateloom.tool(mutates_state=True)
def create_ticket(title: str) -> dict:
    """Side-effecting tool — flagged during strict replay."""
    return api.create(title=title)

@stateloom.tool(session_root=True)
async def run_agent(prompt: str) -> str:
    """Each call gets its own session automatically."""
    return await chain.invoke(prompt)

@stateloom.tool(name="web_search")
def search(query: str) -> list[dict]:
    """Custom tool name (defaults to function name)."""
    return search_api.query(query)
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `mutates_state` | `bool` | `False` | Marks the tool as having side effects (flagged before replay) |
| `name` | `str \| None` | `None` | Override the tool name (defaults to function name) |
| `session_root` | `bool` | `False` | Auto-create a scoped session per call |

---

## Experiments

### `create_experiment(name, variants, ...) -> Experiment`

Create an A/B experiment in DRAFT status.

```python
exp = stateloom.create_experiment(
    name="model-comparison",
    variants=[
        {"name": "control", "weight": 1.0, "model": "gpt-4o"},
        {"name": "challenger", "weight": 1.0, "model": "claude-sonnet-4-20250514"},
    ],
    strategy="random",  # or "hash" for deterministic, "manual" for explicit
    description="Compare GPT-4o vs Claude for summarization",
)

# Agent-scoped experiment — test two agent versions against each other
exp = stateloom.create_experiment(
    name="helpdesk-prompt-test",
    variants=[
        {"name": "v1", "agent_version_id": "agv-abc123"},
        {"name": "v2", "agent_version_id": "agv-def456"},
    ],
    agent_id="agt-helpdesk",
)
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | `str` | required | Experiment name |
| `variants` | `list[dict]` | required | Variant configs with `name`, `weight`, optional `model`, `request_overrides`, `agent_version_id` |
| `description` | `str` | `""` | Experiment description |
| `strategy` | `str` | `"random"` | Assignment strategy: `"random"`, `"hash"`, or `"manual"` |
| `metadata` | `dict \| None` | `None` | Arbitrary metadata |
| `agent_id` | `str \| None` | `None` | Optional agent ID to scope the experiment to |

**Variant config fields:**

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `name` | `str` | required | Variant name |
| `weight` | `float` | `1.0` | Assignment weight (relative) |
| `model` | `str \| None` | `None` | Override model for this variant |
| `request_overrides` | `dict` | `{}` | Override request kwargs (temperature, system_prompt, etc.) |
| `agent_version_id` | `str \| None` | `None` | Use an agent version's model, system prompt, and overrides as the base layer |

When `agent_version_id` is set, the agent version's configuration is applied as the base layer. The variant's explicit `model` and `request_overrides` take priority on top. Agent overrides are snapshotted at assignment time for immutability.

---

### `update_experiment(experiment_id, ...) -> Experiment`

Update a DRAFT experiment. Only experiments in DRAFT status can be edited.

```python
stateloom.update_experiment(
    exp.id,
    name="renamed-experiment",
    variants=[
        {"name": "a", "model": "gpt-4o"},
        {"name": "b", "model": "gpt-4o-mini"},
        {"name": "c", "agent_version_id": "agv-new123"},
    ],
)
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `experiment_id` | `str` | required | Experiment to update |
| `name` | `str \| None` | `None` | New name |
| `description` | `str \| None` | `None` | New description |
| `variants` | `list[dict] \| None` | `None` | Replace variants |
| `strategy` | `str \| None` | `None` | New assignment strategy |
| `metadata` | `dict \| None` | `None` | Replace metadata |
| `agent_id` | `str \| None` | `None` | New agent ID |

Raises `ValueError` if experiment is not found or not in DRAFT status.

---

### `start_experiment(experiment_id) -> Experiment`

Start an experiment — begin assigning sessions to variants.

```python
stateloom.start_experiment(exp.id)
```

---

### `pause_experiment(experiment_id) -> Experiment`

Pause an experiment — stop assigning new sessions.

```python
stateloom.pause_experiment(exp.id)
```

---

### `conclude_experiment(experiment_id) -> dict`

Conclude an experiment and return final metrics.

```python
results = stateloom.conclude_experiment(exp.id)
# {"variants": {"control": {...}, "challenger": {...}}, "winner": "challenger", ...}
```

---

### `feedback(session_id, rating, *, score=None, comment="")`

Record feedback for a session (used by experiment metrics).

```python
stateloom.feedback("session-123", rating="success", score=0.95, comment="Great output")
stateloom.feedback("session-456", rating="failure")
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `session_id` | `str` | required | Session to rate |
| `rating` | `str` | required | `"success"`, `"failure"`, or `"partial"` |
| `score` | `float \| None` | `None` | Numeric quality score |
| `comment` | `str` | `""` | Optional comment |

---

### `experiment_metrics(experiment_id) -> dict`

Get per-variant aggregated metrics.

```python
metrics = stateloom.experiment_metrics(exp.id)
# {"control": {"success_rate": 0.78, "avg_cost": 0.12, "sessions": 50}, ...}
```

---

### `leaderboard() -> list[dict]`

Cross-experiment variant ranking sorted by success_rate desc, avg_cost asc.

```python
board = stateloom.leaderboard()
for entry in board:
    print(f"{entry['variant']}: {entry['success_rate']:.0%} success, ${entry['avg_cost']:.4f} avg")
```

---

### `backtest(sessions, variants, agent_fn, ...) -> list[dict]`

Replay recorded sessions with different variant configs.

```python
results = stateloom.backtest(
    sessions=["session-1", "session-2"],
    variants=[
        {"name": "gpt4o", "model": "gpt-4o"},
        {"name": "claude", "model": "claude-sonnet-4-20250514"},
    ],
    agent_fn=my_agent_function,
    strict=True,
)
for r in results:
    print(f"{r['session_id']} + {r['variant']}: cost=${r['cost']:.4f}")
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `sessions` | `list[str]` | required | Source session IDs to replay |
| `variants` | `list[dict]` | required | Variant configs to test |
| `agent_fn` | `Callable` | required | Function that re-executes agent logic (receives `Session`) |
| `mock_until_step` | `int \| None` | `None` | Mock steps 1..N (defaults to all recorded steps) |
| `strict` | `bool` | `False` | Block outbound HTTP during mocked steps |
| `evaluator` | `Callable \| None` | `None` | Optional scoring callback |

---

## Consensus (Multi-Agent Debate)

### `consensus(prompt, *, models, strategy, agent, ...) -> ConsensusResult`

Run a multi-agent consensus session. Multiple models debate a question across rounds to reduce hallucinations and improve reasoning.

```python
# Debate — multi-round with judge synthesis
result = await stateloom.consensus(
    prompt="What are the key risks of deploying LLMs in healthcare?",
    models=["gpt-4o", "claude-sonnet-4-20250514", "gemini-2.0-flash"],
    strategy="debate",
    rounds=2,
    budget=1.00,
)
print(result.answer)       # Final synthesized answer
print(result.confidence)   # 0.0-1.0 confidence score
print(result.cost)         # Total cost across all models and rounds
print(result.session_id)   # Parent session ID for dashboard drill-down

# Vote — cheapest, one round
result = await stateloom.consensus(
    prompt="What is the capital of France?",
    models=["gpt-4o", "claude-sonnet-4-20250514"],
    strategy="vote",
)

# Self-consistency — multiple samples from one model
result = await stateloom.consensus(
    prompt="Solve: 2x + 5 = 13",
    models=["gpt-4o"],
    strategy="self_consistency",
    samples=5,
    temperature=0.7,
)

# Greedy mode — auto-downgrade to cheaper models when consensus is easy
result = await stateloom.consensus(
    prompt="What is 2+2?",
    models=["gpt-4o", "claude-sonnet-4-20250514"],
    strategy="debate",
    greedy=True,
)

# Agent-guided consensus — agent's system prompt guides all debaters
result = await stateloom.consensus(
    agent="medical-advisor",
    prompt="Should we use AI for radiology screening?",
    models=["gpt-4o", "claude-sonnet-4-20250514"],
    strategy="debate",
)
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `prompt` | `str` | `""` | The question or prompt to debate |
| `messages` | `list[dict] \| None` | `None` | Chat messages (alternative to `prompt`) |
| `models` | `list[str] \| None` | `None` | Models to use (falls back to agent model or config default) |
| `rounds` | `int` | `2` | Number of debate rounds |
| `strategy` | `str` | `"debate"` | `"vote"`, `"debate"`, or `"self_consistency"` |
| `budget` | `float \| None` | `None` | Total budget cap across all models and rounds |
| `session_id` | `str \| None` | `None` | Explicit session ID (for durable resume) |
| `greedy` | `bool` | `False` | Auto-downgrade models after high-agreement Round 1 |
| `greedy_agreement_threshold` | `float` | `0.7` | Agreement threshold for greedy downgrade |
| `early_stop_enabled` | `bool` | `True` | Stop early when confidence is high |
| `early_stop_threshold` | `float` | `0.9` | Confidence threshold for early stop |
| `samples` | `int` | `5` | Number of samples (self_consistency only) |
| `temperature` | `float` | `0.7` | Temperature (self_consistency only) |
| `judge_model` | `str \| None` | `None` | Model for judge synthesis (defaults to first model) |
| `aggregation` | `str` | `"confidence_weighted"` | `"majority_vote"` or `"confidence_weighted"` |
| `agent` | `str \| None` | `None` | Agent slug or ID — uses the agent's system prompt for debaters |

**Returns:** `ConsensusResult`

---

### `consensus_sync(...) -> ConsensusResult`

Synchronous wrapper for `consensus()` — same parameters.

```python
result = stateloom.consensus_sync(
    prompt="What is the capital of France?",
    models=["gpt-4o", "claude-sonnet-4-20250514"],
    strategy="vote",
)
```

---

### `ConsensusResult`

| Field | Type | Description |
|-------|------|-------------|
| `answer` | `str` | Final synthesized answer |
| `confidence` | `float` | Confidence score (0.0–1.0) |
| `cost` | `float` | Total cost across all models and rounds |
| `session_id` | `str` | Parent session ID |
| `strategy` | `str` | Strategy used |
| `models` | `list[str]` | Models used |
| `rounds` | `list[DebateRound]` | Per-round data |
| `total_rounds` | `int` | Number of rounds completed |
| `early_stopped` | `bool` | Whether debate stopped early |
| `aggregation_method` | `str` | Aggregation method used |
| `winner_model` | `str` | Model with highest confidence |
| `duration_ms` | `float` | Total duration in milliseconds |

---

### `DebateRound`

| Field | Type | Description |
|-------|------|-------------|
| `round_number` | `int` | Round number (1-indexed) |
| `responses` | `list[DebaterResponse]` | Per-model responses |
| `consensus_reached` | `bool` | Whether agreement exceeded threshold |
| `agreement_score` | `float` | Pairwise agreement score (0.0–1.0) |
| `cost` | `float` | Round cost |
| `duration_ms` | `float` | Round duration |

---

## Replay & Durable Sessions

### `replay(session, mock_until_step, strict=True, allow_hosts=None)`

Time-travel debugging: replay a session, mocking the first N steps with cached responses, then run live from the failure point.

```python
stateloom.replay(
    session="failed-session-id",
    mock_until_step=13,
    strict=True,
    allow_hosts=["api.example.com"],
)
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `session` | `str` | required | Session ID to replay |
| `mock_until_step` | `int` | required | Mock steps 1 through N with cached responses |
| `strict` | `bool` | `True` | Block outbound HTTP during mocked steps |
| `allow_hosts` | `list[str] \| None` | `None` | Hosts to allow through the network blocker |

---

### `durable_task(retries=3, **kwargs)`

Decorator combining a durable session with automatic retry. If the function raises, it retries up to N times. On crash recovery, cached LLM responses replay automatically.

```python
@stateloom.durable_task(retries=3, session_id="report-42")
def generate_report(prompt: str) -> dict:
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
    )
    return json.loads(response.choices[0].message.content)  # Retries on ValueError

# With validation callback
@stateloom.durable_task(retries=3, validate=lambda r: "summary" in r)
def summarize(text: str) -> dict:
    response = client.chat.completions.create(...)
    return json.loads(response.choices[0].message.content)

# Async
@stateloom.durable_task(retries=3, session_id="async-task-1")
async def async_generate(prompt: str) -> dict:
    response = await client.chat.completions.create(...)
    return json.loads(response.choices[0].message.content)
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `retries` | `int` | `3` | Maximum retry attempts |
| `session_id` | `str` | auto | Fixed session ID (enables durable resume across restarts) |
| `name` | `str` | function name | Session name |
| `budget` | `float \| None` | `None` | Per-session budget in USD |
| `validate` | `Callable \| None` | `None` | Callback — returns `False` to trigger retry |
| `on_retry` | `Callable \| None` | `None` | Callback `(attempt, error)` fired on each retry |

---

### `retry_loop(retries=3, **kwargs) -> RetryLoop`

Create an iterable retry loop for use inside an existing session.

```python
with stateloom.session("task-123", durable=True) as s:
    for attempt in stateloom.retry_loop(retries=3):
        with attempt:
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": "Return valid JSON"}],
            )
            data = json.loads(response.choices[0].message.content)
    # data is available here after success
    print(data)
```

Each iteration yields a `RetryAttempt` context manager. Exceptions inside `with attempt:` are suppressed for retry. Non-retryable errors (budget, PII, kill switch, etc.) always propagate immediately.

---

### `mock(session_id=None, *, force_record=False, network_block=True, allow_hosts=None) -> MockSession`

VCR-cassette mock: record LLM calls once, replay forever. Returns a `MockSession` usable as both a decorator and a context manager.

```python
# As a decorator — auto-records on first run, replays on subsequent runs
@stateloom.mock()
def test_my_agent():
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": "Hello!"}],
    )
    assert "hello" in response.choices[0].message.content.lower()

# As a context manager with explicit session ID
with stateloom.mock("my-cassette") as m:
    response = client.chat.completions.create(...)
    print(m.is_replay)  # True on subsequent runs

# Force re-recording
with stateloom.mock("my-cassette", force_record=True) as m:
    response = client.chat.completions.create(...)  # Always live
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `session_id` | `str \| None` | `None` | Cassette ID (auto-derived from function name if omitted) |
| `force_record` | `bool` | `False` | Re-record even if cached responses exist |
| `network_block` | `bool` | `True` | Block outbound HTTP during replay |
| `allow_hosts` | `list[str] \| None` | `None` | Hosts to allow through the blocker |

**`MockSession` fields:**

| Field | Type | Description |
|-------|------|-------------|
| `is_replay` | `bool` | Whether responses are coming from cache |

---

## Local Models

### `pull_model(model, *, progress=None)`

Download a local model via Ollama.

```python
stateloom.pull_model("llama3.2")

# With progress callback
def on_progress(info):
    print(f"{info.get('status')}: {info.get('completed', 0)}/{info.get('total', 0)}")

stateloom.pull_model("mistral:7b", progress=on_progress)
```

---

### `list_local_models() -> list[dict]`

List locally downloaded Ollama models.

```python
models = stateloom.list_local_models()
for m in models:
    print(f"{m['name']} ({m['size'] / 1e9:.1f}GB)")
```

---

### `recommend_models() -> list[dict]`

Get hardware-aware model recommendations based on available RAM and GPU.

```python
recs = stateloom.recommend_models()
for m in recs:
    print(f"{m['model']} ({m['tier']}, {m['size_gb']}GB) — {m['description']}")
```

---

### `delete_local_model(model)`

Delete a locally downloaded Ollama model.

```python
stateloom.delete_local_model("llama3.2")
```

---

### `set_local_model(model)`

Set the active local model for auto-routing and model testing. Updates `local_model_default`, `auto_route_model`, and `shadow_model`.

```python
stateloom.set_local_model("mistral:7b")
```

---

### `hot_swap_model(new_model, *, delete_old=True, progress=None)`

Hot-swap local model with zero downtime. Pulls the new model, atomically switches config, then deletes the old model.

```python
stateloom.hot_swap_model("llama3.2:latest", delete_old=True)
```

---

### `force_local(enabled=True)`

Force all eligible LLM traffic to route through local models. Incompatible requests (streaming, tools, images) still fall back to cloud.

```python
stateloom.force_local()           # Enable
stateloom.force_local(False)      # Disable
```

---

## Auto-Routing

### `set_auto_route_scorer(scorer)`

Set or clear a custom routing scorer function for auto-routing decisions.

```python
# Simple scorer — receives prompt string
def my_scorer(prompt: str) -> bool | float | None:
    if "translate" in prompt.lower():
        return True   # Route to local
    if "analyze" in prompt.lower():
        return False  # Force cloud
    return None       # Fall through to default scoring

stateloom.set_auto_route_scorer(my_scorer)

# Rich scorer — receives RoutingContext
def rich_scorer(ctx: stateloom.RoutingContext) -> bool | float | None:
    if ctx.total_cost > 5.0:
        return True  # Route local when budget is running low
    return None

stateloom.set_auto_route_scorer(rich_scorer)

# Clear custom scorer
stateloom.set_auto_route_scorer(None)
```

**Scorer return values:**

| Return | Meaning |
|--------|---------|
| `True` | Route to local model |
| `False` | Force cloud |
| `float` | Complexity score 0.0–1.0 (below threshold → local) |
| `None` | Fall through to default scoring |

---

### `RoutingContext`

Dataclass passed to custom scorers when they accept a typed parameter.

| Field | Type | Description |
|-------|------|-------------|
| `prompt` | `str` | The user's prompt text |
| `messages` | `list[dict]` | Full messages array |
| `model` | `str` | Requested model |
| `provider` | `str` | Resolved provider |
| `session_id` | `str` | Active session ID |
| `session_metadata` | `dict` | Session metadata |
| `org_id` | `str` | Organization ID |
| `team_id` | `str` | Team ID |
| `total_cost` | `float` | Session cost so far |
| `budget` | `float \| None` | Session budget |
| `call_count` | `int` | Number of LLM calls in this session |
| `is_streaming` | `bool` | Whether the request is streaming |
| `has_tools` | `bool` | Whether the request includes tool definitions |
| `has_images` | `bool` | Whether the request includes image content |
| `local_model` | `str` | Configured local model name |

---

## Kill Switch

### `kill_switch(active=True, *, message=None)`

Activate or deactivate the global kill switch.

```python
# Block all LLM traffic
stateloom.kill_switch(active=True, message="Maintenance in progress")

# Resume traffic
stateloom.kill_switch(active=False)
```

---

### `add_kill_switch_rule(...)`

Add a granular kill switch rule. Rules support glob patterns for model matching.

```python
stateloom.add_kill_switch_rule(model="gpt-4*", reason="Cost overrun on GPT-4 family")
stateloom.add_kill_switch_rule(provider="anthropic", environment="production")
stateloom.add_kill_switch_rule(agent_version="v2.1.0", message="Known bug in v2.1.0")
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | `str \| None` | `None` | Model glob pattern (e.g. `"gpt-4*"`) |
| `provider` | `str \| None` | `None` | Provider name |
| `environment` | `str \| None` | `None` | Environment filter |
| `agent_version` | `str \| None` | `None` | Agent version filter |
| `message` | `str` | `""` | Message for blocked requests |
| `reason` | `str` | `""` | Internal reason for the rule |

---

### `kill_switch_rules() -> list[dict]`

Get the current kill switch rules.

```python
rules = stateloom.kill_switch_rules()
for rule in rules:
    print(f"Blocking model={rule.get('model')} provider={rule.get('provider')}")
```

---

### `clear_kill_switch_rules()`

Remove all kill switch rules.

```python
stateloom.clear_kill_switch_rules()
```

---

## Blast Radius

### `blast_radius_status() -> dict`

Get blast radius containment status including paused sessions, paused agents, and failure counts.

```python
status = stateloom.blast_radius_status()
print(status["paused_sessions"])   # ["session-123", ...]
print(status["paused_agents"])     # ["agent:ticket-bot", ...]
print(status["session_failure_counts"])  # {"session-123": 5, ...}
```

---

### `unpause_session(session_id) -> bool`

Unpause a blast-radius-paused session.

```python
was_paused = stateloom.unpause_session("session-123")
print(was_paused)  # True
```

---

### `unpause_agent(agent_id) -> bool`

Unpause a blast-radius-paused agent (unpauses all sessions for that agent).

```python
stateloom.unpause_agent("agent:ticket-bot")
```

---

## Rate Limiting

### `rate_limiter_status() -> dict`

Get rate limiter status — per-team queue depths, token counts, and configuration.

```python
status = stateloom.rate_limiter_status()
for team_id, info in status["teams"].items():
    print(f"{team_id}: {info['tps']} TPS, queue={info['queue_size']}")
```

---

## Circuit Breaker

### `circuit_breaker_status() -> dict`

Get circuit breaker status for all tracked providers.

```python
status = stateloom.circuit_breaker_status()
for provider, info in status.items():
    print(f"{provider}: state={info['state']}, failures={info['failures']}")
# {"openai": {"state": "closed", "failures": 0}, "anthropic": {"state": "open", "failures": 5}}
```

---

### `reset_circuit_breaker(provider) -> bool`

Manually reset a provider's circuit breaker to closed state.

```python
stateloom.reset_circuit_breaker("openai")
```

**Returns:** `True` if the circuit was found and reset.

---

## Session Suspension

### `suspend_session(session_id, reason="", data=None) -> bool`

Suspend an active session externally (human-in-the-loop). The next LLM call raises `StateLoomSuspendedError`.

```python
stateloom.suspend_session("session-123", reason="Needs manager approval", data={"draft": "..."})
```

---

### `signal_session(session_id, payload=None) -> bool`

Resume a suspended session with an optional payload.

```python
stateloom.signal_session("session-123", payload={"approved": True, "reviewer": "jane"})
```

---

### `suspend(reason="", data=None, timeout=None) -> Any`

Suspend the **current** session and block until signaled. Use inside a session to pause execution and wait for human input.

```python
with stateloom.session("approval-task") as s:
    response = client.chat.completions.create(...)

    # Pause and wait for human input
    payload = stateloom.suspend(
        reason="Needs manager approval",
        data={"draft": response.choices[0].message.content},
        timeout=300.0,  # Wait up to 5 minutes
    )

    if payload and payload.get("approved"):
        # Continue with approved response
        ...
```

**Returns:** The signal payload from the human, or `None` on timeout.

---

### `async_suspend(reason="", data=None, timeout=None) -> Any`

Async version of `suspend()`. Non-blocking wait.

```python
async with stateloom.async_session("async-approval") as s:
    payload = await stateloom.async_suspend(reason="Needs review", timeout=300.0)
```

---

## Compliance

### `purge_user_data(user_identifier, standard="gdpr") -> dict`

Purge all data matching a user identifier (Right to Be Forgotten).

```python
result = stateloom.purge_user_data("user@example.com", standard="gdpr")
print(result)
# {
#     "user_identifier": "user@example.com",
#     "sessions_deleted": 5,
#     "events_deleted": 42,
#     "cache_entries_deleted": 3,
#     "jobs_deleted": 0,
#     "virtual_keys_deleted": 0,
#     "audit_event_id": "evt-abc123",
# }
```

---

### `compliance_cleanup() -> int`

Run session TTL enforcement for all compliance-configured organizations.

```python
deleted_count = stateloom.compliance_cleanup()
print(f"Purged {deleted_count} expired sessions")
```

**Returns:** Number of sessions purged.

---

## Organizations & Teams

### `create_organization(name="", *, budget=None, pii_rules=None, compliance_profile=None) -> Organization`

Create a new organization.

```python
org = stateloom.create_organization(
    name="Acme Corp",
    budget=1000.0,
    compliance_profile="gdpr",
)
print(org.id)  # "org-abc123"
```

---

### `get_organization(org_id) -> Organization | None`

```python
org = stateloom.get_organization("org-abc123")
```

---

### `list_organizations() -> list[Organization]`

```python
orgs = stateloom.list_organizations()
for org in orgs:
    print(f"{org.name} ({org.id})")
```

---

### `create_team(org_id, name="", *, budget=None, compliance_profile=None) -> Team`

Create a new team within an organization.

```python
team = stateloom.create_team(
    org.id,
    name="Customer Support",
    budget=100.0,
)
print(team.id)  # "team-xyz789"
```

---

### `get_team(team_id) -> Team | None`

```python
team = stateloom.get_team("team-xyz789")
```

---

### `list_teams(org_id=None) -> list[Team]`

List teams, optionally filtered by organization.

```python
# All teams
teams = stateloom.list_teams()

# Teams in a specific org
teams = stateloom.list_teams(org_id=org.id)
```

---

### `org_stats(org_id) -> dict`

Get aggregated stats for an organization.

```python
stats = stateloom.org_stats(org.id)
# {"total_sessions": 150, "total_cost": 45.23, "total_tokens": 120000, ...}
```

---

### `team_stats(team_id) -> dict`

Get aggregated stats for a team.

```python
stats = stateloom.team_stats(team.id)
```

---

## Managed Agents

### `create_agent(slug, team_id, ...) -> Agent`

Create a managed agent definition with an initial version (v1).

```python
agent = stateloom.create_agent(
    slug="legal-bot",
    team_id="team-1",
    model="gpt-4o",
    system_prompt="You are a legal assistant. Be concise and cite sources.",
    request_overrides={"temperature": 0.2},
    budget_per_session=5.0,
    description="Legal Q&A agent",
)
print(agent.id)                # "agt-abc123"
print(agent.active_version_id) # "agv-xyz789"

# The agent is now accessible via the proxy:
# POST http://localhost:4781/v1/agents/legal-bot/chat/completions
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `slug` | `str` | required | URL-friendly ID (3–64 chars, `^[a-z0-9][a-z0-9-]*[a-z0-9]$`) |
| `team_id` | `str` | required | Owning team ID |
| `name` | `str` | `""` | Human-readable name |
| `model` | `str` | `""` | Model for the initial version |
| `system_prompt` | `str` | `""` | System prompt for the initial version |
| `description` | `str` | `""` | Agent description |
| `request_overrides` | `dict \| None` | `None` | Default overrides (temperature, max_tokens, etc.) |
| `budget_per_session` | `float \| None` | `None` | Per-session budget cap in USD |
| `metadata` | `dict \| None` | `None` | Arbitrary metadata |
| `created_by` | `str` | `""` | Creator identifier |

---

### `get_agent(agent_id) -> Agent`

```python
agent = stateloom.get_agent("agt-abc123")
print(agent.slug, agent.status)
```

---

### `list_agents(team_id=None, org_id=None) -> list[Agent]`

```python
# All agents
agents = stateloom.list_agents()

# Filtered
agents = stateloom.list_agents(team_id="team-1")
```

---

### `create_agent_version(agent_id, ...) -> AgentVersion`

Create a new immutable version for an agent.

```python
v2 = stateloom.create_agent_version(
    agent.id,
    model="gpt-4o-mini",
    system_prompt="You are a legal assistant. Be brief.",
    request_overrides={"temperature": 0.1},
)
print(v2.version_number)  # 2
```

---

### `activate_agent_version(agent_id, version_id) -> Agent`

Activate a specific version (rollback).

```python
# Roll back to v1
updated = stateloom.activate_agent_version(agent.id, "agv-original-id")
print(updated.active_version_id)  # "agv-original-id"
```

---

## Virtual Keys

### `create_virtual_key(team_id, name="", ...) -> dict`

Create a virtual API key for proxy authentication. The full key is only returned once at creation time.

```python
key_info = stateloom.create_virtual_key(
    team_id="team-1",
    name="dev-key",
    allowed_models=["gpt-4o-mini", "claude-*"],  # Glob patterns
    budget_limit=100.0,
    rate_limit_tps=5.0,
    agent_ids=["agt-abc123"],  # Restrict to specific agents
)
print(key_info["key"])          # "ag-abc123..." (shown only once!)
print(key_info["key_preview"])  # "ag-abc...xyz"
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `team_id` | `str` | required | Team to associate the key with |
| `name` | `str` | `""` | Human-readable name |
| `allowed_models` | `list[str] \| None` | `None` | Glob patterns restricting models (empty = all) |
| `budget_limit` | `float \| None` | `None` | Per-key spend cap in USD |
| `rate_limit_tps` | `float \| None` | `None` | Per-key TPS limit |
| `agent_ids` | `list[str] \| None` | `None` | Restrict to specific agents (empty = all) |

**Returns:** Dict with `id`, `key`, `key_preview`, `team_id`, `org_id`, `name`, `created_at`.

---

### `list_virtual_keys(team_id=None) -> list[dict]`

List virtual keys (previews only — full keys are never returned after creation).

```python
keys = stateloom.list_virtual_keys(team_id="team-1")
for k in keys:
    print(f"{k['name']}: {k['key_preview']} (revoked={k['revoked']})")
```

---

### `revoke_virtual_key(key_id) -> bool`

Revoke a virtual API key.

```python
stateloom.revoke_virtual_key(key_info["id"])
```

**Returns:** `True` if the key was found and revoked.

---

## Async Jobs

### `submit_job(...) -> Job`

Submit an async job for background processing through the full middleware pipeline.

```python
stateloom.init(async_jobs_enabled=True)

job = stateloom.submit_job(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Summarize this document..."}],
    webhook_url="https://hooks.example.com/results",
    webhook_secret="my-secret",
    max_retries=3,
    ttl_seconds=3600,
)
print(job.id)  # "job-abc123"

# Agent-powered job — resolves the agent's model and system prompt
job = stateloom.submit_job(
    agent="summarizer",
    messages=[{"role": "user", "content": "Summarize this document..."}],
)
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | `str` | `""` | Model to use |
| `messages` | `list[dict] \| None` | `None` | Chat messages |
| `webhook_url` | `str` | `""` | URL for result delivery (HMAC-SHA256 signed) |
| `webhook_secret` | `str` | `""` | Webhook signing secret |
| `session_id` | `str` | `""` | Associate with an existing session |
| `org_id` | `str` | `""` | Organization scope |
| `team_id` | `str` | `""` | Team scope |
| `max_retries` | `int` | `3` | Max retry attempts on failure |
| `ttl_seconds` | `int \| None` | `None` | Job TTL (default from config) |
| `metadata` | `dict \| None` | `None` | Arbitrary metadata |
| `agent` | `str \| None` | `None` | Agent slug or ID — resolves the agent's model and system prompt |

---

### `get_job(job_id) -> Job`

```python
job = stateloom.get_job("job-abc123")
print(job.status)  # "pending", "running", "completed", "failed", "cancelled"
```

---

### `list_jobs(status=None, session_id=None, limit=100, offset=0) -> list[Job]`

```python
# All completed jobs
jobs = stateloom.list_jobs(status="completed")

# Jobs for a session
jobs = stateloom.list_jobs(session_id="session-123")
```

---

### `cancel_job(job_id) -> bool`

Cancel a pending job.

```python
stateloom.cancel_job("job-abc123")
```

**Returns:** `True` if cancelled.

---

### `job_stats() -> dict`

Get aggregate job statistics.

```python
stats = stateloom.job_stats()
# {"total": 100, "completed": 85, "failed": 5, "pending": 10, ...}
```

---

## Config Locking

### `lock_setting(setting, value=None, *, reason="") -> dict`

Lock a config setting to prevent developer overrides. Developers who try to override a locked setting get `StateLoomConfigLockedError`.

```python
stateloom.lock_setting("blast_radius_enabled", value=True, reason="Security policy")
stateloom.lock_setting("pii_enabled", value=True, reason="Compliance requirement")

# This now raises StateLoomConfigLockedError:
stateloom.init(blast_radius_enabled=False)
```

---

### `unlock_setting(setting) -> bool`

```python
stateloom.unlock_setting("blast_radius_enabled")
```

**Returns:** `True` if a lock was removed.

---

### `list_locked_settings() -> list[dict]`

```python
locks = stateloom.list_locked_settings()
for lock in locks:
    print(f"{lock['setting']} = {lock['value']} ({lock['reason']})")
```

---

## Prompt File Watcher

### `prompt_watcher_status() -> dict`

Get prompt file watcher status.

```python
status = stateloom.prompt_watcher_status()
# {"enabled": True, "prompts_dir": "/path/to/prompts", "tracked_files": 3, "errors": []}
```

---

### `rescan_prompts() -> dict`

Force an immediate full scan of the prompts directory.

```python
status = stateloom.rescan_prompts()
print(f"Tracking {status['tracked_files']} prompt files")
```

---

## Guardrails

### `guardrails_status() -> dict`

Get guardrails configuration and detection status.

```python
status = stateloom.guardrails_status()
# {
#     "enabled": True, "mode": "enforce", "pattern_count": 32,
#     "nli_enabled": True, "nli_available": True,
#     "local_model_available": True, ...
# }
```

---

### `configure_guardrails(**kwargs)`

Configure guardrails at runtime — no restart needed. Changes take effect on the next request.

```python
# Enable NLI classifier
stateloom.configure_guardrails(nli_enabled=True)

# Adjust NLI threshold
stateloom.configure_guardrails(nli_threshold=0.8)

# Change mode
stateloom.configure_guardrails(mode="enforce")

# Multiple settings at once
stateloom.configure_guardrails(
    nli_enabled=True,
    nli_threshold=0.8,
    mode="enforce",
)
```

**Parameters** (all optional):

| Parameter | Type | Description |
|-----------|------|-------------|
| `nli_enabled` | `bool` | Enable/disable NLI injection classifier |
| `nli_threshold` | `float` | NLI score threshold (0.0-1.0) |
| `mode` | `str` | `"audit"` or `"enforce"` |

---

## Security

### `security_status() -> dict`

Get combined security status (audit hooks + secret vault).

```python
status = stateloom.security_status()
```

---

### `vault_store(key, value)`

Store a secret in the in-memory vault.

```python
stateloom.vault_store("CUSTOM_SECRET", "value")
```

---

### `vault_retrieve(key) -> str | None`

Retrieve a secret from the vault.

```python
secret = stateloom.vault_retrieve("CUSTOM_SECRET")
```

---

## Threading & Integrations

### `patch_threading()`

Patch `threading.Thread` to propagate StateLoom session context to child threads. Call once at startup. Restored on `stateloom.shutdown()`.

```python
stateloom.init()
stateloom.patch_threading()

# Now child threads inherit the parent's session context
with stateloom.session("task-123") as s:
    def worker():
        # This LLM call is tracked in "task-123"
        client.chat.completions.create(...)

    thread = threading.Thread(target=worker)
    thread.start()
    thread.join()
```

---

### `langchain_callback(gate=None, *, tools_only=None)`

Create a LangChain callback handler for StateLoom observability.

**Recommended** (auto_patch=True + callback — full middleware pipeline):
```python
stateloom.init()  # auto_patch=True by default
handler = stateloom.langchain_callback()  # auto-detects tools_only
chain.invoke(input, config={"callbacks": [handler]})
```

**Standalone** (callback only, no middleware enforcement):
```python
stateloom.init(auto_patch=False)
handler = stateloom.langchain_callback(tools_only=False)
chain.invoke(input, config={"callbacks": [handler]})
```

**Parameters:**
- `gate` — Explicit Gate reference. Uses the global singleton when `None`.
- `tools_only` — Controls whether LLM events are recorded by the callback:
  - `True` — only record tool events (LLM tracking via middleware pipeline)
  - `False` — record both LLM and tool events (standalone mode)
  - `None` (default) — auto-detect from `gate.config.auto_patch`

**Framework context bridge:** In recommended mode (`tools_only=True`), the callback
automatically annotates middleware pipeline events with LangChain-specific metadata
(`run_id`, `chain_name`, `tags`, `model`, `provider`). This metadata appears in
`event.metadata["langchain"]` and is visible in the dashboard session detail view.
The bridge uses a ContextVar — `on_llm_start`/`on_chat_model_start` sets it before
the SDK call fires, and `on_llm_end`/`on_llm_error` clears it after.

---

## Error Reference

All errors inherit from `StateLoomError`. Each includes:
- `error_code` — machine-readable code string
- `help_url` — link to relevant documentation
- `details` — structured string with actionable next steps

```python
from stateloom import StateLoomBudgetError

try:
    response = client.chat.completions.create(...)
except StateLoomBudgetError as e:
    print(e.error_code)    # "BUDGET_EXCEEDED"
    print(e.limit)         # 5.0
    print(e.spent)         # 5.12
    print(e.session_id)    # "task-123"
```

| Error Class | Code | When Raised |
|-------------|------|-------------|
| `StateLoomError` | `STATELOOM_ERROR` | Base error for all StateLoom exceptions |
| `StateLoomBudgetError` | `BUDGET_EXCEEDED` | Session exceeds its budget limit |
| `StateLoomLoopError` | `LOOP_DETECTED` | Repeated request pattern detected |
| `StateLoomPIIBlockedError` | `PII_BLOCKED` | PII block rule prevents an LLM call |
| `StateLoomGuardrailError` | `GUARDRAIL_BLOCKED` | Prompt injection or jailbreak detected (enforce mode) |
| `StateLoomKillSwitchError` | `KILL_SWITCH` | Global kill switch or matching rule is active |
| `StateLoomBlastRadiusError` | `BLAST_RADIUS` | Session/agent paused by blast radius |
| `StateLoomRateLimitError` | `RATE_LIMITED` | Team rate limit exceeded |
| `StateLoomRetryError` | `RETRY_EXHAUSTED` | All retry attempts exhausted |
| `StateLoomTimeoutError` | `SESSION_TIMED_OUT` | Session exceeded duration or idle timeout |
| `StateLoomCancellationError` | `SESSION_CANCELLED` | Session was explicitly cancelled |
| `StateLoomSuspendedError` | `SESSION_SUSPENDED` | LLM call on a suspended session |
| `StateLoomCircuitBreakerError` | `CIRCUIT_BREAKER_OPEN` | Provider circuit breaker is open |
| `StateLoomComplianceError` | `COMPLIANCE_BLOCKED` | Compliance policy violation |
| `StateLoomConfigLockedError` | `CONFIG_LOCKED` | Admin-locked setting override attempted |
| `StateLoomJobError` | `ASYNC_JOB_ERROR` | Async job processing error |
| `StateLoomReplayError` | `REPLAY_ERROR` | Replay engine error |
| `StateLoomSideEffectError` | `SIDE_EFFECT_BLOCKED` | Outbound HTTP blocked during strict replay |
| `StateLoomSecurityError` | `SECURITY_BLOCKED` | Security policy (audit hook) blocked an operation |
| `StateLoomAuthError` | `AUTH_ERROR` | Authentication failure |
| `StateLoomPermissionError` | `PERMISSION_DENIED` | Insufficient permissions (RBAC) |
| `StateLoomFeatureError` | `FEATURE_UNAVAILABLE` | Enterprise feature requires license key |

### Non-Retryable Errors

These errors propagate immediately through `durable_task()` and `retry_loop()` — they are never retried:

- `StateLoomBudgetError`
- `StateLoomPIIBlockedError`
- `StateLoomGuardrailError`
- `StateLoomKillSwitchError`
- `StateLoomBlastRadiusError`
- `StateLoomRateLimitError`
- `StateLoomRetryError`
- `StateLoomTimeoutError`
- `StateLoomCancellationError`
- `StateLoomComplianceError`
- `StateLoomSuspendedError`
- `StateLoomSecurityError`

---

## Type Reference

Key types importable from `stateloom` or their source modules.

| Type | Import | Description |
|------|--------|-------------|
| `Gate` | `stateloom` | Singleton orchestrator — owns pipeline, store, pricing, sessions |
| `Session` | `stateloom` | Session with cost, token, and step accumulators |
| `Client` | `stateloom` | Unified provider-agnostic chat client |
| `ChatResponse` | `stateloom` | Response wrapper from `chat()` / `achat()` |
| `MockSession` | `stateloom` | VCR-cassette record/replay for testing |
| `RoutingContext` | `stateloom` | Context passed to custom auto-routing scorers |
| `Organization` | `stateloom` | Multi-tenant organization model |
| `Team` | `stateloom` | Multi-tenant team model |
| `Agent` | `stateloom` | Managed agent definition (prefix `agt-`) |
| `AgentVersion` | `stateloom` | Immutable agent version (prefix `agv-`) |
| `AgentStatus` | `stateloom` | Enum: `ACTIVE`, `PAUSED`, `ARCHIVED` |
| `ConsensusResult` | `stateloom` | Result from `consensus()` — answer, confidence, cost, rounds |
| `DebateRound` | `stateloom` | Per-round data in a consensus run |
| `ConsensusStrategy` | `stateloom` | Enum: `VOTE`, `DEBATE`, `MOA`, `SELF_CONSISTENCY` |
| `StateLoomConfig` | `stateloom.core.config` | Pydantic-settings configuration model |
| `PIIRule` | `stateloom` | PII rule configuration |
| `KillSwitchRule` | `stateloom` | Kill switch rule (model glob, provider, environment) |
| `ComplianceProfile` | `stateloom` | Compliance profile configuration |
| `ComplianceStandard` | `stateloom` | Enum: `NONE`, `GDPR`, `HIPAA`, `CCPA` |
| `DataRegion` | `stateloom` | Enum: `GLOBAL`, `EU`, `US_EAST`, `US_WEST`, `APAC` |
| `FailureAction` | `stateloom` | Enum: `BLOCK`, `PASS` |
| `Provider` | `stateloom.core.types` | Enum: `OPENAI`, `ANTHROPIC`, `GEMINI`, `MISTRAL`, `COHERE`, `LITELLM`, `LOCAL`, `UNKNOWN` |
| `BillingMode` | `stateloom.core.types` | Enum: `API`, `SUBSCRIPTION` |
| `SessionStatus` | `stateloom.core.types` | Enum: `ACTIVE`, `COMPLETED`, `BUDGET_EXCEEDED`, `LOOP_KILLED`, `ERROR`, `PAUSED`, `SUSPENDED`, `TIMED_OUT`, `CANCELLED` |
| `EventType` | `stateloom.core.types` | Enum: `LLM_CALL`, `TOOL_CALL`, `CACHE_HIT`, `PII_DETECTION`, `SHADOW_DRAFT`, `LOCAL_ROUTING`, `KILL_SWITCH`, `BLAST_RADIUS`, `RATE_LIMIT`, `CHECKPOINT`, `CIRCUIT_BREAKER`, `COMPLIANCE_AUDIT`, `SEMANTIC_RETRY`, `SUSPENSION`, `ASYNC_JOB`, `DEBATE_ROUND`, `CONSENSUS`, and more |
