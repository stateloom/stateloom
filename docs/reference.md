# StateLoom — Full Feature Reference

> This is the detailed feature reference. For a quick overview, see the [README](../README.md). For the complete public API index, see the [API Reference](api-reference.md).

## Table of Contents

- [What It Does](#what-it-does)
- [Quick Start](#quick-start)
- [Providers](#providers)
- [Sessions](#sessions)
- [Budget Enforcement](#budget-enforcement)
- [PII Detection](#pii-detection)
- [Guardrails](#guardrails-prompt-injection-protection)
- [Tool Tracking](#tool-tracking)
- [Caching](#caching)
- [Local Models (Ollama)](#local-models-ollama)
- [Model Testing](#model-testing)
- [Intelligent Auto-Routing](#intelligent-auto-routing)
- [Kill Switch](#kill-switch)
- [Blast Radius Containment](#blast-radius-containment)
- [Rate Limiting](#rate-limiting)
- [Proxy (Multi-Protocol Gateway)](#proxy-multi-protocol-gateway)
- [Managed Agents (Prompts-as-an-API)](#managed-agents-prompts-as-an-api)
- [File-Based Prompt Versioning](#file-based-prompt-versioning)
- [A/B Experiments](#ab-experiments)
- [Multi-Agent Consensus](#multi-agent-consensus)
- [Time-Travel Debugging](#time-travel-debugging)
- [Durable Resumption (Checkpointing)](#durable-resumption-checkpointing)
- [Semantic Retries (Self-Healing)](#semantic-retries-self-healing)
- [Named Checkpoints](#named-checkpoints)
- [Parent-Child Sessions](#parent-child-sessions)
- [Session Timeouts & Heartbeats](#session-timeouts--heartbeats)
- [Session Cancellation](#session-cancellation)
- [Session Suspension (Human-in-the-Loop)](#session-suspension-human-in-the-loop)
- [VCR-Cassette Mock (Zero-Cost Testing)](#vcr-cassette-mock-zero-cost-testing)
- [Unified Chat API](#unified-chat-api)
- [Multi-Model Workflows](#multi-model-workflows)
- [Session Export & Import](#session-export--import)
- [Config Locking (Admin Controls)](#config-locking-admin-controls)
- [Circuit Breaker (Provider Failover)](#circuit-breaker-provider-failover)
- [Compliance Enforcement (GDPR/HIPAA/CCPA)](#compliance-enforcement-gdprhipaaccpa)
- [Billing Mode (API vs Subscription)](#billing-mode-api-vs-subscription)
- [Anthropic-Native Proxy](#anthropic-native-proxy)
- [Gemini-Native Proxy](#gemini-native-proxy)
- [Authentication & RBAC](#authentication--rbac)
- [Multi-Tenant Hierarchy](#multi-tenant-hierarchy)
- [Async Jobs](#async-jobs)
- [LangChain Integration](#langchain-integration)
- [Zero-Trust Security Engine](#zero-trust-security-engine)
- [Cross-Process Config Sync](#cross-process-config-sync)
- [Dashboard](#dashboard)
- [Configuration](#configuration)
- [Middleware Pipeline](#middleware-pipeline)
- [Error Handling](#error-handling)
- [Project Structure](#project-structure)

## What It Does

- **Session-scoped cost tracking** — see cost per agent run, not just per API call
- **Per-model cost breakdown** — track cost and tokens per model within multi-model sessions
- **PII detection** — detect emails, credit cards, SSNs, API keys in prompts (audit/redact/block modes)
- **Guardrails** — prompt injection detection, jailbreak prevention, system prompt leak protection (heuristic + local Llama-Guard + enterprise API)
- **Budget enforcement** — hard stop or warn when a session exceeds its spend limit
- **Loop detection** — catch agents spinning on the same query
- **Exact-match caching** — skip duplicate LLM calls, save money
- **A/B experiments** — test model variants with built-in assignment, metrics, and backtesting
- **Time-travel debugging** — replay failed sessions from any step with network safety
- **Durable resumption** — Temporal-like checkpointing: crash mid-workflow, restart, and resume from cached LLM responses (including streaming) with zero re-execution cost
- **Semantic retries (self-healing)** — automatic retries for LLM output failures (bad JSON, hallucinated tool calls, missing fields) with durable-aware crash recovery
- **Local model support** — run Ollama models locally with hardware-aware recommendations and model management
- **Model testing** — test candidate models against your production model in parallel with every cloud call, compare responses with automatic similarity scoring, and generate migration readiness scores
- **Intelligent auto-routing** — automatically route simple requests to local models based on semantic NLI complexity scoring, budget pressure, realtime-data detection, and learned success rates
- **Global kill switch** — halt all LLM traffic instantly, or block specific models/providers/environments with granular rules and glob patterns
- **Blast radius containment** — auto-pause sessions and agents after repeated failures or budget violations, with cross-session agent tracking and webhooks
- **Priority-aware rate limiting** — per-team TPS limits with request queueing and priority lanes, configured via dashboard API
- **Named checkpoints** — mark milestones within sessions for observability and debugging
- **Parent-child sessions** — hierarchical session management with auto-derived parent, org/team inheritance, and child listing
- **Session timeouts & heartbeats** — per-session max duration and idle timeout with automatic heartbeat tracking
- **Session cancellation** — cancel running sessions gracefully with cooperative cancellation checking
- **Managed agent definitions (Prompts-as-an-API)** — deploy AI agents without code: pick a model, write a system prompt, set a budget, get a URL. Immutable versioning with instant rollback, virtual key scoping, and full middleware pipeline
- **File-based prompt versioning** — drop `.md`/`.yaml`/`.txt` files in a `prompts/` folder; StateLoom auto-detects changes as new agent versions. Edit a file → new version auto-activates instantly. Delete a file → agent archived. Full version history and rollback in the dashboard
- **Circuit breaker** — automatic provider failover on outages with tier-based model fallback and synthetic health probes
- **Compliance enforcement** — declarative GDPR/HIPAA/CCPA profiles with tamper-proof audit trails, zero-retention modes, data region controls, and "Right to Be Forgotten" purge engine
- **Session suspension (human-in-the-loop)** — pause agent sessions to wait for human approval, resume with signal payload
- **Billing mode detection** — distinguish API-billed vs subscription (Claude Max, ChatGPT Plus) users; tracks both actual cost ($0 for subscriptions) and estimated API cost
- **Async jobs** — fire-and-forget LLM calls with webhook notifications (HMAC-SHA256 signed), retry logic, and pluggable queue backends (Redis Streams, Kafka, SQS)
- **Multi-tenant hierarchy** — Organizations and Teams with per-level budgets, PII rules, compliance profiles, and rate limits
- **Sticky sessions** — automatic session grouping for proxy clients via IP + user-agent fingerprinting
- **Observability** — Prometheus metrics, time-series aggregation, webhook-based alerting, and optional OpenTelemetry distributed tracing
- **Semantic caching** — beyond exact-match: embedding-based similarity matching with FAISS/Redis vector backends and request normalization
- **HTTP reverse proxy** — true passthrough proxy that forwards requests to upstream APIs via httpx, enabling subscription users (Claude Max, Gemini Ultra, ChatGPT Plus) whose CLIs use OAuth/session tokens instead of API keys
- **OpenAI-compatible proxy** — drop-in `/v1/chat/completions` endpoint with virtual keys and BYOK (Bring Your Own Key) header support
- **Anthropic-native proxy** — drop-in `/v1/messages` endpoint for Claude CLI and Anthropic SDK clients, with transparent auth passthrough for subscription users
- **Gemini-native proxy** — drop-in `/v1beta/models/{model}:generateContent` endpoint for Gemini CLI and Google AI SDK clients, with transparent auth passthrough for subscription users
- **Codex CLI proxy** — OpenAI Responses API (`/v1/responses`) via HTTP POST + WebSocket, with ChatGPT OAuth routing, zstd body decompression, and middleware enforcement on WebSocket traffic
- **Additional providers** — Cohere, Mistral, and LiteLLM adapters alongside OpenAI, Anthropic, and Gemini
- **NER-based PII detection** — GLiNER zero-shot entity recognition in addition to regex-based scanning
- **Streaming PII buffer** — hold back stream chunks for PII scanning before releasing to the client
- **Tool tracking** — `@stateloom.tool()` decorator for visibility into tool calls and safe replay
- **VCR-cassette mock** — record LLM calls once, replay forever for zero-cost testing (`stateloom.mock()`)
- **Unified chat API** — provider-agnostic `stateloom.chat()` / `stateloom.achat()` with BYOK support
- **Session export/import** — portable JSON bundles for sharing, debugging, and migration
- **Config locking** — admin-lock settings to prevent developer overrides
- **LangChain integration** — first-class callback handler for any LangChain runnable
- **LangGraph integration** — callback handler for LangGraph workflows
- **Authentication & RBAC** — unified role-based access control with local email/password auth, JWT tokens, OIDC federation (Google, Okta, etc.), and a five-tier role hierarchy (`org_admin`, `org_auditor`, `team_admin`, `team_editor`, `team_viewer`). Optional — disabled by default
- **VK scope enforcement** — restrict virtual keys to specific proxy endpoints (`chat`, `messages`, `responses`, `generate`, `agents`). Empty scopes = all allowed (backward compatible)
- **End-user attribution** — track downstream users via `X-StateLoom-End-User` header. Sanitized, stored as a first-class session field, filterable in dashboard
- **Local dashboard** — live session viewer + REST API + WebSocket at localhost:4781

## Quick Start

```python
import stateloom
import openai

stateloom.init()

client = openai.OpenAI()
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Hello!"}],
)
# Terminal: [StateLoom] gpt-4o | 28 tok | $0.0004 | 312ms | session:a1b2c3...
```

## Providers

StateLoom auto-detects and patches installed LLM clients:

| Provider | Package | Auto-patched |
|----------|---------|:------------:|
| OpenAI | `openai` | Yes |
| Anthropic | `anthropic` | Yes |
| Google Gemini | `google-generativeai` | Yes |
| Cohere | `cohere` | Yes |
| Mistral | `mistralai` | Yes |
| LiteLLM | `litellm` | Yes |
| Ollama (local) | — | Via `local_model=` |
| Custom | any | Via `register_provider()` |

### Explicit wrapping (no monkey-patching)

```python
gate = stateloom.init(auto_patch=False)
client = stateloom.wrap(openai.OpenAI())
```

### Custom providers

```python
from stateloom.intercept.provider_adapter import BaseProviderAdapter

class MistralAdapter(BaseProviderAdapter):
    @property
    def name(self) -> str:
        return "mistral"
    # ... implement extract_tokens, get_patch_targets, etc.

stateloom.register_provider(
    MistralAdapter(),
    pricing={"mistral-large-latest": (0.000002, 0.000006)},
)
stateloom.init()
```

## Sessions

Sessions group related LLM calls into a single tracked unit with shared cost, token counts, and metadata.

```python
with stateloom.session("task-123", name="Summarize docs", budget=5.0) as s:
    response = client.chat.completions.create(...)
    print(s.total_cost, s.total_tokens)

# Async
async with stateloom.async_session("task-456", budget=2.0) as s:
    response = await client.chat.completions.create(...)

# Durable mode — see "Durable Resumption" section below
with stateloom.session(session_id="task-789", durable=True) as s:
    res1 = client.chat.completions.create(...)  # Cache hit (free) on resume
    res2 = client.chat.completions.create(...)  # Live call (picks up here)
```

## Budget Enforcement

```python
stateloom.init(budget=10.0)  # $10 per session, hard stop on exceed

# Or configure failure behavior explicitly
stateloom.init(budget=10.0, budget_on_middleware_failure="block")
```

When a session exceeds its budget, `StateLoomBudgetError` is raised before the next LLM call.

## PII Detection

Three modes: **audit** (log only), **redact** (mask before LLM, restore after), **block** (prevent the call).

```python
from stateloom import PIIRule

stateloom.init(
    pii=True,  # Enable with audit-all default
    pii_rules=[
        PIIRule(pattern="credit_card", mode="block", on_middleware_failure="block"),
        PIIRule(pattern="email", mode="redact"),
        PIIRule(pattern="api_key_openai", mode="block", on_middleware_failure="block"),
    ],
)
```

Detected patterns: `email`, `credit_card`, `ssn`, `phone_us`, `ip_address`, `api_key_openai`, `api_key_anthropic`, `api_key_aws`, `api_key_bearer`.

## Guardrails (Prompt Injection Protection)

Four protection levels: **heuristic** (32 regex patterns, ~0ms), **NLI classifier** (CrossEncoder semantic classification, ~5-20ms, opt-in), **local model** (Llama-Guard via Ollama, 100-500ms), and **enterprise API** (pluggable protocol). Plus output scanning for system prompt leak detection.

Two modes: **audit** (log only, never block) and **enforce** (block on violation).

```python
# Audit mode — log all violations without blocking
stateloom.init(
    guardrails_enabled=True,
    guardrails_mode="audit",
)

# Enforce mode — block high/critical severity violations
stateloom.init(
    guardrails_enabled=True,
    guardrails_mode="enforce",
    guardrails_heuristic_enabled=True,       # Fast regex patterns (default: True)
    guardrails_nli_enabled=True,             # NLI classifier (~5-20ms, opt-in)
    guardrails_local_model_enabled=True,     # Llama-Guard via Ollama
    guardrails_output_scanning_enabled=True, # System prompt leak detection
)

# Runtime toggle — no restart needed
stateloom.configure_guardrails(nli_enabled=True, nli_threshold=0.8, mode="enforce")
```

When a violation is detected in enforce mode, `StateLoomGuardrailError` is raised before the LLM call.

### Heuristic patterns (Level 1)

32 built-in regex patterns across categories:

- **Prompt injection** — ignore instructions, system prompt override, fake system role, instruction delimiters, conversation reset, completion manipulation, indirect injection, few-shot poisoning, recursive override
- **Jailbreak** — DAN mode, developer mode, "do anything now", unfiltered mode, evil twin, roleplay exploit, hypothetical framing
- **Encoding attacks** — base64/hex/ROT13 hidden instructions, zero-width unicode smuggling, homograph attacks (Cyrillic lookalikes), token smuggling (ChatML special tokens)
- **Code injection** — SQL injection, shell injection, XPath injection, markdown/HTML injection
- **Evasion** — spaced character evasion, multi-turn manipulation, payload splitting
- **Meta** — system prompt extraction attempts

```python
# Disable specific patterns
stateloom.init(
    guardrails_enabled=True,
    guardrails_disabled_rules=["xpath_injection", "rot13_instruction"],
)
```

### NLI classifier (Level 2)

Zero-shot Natural Language Inference classifier using CrossEncoder (`cross-encoder/nli-MiniLM2-L6-H768`). Scores prompts as 0.0–1.0 injection probability in ~5-20ms. Opt-in — disabled by default. Requires `stateloom[semantic]` (no new dependencies if already using semantic caching or auto-routing).

```python
stateloom.init(
    guardrails_enabled=True,
    guardrails_nli_enabled=True,             # Enable NLI classifier
    guardrails_nli_threshold=0.75,           # Score threshold for flagging
    guardrails_nli_model="cross-encoder/nli-MiniLM2-L6-H768",  # default
)

# Or toggle at runtime via dashboard API or programmatically:
stateloom.configure_guardrails(nli_enabled=True, nli_threshold=0.8)
```

Fail-open: returns `None` on error (model unavailable, timeout). Lazy-initialized with thread-safe double-checked locking. Truncates input to 2000 chars (model token limit). NLI score maps to severity: >0.9 → critical, >0.8 → high, else → medium.

### Local model validation (Level 3)

Uses Llama-Guard 3 via Ollama for ML-based classification. Auto-pulls the model on first use.

```python
stateloom.init(
    guardrails_enabled=True,
    guardrails_local_model_enabled=True,
    guardrails_local_model="llama-guard3:1b",  # default
    guardrails_local_model_timeout=10.0,       # seconds
)
```

Requires Ollama running locally. Fail-open: if Ollama is unavailable, the request proceeds.

### Output scanning

Detects if the LLM response leaks the system prompt back to the user.

```python
stateloom.init(
    guardrails_enabled=True,
    guardrails_output_scanning_enabled=True,
    guardrails_system_prompt_leak_threshold=0.6,  # 0.0-1.0 sensitivity
)
```

### Status

```python
status = stateloom.guardrails_status()
# {"enabled": True, "mode": "enforce", "pattern_count": 32,
#  "nli_enabled": True, "nli_available": True, "local_model_available": True, ...}
```

### Runtime configuration

```python
# Toggle NLI, adjust threshold, change mode — takes effect on the next request
stateloom.configure_guardrails(
    nli_enabled=True,
    nli_threshold=0.8,
    mode="enforce",
)
```

Dashboard API: `POST /api/security/guardrails/configure` with JSON body `{"nli_enabled": true, "nli_threshold": 0.8, "mode": "enforce"}`.

### Dashboard

Guardrail detections are visible in:
- **Live Overview** — "Guardrail Detections" stat card (total across all sessions)
- **Session detail** — per-session guardrail detection count
- **Security tab → Guardrails sub-tab** — config status (enabled/mode, heuristic/NLI/local model/output scanning), NLI toggle switch, aggregated detection stats (total, blocked, by category, by severity), and a recent violations table with severity/action badges

Dashboard API endpoints: `GET /api/security/guardrails` (config + stats), `GET /api/security/guardrails/events` (recent events, `?limit=N`), `POST /api/security/guardrails/configure` (runtime toggle).

### Configuration

| Option | Default | Description |
|--------|---------|-------------|
| `guardrails_enabled` | `False` | Enable guardrails middleware |
| `guardrails_mode` | `"audit"` | `"audit"` (log only) or `"enforce"` (block) |
| `guardrails_heuristic_enabled` | `True` | Enable regex pattern scanning |
| `guardrails_local_model_enabled` | `False` | Enable Llama-Guard via Ollama |
| `guardrails_local_model` | `"llama-guard3:1b"` | Llama-Guard model name |
| `guardrails_local_model_timeout` | `10.0` | Local model timeout in seconds |
| `guardrails_output_scanning_enabled` | `True` | Enable system prompt leak detection |
| `guardrails_system_prompt_leak_threshold` | `0.6` | Leak detection sensitivity (0.0-1.0) |
| `guardrails_nli_enabled` | `False` | Enable NLI injection classifier (requires `stateloom[semantic]`) |
| `guardrails_nli_model` | `"cross-encoder/nli-MiniLM2-L6-H768"` | NLI CrossEncoder model |
| `guardrails_nli_threshold` | `0.75` | NLI score threshold for flagging (0.0-1.0) |
| `guardrails_disabled_rules` | `[]` | Pattern names to skip |
| `guardrails_webhook_url` | `""` | Webhook URL for violation notifications |

### YAML config

```yaml
guardrails:
  enabled: true
  mode: enforce
  heuristic_enabled: true
  nli_enabled: true                              # NLI classifier (opt-in)
  nli_model: cross-encoder/nli-MiniLM2-L6-H768  # default
  nli_threshold: 0.75                            # injection score threshold
  local_model_enabled: true
  local_model: llama-guard3:1b
  local_model_timeout: 10.0
  output_scanning_enabled: true
  system_prompt_leak_threshold: 0.6
  disabled_rules:
    - xpath_injection
  webhook_url: https://hooks.example.com/guardrails
```

## Tool Tracking

```python
@stateloom.tool(mutates_state=True)
def create_ticket(title: str) -> dict:
    """Side-effecting tool — marked for safe replay."""
    return api.create(title=title)

@stateloom.tool(session_root=True)
async def run_agent(prompt: str) -> str:
    """Each call gets its own session automatically."""
    return await chain.invoke(prompt)
```

## Caching

Enabled by default. Exact-match request deduplication per session.

```python
stateloom.init(cache=True)  # default

# Same request in the same session returns cached response instantly
# Terminal: [StateLoom] CACHE HIT | exact match | saved $0.0010
```

## Local Models (Ollama)

Run models locally via [Ollama](https://ollama.com) as a cheaper alternative for simple tasks. StateLoom provides hardware detection, model recommendations, and download management.

```python
stateloom.init(local_model="llama3.2")

# Discover what your hardware can run
recs = stateloom.recommend_models()
for m in recs:
    print(f"{m['model']} ({m['tier']}, {m['size_gb']}GB) — {m['description']}")

# Download a model
stateloom.pull_model("llama3.2")

# List downloaded models
models = stateloom.list_local_models()

# Delete a model
stateloom.delete_local_model("llama3.2")
```

Requires Ollama running locally (`ollama serve`). No additional Python dependencies needed — communication uses httpx (already a dependency).

## Model Testing

Test candidate models against your production model before committing to a switch. StateLoom runs candidate models in parallel with your cloud calls, compares responses, and generates a migration readiness score.

```python
stateloom.init(
    shadow=True,
    shadow_model="llama3.2",          # candidate model to test
    shadow_sample_rate=0.5,           # test 50% of traffic (default: 1.0 = all)
    shadow_max_context_tokens=4096,   # skip requests exceeding this context length
    shadow_models=["llama3.2", "mistral:7b"],  # test multiple candidates at once
)

# Every sampled cloud call now also runs against candidate model(s) locally
# Terminal: [StateLoom] gpt-4o | 28 tok | $0.0004 | 312ms | session:a1b2c3...
# Terminal: [StateLoom]   SHADOW | llama3.2 | 450ms | 120 tok | similarity:0.87
```

Per-session overrides via session metadata:

```python
with stateloom.session("task-1") as s:
    s.metadata["shadow_model"] = "mistral:7b"     # Override candidate model
    s.metadata["shadow_enabled"] = False           # Disable for this session
```

Model testing metrics are available via the dashboard API at `GET /api/shadow/metrics`. Readiness scores per candidate model are available at `GET /api/shadow/readiness`.

**PII safety**: If the PII pre-scan raises an exception, the candidate model call is **skipped** (fail-closed). This prevents accidental PII leakage to local models — the cloud call still proceeds normally.

### Readiness Scoring

StateLoom automatically computes a **migration readiness score** (0.0–1.0) for each candidate model based on accumulated test results:

- **Similarity distribution** — how closely candidate responses match cloud responses across all sampled traffic
- **Latency ratio** — candidate vs. cloud response time (lower is better for the candidate)
- **Error rate** — percentage of candidate calls that fail or time out
- **Coverage** — what fraction of your traffic the candidate can handle (excludes skipped requests)

```
GET /api/shadow/readiness

{
  "candidates": {
    "llama3.2": {
      "readiness_score": 0.82,
      "total_comparisons": 1247,
      "avg_similarity": 0.87,
      "p50_similarity": 0.91,
      "p10_similarity": 0.64,
      "avg_latency_ratio": 1.3,
      "error_rate": 0.02,
      "coverage": 0.78,
      "recommendation": "good_candidate"
    },
    "mistral:7b": {
      "readiness_score": 0.65,
      "total_comparisons": 1183,
      "avg_similarity": 0.72,
      "recommendation": "needs_improvement"
    }
  }
}
```

Recommendations: `"ready"` (≥0.9), `"good_candidate"` (≥0.75), `"needs_improvement"` (≥0.5), `"not_ready"` (<0.5).

### Smart Filtering

Not every request is a good candidate for model testing. StateLoom automatically skips:

- **Requests exceeding `shadow_max_context_tokens`** — very long contexts are expensive to test and often provider-specific
- **Requests with tool use / function calling** — tool schemas vary across providers and local models rarely support them
- **Requests with image content** — multimodal inputs require vision-capable candidates
- **Requests with structured output (`response_format`)** — JSON mode support varies across models
- **Requests containing PII** (detected by pre-scan) — prevents leaking sensitive data to candidate models
- **Requests outside the sample rate** — `shadow_sample_rate=0.5` randomly tests 50% of traffic to reduce local compute cost

Skipped requests are not counted against readiness coverage, so the score reflects only comparable traffic.

## Enterprise Roadmap: Dark Launching & Distillation

The following features build on top of model testing and are planned for the enterprise tier.

### Dark Launching

Zero-risk model migration with functional equivalence checking. Dark launching extends model testing by adding **automated acceptance criteria** — instead of just comparing similarity scores, it evaluates whether the candidate model produces functionally equivalent outputs for your specific use case.

**How it works:**

1. **Define acceptance criteria** — specify what "equivalent" means for your workload: exact JSON schema match, semantic similarity threshold, key-value extraction accuracy, or custom validator functions
2. **Run a dark launch campaign** — StateLoom routes a configurable percentage of production traffic to both the current and candidate models, collecting structured comparison data
3. **Functional equivalence report** — beyond similarity scores, dark launching tracks schema compliance, factual consistency, latency budgets, and cost savings per request class
4. **Graduated rollout gates** — automatically promote the candidate to handle live traffic once acceptance criteria are met across a statistically significant sample (configurable confidence interval)

Dark launching is designed for teams migrating from expensive frontier models (GPT-4o, Claude Opus) to cheaper alternatives (GPT-4o-mini, Claude Haiku, local models) where response format and factual accuracy matter more than stylistic similarity.

### Distillation Flywheel

Automated fine-tuning dataset generation from production traffic. The distillation flywheel turns your model testing data into a continuous improvement pipeline.

**How it works:**

1. **Collect high-quality pairs** — model testing already produces (prompt, cloud_response, candidate_response, similarity_score) tuples for every sampled request. The flywheel filters for high-similarity pairs where the cloud model's response is correct and well-formed.
2. **Export training datasets** — automatically generate fine-tuning datasets in OpenAI JSONL, Hugging Face, or Axolotl formats, filtered by quality threshold and deduplicated
3. **Track dataset lineage** — every training example links back to the original session, request, and comparison event for full auditability
4. **Feedback loop** — after fine-tuning and redeploying the candidate, run another model testing campaign to measure improvement. The readiness score tracks progress across fine-tuning iterations.

The distillation flywheel closes the loop between model testing (evaluation) and model improvement (training), enabling teams to progressively replace cloud models with fine-tuned local alternatives while maintaining quality guarantees.

## Intelligent Auto-Routing

When you enable local models, auto-routing activates automatically behind the scenes — no separate flag needed. The router analyzes each request's complexity using heuristics, factors in budget pressure, and learns from past routing outcomes. Routing stats are persisted across restarts so the router gets smarter over time.

```python
stateloom.init(local_model="llama3.2")

# Simple requests route to local automatically
# Terminal: [StateLoom]   ROUTED LOCAL | llama3.2 | complexity:0.12 | reason:low complexity

# Complex requests (long prompts, tools, images, multi-turn) go to cloud as usual
# Budget pressure increases local routing as spend approaches the limit

# To explicitly disable auto-routing while keeping local model support:
stateloom.init(local_model="llama3.2", auto_route=False)
```

The router skips local routing for streaming, tool use, image content, structured output (`response_format`), requests already hitting cache, and **realtime data requests** (weather, stock prices, latest news, live scores, etc.). If the local model fails or returns an **inadequate response** (e.g., "I don't have access to real-time data", "my knowledge cutoff"), the request silently falls back to cloud with no user-visible error.

Per-session overrides:

```python
with stateloom.session("task-1") as s:
    s.metadata["auto_route_model"] = "mistral:7b"   # Override model
    s.metadata["auto_route_enabled"] = False         # Disable for this session
```

### How routing decisions work

1. **Complexity scoring** — weighted composite of token count, conversation depth, message count, max message length, system prompt presence, and model tier
2. **Budget pressure** — kicks in at 50% spend, linearly increases to 1.0 at 100%, lowers the routing threshold
3. **Historical learning** — tracks success/failure per cloud model, adjusts threshold after 5+ data points
4. **Probe (optional)** — when the score is in the uncertain zone, asks the local model to self-assess confidence before routing

### Configuration

```python
stateloom.init(
    local_model="llama3.2",             # Auto-enables routing
    auto_route_model="mistral:7b",      # Override routing model (defaults to local_model)
)
```

| Option | Default | Description |
|--------|---------|-------------|
| `auto_route_enabled` | auto | Auto-enabled when `local_model` is set; set `auto_route=False` to disable |
| `auto_route_model` | `""` | Local model for routing (falls back to `local_model_default`) |
| `auto_route_timeout` | `30.0` | Timeout for local model calls (seconds) |
| `auto_route_complexity_threshold` | `0.35` | Below this score, route to local |
| `auto_route_complex_floor` | `0.70` | Above this score, always use cloud |
| `auto_route_probe_enabled` | `True` | Enable probe in uncertain zone |
| `auto_route_probe_timeout` | `5.0` | Timeout for probe calls (seconds) |
| `auto_route_probe_threshold` | `0.6` | Probe confidence needed to route local |

## Kill Switch

Emergency stop for all LLM traffic — or granular rules to block specific models, providers, environments, or agent versions.

```python
# Global kill switch — block everything
stateloom.kill_switch(active=True, message="Maintenance in progress")

# Resume traffic
stateloom.kill_switch(active=False)

# Granular rules with glob patterns
stateloom.add_kill_switch_rule(model="gpt-4*", reason="Cost overrun on GPT-4 family")
stateloom.add_kill_switch_rule(provider="anthropic", environment="production")
stateloom.add_kill_switch_rule(agent_version="v2.1.0", message="Known bug in v2.1.0")

# View and manage rules
rules = stateloom.kill_switch_rules()
stateloom.clear_kill_switch_rules()
```

### Response modes

By default, blocked requests raise `StateLoomKillSwitchError`. Use `response` mode to return a static response instead (useful for graceful degradation):

```python
stateloom.init(kill_switch_response_mode="response")
# Blocked calls return {"kill_switch": True, "message": "..."} instead of raising
```

### Configuration

| Option | Default | Description |
|--------|---------|-------------|
| `kill_switch_active` | `False` | Global kill switch on/off |
| `kill_switch_message` | `"Service temporarily unavailable..."` | Message for blocked requests |
| `kill_switch_response_mode` | `"error"` | `"error"` (raise) or `"response"` (return static) |
| `kill_switch_rules` | `[]` | List of `KillSwitchRule` for granular blocking |
| `kill_switch_environment` | `""` | Current environment (matched against rule filters) |
| `kill_switch_agent_version` | `""` | Current agent version (matched against rule filters) |

### YAML config

```yaml
kill_switch:
  active: false
  message: "Service temporarily unavailable"
  response_mode: error
  environment: production
  agent_version: v2.1.0
  rules:
    - model: "gpt-4*"
      reason: "Cost overrun"
    - provider: anthropic
      environment: production
      message: "Anthropic blocked in prod"
```

## Blast Radius Containment

Auto-pause sessions and agents after repeated failures or budget violations. Prevents a single bad agent from burning through your entire LLM budget.

```python
stateloom.init(
    blast_radius_enabled=True,
    blast_radius_consecutive_failures=5,        # Pause after 5 consecutive failures
    blast_radius_budget_violations_per_hour=10,  # Pause after 10 budget violations/hour
    blast_radius_webhook_url="https://hooks.example.com/alerts",  # Optional webhook
)
```

### Cross-session agent tracking

Blast radius tracks failures per session **and** per agent identity. An agent's identity is derived from `session.agent_name` (typed field, or falls back to the model name). If the same agent fails across multiple sessions, all sessions for that agent are paused.

```python
with stateloom.session("task-1") as s:
    s.agent_name = "ticket-agent"
    # If ticket-agent fails 5 times across ANY session, all ticket-agent sessions are paused
```

### Session cleanup

Per-session failure counts and budget violation timestamps are automatically cleaned up when a session ends. This prevents unbounded memory growth in long-running processes.

### Monitoring and recovery

```python
# Check status
status = stateloom.blast_radius_status()
# {"paused_sessions": [...], "paused_agents": [...], "session_failure_counts": {...}, ...}

# Unpause
stateloom.unpause_session("session-id")
stateloom.unpause_agent("agent:ticket-agent")
```

### Configuration

| Option | Default | Description |
|--------|---------|-------------|
| `blast_radius_enabled` | `False` | Enable blast radius containment |
| `blast_radius_consecutive_failures` | `5` | Consecutive failures before pausing |
| `blast_radius_budget_violations_per_hour` | `10` | Budget violations per hour before pausing |
| `blast_radius_webhook_url` | `""` | Webhook URL for pause notifications |

### YAML config

```yaml
blast_radius:
  enabled: true
  consecutive_failures: 5
  budget_violations_per_hour: 10
  webhook_url: https://hooks.example.com/alerts
```

## Rate Limiting

Per-team TPS (Transactions Per Second) limits with priority-aware request queueing. Prevents one bursty team from starving others sharing the same API keys. Instead of 429 errors, requests are held in a virtual queue until a slot opens.

### Configuration via dashboard API

```bash
# Set rate limit for a team (2 TPS, priority 10, max 50 queued, 15s timeout)
curl -X PUT localhost:4781/api/teams/team-123/rate-limit \
  -H 'Content-Type: application/json' \
  -d '{"rate_limit_tps": 2, "rate_limit_priority": 10, "rate_limit_max_queue": 50, "rate_limit_queue_timeout": 15}'

# Check rate limit config and live status
curl localhost:4781/api/teams/team-123/rate-limit

# Remove rate limit (unlimited)
curl -X DELETE localhost:4781/api/teams/team-123/rate-limit

# Global rate limiter status across all teams
curl localhost:4781/api/rate-limiter
```

### Configuration via Python

Rate limits are set on `Team` objects (not `init()`), since they're per-team:

```python
team = stateloom.create_team("org-1", name="Customer Support")
team.rate_limit_tps = 5.0         # 5 requests/second
team.rate_limit_priority = 10     # Higher priority = served first when queued
team.rate_limit_max_queue = 100   # Max queued requests before rejection
team.rate_limit_queue_timeout = 30.0  # Seconds before queued request times out

# Check global status
status = stateloom.rate_limiter_status()
# {"teams": {"team-123": {"tps": 5.0, "tokens_available": 4.2, "queue_size": 0, ...}}}
```

### How it works

1. **Token bucket per team** — capacity equals TPS, refills lazily at `tps` tokens/second
2. **When a slot is available** — request proceeds immediately, no queuing
3. **When the bucket is empty** — request enters a priority queue (higher `priority` → served first, FIFO within same priority)
4. **Release mechanism** — when a request completes, the next highest-priority waiter is released
5. **Queue full** — raises `StateLoomRateLimitError` immediately
6. **Queue timeout** — raises `StateLoomRateLimitError` after `queue_timeout` seconds
7. **Hot-reload** — changing TPS/priority via dashboard takes effect on the next request

Rate limit errors are excluded from blast radius failure counting.

## Proxy (Multi-Protocol Gateway)

StateLoom includes a multi-protocol proxy server supporting OpenAI, Anthropic, and Gemini native formats. All requests flow through the full middleware pipeline (PII, budget, rate limiting, etc.). The proxy uses HTTP reverse proxy (passthrough) by default — forwarding requests directly to upstream APIs via httpx instead of instantiating SDK clients. This enables subscription users (Claude Max, Gemini Ultra) whose CLIs use OAuth/session tokens to work transparently without API keys.

```python
stateloom.init(proxy_enabled=True)

# Client side — any language, any SDK
import openai
client = openai.OpenAI(base_url="http://localhost:4781/v1", api_key="ag-...")
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Hello!"}],
)
```

### Virtual Keys

Issue virtual keys to teams and orgs. Keys are hashed and stored — the plaintext is only shown once at creation time.

```python
key_info = stateloom.create_virtual_key(
    team_id="team-1",
    name="dev-key",
    allowed_models=["gpt-4o-mini", "claude-*"],  # Glob patterns
    budget_limit=100.0,
    rate_limit_tps=5.0,
    agent_ids=["agt-abc123"],  # Restrict to specific agents
)
# Use key_info["key"] as the Bearer token

# List keys (previews only, never full keys)
keys = stateloom.list_virtual_keys(team_id="team-1")

# Revoke a key
stateloom.revoke_virtual_key(key_info["id"])
```

### Bring Your Own Key (BYOK)

Users can pass their own provider API keys via headers. StateLoom still tracks the session and enforces all middleware (PII, budget, rate limiting), but uses the caller's key for the actual LLM call.

```bash
curl http://localhost:4781/v1/chat/completions \
  -H "Authorization: Bearer ag-..." \
  -H "X-StateLoom-OpenAI-Key: sk-my-personal-key" \
  -H "Content-Type: application/json" \
  -d '{"model": "gpt-4o", "messages": [{"role": "user", "content": "Hello"}]}'
```

Supported headers:

| Header | Provider |
|--------|----------|
| `X-StateLoom-OpenAI-Key` | OpenAI |
| `X-StateLoom-Anthropic-Key` | Anthropic |
| `X-StateLoom-Google-Key` | Google Gemini |

**Key resolution priority** (highest wins):

1. BYOK header (`X-StateLoom-{Provider}-Key`)
2. Org-level secret (stored via `save_secret()`)
3. Global config (`provider_api_key_openai`, etc.)
4. Environment variable (SDK default behavior)

### Native Protocol Endpoints

In addition to the OpenAI-compatible endpoint, StateLoom provides native endpoints for other providers:

| Provider | Endpoint | Base URL env var |
|----------|----------|------------------|
| OpenAI | `POST /v1/chat/completions` | `OPENAI_BASE_URL` |
| OpenAI Responses | `POST /v1/responses` (+ WebSocket) | `OPENAI_BASE_URL` |
| Anthropic | `POST /v1/messages` | `ANTHROPIC_BASE_URL` |
| Gemini | `POST /v1beta/models/{model}:generateContent` | `GOOGLE_GEMINI_BASE_URL` |
| Gemini (stream) | `POST /v1beta/models/{model}:streamGenerateContent` | `GOOGLE_GEMINI_BASE_URL` |

```bash
# Codex CLI through StateLoom
export OPENAI_BASE_URL=http://localhost:4782/v1
codex "explain this code"

# Claude CLI through StateLoom
export ANTHROPIC_BASE_URL=http://localhost:4782
claude "explain this code"

# Gemini CLI through StateLoom
export GOOGLE_GEMINI_BASE_URL=http://localhost:4782
gemini "explain this code"
```

## Managed Agents (Prompts-as-an-API)

Deploy AI agents without writing code. Select a model, write a system prompt, attach a budget, and get a unique URL. Each agent gets an OpenAI-compatible endpoint that routes through the full middleware pipeline (PII, budget, rate limiting, etc.).

### Create an agent

```python
agent = stateloom.create_agent(
    slug="legal-bot",
    team_id="team-1",
    model="gpt-4o",
    system_prompt="You are a legal assistant. Be concise and cite sources.",
    request_overrides={"temperature": 0.2},
    budget_per_session=5.0,
)
```

Or via the dashboard API:

```bash
curl -X POST localhost:4781/api/agents \
  -H 'Content-Type: application/json' \
  -d '{
    "slug": "legal-bot",
    "team_id": "team-1",
    "model": "gpt-4o",
    "system_prompt": "You are a legal assistant. Be concise and cite sources.",
    "request_overrides": {"temperature": 0.2},
    "budget_per_session": 5.0
  }'
```

### Call the agent

```bash
curl http://localhost:4781/v1/agents/legal-bot/chat/completions \
  -H "Authorization: Bearer ag-..." \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "What is force majeure?"}]}'
```

The `model` field in the request body is optional and silently ignored — the agent's model always takes precedence (governance). The agent's system prompt is prepended before any client-supplied system prompt.

### Versioning and rollback

Versions are immutable. "Update" = create a new version. "Rollback" = activate an old version.

```python
# Create a new version (v2)
v2 = stateloom.create_agent_version(
    agent.id,
    model="gpt-4o-mini",
    system_prompt="You are a legal assistant. Be brief.",
)

# Roll back to v1
stateloom.activate_agent_version(agent.id, agent.active_version_id)
```

### Virtual key scoping

Virtual keys can be restricted to specific agents:

```bash
curl -X POST localhost:4781/api/virtual-keys \
  -H 'Content-Type: application/json' \
  -d '{"team_id": "team-1", "name": "legal-only", "agent_ids": ["agt-abc123"]}'
```

A key with `agent_ids` set can only access the listed agents. An empty list allows access to all agents.

### Agent lifecycle

| Status | Proxy behavior | Description |
|--------|---------------|-------------|
| `active` | 200 | Normal operation |
| `paused` | 403 | Temporarily disabled |
| `archived` | 410 | Soft-deleted, session history preserved |

```python
# Pause via dashboard API
curl -X PATCH localhost:4781/api/agents/legal-bot?team_id=team-1 \
  -H 'Content-Type: application/json' \
  -d '{"status": "paused"}'

# Archive (soft delete)
curl -X DELETE localhost:4781/api/agents/legal-bot?team_id=team-1
```

## File-Based Prompt Versioning

For individual users who want a simpler workflow than the dashboard API: drop `.md` files in a `prompts/` folder, and StateLoom auto-detects changes as new agent versions.

```python
stateloom.init(prompts_dir="prompts/")
```

### Prompt file format

**Markdown with YAML frontmatter** (`.md`) — the primary format:

```markdown
---
model: gpt-4o
temperature: 0.3
max_tokens: 4096
budget_per_session: 2.0
description: Customer support agent
---
You are a helpful customer support agent for Acme Corp.
Always be polite and professional.
```

**YAML** (`.yaml`/`.yml`) — for structured configs:

```yaml
model: gpt-4o
system_prompt: You are a helpful assistant.
request_overrides:
  temperature: 0.5
  max_tokens: 2000
budget_per_session: 1.5
```

**Plain text** (`.txt`) — entire content is the system prompt, model from `default_model`.

The filename stem becomes the agent slug: `support-bot.md` → slug `support-bot`.

### How it works

1. **New file** → creates agent + version v1, auto-activated
2. **Edit file** → creates new version, auto-activated (previous versions preserved for rollback)
3. **Delete file** → agent archived (version history preserved)
4. **Rename file** → old agent archived, new agent created
5. **No change** → no-op (SHA-256 content hash comparison)

File-based agents never overwrite API-created agents with the same slug. All file-sourced agents belong to an auto-created `_local` org and team.

### Monitoring

```python
# Check watcher status
status = stateloom.prompt_watcher_status()
# {"enabled": True, "prompts_dir": "/path/to/prompts", "tracked_files": 3, ...}

# Force immediate rescan
stateloom.rescan_prompts()
```

Dashboard API: `GET /api/prompts/status`, `POST /api/prompts/rescan`.

### Configuration

| Option | Default | Description |
|--------|---------|-------------|
| `prompts_dir` | `""` | Directory to watch (empty = disabled). Relative paths resolve from cwd |
| `prompts_poll_interval` | `2.0` | Watchdog polling interval in seconds |

Requires `watchdog` for live file monitoring: `pip install stateloom[prompts]`. Without it, the initial scan still runs but file changes aren't detected automatically.

### YAML config

```yaml
prompts:
  dir: prompts/
  poll_interval: 2.0
```

## A/B Experiments

Create experiments, assign sessions to variants, collect feedback, and compare metrics. Experiments can test model variants, prompt variants, or agent version variants.

```python
# Create an experiment
exp = stateloom.create_experiment(
    name="model-comparison",
    variants=[
        {"name": "control", "weight": 1.0, "model": "gpt-4o"},
        {"name": "challenger", "weight": 1.0, "model": "claude-sonnet-4-20250514"},
    ],
    strategy="random",  # or "hash" for deterministic, "manual" for explicit
)

stateloom.start_experiment(exp.id)

# Sessions are auto-assigned to variants
with stateloom.session("user-request-1", experiment=exp.id) as s:
    response = client.chat.completions.create(...)

# Record outcome
stateloom.feedback("user-request-1", rating="success", score=0.95)

# View metrics
metrics = stateloom.experiment_metrics(exp.id)
board = stateloom.leaderboard()
stateloom.conclude_experiment(exp.id)
```

### Agent Version Experiments

Variants can reference managed agent versions, inheriting their model, system prompt, and request overrides. The agent's config is applied as the base layer, with the variant's explicit overrides taking priority on top.

```python
# Compare two agent versions side by side
exp = stateloom.create_experiment(
    name="helpdesk-prompt-test",
    variants=[
        {"name": "current", "agent_version_id": "agv-abc123"},
        {"name": "new-prompt", "agent_version_id": "agv-def456"},
    ],
    agent_id="agt-helpdesk",  # informational — marks which agent this tests
)

# Mix agent versions with model overrides
exp = stateloom.create_experiment(
    name="agent-model-test",
    variants=[
        {"name": "agent-gpt4", "agent_version_id": "agv-abc123", "model": "gpt-4o"},
        {"name": "agent-claude", "agent_version_id": "agv-abc123", "model": "claude-sonnet-4-20250514"},
    ],
)
```

Agent overrides are snapshotted at assignment time — even if the agent version is updated later, existing experiment sessions use the original snapshot.

### Editing Experiments

DRAFT experiments can be updated before starting:

```python
stateloom.update_experiment(
    exp.id,
    name="renamed-experiment",
    variants=[{"name": "a", "model": "gpt-4o"}, {"name": "b", "model": "gpt-4o-mini"}],
    strategy="hash",
)
```

### Backtesting

Replay recorded sessions with different variant configs:

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
```

## Multi-Agent Consensus

Run multiple LLMs on the same question and aggregate their answers. Research shows this dramatically reduces hallucinations and improves reasoning. Three strategies are available:

- **Vote** — all models answer independently in parallel (one round, cheapest)
- **Debate** — multi-round discussion where each model sees others' responses, with a judge model synthesizing the final answer
- **Self-consistency** — multiple samples from a single model at higher temperature, aggregated by majority vote

```python
# Debate strategy — multi-round with judge synthesis
result = await stateloom.consensus(
    prompt="What are the key risks of deploying LLMs in healthcare?",
    models=["gpt-4o", "claude-sonnet-4-20250514", "gemini-2.0-flash"],
    strategy="debate",
    rounds=2,
    budget=1.00,
)
print(result.answer)       # Final synthesized answer
print(result.confidence)   # 0.0-1.0
print(result.cost)         # Total cost across all models and rounds
print(result.session_id)   # Parent session ID for dashboard drill-down

# Vote strategy — cheapest, one round
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

# Sync wrapper
result = stateloom.consensus_sync(prompt="...", models=[...])

# Agent-guided consensus — use a managed agent's system prompt for all debaters
result = await stateloom.consensus(
    agent="medical-advisor",           # slug or ID
    prompt="Should we use AI for radiology screening?",
    models=["gpt-4o", "claude-sonnet-4-20250514"],
    strategy="debate",
)
# The agent's system prompt replaces the default debater prompt.
# If models is omitted, defaults to the agent's configured model.
```

### Core vs Enterprise

Consensus is split into a free core tier and an enterprise tier:

| Feature | Core (OSS) | Enterprise |
|---------|-----------|------------|
| Strategies | Vote, Debate, Self-Consistency | All |
| Model limit | Up to 3 | Unlimited (10+) |
| Aggregation | Majority vote, confidence-weighted | + Judge synthesis |
| Optimization | Static early stop | + Greedy model downgrade |
| Persistence | Session events | + Durable time-travel replay |

Without an enterprise license, requesting enterprise-only features raises
`StateLoomFeatureError` with actionable guidance. Set `STATELOOM_LICENSE_KEY`
or use `STATELOOM_ENV=development` for local development.

### Greedy mode (Enterprise)

When `greedy=True`, if models reach high agreement after Round 1, they are automatically swapped to cheaper tier equivalents (e.g., gpt-4o → gpt-4o-mini, claude-sonnet → claude-haiku) for remaining rounds. This uses the existing tier mapping from the circuit breaker. Requires enterprise license.

```python
result = await stateloom.consensus(
    prompt="What is 2+2?",
    models=["gpt-4o", "claude-sonnet-4-20250514"],
    strategy="debate",
    greedy=True,                        # Enable greedy downgrade
    greedy_agreement_threshold=0.7,     # Agreement threshold (default)
)
```

### How it works

1. `ConsensusOrchestrator` creates a parent session (durable with enterprise, non-durable in core)
2. Dispatches to the chosen strategy (vote/debate/self_consistency)
3. Each debater runs in a child session auto-linked to the parent via nesting
4. Every debater call flows through the full middleware pipeline (PII, guardrails, budget, cost tracking, circuit breaker)
5. After each round: `DebateRoundEvent` recorded on the parent session
6. Debate final aggregation: judge synthesis (enterprise) or confidence-weighted (core)
7. On completion: `ConsensusEvent` recorded with final answer, confidence, cost
8. On `StateLoomBudgetError` mid-debate: returns best answer from completed rounds

### Adapter-level confidence

Each provider adapter has optional `confidence_instruction()` and `extract_confidence()` methods. The consensus layer uses these for provider-specific confidence parsing, falling back to regex `[Confidence: X.XX]` extraction.

### Dashboard

The Consensus section in the dashboard shows:
- Stats: total runs, average confidence, total cost, early-stopped count
- Runs table: session, strategy, models, rounds, confidence, cost, duration
- Detail panel: answer preview, winner model, round-by-round breakdown with per-model responses, child debater sessions

Dashboard API: `GET /consensus-runs` (with optional `?strategy=` filter), `GET /consensus-runs/{session_id}`.

### Configuration

```python
stateloom.init(
    consensus_default_models=["gpt-4o", "claude-sonnet-4-20250514"],
    consensus_default_strategy="debate",
    consensus_default_rounds=2,
    consensus_default_budget=1.0,
    consensus_max_rounds=10,
    consensus_max_models=10,
    consensus_early_stop_enabled=True,
    consensus_early_stop_threshold=0.9,
    consensus_greedy=False,
    consensus_greedy_agreement_threshold=0.7,
)
```

## Time-Travel Debugging

Replay a failed session, mocking the first N steps with cached responses, then run live from the failure point.

```python
stateloom.replay(
    session="failed-session-id",
    mock_until_step=13,
    strict=True,            # Block outbound HTTP during mocked steps
    allow_hosts=["api.example.com"],  # Exceptions to the block
)
```

In strict mode, a network blocker prevents unintended side effects. Tools marked with `mutates_state=True` are flagged before replay.

## Durable Resumption (Checkpointing)

Temporal-like crash recovery for LLM workflows. Pass `durable=True` to a session, and if the process crashes mid-workflow, restarting with the same session ID automatically replays cached responses for completed steps and resumes live execution from where it left off. Both streaming and non-streaming calls are supported.

```python
with stateloom.session(session_id="task-123", durable=True) as s:
    res1 = client.chat.completions.create(...)  # Step 1
    res2 = client.chat.completions.create(...)  # Step 2
    res3 = client.chat.completions.create(...)  # Step 3
    # Process crashes here...
    res4 = client.chat.completions.create(...)  # Step 4
```

On restart with the same code:

```python
with stateloom.session(session_id="task-123", durable=True) as s:
    res1 = client.chat.completions.create(...)  # Cache hit (free)
    res2 = client.chat.completions.create(...)  # Cache hit (free)
    res3 = client.chat.completions.create(...)  # Cache hit (free)
    res4 = client.chat.completions.create(...)  # Live call (resumes here)
```

### Durable streaming

Streaming calls work natively in durable sessions. On the first run, chunks are yielded to the caller in real time while being accumulated behind the scenes. On resume, cached chunks replay as an iterable — same developer experience, zero API cost.

```python
with stateloom.session(session_id="stream-task-1", durable=True) as s:
    stream = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": "Explain quantum computing"}],
        stream=True,
    )
    for chunk in stream:  # First run: live tokens; resume: cached replay
        if chunk.choices[0].delta.content:
            print(chunk.choices[0].delta.content, end="")
```

For interactive replay that simulates the original streaming feel, configure an inter-chunk delay:

```python
stateloom.init(durable_stream_delay_ms=30)  # 30ms between cached chunks
```

### How it works

1. **First run**: Each LLM response is serialized and stored alongside the event in SQLite. For streaming calls, chunks are accumulated as they arrive and serialized after the stream completes.
2. **Crash recovery**: On resume, cached responses are loaded and returned instantly for completed steps. Cached streaming responses are returned as iterables that support both `for chunk in ...` (sync) and `async for chunk in ...` (async).
3. **Idempotent re-run**: A fully completed session returns all steps from cache (zero cost)

### Limitations

- **Tool calls re-execute**: Only LLM calls are cached; `@stateloom.tool()` functions re-execute on every run
- **Code must be deterministic**: Like Temporal, the code path must be the same on every run for step numbers to match
- **Session ID required**: `durable=True` without a `session_id` is a no-op

### Async support

```python
async with stateloom.async_session(session_id="task-123", durable=True) as s:
    res1 = await client.chat.completions.create(...)  # Cache hit on resume
    res2 = await client.chat.completions.create(...)  # Live call
```

## Semantic Retries (Self-Healing)

LLMs often produce semantically invalid output — bad JSON, hallucinated tool calls, missing required fields. StateLoom provides automatic retries that compose naturally with durable sessions.

### `durable_task` decorator

Combines a durable session with automatic retry logic. If the function raises, it retries up to N times. On crash recovery, cached LLM responses replay automatically — zero wasted API calls.

```python
@stateloom.durable_task(retries=3, session_id="report-42")
def generate_report(prompt: str) -> dict:
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
    )
    return json.loads(response.choices[0].message.content)  # Retries on ValueError
```

With explicit validation:

```python
@stateloom.durable_task(retries=3, validate=lambda r: "summary" in r)
def summarize(text: str) -> dict:
    response = client.chat.completions.create(...)
    return json.loads(response.choices[0].message.content)
```

Async support:

```python
@stateloom.durable_task(retries=3, session_id="async-task-1")
async def async_generate(prompt: str) -> dict:
    response = await client.chat.completions.create(...)
    return json.loads(response.choices[0].message.content)
```

### `retry_loop` — inline retry inside existing sessions

For finer control, use `retry_loop()` inside an existing session:

```python
with stateloom.session("task-123", durable=True) as s:
    for attempt in stateloom.retry_loop(retries=3):
        with attempt:
            response = client.chat.completions.create(...)
            data = json.loads(response.choices[0].message.content)
    # data is available here after success
```

### How it works

- Each LLM call gets a unique step number (step counter advances on every call, including retries)
- Failed attempts' LLM responses are cached (valid API responses, just semantically wrong)
- On durable resume: cached bad responses trigger the same validation failures → retry logic kicks in → cached good responses succeed on the correct attempt
- Infrastructure errors (`StateLoomBudgetError`, `StateLoomPIIBlockedError`, `StateLoomKillSwitchError`, etc.) are **never retried** — they propagate immediately
- When all attempts are exhausted, `StateLoomRetryError` is raised with the original exception chained

### Configuration

| Option | Default | Description |
|--------|---------|-------------|
| `retries` | `3` | Maximum retry attempts |
| `validate` | `None` | Optional callback — returns `False` to trigger retry |
| `session_id` | auto | Fixed session ID (enables durable resume across restarts) |
| `name` | function name | Session name |
| `budget` | `None` | Per-session budget in USD |
| `on_retry` | `None` | Callback `(attempt, error)` fired on each retry |

## Named Checkpoints

Mark milestones within a session for observability and debugging. Checkpoint events appear in the dashboard waterfall timeline as labeled dividers.

```python
with stateloom.session("task-123") as s:
    response1 = client.chat.completions.create(...)
    stateloom.checkpoint("data-loaded", description="Raw data fetched from API")

    response2 = client.chat.completions.create(...)
    stateloom.checkpoint("analysis-complete")
```

## Parent-Child Sessions

Sessions support hierarchical relationships. Just nest `session()` calls — the parent is **auto-derived** from the enclosing session context. Children also inherit `org_id` and `team_id` from the parent when not explicitly set.

```python
with stateloom.session("parent-task", org_id="org-1", team_id="team-1") as parent:
    response = client.chat.completions.create(...)

    # Parent auto-derived from enclosing session — no need to pass parent=
    with stateloom.session("subtask-1") as child:
        # child.parent_session_id == "parent-task"
        # child.org_id == "org-1" (inherited)
        sub_response = client.chat.completions.create(...)

        # Nesting works to any depth
        with stateloom.session("sub-subtask") as grandchild:
            # grandchild.parent_session_id == "subtask-1"
            deep_response = client.chat.completions.create(...)
```

> **Explicit parent ID.** If you need to link a child session outside the parent's `with` block (e.g., in a different thread or async task), pass `parent=` explicitly:
>
> ```python
> with stateloom.session("child-task", parent="parent-task") as child:
>     response = client.chat.completions.create(...)
> ```

List child sessions via the API:

```bash
curl localhost:4781/api/sessions/parent-task/children
```

## Session Timeouts & Heartbeats

Set per-session max duration and idle timeout. The session heartbeat is updated automatically after each successful LLM call.

```python
# Max 60 seconds total, max 30 seconds idle between calls
with stateloom.session("task-123", timeout=60.0, idle_timeout=30.0) as s:
    response1 = client.chat.completions.create(...)  # Heartbeat updated
    time.sleep(35)  # Exceeds idle_timeout
    response2 = client.chat.completions.create(...)  # Raises StateLoomTimeoutError
```

On timeout, `StateLoomTimeoutError` is raised and the session status is set to `TIMED_OUT`.

## Session Cancellation

Cancel a running session from another thread or process. The next LLM call in the cancelled session will raise `StateLoomCancellationError`.

```python
# In the main thread
with stateloom.session("long-task") as s:
    response = client.chat.completions.create(...)  # Works fine
    # ... another thread calls stateloom.cancel_session("long-task")
    response2 = client.chat.completions.create(...)  # Raises StateLoomCancellationError

# Cancel from anywhere
stateloom.cancel_session("long-task")

# Or via the dashboard API
# curl -X POST localhost:4781/api/sessions/long-task/cancel
```

## Session Suspension (Human-in-the-Loop)

Pause an agent session to wait for human approval, then resume with a signal payload.

```python
# Inside a session — block until human approves
with stateloom.session("approval-task") as s:
    response = client.chat.completions.create(...)

    # Pause and wait for human input (shows in dashboard)
    payload = stateloom.suspend(reason="Needs manager approval", data={"draft": response})
    # Execution blocks here until signaled

    if payload.get("approved"):
        # Continue with approved response
        ...

# From another thread/process — signal the suspended session
stateloom.signal_session("approval-task", payload={"approved": True})

# Or suspend/signal externally
stateloom.suspend_session("session-id", reason="Review needed")
stateloom.signal_session("session-id", payload={"decision": "approved"})
```

Async version:

```python
async with stateloom.async_session("async-approval") as s:
    payload = await stateloom.async_suspend(reason="Needs review", timeout=300.0)
```

Dashboard API: `POST /api/sessions/{id}/suspend`, `POST /api/sessions/{id}/signal`.

## VCR-Cassette Mock (Zero-Cost Testing)

Record LLM calls once, replay forever. Perfect for deterministic tests and CI pipelines.

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

Network blocking is enabled by default during replay to prevent accidental live calls.

## Unified Chat API

Provider-agnostic chat interface — call any model without importing provider SDKs.

```python
import stateloom

stateloom.init(default_model="gpt-4o")

# Sync
response = stateloom.chat("What is the capital of France?")
print(response.content)    # "The capital of France is Paris."
print(response.model)      # "gpt-4o"
print(response.provider)   # "openai"

# Async
response = await stateloom.achat("What is 2+2?", model="claude-sonnet-4-20250514")

# With full message history
response = stateloom.chat(
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello!"},
    ],
    model="gpt-4o",
    temperature=0.3,
)

# With BYOK
client = stateloom.Client(provider_keys={"openai": "sk-my-key"})
response = client.chat("Hello!", model="gpt-4o")
```

## Multi-Model Workflows

StateLoom tracks per-model cost and token breakdowns when a session uses multiple models. Use cheap models for simple tasks and expensive models for complex reasoning — all within a single session with full visibility.

```python
import stateloom

stateloom.init(default_model="gpt-4o-mini")

with stateloom.session(name="content-pipeline") as s:
    # Step 1: cheap model for outline
    outline = stateloom.chat(messages=[{"role": "user", "content": "Outline a blog post about AI agents"}])

    # Step 2: powerful model for writing
    article = stateloom.chat(
        messages=[{"role": "user", "content": f"Write a blog post from this outline: {outline.content}"}],
        model="gpt-4o",
    )

    # Per-model breakdown
    print(s.cost_by_model)
    # {"gpt-4o-mini": 0.000123, "gpt-4o": 0.045600}

    print(s.tokens_by_model)
    # {"gpt-4o-mini": {"prompt_tokens": 20, "completion_tokens": 150, "total_tokens": 170},
    #  "gpt-4o": {"prompt_tokens": 180, "completion_tokens": 1200, "total_tokens": 1380}}
```

The breakdown is available in the dashboard REST API:

```bash
curl localhost:4781/api/sessions/{id} | jq '.cost_by_model'
```

Durable replay works across models — cached responses are stored per-step regardless of which model produced them.

## Session Export & Import

Export sessions as portable JSON bundles for sharing, debugging, or migration between environments.

```python
# Export a session
bundle = stateloom.export_session("session-123", path="session.json")

# Export with children and PII scrubbing
bundle = stateloom.export_session(
    "session-123",
    path="session.json.gz",
    include_children=True,
    scrub_pii=True,
)

# Import into another environment
session = stateloom.import_session("session.json")

# Import with a new session ID to avoid collisions
session = stateloom.import_session("session.json", session_id_override="imported-123")
```

Dashboard API: `GET /api/sessions/{id}/export`, `POST /api/sessions/import`.

## Config Locking (Admin Controls)

Lock configuration settings to prevent developer overrides. Useful for platform teams enforcing governance.

```python
# Lock a setting at a specific value
stateloom.lock_setting("blast_radius_enabled", value=True, reason="Security policy")
stateloom.lock_setting("pii_enabled", value=True, reason="Compliance requirement")

# Developers who try to override locked settings get StateLoomConfigLockedError
stateloom.init(blast_radius_enabled=False)  # Raises!

# Unlock
stateloom.unlock_setting("blast_radius_enabled")

# List all locks
locks = stateloom.list_locked_settings()
```

Dashboard API: `POST /api/admin/locks`, `GET /api/admin/locks`, `DELETE /api/admin/locks/{setting}`.

## Circuit Breaker (Provider Failover)

Automatic provider failover when a provider is experiencing outages. Uses the circuit breaker pattern with three states: closed (normal), open (failing — routes to tier-matched fallback), and half-open (synthetic probe testing recovery). Failure timestamps are bounded (max 10,000 per circuit) to prevent unbounded memory growth in long-running processes.

```python
stateloom.init(circuit_breaker=True)

# If OpenAI is down, requests automatically fail over to a same-tier model
# on another provider (e.g., gpt-4o → claude-sonnet-4-20250514)
response = client.chat.completions.create(model="gpt-4o", ...)

# Check circuit breaker status
status = stateloom.circuit_breaker_status()
# {"openai": {"state": "open", "failures": 5, ...}, ...}

# Manually reset a provider's circuit
stateloom.reset_circuit_breaker("openai")
```

## Compliance Enforcement (GDPR/HIPAA/CCPA)

Declarative compliance profiles with tamper-proof audit trails, zero-retention modes, and data region controls.

```python
stateloom.init(compliance="gdpr")  # or "hipaa", "ccpa"

# All LLM calls are audited with SHA-256 tamper-proof hashes
# PII scanning is auto-enabled per regulation requirements
# Data retention policies are enforced automatically

# Right to Be Forgotten — purge all data for a user
result = stateloom.purge_user_data("user@example.com", standard="gdpr")
# {"sessions_deleted": 5, "events_deleted": 42, "cache_entries_deleted": 3, ...}

# Run periodic compliance cleanup (enforces session TTL)
deleted_count = stateloom.compliance_cleanup()
```

Compliance profiles can be set at org or team level via the dashboard API.

## Billing Mode (API vs Subscription)

Automatically detect whether users are billed per-token (API) or on a flat subscription (Claude Max, ChatGPT Plus). Subscription users see `cost=$0.00` with a separate `estimated_api_cost` showing what they would have paid.

```python
# Virtual keys can be configured with billing_mode
vk = VirtualKey(billing_mode="subscription", ...)

# BYOK detection is automatic:
# sk-ant-* → API billing (Anthropic)
# sk-* → API billing (OpenAI)
# AIzaSy* → API billing (Google)
# Other tokens → subscription billing
```

Budget enforcement is skipped for subscription sessions. Both actual cost and estimated API cost are tracked on every event and session.

## Anthropic-Native Proxy

Drop-in replacement for the Anthropic API. Claude CLI and Anthropic SDK clients can use StateLoom directly:

```bash
export ANTHROPIC_BASE_URL=http://localhost:4782
claude "explain this code"  # Routes through StateLoom middleware
```

Supports `/v1/messages` with real-time SSE streaming, virtual key auth, BYOK, and transparent auth passthrough for subscription users (Claude Max). The proxy forwards requests directly to `api.anthropic.com` via httpx — no API key required from StateLoom's side.

## Gemini-Native Proxy

Drop-in replacement for the Google AI API. Gemini CLI and Google AI SDK clients can use StateLoom directly:

```bash
export GOOGLE_GEMINI_BASE_URL=http://localhost:4782
gemini "explain this code"  # Routes through StateLoom middleware
```

Supports `/v1beta/models/{model}:generateContent` and `/v1beta/models/{model}:streamGenerateContent` with transparent auth passthrough for subscription users (Gemini Ultra).

## Authentication & RBAC

StateLoom provides a unified authentication and role-based access control system that works for both solo developers and enterprise teams. **Disabled by default** — existing deployments are unaffected.

```bash
pip install stateloom[auth]  # Adds pyjwt + argon2-cffi
```

### Quick setup

```python
stateloom.init(auth_enabled=True)

# First user bootstraps as org_admin
# POST /api/v1/auth/bootstrap
# { "email": "admin@example.com", "password": "...", "display_name": "Admin" }
```

Or bootstrap headlessly for K8s/IaC:

```bash
export STATELOOM_ADMIN_EMAIL=admin@example.com
export STATELOOM_ADMIN_PASSWORD_HASH='$argon2id$...'
```

### Two auth planes

| Plane | Auth method | RBAC | Latency impact |
|-------|------------|------|----------------|
| **Dashboard** (management) | JWT access token | Full role/permission checks | Acceptable (not latency-critical) |
| **Proxy** (data) | Virtual key hash lookup | Scope + model + budget checks only | Zero (no JWT validation) |

Dashboard API uses JWT tokens (15-min access + 7-day refresh). Proxy endpoints continue using virtual key hash lookup — no JWT overhead on the hot path.

### Roles

Five hierarchical roles across org and team scopes:

| Role | Scope | Description |
|------|-------|-------------|
| `org_admin` | Organization | Full access — users, teams, config, OIDC, kill switch, compliance |
| `org_auditor` | Organization | Read-only access across the entire org |
| `team_admin` | Team | Full CRUD within the team — agents, experiments, virtual keys |
| `team_editor` | Team | Read + write agents and experiments |
| `team_viewer` | Team | Read-only within the team |

Solo developers are `org_admin` of a "Default Organization" — same system, no special case.

### Local auth

```bash
# Login
curl -X POST localhost:4781/api/v1/auth/login \
  -H 'Content-Type: application/json' \
  -d '{"email": "admin@example.com", "password": "..."}'
# Returns: { "access_token": "...", "refresh_token": "...", "user": {...} }

# Use the token
curl localhost:4781/api/sessions \
  -H 'Authorization: Bearer <access_token>'

# CLI login
stateloom login --host 127.0.0.1 --port 4781
```

### OIDC federation (SSO)

Connect to Google Workspace, Okta, Azure AD, or any OIDC provider:

```bash
# Register a provider (org_admin only)
curl -X POST localhost:4781/api/v1/oidc/providers \
  -H 'Authorization: Bearer <token>' \
  -H 'Content-Type: application/json' \
  -d '{
    "name": "Google Workspace",
    "issuer_url": "https://accounts.google.com",
    "client_id": "...",
    "client_secret": "...",
    "group_claim": "groups",
    "group_role_mapping": {
      "engineering": {"team_id": "team-eng", "role": "team_editor"},
      "platform": {"team_id": "team-platform", "role": "team_admin"}
    }
  }'
```

TOFU (Trust On First Use) prevents email-based account takeover: if an OIDC login matches an existing local email, the user must prove ownership with their local password before the accounts merge.

### VK scope enforcement

Virtual keys can be restricted to specific proxy endpoints:

```python
key_info = stateloom.create_virtual_key(
    team_id="team-1",
    name="chat-only",
    scopes=["chat", "messages"],  # Only /v1/chat/completions and /v1/messages
)
```

| Scope | Endpoint |
|-------|----------|
| `chat` | `POST /v1/chat/completions` |
| `messages` | `POST /v1/messages` |
| `responses` | `POST /v1/responses` |
| `generate` | `POST /v1beta/models/{model}:generateContent` |
| `agents` | `POST /v1/agents/{ref}/chat/completions` |

Empty scopes = all allowed (backward compatible). Non-matching scope returns 403.

### End-user attribution

Track which downstream user made each request:

```bash
curl http://localhost:4781/v1/chat/completions \
  -H "Authorization: Bearer ag-..." \
  -H "X-StateLoom-End-User: user@example.com" \
  -H "Content-Type: application/json" \
  -d '{"model": "gpt-4o", "messages": [{"role": "user", "content": "Hello"}]}'
```

The header is sanitized (non-printable chars stripped, 256 char max), stored on the session's `end_user` field, and stripped before upstream forwarding. Filter sessions by end-user in the dashboard:

```bash
curl "localhost:4781/api/sessions?end_user=user@example.com"
```

### Backward compatibility

| Scenario | `auth_enabled=false` (default) | `auth_enabled=true` |
|----------|-------------------------------|---------------------|
| Dashboard API key | Works (existing behavior) | Works (key = system admin) |
| JWT token | Ignored | Required for dashboard API |
| Proxy VK (empty scopes) | All allowed | All allowed |
| Proxy VK (scopes set) | Enforced | Enforced |
| X-StateLoom-End-User | Captured | Captured |

### Configuration

| Option | Default | Description |
|--------|---------|-------------|
| `auth_enabled` | `False` | Enable JWT auth for dashboard |
| `auth_jwt_algorithm` | `"HS256"` | JWT signing algorithm |
| `auth_jwt_access_ttl` | `900` | Access token TTL in seconds (15 min) |
| `auth_jwt_refresh_ttl` | `604800` | Refresh token TTL in seconds (7 days) |

## Multi-Tenant Hierarchy

Organizations and Teams provide multi-tenant isolation with per-level budgets, PII rules, compliance profiles, and rate limits.

```python
# Create organization and team
org = stateloom.create_organization(
    name="Acme Corp",
    budget=1000.0,
    compliance_profile="gdpr",
)
team = stateloom.create_team(org.id, name="Customer Support", budget=100.0)

# Sessions inherit org_id and team_id from virtual keys or explicit params
with stateloom.session("task-1", org_id=org.id, team_id=team.id) as s:
    response = client.chat.completions.create(...)

# Query stats
org_stats = stateloom.org_stats(org.id)
team_stats = stateloom.team_stats(team.id)

# List
orgs = stateloom.list_organizations()
teams = stateloom.list_teams(org_id=org.id)
```

## Async Jobs

Fire-and-forget LLM calls with webhook notifications and retry logic. Useful for batch processing and background workflows.

```python
stateloom.init(async_jobs_enabled=True)

# Submit a job
job = stateloom.submit_job(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Summarize this document..."}],
    webhook_url="https://hooks.example.com/results",
    webhook_secret="my-secret",
    max_retries=3,
    ttl_seconds=3600,
)
print(job.id)  # job-abc123

# Agent-powered job — resolves the agent's model and system prompt automatically
job = stateloom.submit_job(
    agent="summarizer",
    messages=[{"role": "user", "content": "Summarize this document..."}],
)

# Check job status
job = stateloom.get_job(job.id)
print(job.status)  # "pending", "processing", "completed", "failed"

# List and filter jobs
jobs = stateloom.list_jobs(status="completed")
stats = stateloom.job_stats()

# Cancel a pending job
stateloom.cancel_job(job.id)
```

Results are delivered via HMAC-SHA256 signed webhooks. Supports pluggable queue backends: in-process (default), Redis Streams.

## LangChain Integration

**Recommended: auto_patch + callback handler** — LangChain's underlying SDK calls
flow through the full middleware pipeline (PII scanning, guardrails, budget
enforcement, etc.). The callback handler adds LangChain-specific observability
(tool names, chain tracking) without duplicating LLM events.

```python
import stateloom
from stateloom.ext.langchain import StateLoomCallbackHandler

stateloom.init()  # auto_patch=True by default
handler = StateLoomCallbackHandler()  # auto-detects tools_only mode

chain.invoke(input, config={"callbacks": [handler]})
```

Or use the convenience function:

```python
handler = stateloom.langchain_callback()
```

**Standalone mode** (observability only, no middleware enforcement):

```python
stateloom.init(auto_patch=False)
handler = StateLoomCallbackHandler(tools_only=False)
chain.invoke(input, config={"callbacks": [handler]})
```

Note: In standalone mode, LLM calls are recorded directly to the store, bypassing
the middleware pipeline. PII scanning, guardrails, and budget enforcement will not
run on LangChain LLM calls.

**Framework context bridge:** In recommended mode, the callback handler sets a
ContextVar before each SDK call with LangChain-specific metadata (run ID, chain
name, tags). The middleware pipeline reads this ContextVar and flows it into
`event.metadata["langchain"]`, so LLM events recorded by the pipeline include
full LangChain context for dashboard visibility:

```python
# event.metadata after pipeline processing:
{
    "langchain": {
        "run_id": "abc-123",
        "chain_name": "RetrievalQA",
        "tags": ["prod", "v2"],
        "model": "gpt-4o",
        "provider": "openai",
    }
}
```

This bridge pattern is generic — any framework integration can set
`stateloom.core.context.set_framework_context({"my_framework": {...}})` to
annotate pipeline events with framework-specific metadata.

## Zero-Trust Security Engine

CPython audit hooks (PEP 578) for interpreter-level operation interception, plus an in-memory secret vault with `os.environ` scrubbing. Catches supply-chain attacks in third-party libraries at the C level — not just during LLM calls.

```python
stateloom.init(
    security_audit_hooks_enabled=True,   # CPython audit hooks (PEP 578)
    security_audit_hooks_mode="enforce", # "audit" (log) or "enforce" (block)
    security_audit_hooks_deny_events=["subprocess.Popen", "os.system"],
    security_audit_hooks_allow_paths=["/usr/lib/*", "/tmp/safe/*"],
    security_secret_vault_enabled=True,  # Move API keys to protected vault
    security_secret_vault_scrub_environ=True,  # Scrub keys from os.environ
)
```

### Audit Hooks

The hook intercepts dangerous interpreter operations: `open`, `socket.connect`, `subprocess.Popen`, `os.system`, `exec`, `import`, `ctypes.dlopen`, etc.

- **Audit mode** — log violations as `SecurityAuditEvent`, never block
- **Enforce mode** — raise `RuntimeError` for denied events (CPython hooks can only propagate `RuntimeError`)
- **Allow list** — glob patterns for file paths that bypass `open` checks
- **Global and irreversible** — `sys.addaudithook` is installed once; toggled via mutable state

### Secret Vault

Thread-safe in-memory secret storage. Moves API keys out of `os.environ` into a protected vault.

```python
# Store and retrieve secrets
stateloom.vault_store("CUSTOM_SECRET", "my-value")
secret = stateloom.vault_retrieve("CUSTOM_SECRET")

# Check status
status = stateloom.security_status()
```

Default keys: `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GOOGLE_API_KEY`, `STATELOOM_LICENSE_KEY`, `STATELOOM_JWT_SECRET`. Vault scrub is opt-in — scrubbing can break SDKs that read keys lazily. `restore_environ()` on shutdown restores scrubbed keys.

### Configuration

| Option | Default | Description |
|--------|---------|-------------|
| `security_audit_hooks_enabled` | `False` | Enable CPython audit hooks |
| `security_audit_hooks_mode` | `"audit"` | `"audit"` (log) or `"enforce"` (block) |
| `security_audit_hooks_deny_events` | `[]` | Audit events to deny (e.g. `subprocess.Popen`) |
| `security_audit_hooks_allow_paths` | `[]` | Glob patterns for allowed file paths |
| `security_secret_vault_enabled` | `False` | Enable in-memory secret vault |
| `security_secret_vault_scrub_environ` | `False` | Scrub API keys from `os.environ` |
| `security_secret_vault_keys` | `[]` | Extra env var names to move to vault |
| `security_webhook_url` | `""` | Webhook URL for security events |

### YAML config

```yaml
security:
  audit_hooks_enabled: true
  audit_hooks_mode: enforce
  audit_hooks_deny_events:
    - subprocess.Popen
    - os.system
  audit_hooks_allow_paths:
    - /usr/lib/*
  secret_vault_enabled: true
  secret_vault_scrub_environ: true
  webhook_url: https://hooks.example.com/security
```

No new dependencies — uses stdlib only (`sys.addaudithook`, `threading`, `collections.deque`, `fnmatch`).

## Cross-Process Config Sync

When StateLoom runs across multiple processes (e.g., a dashboard server and an SDK script sharing the same SQLite store), configuration changes made via the dashboard API are automatically propagated to all processes. Each middleware polls the shared store every 2 seconds for updated config.

**Synced middleware:**

| Middleware | Store Key | Synced Fields |
|-----------|-----------|---------------|
| Kill Switch | `kill_switch_state` | `active`, `message`, `rules` |
| PII Scanner | `pii_config_json` | `enabled`, `default_mode`, `rules` |
| Budget Enforcer | `budget_config_json` | `budget_per_session`, `budget_global`, `budget_action` |
| Blast Radius | `blast_radius_config_json` | `enabled`, `consecutive_failures`, `budget_violations_per_hour` |
| Guardrails | `guardrails_config_json` | `enabled`, `mode`, `heuristic_enabled`, `nli_enabled`, `nli_threshold`, `local_model_enabled`, `output_scanning_enabled`, `disabled_rules` |

**How it works:**

1. **Write side** (dashboard API): When you change a config value via `PATCH /api/config` or a feature-specific endpoint, the API serializes the relevant config fields as JSON and persists them to the store via `save_secret(key, json_blob)`.
2. **Read side** (middleware): At the top of each `process()` call, middleware calls `_sync_from_store()` which is gated by a 2-second interval using `time.monotonic()`. If enough time has elapsed, it reads the JSON blob from the store and updates the in-memory config.
3. **Fail-open**: All sync operations use `except Exception: logger.debug(...)` — a sync failure never breaks LLM calls.

**Latency impact:** Negligible. Between polls, the interval check is a `time.monotonic()` comparison (~nanoseconds). When the poll fires, it's one `get_secret()` call per middleware (~0.05ms on SQLite, dict lookup on MemoryStore).

## Dashboard

Starts automatically at `localhost:4781` (disable with `dashboard=False`).

### REST API

**Sessions**

| Endpoint | Description |
|----------|-------------|
| `GET /api/sessions` | List sessions (paginated) |
| `GET /api/sessions/{id}` | Session details |
| `GET /api/sessions/{id}/events` | Events for a session |
| `GET /api/sessions/{id}/children` | List child sessions |
| `POST /api/sessions/{id}/cancel` | Cancel an active session |
| `POST /api/sessions/{id}/suspend` | Suspend session (human-in-the-loop) |
| `POST /api/sessions/{id}/signal` | Signal/resume a suspended session |
| `POST /api/sessions/{id}/feedback` | Record feedback |
| `GET /api/sessions/{id}/feedback` | Get feedback for a session |
| `GET /api/sessions/{id}/export` | Export session as JSON bundle |
| `POST /api/sessions/import` | Import a session bundle |

**Experiments**

| Endpoint | Description |
|----------|-------------|
| `GET /api/experiments` | List experiments |
| `POST /api/experiments` | Create experiment |
| `GET /api/experiments/{id}` | Experiment detail |
| `PATCH /api/experiments/{id}` | Update DRAFT experiment |
| `POST /api/experiments/{id}/start` | Start experiment |
| `POST /api/experiments/{id}/pause` | Pause experiment |
| `POST /api/experiments/{id}/conclude` | Conclude experiment |
| `GET /api/experiments/{id}/metrics` | Experiment metrics |
| `GET /api/leaderboard` | Cross-experiment ranking |

**Consensus**

| Endpoint | Description |
|----------|-------------|
| `GET /api/consensus-runs` | List consensus runs (optional `?strategy=` filter) |
| `GET /api/consensus-runs/{session_id}` | Consensus run detail with rounds and children |

**Stats & Health**

| Endpoint | Description |
|----------|-------------|
| `GET /api/health` | Health check |
| `GET /api/stats` | Global stats |
| `GET /api/stats/cost-by-model` | Cost breakdown by model |
| `GET /api/models/cloud` | List available cloud models |

**PII**

| Endpoint | Description |
|----------|-------------|
| `GET /api/pii` | PII scan status and statistics |
| `GET /api/pii/rules` | List PII rules |
| `POST /api/pii/rules` | Add a PII rule |
| `PUT /api/pii/rules` | Replace all PII rules |
| `DELETE /api/pii/rules` | Clear all PII rules |
| `DELETE /api/pii/rules/{pattern}` | Remove a specific PII rule |

**Local Models**

| Endpoint | Description |
|----------|-------------|
| `GET /api/local/status` | Ollama availability and config |
| `GET /api/local/models` | List downloaded local models |
| `GET /api/local/recommend` | Hardware-aware model recommendations |
| `POST /api/local/pull` | Trigger model download |
| `GET /api/local/pull/{model}/progress` | Model download progress (SSE stream) |
| `DELETE /api/local/models/{model}` | Delete a local model |
| `POST /api/local/hot-swap` | Hot-swap local model with zero downtime |
| `GET /api/local/hot-swap` | Hot-swap operation status |
| `GET /api/shadow/metrics` | Model testing aggregate metrics |
| `GET /api/shadow/readiness` | Model testing readiness scores per candidate model |

**Kill Switch**

| Endpoint | Description |
|----------|-------------|
| `GET /api/kill-switch` | Kill switch status, rules, config |
| `POST /api/kill-switch/activate` | Activate global kill switch |
| `POST /api/kill-switch/deactivate` | Deactivate global kill switch |
| `POST /api/kill-switch/rules` | Add a kill switch rule |
| `PUT /api/kill-switch/rules` | Replace all kill switch rules |
| `DELETE /api/kill-switch/rules` | Clear all kill switch rules |

**Blast Radius**

| Endpoint | Description |
|----------|-------------|
| `GET /api/blast-radius` | Blast radius status (paused sessions/agents) |
| `POST /api/blast-radius/unpause-session/{id}` | Unpause a session |
| `POST /api/blast-radius/unpause-agent/{id}` | Unpause an agent |
| `GET /api/blast-radius/events` | List blast radius events |

**Circuit Breaker**

| Endpoint | Description |
|----------|-------------|
| `GET /api/circuit-breaker` | Circuit breaker status for all providers |
| `POST /api/circuit-breaker/{provider}/reset` | Reset a provider's circuit breaker |

**Rate Limiting**

| Endpoint | Description |
|----------|-------------|
| `PUT /api/teams/{id}/rate-limit` | Set team rate limit (TPS, priority, queue config) |
| `GET /api/teams/{id}/rate-limit` | Get team rate limit config + live status |
| `DELETE /api/teams/{id}/rate-limit` | Remove team rate limit (unlimited) |
| `GET /api/rate-limiter` | Global rate limiter status across all teams |

**Organizations & Teams**

| Endpoint | Description |
|----------|-------------|
| `GET /api/organizations` | List organizations |
| `POST /api/organizations` | Create organization |
| `GET /api/organizations/{id}` | Organization detail |
| `PATCH /api/organizations/{id}` | Update organization |
| `GET /api/organizations/{id}/teams` | List teams in organization |
| `GET /api/organizations/{id}/sessions` | List sessions for organization |
| `GET /api/teams` | List teams (filter by org) |
| `POST /api/teams` | Create team |
| `GET /api/teams/{id}` | Team detail |
| `PATCH /api/teams/{id}` | Update team |
| `GET /api/teams/{id}/sessions` | List sessions for team |

**Compliance**

| Endpoint | Description |
|----------|-------------|
| `GET /api/compliance/profiles` | List available compliance presets (GDPR/HIPAA/CCPA) |
| `GET /api/compliance/global` | Get global compliance profile |
| `PUT /api/compliance/global` | Set global compliance profile |
| `DELETE /api/compliance/global` | Remove global compliance profile |
| `GET /api/compliance/audit` | Query compliance audit trail |
| `POST /api/compliance/purge` | Purge user data (Right to Be Forgotten) |
| `GET /api/organizations/{id}/compliance` | Get org compliance profile |
| `PUT /api/organizations/{id}/compliance` | Set org compliance profile |
| `GET /api/teams/{id}/compliance` | Get team compliance profile |
| `PUT /api/teams/{id}/compliance` | Set team compliance profile |

**Async Jobs**

| Endpoint | Description |
|----------|-------------|
| `POST /api/jobs` | Submit async job |
| `GET /api/jobs` | List jobs (filter by status, session) |
| `GET /api/jobs/stats` | Job statistics |
| `GET /api/jobs/{id}` | Job detail |
| `DELETE /api/jobs/{id}` | Cancel a pending job |

**Virtual Keys**

| Endpoint | Description |
|----------|-------------|
| `POST /api/virtual-keys` | Create virtual key |
| `GET /api/virtual-keys` | List virtual keys (previews only) |
| `DELETE /api/virtual-keys/{id}` | Revoke a virtual key |
| `PUT /api/virtual-keys/{id}/rate-limit` | Set per-key rate limit |
| `GET /api/virtual-keys/{id}/rate-limit` | Get per-key rate limit |
| `DELETE /api/virtual-keys/{id}/rate-limit` | Remove per-key rate limit |

**Agents**

| Endpoint | Description |
|----------|-------------|
| `POST /api/agents` | Create agent + initial version |
| `GET /api/agents` | List agents (filter by team, org, status) |
| `GET /api/agents/{ref}` | Agent detail with active version |
| `PATCH /api/agents/{ref}` | Update agent name/description/status |
| `DELETE /api/agents/{ref}` | Archive agent (soft delete) |
| `POST /api/agents/{ref}/versions` | Create new version |
| `GET /api/agents/{ref}/versions` | List versions (newest first) |
| `PUT /api/agents/{ref}/versions/{id}/activate` | Activate version (rollback) |
| `GET /api/agents/{ref}/sessions` | List sessions for agent |

**Prompts**

| Endpoint | Description |
|----------|-------------|
| `GET /api/prompts/status` | Prompt watcher status (tracked files, errors) |
| `POST /api/prompts/rescan` | Force prompt directory rescan |

**Authentication** (when `auth_enabled=True`)

| Endpoint | Description |
|----------|-------------|
| `POST /api/v1/auth/bootstrap` | First-user setup (only when 0 users exist) |
| `POST /api/v1/auth/login` | Email + password login → JWT tokens |
| `POST /api/v1/auth/refresh` | Rotate refresh token |
| `POST /api/v1/auth/logout` | Revoke refresh token |
| `GET /api/v1/auth/me` | Current user + roles + permissions |
| `POST /api/v1/auth/change-password` | Update password |

**Users** (when `auth_enabled=True`)

| Endpoint | Description |
|----------|-------------|
| `POST /api/v1/users` | Create user |
| `GET /api/v1/users` | List users |
| `GET /api/v1/users/{id}` | Get user + team roles |
| `PATCH /api/v1/users/{id}` | Update user |
| `DELETE /api/v1/users/{id}` | Soft-delete user |
| `POST /api/v1/users/{id}/team-roles` | Assign team role |
| `DELETE /api/v1/users/{id}/team-roles/{team_id}` | Remove team role |

**OIDC** (when `auth_enabled=True`)

| Endpoint | Description |
|----------|-------------|
| `GET /api/v1/auth/oidc/providers` | List OIDC providers (public, for login UI) |
| `GET /api/v1/auth/oidc/authorize/{provider_id}` | Get authorization URL + state |
| `POST /api/v1/auth/oidc/callback` | Exchange code → JWT (JIT provision or login) |
| `POST /api/v1/auth/oidc/link` | TOFU: verify local password, merge OIDC identity |
| `POST /api/v1/oidc/providers` | Register OIDC provider |
| `GET /api/v1/oidc/providers` | List OIDC providers (admin) |
| `PATCH /api/v1/oidc/providers/{id}` | Update OIDC provider |
| `DELETE /api/v1/oidc/providers/{id}` | Delete OIDC provider |

**Config & Admin**

| Endpoint | Description |
|----------|-------------|
| `GET /api/config` | Current configuration (includes all settings) |
| `PATCH /api/config` | Update configuration at runtime |
| `POST /api/admin/locks` | Lock a config setting |
| `GET /api/admin/locks` | List all locked settings |
| `DELETE /api/admin/locks/{setting}` | Unlock a config setting |
| `GET /api/provider-keys` | List configured provider API keys (masked) |
| `POST /api/restart` | Restart the gateway |

**Security**

| Endpoint | Description |
|----------|-------------|
| `GET /api/security` | Combined security status (audit hooks + vault) |
| `POST /api/security/audit-hooks/configure` | Configure audit hooks (enable/disable, mode, deny events, allow paths) |
| `GET /api/security/vault` | Vault status (key names, NOT values) |
| `POST /api/security/vault/store` | Store a new secret in the vault |
| `GET /api/security/events` | Recent security audit events |
| `GET /api/security/guardrails` | Guardrails config status + aggregated detection stats |
| `GET /api/security/guardrails/events` | Recent guardrail violation events |
| `POST /api/security/guardrails/configure` | Runtime guardrails configuration (mode, NLI toggle, threshold) |

**Observability**

| Endpoint | Description |
|----------|-------------|
| `GET /api/observability/timeseries` | Time-series metrics data |
| `GET /api/observability/latency` | Latency percentile data |
| `GET /api/observability/breakdown` | Metrics breakdown by model/provider |

Real-time event streaming is available via WebSocket at `ws://localhost:4781/ws`.

## Configuration

### Via `init()`

```python
stateloom.init(
    auto_patch=True,
    default_model="",        # default model for stateloom.chat()
    budget=None,
    pii=False,
    pii_rules=[],
    dashboard=True,
    dashboard_port=4781,
    console_output=True,
    fail_open=True,
    store_backend="sqlite",  # or "memory" or "postgres"
    cache=True,
    cache_backend="memory",  # or "sqlite", "redis"
    cache_semantic=False,    # semantic similarity matching
    local_model=None,        # e.g. "llama3.2" to enable local models
    shadow=False,            # enable model testing
    shadow_model=None,       # defaults to local_model
    auto_route=None,         # auto-enabled when local_model is set; False to disable
    auto_route_model=None,   # defaults to local_model
    circuit_breaker=False,   # provider failover
    metrics_enabled=False,   # Prometheus metrics
    async_jobs_enabled=False, # background job processing
    proxy=False,             # proxy mode
    guardrails_enabled=False, # prompt injection / jailbreak protection
    guardrails_mode="audit", # "audit" or "enforce"
    prompts_dir="",          # e.g. "prompts/" for file-based agent management
    durable_stream_delay_ms=0, # replay delay for cached streams
)
```

### Via YAML

```python
from stateloom.core.config import StateLoomConfig
config = StateLoomConfig.from_yaml("stateloom.yaml")
```

### All config options

**Core**

| Option | Default | Description |
|--------|---------|-------------|
| `auto_patch` | `True` | Auto-detect and patch LLM clients |
| `fail_open` | `True` | Middleware errors never break LLM calls |
| `log_level` | `"INFO"` | Logging level |
| `default_model` | `""` | Default model for `stateloom.chat()` |
| `console_output` | `True` | Print per-call summary to terminal |
| `console_verbose` | `False` | Verbose console output |

**Dashboard**

| Option | Default | Description |
|--------|---------|-------------|
| `dashboard` | `True` | Start dashboard server |
| `dashboard_port` | `4781` | Dashboard port |
| `dashboard_host` | `"127.0.0.1"` | Dashboard bind address |
| `dashboard_api_key` | `""` | API key for dashboard authentication |

**Budget**

| Option | Default | Description |
|--------|---------|-------------|
| `budget_per_session` | `None` | Per-session budget in USD |
| `budget_global` | `None` | Global budget cap in USD |
| `budget_action` | `hard_stop` | `warn` or `hard_stop` |

**PII**

| Option | Default | Description |
|--------|---------|-------------|
| `pii_enabled` | `False` | Enable PII scanning |
| `pii_default_mode` | `audit` | `audit`, `redact`, or `block` |
| `pii_ner_enabled` | `False` | Enable GLiNER NER-based PII detection |
| `pii_ner_model` | `"urchade/gliner_small-v2.1"` | GLiNER model name |
| `pii_ner_labels` | `["person", "location", ...]` | NER entity labels to detect |
| `pii_ner_threshold` | `0.5` | NER confidence threshold |
| `pii_stream_buffer_enabled` | `False` | Hold back stream chunks for PII scanning |
| `pii_stream_buffer_size` | `3` | Number of chunks to hold back |

**Guardrails**

| Option | Default | Description |
|--------|---------|-------------|
| `guardrails_enabled` | `False` | Enable guardrails middleware |
| `guardrails_mode` | `"audit"` | `"audit"` (log only) or `"enforce"` (block) |
| `guardrails_heuristic_enabled` | `True` | Enable regex pattern scanning |
| `guardrails_local_model_enabled` | `False` | Enable Llama-Guard via Ollama |
| `guardrails_local_model` | `"llama-guard3:1b"` | Llama-Guard model name |
| `guardrails_local_model_timeout` | `10.0` | Local model timeout in seconds |
| `guardrails_output_scanning_enabled` | `True` | Enable system prompt leak detection |
| `guardrails_system_prompt_leak_threshold` | `0.6` | Leak detection sensitivity (0.0-1.0) |
| `guardrails_nli_enabled` | `False` | Enable NLI injection classifier (requires `stateloom[semantic]`) |
| `guardrails_nli_model` | `"cross-encoder/nli-MiniLM2-L6-H768"` | NLI CrossEncoder model |
| `guardrails_nli_threshold` | `0.75` | NLI score threshold for flagging (0.0-1.0) |
| `guardrails_disabled_rules` | `[]` | Pattern names to skip |
| `guardrails_webhook_url` | `""` | Webhook URL for violation notifications |

**Cache**

| Option | Default | Description |
|--------|---------|-------------|
| `cache_enabled` | `True` | Enable request caching |
| `cache_max_size` | `1000` | Max cached entries |
| `cache_ttl_seconds` | `0` | Cache entry TTL (0 = no expiry) |
| `cache_backend` | `"memory"` | `"memory"`, `"sqlite"`, or `"redis"` |
| `cache_scope` | `"global"` | `"session"` or `"global"` |
| `cache_semantic_enabled` | `False` | Enable semantic similarity matching |
| `cache_similarity_threshold` | `0.95` | Cosine similarity threshold for semantic hits |
| `cache_embedding_model` | `"all-MiniLM-L6-v2"` | Embedding model for semantic matching |
| `cache_redis_url` | `"redis://localhost:6379"` | Redis URL for cache backend |
| `cache_vector_backend` | `"faiss"` | `"faiss"` or `"redis"` for vector storage |
| `cache_normalize_patterns` | `[]` | Regex patterns for request normalization |
| `cache_db_path` | `".stateloom/cache.db"` | SQLite cache database path |

**Loop Detection**

| Option | Default | Description |
|--------|---------|-------------|
| `loop_exact_threshold` | `3` | Repeated request threshold |
| `loop_semantic_enabled` | `False` | Enable semantic loop detection |
| `loop_semantic_threshold` | `0.92` | Similarity threshold for semantic loop detection |

**Storage**

| Option | Default | Description |
|--------|---------|-------------|
| `store_backend` | `"sqlite"` | `"sqlite"`, `"memory"`, or `"postgres"` |
| `store_path` | `".stateloom/data.db"` | SQLite database path |
| `store_retention_days` | `30` | Auto-cleanup after N days |
| `store_payloads` | `False` | Store request/response payloads (privacy-sensitive) |
| `store_postgres_url` | `"postgresql://localhost:5432/stateloom"` | PostgreSQL connection URL |
| `store_postgres_pool_min` | `2` | Min connection pool size |
| `store_postgres_pool_max` | `10` | Max connection pool size |

**Local Models (Ollama)**

| Option | Default | Description |
|--------|---------|-------------|
| `local_model_enabled` | `False` | Enable local model support via Ollama |
| `local_model_host` | `"http://localhost:11434"` | Ollama server URL |
| `local_model_default` | `""` | Default local model name |
| `local_model_timeout` | `30.0` | Timeout for local model calls (seconds) |
| `ollama_managed` | `False` | Auto-download and manage Ollama |
| `ollama_managed_port` | `11435` | Managed Ollama port |
| `ollama_auto_pull` | `True` | Auto-pull models on first use |

**Model Testing**

| Option | Default | Description |
|--------|---------|-------------|
| `shadow_enabled` | `False` | Enable model testing |
| `shadow_model` | `""` | Candidate model to test against production |
| `shadow_models` | `[]` | List of candidate models to test simultaneously |
| `shadow_timeout` | `30.0` | Timeout for candidate model calls (seconds) |
| `shadow_max_workers` | `2` | Thread pool size for candidate model calls |
| `shadow_sample_rate` | `1.0` | Fraction of traffic to test (0.0–1.0); lower values reduce local compute cost |
| `shadow_max_context_tokens` | `0` | Skip requests exceeding this token count (0 = no limit) |
| `shadow_similarity_timeout` | `5.0` | Timeout for similarity scoring |
| `shadow_similarity_method` | `"difflib"` | `"difflib"`, `"semantic"`, or `"auto"` |
| `shadow_similarity_model` | `"all-MiniLM-L6-v2"` | Model for semantic similarity |

**Auto-Routing**

| Option | Default | Description |
|--------|---------|-------------|
| `auto_route_enabled` | `False` | Auto-enabled when `local_model` is set; use `auto_route=False` to disable |
| `auto_route_force_local` | `False` | Force all eligible requests to local models |
| `auto_route_model` | `""` | Local model to route to (falls back to `local_model_default`) |
| `auto_route_timeout` | `30.0` | Timeout for auto-routed local calls (seconds) |
| `auto_route_complexity_threshold` | `0.15` | Complexity score below which requests route local |
| `auto_route_complex_floor` | `0.15` | Complexity score above which requests always use cloud |
| `auto_route_probe_enabled` | `True` | Probe local model in uncertain zone |
| `auto_route_probe_timeout` | `5.0` | Timeout for probe calls (seconds) |
| `auto_route_probe_threshold` | `0.6` | Probe confidence needed to route local |
| `auto_route_semantic_enabled` | `True` | Enable semantic NLI complexity classification |
| `auto_route_semantic_model` | `"cross-encoder/nli-MiniLM2-L6-H768"` | CrossEncoder model for semantic classification |

**Kill Switch**

| Option | Default | Description |
|--------|---------|-------------|
| `kill_switch_active` | `False` | Global kill switch on/off |
| `kill_switch_message` | `"Service temporarily unavailable..."` | Message for blocked requests |
| `kill_switch_response_mode` | `"error"` | `"error"` (raise) or `"response"` (return static) |
| `kill_switch_rules` | `[]` | List of `KillSwitchRule` for granular blocking |
| `kill_switch_environment` | `""` | Current environment (matched by rule filters) |
| `kill_switch_agent_version` | `""` | Current agent version (matched by rule filters) |
| `kill_switch_webhook_url` | `""` | Webhook URL for kill switch events |

**Blast Radius**

| Option | Default | Description |
|--------|---------|-------------|
| `blast_radius_enabled` | `False` | Enable blast radius containment |
| `blast_radius_consecutive_failures` | `5` | Consecutive failures before pausing |
| `blast_radius_budget_violations_per_hour` | `10` | Budget violations per hour before pausing |
| `blast_radius_webhook_url` | `""` | Webhook URL for pause notifications |

**Circuit Breaker**

| Option | Default | Description |
|--------|---------|-------------|
| `circuit_breaker_enabled` | `False` | Enable circuit breaker failover |
| `circuit_breaker_failure_threshold` | `5` | Failures to open circuit |
| `circuit_breaker_window_seconds` | `300` | Failure tracking window |
| `circuit_breaker_recovery_timeout` | `60` | Seconds before half-open probe |
| `circuit_breaker_fallback_map` | `{}` | Manual model fallback overrides |

**Authentication**

| Option | Default | Description |
|--------|---------|-------------|
| `auth_enabled` | `False` | Enable JWT auth for dashboard API |
| `auth_jwt_algorithm` | `"HS256"` | JWT signing algorithm (`HS256` or `RS256`) |
| `auth_jwt_access_ttl` | `900` | Access token TTL in seconds (15 min) |
| `auth_jwt_refresh_ttl` | `604800` | Refresh token TTL in seconds (7 days) |

**Proxy**

| Option | Default | Description |
|--------|---------|-------------|
| `proxy_enabled` | `False` | Enable proxy mode |
| `proxy_require_virtual_key` | `True` | Require virtual key for proxy requests |
| `proxy_upstream_anthropic` | `"https://api.anthropic.com"` | Upstream URL for Anthropic API |
| `proxy_upstream_openai` | `"https://api.openai.com"` | Upstream URL for OpenAI API |
| `proxy_upstream_gemini` | `"https://generativelanguage.googleapis.com"` | Upstream URL for Gemini API |
| `proxy_timeout` | `600.0` | Upstream request timeout in seconds |
| `provider_api_key_openai` | `""` | Centrally managed OpenAI API key |
| `provider_api_key_anthropic` | `""` | Centrally managed Anthropic API key |
| `provider_api_key_google` | `""` | Centrally managed Google API key |

**Compliance**

| Option | Default | Description |
|--------|---------|-------------|
| `compliance_profile` | `None` | Global compliance profile (GDPR/HIPAA/CCPA) |

**Async Jobs**

| Option | Default | Description |
|--------|---------|-------------|
| `async_jobs_enabled` | `False` | Enable background job processing |
| `async_jobs_max_workers` | `4` | Worker pool size |
| `async_jobs_default_ttl` | `3600` | Default job TTL in seconds |
| `async_jobs_webhook_timeout` | `30.0` | Webhook delivery timeout |
| `async_jobs_webhook_retries` | `3` | Webhook retry count |
| `async_jobs_webhook_secret` | `""` | HMAC-SHA256 webhook signing secret |
| `async_jobs_queue_backend` | `"in_process"` | `"in_process"` or `"redis"` |
| `async_jobs_redis_url` | `"redis://localhost:6379"` | Redis URL for job queue |

**Observability**

| Option | Default | Description |
|--------|---------|-------------|
| `metrics_enabled` | `False` | Enable Prometheus metrics collection |

**Durable Sessions**

| Option | Default | Description |
|--------|---------|-------------|
| `durable_stream_delay_ms` | `0` | Inter-chunk delay (ms) when replaying cached streams (0 = instant) |

**Prompts**

| Option | Default | Description |
|--------|---------|-------------|
| `prompts_dir` | `""` | Directory for file-based prompt versioning (empty = disabled) |
| `prompts_poll_interval` | `2.0` | Watchdog polling interval in seconds |

**Security**

| Option | Default | Description |
|--------|---------|-------------|
| `security_audit_hooks_enabled` | `False` | Enable CPython audit hooks (PEP 578) |
| `security_audit_hooks_mode` | `"audit"` | `"audit"` (log only) or `"enforce"` (block operations) |
| `security_audit_hooks_deny_events` | `[]` | Audit events to deny (e.g. `["subprocess.Popen", "os.system"]`) |
| `security_audit_hooks_allow_paths` | `[]` | Glob patterns for allowed file paths |
| `security_secret_vault_enabled` | `False` | Enable in-memory secret vault |
| `security_secret_vault_scrub_environ` | `False` | Scrub API keys from `os.environ` |
| `security_secret_vault_keys` | `[]` | Additional env var names to move to vault |
| `security_webhook_url` | `""` | Webhook URL for security events |

## Threading

For applications that spawn threads inside sessions:

```python
stateloom.patch_threading()
```

This patches `threading.Thread` to propagate session context to child threads. Restored on `stateloom.shutdown()`.

## Middleware Pipeline

Requests flow through a middleware chain in order:

1. **KillSwitchMiddleware** — global/granular emergency stop (position 0)
2. **ComplianceMiddleware** — regulatory compliance enforcement (GDPR/HIPAA/CCPA)
3. **BlastRadiusMiddleware** — auto-pause failing sessions/agents (if enabled)
4. **CircuitBreakerMiddleware** — provider failover with tier-based fallback (if enabled)
5. **RateLimiterMiddleware** — per-team TPS throttling with priority queue
6. **TimeoutCheckerMiddleware** — session timeout, idle timeout, and cancellation checking
7. **ExperimentMiddleware** — apply variant overrides (model, params, system prompt)
8. **ShadowMiddleware** — fire-and-forget model testing with candidate models (if enabled)
9. **PIIScannerMiddleware** — scan/redact/block PII
10. **GuardrailMiddleware** — prompt injection, jailbreak, and system prompt leak detection (if enabled)
11. **BudgetEnforcer** — check session budget (skipped for subscription billing mode)
12. **CacheMiddleware** — exact-match cache lookup/store
13. **LoopDetector** — detect repeated requests
14. **AutoRouterMiddleware** — route simple requests to local models (if enabled)
15. **EventRecorder** — persist events and session state
16. **ConsoleOutput** — terminal output
17. **CostTracker** — extract tokens, calculate cost (dual tracking: actual + estimated API cost)
18. **LatencyTracker** — record latency

Each middleware calls `call_next(ctx)` to continue the chain. Middleware can short-circuit (e.g., cache hit sets `skip_call=True`).

### Lifecycle hooks

Middleware can implement an optional `on_session_end(session_id)` method to clean up per-session in-memory state when a session ends. The pipeline dispatches this cleanup automatically via `Pipeline.notify_session_end()`, which is called in the `finally` block of both `session()` and `async_session()`. Cleanup errors are swallowed — they must never crash the session teardown path.

Built-in middleware with session cleanup:
- **BlastRadiusMiddleware** — evicts per-session failure counts and budget violation timestamps
- **LoopDetector** — evicts per-session loop counts

## Error Handling

StateLoom is **fail-open by default** — observability middleware errors never break your LLM calls. Security middleware (PII block, budget hard-stop) requires explicit `on_middleware_failure` configuration.

### Exception handler logging guidelines

Exception handlers follow severity-based logging:
- **CRITICAL path** (security bypass, data loss risk): `logger.warning(...)` + fail-closed behavior where appropriate
- **Observability** (cost tracking, event recording): `logger.warning(...)` — failures are visible but don't block LLM calls
- **Reconstruction fallbacks** (replay schema, SDK-optional paths): `logger.debug(...)` — fire during normal operation when optional SDKs aren't installed
- **Cleanup** (session teardown, middleware lifecycle): `pass` only for truly harmless cases

```python
from stateloom import (
    StateLoomError,              # Base
    StateLoomBudgetError,        # Budget exceeded
    StateLoomLoopError,          # Loop detected
    StateLoomPIIBlockedError,    # PII block rule triggered
    StateLoomGuardrailError,     # Guardrail violation (prompt injection, jailbreak)
    StateLoomKillSwitchError,    # Kill switch active (global or rule match)
    StateLoomBlastRadiusError,   # Session/agent paused by blast radius
    StateLoomRateLimitError,     # Team rate limit exceeded (queue full or timeout)
    StateLoomRetryError,         # All retry attempts exhausted (semantic retries)
    StateLoomTimeoutError,       # Session timed out (duration or idle)
    StateLoomCancellationError,  # Session cancelled
    StateLoomSuspendedError,     # Session suspended (human-in-the-loop)
    StateLoomCircuitBreakerError,# Provider circuit breaker open
    StateLoomComplianceError,    # Compliance violation (GDPR/HIPAA/CCPA)
    StateLoomConfigLockedError,  # Admin-locked setting override attempted
    StateLoomJobError,           # Async job processing error
    StateLoomReplayError,        # Replay engine error
    StateLoomSideEffectError,    # HTTP blocked during strict replay
    StateLoomSecurityError,      # Security policy blocked operation (audit hooks)
    StateLoomAuthError,          # Authentication failure
    StateLoomPermissionError,    # Insufficient permissions (RBAC)
    StateLoomFeatureError,       # Enterprise feature requires license
)
```

## API Reference

For a complete index of all public functions, classes, error types, and enums, see the **[API Reference](api-reference.md)**.

## Project Structure

```
src/stateloom/
  core/           Config, session, events, errors, types (BillingMode, AgentStatus, Role, etc.), context vars, pricing, jobs, org/team hierarchy
  auth/           Authentication & RBAC: User/UserTeamRole models, permissions (Role→Permission mapping), password hashing (argon2/pbkdf2), JWT issuance/verification, auth endpoints, OIDC client, permission dependencies
  agent/          Managed agent definitions: Agent/AgentVersion models, resolver, override application, prompt file parser, file watcher
  retry.py        Semantic retries: RetryLoop, durable_task() decorator
  guardrails/     Prompt injection/jailbreak protection: 32 heuristic patterns, NLI classifier, Llama-Guard local validator, output scanner, validator protocol
  middleware/     Pipeline chain: kill switch, compliance, circuit breaker, blast radius, rate limiter, timeout/cancellation, PII, guardrails, budget, cache, loop, auto-routing, cost, latency, experiments, model testing
  intercept/     Provider adapters (OpenAI, Anthropic, Gemini, Cohere, Mistral, LiteLLM), monkey-patching, generic interceptor
  local/         Ollama client, adapter, hardware detection, model catalog
  cache/         Pluggable cache backends: exact-match, semantic similarity (embeddings), Redis, SQLite, FAISS vector search
  compliance/    GDPR/HIPAA/CCPA profiles, audit hashing, legal rules, purge engine
  jobs/          Async job processing: queue protocol, Redis Streams backend, webhook delivery, processor pool
  observability/ Prometheus metrics, time-series aggregation, alerting, OpenTelemetry tracing
  experiment/    A/B testing: models, assigner, manager, backtest runner
  replay/        Time-travel: engine, durable replay engine, network blocker, schema, safety checks
  pii/           PII scanner (regex + GLiNER NER), patterns, rehydrator, stream buffer
  store/         Persistence: SQLite, memory, base protocol (sessions, events, experiments, agents, users, team roles, refresh tokens, OIDC providers)
  pricing/       Model cost calculation, bundled price data
  dashboard/     FastAPI server, REST API (incl. agent CRUD, user management, observability), WebSocket
  proxy/         Multi-protocol proxy: HTTP reverse proxy (passthrough), OpenAI, Anthropic-native, Gemini-native, Responses API (Codex CLI) routers, auth (VK + scope enforcement), virtual keys, BYOK, billing mode, sticky sessions, per-key rate limiting
  ext/           LangChain and LangGraph callback handlers
  export/        Console output formatting
```

## License

Apache 2.0
