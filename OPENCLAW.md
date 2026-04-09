# Using OpenClaw with StateLoom

OpenClaw can also route LLM requests through StateLoom's proxy, giving you cost tracking, PII detection, guardrails, and all other middleware features for every agent call.

## Prerequisites

- StateLoom running with the dashboard/proxy enabled (default port `4782`)
- OpenClaw installed and configured (`~/.openclaw/openclaw.json`)

## Demo

[![StateLoom + OpenClaw Demo](https://img.youtube.com/vi/r2qr3qdgb8U/maxresdefault.jpg)](https://www.youtube.com/watch?v=r2qr3qdgb8U)

## Setup

### 1. Start StateLoom

Start StateLoom via environment variables you want to use with OpenClaw:

```bash
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."
export GOOGLE_API_KEY="AIza..."
stateloom serve
```

### 2. Configure `~/.openclaw/openclaw.json`

Add a `stateloom` provider under `models.providers` and set your agent defaults to use it.
```json
{
  "agents": {
    "defaults": {
      "model": {
        "primary": "stateloom/gemini-2.5-flash"
      },
      "models": {
        "stateloom/gemini-2.5-flash": {},
        "stateloom/claude-haiku-4-5-20251001": {},
        "stateloom/gpt-4o-mini": {}
      }
    }
  },
  "models": {
    "mode": "merge",
    "providers": {
      "stateloom": {
        "baseUrl": "http://localhost:4782/v1",
        "apiKey": "unused",
        "api": "openai-completions",
        "models": [
          {
            "id": "gemini-2.5-flash",
            "name": "Gemini 2.5 Flash (via StateLoom)",
            "contextWindow": 1048576,
            "maxTokens": 65536
          },
          {
            "id": "claude-haiku-4-5-20251001",
            "name": "Claude Haiku 4.5 (via StateLoom)",
            "contextWindow": 200000,
            "maxTokens": 8192
          },
          {
            "id": "gpt-4o-mini",
            "name": "GPT-4o Mini (via StateLoom)",
            "contextWindow": 128000,
            "maxTokens": 16384
          }
        ]
      }
    }
  }
}
```

### Key fields

| Field | Purpose |
|-------|---------|
| `models.providers.stateloom.baseUrl` | Points to StateLoom's OpenAI-compatible proxy endpoint |
| `models.providers.stateloom.apiKey` | Set to `"unused"` |
| `models.providers.stateloom.api` | Must be `"openai-completions"` — StateLoom exposes `/v1/chat/completions` |
| `models.mode` | `"merge"` keeps OpenClaw's built-in providers alongside StateLoom |
| `agents.defaults.model.primary` | The default model for new agents, prefixed with `stateloom/` |
| `agents.defaults.models` | Available models in the agent model picker |

### 3. Adding more models

To route additional models through StateLoom, add them to both sections:

1. **`models.providers.stateloom.models`** — registers the model with OpenClaw
2. **`agents.defaults.models`** — makes it available in the agent picker

The model `id` must match a model name that StateLoom can route to the correct upstream provider (e.g. `gpt-4o`, `claude-sonnet-4-20250514`, `gemini-2.5-flash`).

## Verifying

Once configured, open the StateLoom dashboard at `http://localhost:4782` and run an OpenClaw agent. You should see sessions appear with per-call cost tracking, PII detections, and model breakdowns.
