# Using OpenClaw with StateLoom

OpenClaw can also route LLM requests through StateLoom's proxy, giving you cost tracking, PII detection, guardrails, and all other middleware features for every agent call.

## Prerequisites

- Python 3.10+
- StateLoom installed (`pip install stateloom`)
- OpenClaw installed (`npm install -g @anthropic-ai/openclaw`, or `pip install 'stateloom[openclaw]'` for automatic setup)

## Demo

[![StateLoom + OpenClaw Demo](https://img.youtube.com/vi/r2qr3qdgb8U/maxresdefault.jpg)](https://www.youtube.com/watch?v=r2qr3qdgb8U)

## Quick Start

The `stateloom openclaw launch` command handles everything — starts StateLoom, configures OpenClaw, and prompts for API keys.

### Cloud models

```bash
stateloom openclaw launch --model gemini-2.5-flash
```

### Local models (Ollama)

Run Ollama models through OpenClaw with the `ollama:` prefix. No API key needed:

```bash
stateloom openclaw launch --model ollama:llama3.2:3b
```

Requires Ollama running locally (`ollama serve` or `stateloom ollama start`).

### Mixed cloud + local

```bash
stateloom openclaw launch --model gemini-2.5-flash,ollama:llama3.2:3b
```

### Multiple cloud models

```bash
stateloom openclaw launch --model gemini-2.5-flash,gpt-4o,claude-haiku-4-5
```

Then in another terminal:

```bash
openclaw gateway start
```

Models appear in OpenClaw's picker as `stateloom/<model>` (e.g. `stateloom/gemini-2.5-flash`, `stateloom/ollama:llama3.2:3b`).

### Options

| Flag | Purpose |
|------|---------|
| `--model` | Comma-separated model list. Cloud models and `ollama:*` local models |
| `--host` | StateLoom host (default: `127.0.0.1`) |
| `--port` | StateLoom port (default: `4782`) |
| `--no-auth` | Skip virtual key requirement (solo dev mode) |
| `--verbose` | Print a one-liner for each LLM call |

## Local Models with Ollama

StateLoom can route to any locally pulled Ollama model using the `ollama:` prefix.

### Setup

```bash
# Install and start managed Ollama
stateloom ollama install
stateloom ollama start

# Pull a model
stateloom ollama pull llama3.2:3b

# List available models
stateloom ollama list
```

Or use a standalone Ollama install:

```bash
ollama serve
ollama pull llama3.2:3b
```

### How it works

The `ollama:` prefix tells StateLoom to route the request to Ollama's OpenAI-compatible endpoint instead of a cloud provider. The full middleware pipeline still applies — cost tracking, PII scanning, guardrails, events, and dashboard visibility.

`tools` and `tool_choice` parameters are automatically stripped since most local models don't support tool calling.

### Direct proxy test

```bash
curl http://localhost:4782/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "ollama:llama3.2:3b", "messages": [{"role": "user", "content": "hi"}]}'
```

## Manual Setup

If you prefer to configure OpenClaw manually instead of using `stateloom openclaw launch`:

### 1. Start StateLoom

```bash
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
        "stateloom/ollama:llama3.2:3b": {}
      }
    }
  },
  "models": {
    "mode": "merge",
    "providers": {
      "stateloom": {
        "baseUrl": "http://localhost:4782/v1",
        "apiKey": "sk-stateloom",
        "api": "openai-completions",
        "models": [
          {
            "id": "gemini-2.5-flash",
            "name": "Gemini 2.5 Flash (via StateLoom)",
            "contextWindow": 1048576,
            "maxTokens": 65536
          },
          {
            "id": "ollama:llama3.2:3b",
            "name": "Ollama llama3.2:3b (Local via StateLoom)",
            "contextWindow": 128000,
            "maxTokens": 8192
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
| `models.providers.stateloom.apiKey` | Set to `"sk-stateloom"` (or any non-empty string) |
| `models.providers.stateloom.api` | Must be `"openai-completions"` — StateLoom exposes `/v1/chat/completions` |
| `models.mode` | `"merge"` keeps OpenClaw's built-in providers alongside StateLoom |
| `agents.defaults.model.primary` | The default model, prefixed with `stateloom/` |
| `agents.defaults.models` | Available models in the agent picker |

## Verifying

Once configured, open the StateLoom dashboard at `http://localhost:4782` and run an OpenClaw agent. You should see sessions appear with per-call cost tracking, PII detections, and model breakdowns.
