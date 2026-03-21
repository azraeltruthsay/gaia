# gaia-nano — The Reflex

**Port:** 8090 | **GPU:** Yes | **Dependencies:** llama-server

gaia-nano runs a tiny 0.8B Qwen model via llama-server for sub-second triage classification and transcript cleanup.

## Design Principles

- **Minimal surface**: No application code — just llama-server with a GGUF model
- **OpenAI-compatible API**: Drop-in integration with gaia-core's model clients
- **GPU primary, CPU fallback**: Resilience over raw speed

## Key Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check (llama-server built-in) |
| `/v1/chat/completions` | POST | OpenAI-compatible chat completions |
| `/v1/completions` | POST | OpenAI-compatible text completions |
| `/slots` | GET | KV cache slot status (used by doctor for pressure monitoring) |

## Role in Cascade Routing

Nano is the first stage of GAIA's three-tier cascade:

1. **Nano** (0.8B) — classifies input as SIMPLE or COMPLEX in <100ms
2. **Core** (2B) — handles SIMPLE tasks, escalates COMPLEX to Prime
3. **Prime** (8B) — heavyweight reasoning, code, complex tasks

This routing saves GPU time and reduces latency for simple queries.
