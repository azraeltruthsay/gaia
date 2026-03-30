# gaia-nano — The Reflex

**Port:** 8090 | **GPU:** Yes | **Dependencies:** GAIA Engine (managed mode)

gaia-nano runs a tiny 0.8B Qwen model via GAIA Engine managed mode for sub-second triage classification and transcript cleanup.

## Design Principles

- **Minimal surface**: GAIA Engine managed mode with a model file
- **OpenAI-compatible API**: Drop-in integration with gaia-core's model clients
- **GPU primary, CPU fallback**: Resilience over raw speed

## Key Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check (GAIA Engine built-in) |
| `/v1/chat/completions` | POST | OpenAI-compatible chat completions |
| `/v1/completions` | POST | OpenAI-compatible text completions |
| `/slots` | GET | KV cache slot status (used by doctor for pressure monitoring) |

## Role in Cascade Routing

Nano is the first stage of GAIA's three-tier cascade:

1. **Nano** (0.8B) — classifies input as SIMPLE or COMPLEX in <100ms
2. **Core** (2B) — handles SIMPLE tasks, escalates COMPLEX to Prime
3. **Prime** (8B) — heavyweight reasoning, code, complex tasks

This routing saves GPU time and reduces latency for simple queries.

## Container

- **Non-root**: Runs as `gaia` user (Dockerfile updated 2026-03-25)
- **GPU primary, CPU fallback**: GGUF variant available for CPU-only resilience
