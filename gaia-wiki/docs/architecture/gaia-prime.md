# gaia-prime — The Voice

Standalone GAIA Engine inference server. Runs CPU/GGUF by default; gets GPU during FOCUSING transitions via the Consciousness Matrix.

## Responsibilities

- Serve inference requests from gaia-core via OpenAI-compatible HTTP API
- Run in CPU/GGUF mode as subconscious observer (default state)
- Accept GPU handoff during FOCUSING for heavyweight tasks
- Support LoRA adapter hot-loading
- Identity-baked model (Huihui-Qwen3-8B-GAIA-Prime-adaptive)

## Runtime Configuration

Prime uses GAIA Engine managed mode (not vLLM). Key settings:

| Setting | Value | Rationale |
|---------|-------|-----------|
| Backend | GAIA Engine | Custom engine with polygraph, KV cache, lifecycle |
| Default state | Subconscious (CPU/GGUF) | Conserve GPU for Nano/Core |
| GPU mode | On FOCUSING transition | Orchestrator swaps GPU when escalation needed |
| Port | 7777 | OpenAI-compatible API |
| `GAIA_AUTOLOAD_MODEL` | 0 | Standby until orchestrator loads |

## Consciousness States

| State | Mode | Triggered By |
|-------|------|-------------|
| Subconscious | CPU/GGUF | Default / after FOCUSING completes |
| Conscious | GPU | FOCUSING transition (escalation from Core) |
| Unconscious | Unloaded | DEEP_SLEEP / explicit unload |

## Container

- **Non-root**: Runs as `gaia` user (Dockerfile updated 2026-03-25)
- **Model path**: `/models/Huihui-Qwen3-8B-GAIA-Prime-adaptive` (safetensors) or GGUF variant

## Model Loading

The active model is an identity-baked 8B model. GGUF Q8_0 variant used for CPU inference. Safetensors variant used when GPU is available during FOCUSING.
