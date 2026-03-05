# GAIA Service Blueprint: `gaia-core` (The Brain)

## Role and Overview

`gaia-core` is the cognitive engine of the GAIA system. It processes user input through a multi-step reasoning pipeline: intent detection, knowledge enhancement, LLM inference, tool routing, self-reflection, and response assembly. In v0.6, gaia-core implements the **Sovereign Shield** hardening suite and the **Smart Immune System** for autonomous triage and irritation awareness.

## Container Configuration

**Base Image**: `python:3.11-slim` (CPU-only, no CUDA)

**Port**: 6415 (live), 6416 (candidate)

**Health Check**: `curl -f http://localhost:6415/health` (30s interval, 3 retries)

**Startup**: `uvicorn gaia_core.main:app --host 0.0.0.0 --port 6415`

**Dependencies**: Waits for `gaia-prime` (healthy) and `gaia-mcp` (healthy) before starting.

### Key Environment Variables

| Variable | Value | Purpose |
|----------|-------|---------|
| `GAIA_BACKEND` | `gpu_prime` | Selects remote vLLM inference backend |
| `GAIA_FORCE_CPU` | `1` | Prevents local GPU usage |
| `N_GPU_LAYERS` | `0` | No GPU layers for llama.cpp fallback |
| `PRIME_ENDPOINT` | `http://gaia-prime:7777` | Remote vLLM server address |
| `PRIME_MODEL` | `/models/Qwen3-8B-abliterated-AWQ` | Model path on gaia-prime |
| `GAIA_LITE_GGUF` | `/models/Qwen3-8B-abliterated-Q4_K_M.gguf` | **Operator** model path |
| `GAIA_NANO_GGUF` | `/models/Qwen3-0.5B-Instruct-GGUF/...` | **Nano** model path |
| `GAIA_AUTOLOAD_MODELS` | `0` | Lazy model loading on first use |
| `HOME` | `/tmp` | Zero-Trust: writable temp home |
| `TRANSFORMERS_CACHE` | `/tmp/.cache` | Zero-Trust: writable cache |

### Volume Mounts

- `./gaia-core:/app:rw` — Source code (editable in dev)
- `./gaia-common:/gaia-common:ro` — Shared library
- `./knowledge:/knowledge:ro` — Knowledge base
- `./gaia-models:/models:ro` — Model files (GGUF for Operator/Nano)
- `gaia-shared:/shared:rw` — Inter-service state
- `/run/secrets:/run/secrets:ro` — Docker Secrets (Zero-Trust Identity)

## Sovereign Shield & Immune System (v0.6)

The Sovereign Shield provides structural hardening and self-protection:

1. **Circuit Breaker** — Fatal halt mechanism in `EthicalSentinel`. If a cognitive loop exceeds thresholds, the system creates `/shared/HEALING_REQUIRED.lock` and freezes the turn loop until manual triage.
2. **Zero-Trust Identity** — API keys and sensitive tokens are prioritized from Docker Secrets (`/run/secrets/`) rather than local environment files.
3. **Blast Shield** — Deterministic pre-flight validation in the MCP layer. Blocks dangerous command patterns (e.g., `rm -rf`, `sudo`) before LLM reasoning is even invoked.
4. **Smart Immune System** — SIEM-lite logic in `gaia-common` that scans service logs, consolidates repetitive errors into unique "irritants," and calculates a weighted systemic health score.
5. **Transparency Logging** — Tool results in the `CognitionPacket` include a `raw_source_data` field, ensuring the "Reflector" persona sees un-summarized machine truth.

## Cognitive Pipeline (v0.6)

The core reasoning loop in `AgentCore.run_turn()`:

1. **Circuit Breaker Check** — Ensure no healing lock is active.
2. **Nano Fast-Path** — Ultra-fast routing for trivial tasks/status checks via **Nano (0.5B)**.
3. **Semantic Probe** — Pre-cognition vector lookup across all knowledge bases.
4. **Interstitial Triage (TCP)** — High-speed triage of "gap audio" heard while busy.
5. **Immune Awareness** — Inject "Smart" health summary into world-state snapshot.
6. **Auditory Environment** — Sense the surroundings (BPM, Key, Energy) via `gaia-audio/analyze`.
7. **Doctor Loop (Self-Healing)** — Failover-aware autonomous diagnosis.
8. **Persona & KB Selection** — Probe/Triage results drive routing.
9. **Model Selection** — Multi-path selection with escalation to **Prime (8B AWQ)**.
10. **Intent Detection** — Classify user intent via **Operator (8B GGUF)**.
11. **Planning** — Generate execution plan for complex queries via **Operator**.
12. **Cognitive Self-Audit** — Epistemic assessment.
13. **Knowledge Enhancement** — Inject relevant context.
14. **Prompt Assembly** — Build LLM prompt with **Conversation Timeline** landmarks.
15. **Generation** — Stream from primary model (**Prime**).
16. **Stream Observation** — Real-time output validation.
17. **Tool Routing** — Route to MCP tools with **Blast Shield** protection.
18. **Saṃvega Integration** — Capture discernment artifacts on low-confidence turns.
19. **Self-Reflection** — Review and refine response.
20. **Output Routing** — Deliver to destination with Transparency Logging.

## Inference Backends (v0.6)

| Backend | Role | Type | Connection | GPU Layers |
|---------|-----------|-------|------------|------------|
| `gpu_prime` | **The Voice** (Prime) | vLLM (AWQ) | HTTP:7777 | Full (vLLM) |
| `operator` | **The Brain** (Operator) | llama.cpp (GGUF) | In-process | Full (All) |
| `nano` | **The Reflex** (Nano) | llama.cpp (GGUF) | In-process | Full (All) |
| `groq_fallback` | Emergency Thinker | Groq API | HTTPS | N/A |

## Meta
**Status**: live
**Blueprint Version**: 0.6
**Last Updated**: 2026-03-02
