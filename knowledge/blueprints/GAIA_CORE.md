# GAIA Service Blueprint: `gaia-core` (The Brain)

## Role and Overview

`gaia-core` is the cognitive engine of the GAIA system. It processes user input through a multi-step reasoning pipeline: intent detection, knowledge enhancement, LLM inference, tool routing, self-reflection, and response assembly. In v0.4, gaia-core runs **CPU-only**, delegates all GPU inference to `gaia-prime` via HTTP, and implements the **Temporal Context Protocol (TCP)** for continuous situational awareness.

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
| `GROQ_API_KEY` | from `.env` | Groq API fallback (free tier) |
| `MCP_ENDPOINT` | `http://gaia-mcp:8765/jsonrpc` | Tool execution endpoint |
| `STUDY_ENDPOINT` | `http://gaia-study:8766` | Knowledge/embedding service |
| `GAIA_AUTOLOAD_MODELS` | `0` | Lazy model loading on first use |

### Volume Mounts

- `./gaia-core:/app:rw` — Source code (editable in dev)
- `./gaia-common:/gaia-common:ro` — Shared library
- `./knowledge:/knowledge:ro` — Knowledge base
- `./gaia-models:/models:ro` — Model files (GGUF for lite backend)
- `gaia-shared:/shared:rw` — Inter-service state

## Cognitive Pipeline (v0.4)

The core reasoning loop in `AgentCore.run_turn()`:

1. **Semantic Probe** — Pre-cognition vector lookup across all knowledge bases (`semantic_probe`)
2. **Interstitial Triage (TCP)** — High-speed triage of "gap audio" (heard while busy) using 0.5B Nano model on CPU. Detects urgent interruptions or pivots.
3. **Persona & KB Selection** — Probe/Triage results drive persona/knowledge base routing.
4. **Model Selection** — Multi-path selection with escalation and fallback. Supports `::lite` and `::thinker` fast-paths.
5. **History Review** — Pre-injection audit strips fabricated citations from conversation history.
6. **Intent Detection** — Classify user intent via lite model. Direct tasks (::lite/::thinker) skip this step.
7. **Slim Prompt Fast Path** — Simple queries (recitation, tool calls) skip planning/reflection.
8. **Planning** — Generate execution plan for complex queries.
9. **Cognitive Self-Audit** — Epistemic assessment between planning and reflection.
10. **Knowledge Enhancement** — Inject relevant context.
11. **Prompt Assembly** — Build LLM prompt within token budget. Injects **Conversation Timeline** landmarks.
12. **Generation** — Stream from primary model with lite fallback on failure.
13. **Stream Observation** — Real-time output validation.
14. **Tool Routing** — Route to MCP tools if needed.
15. **Loop Detection** — Record tool calls and outputs, detect repetitive patterns.
16. **Self-Reflection** — Review and refine response.
17. **Output Routing** — Deliver to web, Discord, CLI, or API.

## Temporal Context Protocol (TCP)

v0.4 introduces the **Temporal Context Protocol**, resolving "AI Deafness":
- **Continuous Hearing**: `VoiceManager` captures audio even while GAIA is reasoning or speaking.
- **Landmark Tracking**: Timeline markers (`reasoning_start`, `speaking_start`, `speaking_end`) ground GAIA in the sequence of events.
- **Urgent Pivots**: If the user interrupts while GAIA is talking, the Interstitial Triage detects it and tags the next packet with `#Interruption`, allowing GAIA to address the new input immediately.

## Inference Backends

| Backend | Config Key | Class | Connection |
|---------|-----------|-------|------------|
| Remote vLLM | `gpu_prime` | `VLLMRemoteModel` | HTTP to gaia-prime:7777 (32k context) |
| Groq API | `groq_fallback` | `GroqAPIModel` | HTTPS to api.groq.com |
| Local GGUF | `lite` | llama-cpp-python | In-process CPU (32k context) |
| Nano-Refiner | `nano` | RefinerEngine | HTTP to gaia-audio:8080 (0.5B model) |

## Dependencies

**Runtime**: fastapi, uvicorn, pydantic, httpx, requests, regex, discord.py, llama-cpp-python, groq
**Shared**: gaia-common (protocols, utils, config)
**Dev**: pytest, pytest-asyncio, ruff, mypy

## Meta
**Status**: live
**Blueprint Version**: 0.4
**Last Updated**: 2026-03-01
