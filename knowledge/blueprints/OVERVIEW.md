# GAIA System Architecture Overview

The GAIA project is a service-oriented architecture designed for advanced AI operations. It comprises several interconnected containerized services, each with distinct responsibilities, working together to process, reason, and act based on user requests and internal cognitive processes.

## Core Services

The system is built around eight primary services plus a shared library:

| Service | Role | Port | GPU | Base Image |
|---------|------|------|-----|------------|
| **`gaia-orchestrator`** | The Coordinator | 6410 | - | python:3.11-slim |
| **`gaia-prime`** | The Voice (Inference) | 7777 | 1x NVIDIA | NGC PyTorch 25.03 |
| **`gaia-core`** | The Brain (Cognition) | 6415 | - (CPU-only) | python:3.11-slim |
| **`gaia-web`** | The Face (UI/Discord/Voice) | 6414 | - | python:3.11-slim |
| **`gaia-mcp`** | The Hands (Tools) | 8765 | - | python:3.11-slim |
| **`gaia-study`** | The Subconscious (Learning) | 8766 | All GPUs | nvidia/cuda:12.4 |
| **`gaia-audio`** | The Ears & Mouth (STT/TTS) | 8080 | 1x NVIDIA | python:3.11-slim |
| **`gaia-wiki`** | The Library (Documentation) | 8080 (internal) | - | python:3.11-slim |

**`gaia-common`** is a shared Python library (not a running service) consumed by all services.

### Service Descriptions

1.  **`gaia-orchestrator`**: Manages Docker containers, GPU resources, and service lifecycle. Coordinates GPU handoffs between gaia-prime and gaia-study.
2.  **`gaia-prime`**: Standalone vLLM OpenAI-compatible inference server. Owns the GPU for LLM inference. Built from source targeting RTX 5080 Blackwell (sm_120) with vLLM v0.15.1 and LoRA adapter support. Currently serves Qwen3-8B-abliterated-AWQ (GPU) with Qwen3-8B-abliterated-Q4_K_M.gguf (CPU lite fallback).
3.  **`gaia-core`**: The cognitive engine. Runs CPU-only and delegates all GPU inference to gaia-prime via `PRIME_ENDPOINT`. Handles reasoning, intent detection, planning, self-reflection, tool routing, sleep/wake cycle, and session management. Falls back to Groq API or local GGUF models when prime is unavailable. Supports HA failover via `MCP_FALLBACK_ENDPOINT`.
4.  **`gaia-web`**: User-facing interface providing HTTP REST API, Discord bot (text + voice), and dashboard. Converts user input to CognitionPackets and routes completed responses back to their origin. Orchestrates Discord voice calls via VoiceManager and gaia-audio. Supports HA failover via `CORE_FALLBACK_ENDPOINT`.
5.  **`gaia-mcp`**: Sandboxed tool execution environment with approval workflows. Provides file operations, shell execution, vector queries, and knowledge management tools. Security-hardened with dropped capabilities.
6.  **`gaia-study`**: Background processing for vector indexing, embedding generation, and QLoRA model fine-tuning. Sole writer to the vector store. Uses GPU for embeddings and training.
7.  **`gaia-audio`**: Sensory microservice providing STT (faster-whisper, GPU-accelerated) and TTS (Coqui XTTS v2, espeak-ng fallback, ElevenLabs cloud fallback). Half-duplex GPU management swaps between STT and TTS models to fit VRAM budget. Called by gaia-web's VoiceManager during Discord voice sessions.
8.  **`gaia-wiki`**: Internal MkDocs Material documentation server. Accessible on the Docker network at `gaia-wiki:8080`, optionally proxied by gaia-web at `/wiki`.

## Architecture & Data Flow

The central data structure facilitating inter-service communication is the **`CognitionPacket`** (v0.3), defined in `gaia-common`.

### Request Flow

```
User (HTTP / Discord text / Discord voice)
    ↓
gaia-web (creates CognitionPacket; VoiceManager for voice calls)
    ↓                                          ↓
    ↓                                   gaia-audio (STT: Whisper → text)
    ↓
gaia-core (cognitive pipeline: intent → knowledge → reasoning → tools → reflection)
    ↓                    ↓
gaia-prime (LLM inference via HTTP)    gaia-mcp (tool execution via JSON-RPC)
    ↓                    ↓
gaia-core (assembles response)
    ↓
gaia-web (routes to Discord/HTTP/Log)
    ↓
    └─→ (voice) gaia-audio (TTS: Coqui → audio) → Discord playback
```

### Dependency Chain (startup order)

```
gaia-orchestrator (independent)
gaia-prime + gaia-mcp → gaia-core → gaia-web
gaia-study (independent, GPU-enabled)
```

### GPU Offload Architecture (v0.3)

In v0.3, GPU inference is fully decoupled from cognition:

- **gaia-core** runs CPU-only (`GAIA_FORCE_CPU=1`, `GAIA_BACKEND=gpu_prime`)
- **gaia-prime** owns 1 GPU exclusively for vLLM inference (port 7777)
- Communication is via HTTP: gaia-core's `VLLMRemoteModel` calls gaia-prime's OpenAI-compatible API
- **Fallback chain**: gpu_prime (remote vLLM, Qwen3) → groq_fallback (Groq API) → lite (local GGUF CPU, Qwen3-8B-abliterated Q4_K_M)

### Inference Backend Options

| Backend | Class | Location | Use Case |
|---------|-------|----------|----------|
| `gpu_prime` | VLLMRemoteModel | gaia-prime (HTTP) | Primary inference |
| `groq_fallback` | GroqAPIModel | Groq cloud API | Free-tier fallback |
| `lite` | llama-cpp-python | gaia-core (local CPU) | Lightweight tasks |
| `oracle_openai` | GPTAPIModel | OpenAI API | High-quality reasoning |
| `oracle_gemini` | GeminiAPIModel | Google API | Alternative oracle |

## Key Design Patterns

*   **Stateful Cognitive Packet**: The `CognitionPacket` (v0.3) maintains state through the entire cognitive pipeline with header, content, context, reasoning, response, governance, metrics, and status sections.
*   **GPU Offload**: Inference is fully decoupled from cognition. gaia-core is CPU-only; all GPU work goes through gaia-prime via HTTP.
*   **Read/Write Segregation**: `gaia-study` is the exclusive writer for the vector store. Other services have read-only access via `VectorClient`.
*   **Hybrid AI Model System**: Dynamic backend selection from local GPU inference (vLLM), local CPU inference (GGUF), cloud APIs (Groq, OpenAI, Gemini).
*   **Continuous Learning Loop**: `gaia-study` processes new information and fine-tunes LoRA adapters using QLoRA.
*   **Secure Sandboxed Tooling**: `gaia-mcp` executes tools with dropped Linux capabilities, approval workflows for sensitive operations, and isolated sandbox volumes.
*   **Candidate/Live SDLC**: Parallel candidate stack for testing. Validated promotion via `promote_candidate.sh` with containerized ruff/mypy/pytest checks.
*   **High Availability**: HA overlay (`docker-compose.ha.yml`) enables hot-standby failover. Candidate services run as warm standbys; gaia-web and gaia-core auto-route to candidates on primary failure.
*   **Sleep/Wake Cycle**: Biologically-inspired activity states (Active → Drowsy → Asleep → REM). Incoming messages or voice calls trigger wake signals. Lite model provides stalling responses while Prime boots.
*   **Voice Pipeline**: Discord voice integration via gaia-web VoiceManager + gaia-audio sensory service. Real-time VAD → STT (Whisper) → cognition → TTS (Coqui) → Discord playback.

## Key Directories and Conventions

*   **`/gaia/GAIA_Project/candidates/`**: Development versions of services for testing before promotion.
*   **`/gaia/GAIA_Project/<service-name>/`**: Live version of each service (e.g., `gaia-core/`, `gaia-prime/`, `gaia-web/`).
*   **`/gaia/GAIA_Project/gaia-common/`**: Shared Python library for protocols, utilities, and config.
*   **`knowledge/Dev_Notebook/`**: Developer journal entries with historical context and design decisions.
*   **`knowledge/blueprints/`**: Architecture documentation (this directory).
*   **`docker-compose.yml`**: Live Docker Compose stack definition.
*   **`docker-compose.candidate.yml`**: Candidate stack for testing and staging.
*   **`docker-compose.ha.yml`**: HA failover overlay (hot-standby candidate services).
*   **`scripts/promote_candidate.sh`**: Formal candidate → live promotion with validation.
*   **`scripts/ha_start.sh`**, **`ha_stop.sh`**, **`ha_sync.sh`**, **`ha_maintenance.sh`**: HA lifecycle scripts.
*   **`gaia.sh`**: Primary stack management CLI (live/candidate start/stop/status).
*   **`gaia_constants.json`**: Runtime configuration in `gaia-core/gaia_core/`, specifying model configs, feature flags, and task instructions.

## Network Configuration

*   **Network**: `gaia-net` (bridge, 172.28.0.0/16)
*   **Shared volumes**: `gaia-shared` (inter-service state), `gaia-sandbox` (MCP workspace)
*   **Secrets**: Discord token via `.env.discord` (gitignored), Groq API key via `.env`
