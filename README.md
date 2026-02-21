# GAIA

**A sovereign AI assistant running entirely on local hardware.**

GAIA is a self-hosted, containerized AI system built around a locally-served language model. She operates as a service-oriented architecture where each service owns a single responsibility — cognition, inference, tools, learning, audio, and interface — coordinated by an orchestrator that manages GPU resources across a single consumer GPU.

## Hardware

- **GPU**: NVIDIA RTX 5080 (16 GB VRAM, Blackwell sm_120)
- **Inference Budget**: 70% VRAM (~11.2 GB) for the primary model, remainder for KV cache
- **CPU Fallback**: GGUF models via llama-cpp-python when GPU is unavailable

## Models

### Primary: Qwen3-4B-Instruct-2507-heretic (GPU)

The main inference model, served by vLLM through gaia-prime.

| Property | Value |
|----------|-------|
| Base | Qwen/Qwen3-4B-Instruct-2507 |
| Variant | Heretic (abliterated/uncensored) |
| Format | SafeTensors (~7.6 GB) |
| Context | 8192 tokens |
| Serving | vLLM with Flash Attention v2, `--enforce-eager` (required for Blackwell) |
| LoRA | Enabled (max 4 adapters, rank 64) |

A Qwen3-8B-AWQ (~5.7 GB, AWQ quantized) is available as an alternative and used in the candidate stack.

### Fallback: Qwen3-4B-Instruct-2507-heretic Q4_K_M (CPU)

Local GGUF fallback for when the GPU is handed off to training.

| Property | Value |
|----------|-------|
| Format | GGUF Q4_K_M (~2.4 GB) |
| Context | 8192 tokens |
| Serving | llama-cpp-python (CPU-only) |

### Cloud Fallbacks

| Backend | Model | Purpose |
|---------|-------|---------|
| `groq_fallback` | llama-3.3-70b-versatile | Free-tier API fallback |
| `oracle_openai` | gpt-4o-mini | High-quality reasoning oracle |

### Embedding: all-MiniLM-L6-v2

Sentence-transformer model used by gaia-study for document embeddings and vector similarity search (RAG retrieval).

## Architecture

```
                   Discord / HTTP
                        |
                   [gaia-web]          Face — user interface, Discord bot, voice commands
                     /     \
              [gaia-wiki] [gaia-core]  Library & Brain — docs, cognition, session memory
                          /    \
               [gaia-prime]  [gaia-mcp]   Voice & Hands — GPU inference, tool execution
                                |
                           [gaia-study]   Subconscious — vector indexing, QLoRA training
                                |
                   [gaia-orchestrator]    Conductor — GPU scheduling, container lifecycle

                   [gaia-audio]          Ears & Mouth — STT (Whisper), TTS (Coqui)
                   (called by gaia-web for Discord voice)
```

### Services

| Service | Role | Port | Runtime |
|---------|------|------|---------|
| **gaia-orchestrator** | GPU scheduling, container lifecycle, handoff state machine | 6410 | Python 3.11 |
| **gaia-prime** | vLLM inference server (primary model) | 7777 | NGC PyTorch 25.03 + NVIDIA GPU |
| **gaia-core** | Cognitive pipeline, model pool, session/memory management | 6415 | Python 3.11 |
| **gaia-web** | Discord bot, HTTP API, voice manager, CognitionPacket routing | 6414 | Python 3.11 |
| **gaia-mcp** | Tool registry (file I/O, shell, knowledge, study gateway) | 8765 | Python 3.11 |
| **gaia-study** | Vector index (sole writer), QLoRA adapter training | 8766 | NVIDIA CUDA 12.4 |
| **gaia-audio** | STT (Whisper), TTS (Coqui/espeak-ng), half-duplex GPU management | 8080 | Python 3.11 + NVIDIA GPU |
| **gaia-wiki** | Internal MkDocs Material documentation server | 8080 (internal) | Python 3.11 |

### Candidate SDLC

A parallel candidate stack runs alongside live services for testing changes before promotion. Candidates use the same images with a `-candidate` suffix and port offsets (+1).

```
Live:      6410  7777  6415  6414  8765  8766
Candidate: 6411  7778  6416  6417  8767  8768  8081 (audio)
```

Both stacks share a Docker network (`172.28.0.0/16`) for hybrid testing.

### High Availability

An HA overlay (`docker-compose.ha.yml`) enables hot-standby failover:

- Candidate gaia-core and gaia-mcp run as warm standbys
- gaia-web auto-routes to candidate-core if the primary fails (`CORE_FALLBACK_ENDPOINT`)
- gaia-core auto-routes to candidate-mcp if the primary fails (`MCP_FALLBACK_ENDPOINT`)
- Session state syncs between live and candidate stacks

```bash
# Start HA overlay
./scripts/ha_start.sh

# Maintenance mode (drain to candidate)
./scripts/ha_maintenance.sh

# Stop HA
./scripts/ha_stop.sh
```

## Discord

GAIA runs a Discord bot (discord.py) with text and voice support.

### Text

- Responds to **DMs**, **@mentions**, and messages starting with `gaia,` or `gaia:`
- Sleep-aware: queues messages when asleep, wakes up, then responds
- 2000-character message splitting respecting newline/word boundaries

### Voice

GAIA can join Discord voice channels for real-time voice conversations.

**Commands:**
- `!call` — GAIA joins your current voice channel
- `!hangup` — GAIA disconnects from voice

**Auto-answer:** Whitelisted users trigger automatic join when they enter a voice channel. Manage the whitelist via the dashboard API (`/api/voice/whitelist`).

**Pipeline:** Discord audio (48kHz stereo) -> VAD segmentation -> gaia-audio STT (Whisper) -> gaia-core cognition -> gaia-audio TTS (Coqui) -> Discord playback

## GPU Handoff

The single GPU is shared between inference (gaia-prime) and training (gaia-study) via an orchestrator-driven container stop/start protocol.

### Release (prime -> study): ~1 second

```
Orchestrator stops gaia-prime container
  -> VRAM drops from ~13 GB to ~2 GB (desktop baseline)
  -> Core demotes gpu_prime from model pool, fallback chain activates
  -> Study receives gpu-ready signal, begins QLoRA training
```

### Reclaim (study -> prime): ~60 seconds

```
Study releases CUDA resources
  -> Orchestrator starts gaia-prime container
  -> vLLM loads model from disk (~40-60s cold start)
  -> Core restores gpu_prime in model pool
```

During handoff, GAIA remains responsive via the CPU GGUF model and cloud API fallbacks.

## Sleep/Wake Cycle

GAIA has a biologically-inspired sleep/wake cycle managed by gaia-core:

- **Active** — fully awake, Prime inference available
- **Drowsy** — winding down, still responsive
- **Asleep** — minimal processing, queues incoming messages
- **REM/Dreaming** — background consolidation (QLoRA self-study)

Incoming Discord messages or voice calls trigger a wake signal. While waking, a lightweight Lite model provides stalling responses until Prime is online.

## MCP Tools

Capabilities exposed to the cognitive pipeline through gaia-mcp:

- **File operations** — read, write, list, search (sandboxed allowlists)
- **Knowledge** — embed documents, vector similarity search, memory management
- **Shell** — sandboxed command execution (safe command allowlist)
- **Study gateway** — start/cancel QLoRA training, manage LoRA adapters
- **Fragmentation** — chunked read/write for large content

Sensitive operations (write, shell, index rebuild) require a challenge-response approval workflow.

## Knowledge Ingestion

GAIA can detect and persist structured knowledge from conversations:

- Explicit save commands ("save this about X", "remember this")
- Auto-detection heuristic for campaign/world-building content
- Semantic deduplication (0.85 similarity threshold)
- YAML front matter + markdown output
- Two-phase pipeline: write file, then embed for vector retrieval

## Running

```bash
# Live stack
docker compose up -d

# Candidate stack (parallel testing)
docker compose -f docker-compose.candidate.yml up -d

# Start prime (GPU inference) — uses the "prime" profile
docker compose --profile prime up -d gaia-prime

# Start audio (voice) — uses the "audio" profile
docker compose -f docker-compose.candidate.yml --profile audio up -d gaia-audio-candidate

# GPU handoff test
curl -X POST http://localhost:6410/handoff/prime-to-study \
  -H 'Content-Type: application/json' \
  -d '{"handoff_type":"prime_to_study","reason":"training","timeout_seconds":90}'

# HA failover overlay
./scripts/ha_start.sh
```

## Configuration

Services are configured via environment variables (prefixed per service) with `gaia_constants.json` providing runtime model configs, feature flags, and task instructions. Environment variables always take precedence.

Key env vars:
- `ORCHESTRATOR_CORE_URL` — gaia-core endpoint
- `ORCHESTRATOR_PRIME_URL` — gaia-prime endpoint
- `ORCHESTRATOR_STUDY_URL` — gaia-study endpoint
- `CORE_FALLBACK_ENDPOINT` — HA fallback for gaia-core
- `MCP_FALLBACK_ENDPOINT` — HA fallback for gaia-mcp
- `AUDIO_ENDPOINT` — gaia-audio endpoint (for voice)

## Project Structure

```
GAIA_Project/
  candidates/          # Candidate service source (active development)
    gaia-audio/        # STT/TTS sensory service
    gaia-core/
    gaia-web/
    gaia-mcp/
    gaia-study/
    gaia-prime/
    gaia-orchestrator/
    gaia-common/       # Shared library (CognitionPacket v0.3 protocol)
  gaia-core/           # Live service source (promoted from candidates)
  gaia-web/
  gaia-mcp/
  gaia-study/
  gaia-orchestrator/
  gaia-common/
  gaia-wiki/           # MkDocs Material documentation source
  gaia-models/         # Model weights (SafeTensors, GGUF, AWQ, embeddings)
  knowledge/           # Persistent knowledge base
    blueprints/        # Architecture documentation
    Dev_Notebook/      # Development journals
  scripts/             # Operational scripts (promote, HA, testing)
  docker-compose.yml           # Live stack
  docker-compose.candidate.yml # Candidate stack
  docker-compose.ha.yml        # HA failover overlay
```
