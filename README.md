# GAIA

**A sovereign AI assistant running entirely on local hardware.**

GAIA is a self-hosted, containerized AI system built around a locally-served language model. She operates as a service-oriented architecture where each service owns a single responsibility — cognition, inference, tools, learning, and interface — coordinated by an orchestrator that manages GPU resources across a single consumer GPU.

## Hardware

- **GPU**: NVIDIA RTX 5080 (16 GB VRAM, Blackwell sm_120)
- **Inference Budget**: 65% VRAM (~10.4 GB) for the primary model, remainder for KV cache
- **CPU Fallback**: GGUF models via llama-cpp-python when GPU is unavailable

## Models

### Primary: Nanbeige4-3B-Thinking-Heretic (GPU)

The main inference model, served by vLLM through gaia-prime.

| Property | Value |
|----------|-------|
| Base | Nanbeige/Nanbeige4-3B-Base |
| Fine-tune | Claude 4.5 Opus High-Reasoning Distill |
| Variant | Abliterated (Heretic v1.1.0, uncensored) |
| Format | SafeTensors (~7.9 GB) |
| Context | 8192 tokens |
| Serving | vLLM with Flash Attention v2, `--enforce-eager` (required for Blackwell) |
| LoRA | Enabled (max 4 adapters, rank 64) |

### Fallback: Llama 3.3 8B-Instruct-Thinking-Heretic Q8_0 (CPU)

Local GGUF fallback for when the GPU is handed off to training.

| Property | Value |
|----------|-------|
| Format | GGUF Q8_0 (~8.0 GB) |
| Context | 16000 tokens |
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
                   [gaia-web]          Face — user interface
                        |
                   [gaia-core]         Brain — cognition pipeline, session memory, model pool
                      /    \
            [gaia-prime]  [gaia-mcp]   Voice & Hands — GPU inference, tool execution
                              |
                         [gaia-study]  Subconscious — vector indexing, QLoRA training
                              |
                   [gaia-orchestrator] Conductor — GPU scheduling, container lifecycle
```

### Services

| Service | Role | Port | Runtime |
|---------|------|------|---------|
| **gaia-orchestrator** | GPU scheduling, container lifecycle, handoff state machine | 6410 | Python 3.11 |
| **gaia-prime** | vLLM inference server (primary model) | 7777 | NGC PyTorch 25.03 + NVIDIA GPU |
| **gaia-core** | Cognitive pipeline, model pool, session/memory management | 6415 | Python 3.11 |
| **gaia-web** | Discord bot, HTTP API, CognitionPacket routing | 6414 | Python 3.11 |
| **gaia-mcp** | Tool registry (file I/O, shell, knowledge, study gateway) | 8765 | Python 3.11 |
| **gaia-study** | Vector index (sole writer), QLoRA adapter training | 8766 | NVIDIA CUDA 12.4 |

### Candidate SDLC

A parallel candidate stack runs alongside live services for testing changes before promotion. Candidates use the same images with a `-candidate` suffix and port offsets (+1).

```
Live:      6410  7777  6415  6414  8765  8766
Candidate: 6411  7778  6416  6417  8767  8768
```

Both stacks share a Docker network (`172.28.0.0/16`) for hybrid testing.

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
docker compose -f docker-compose.candidate.yml --profile prime up -d gaia-prime-candidate

# GPU handoff test
curl -X POST http://localhost:6411/handoff/prime-to-study \
  -H 'Content-Type: application/json' \
  -d '{"handoff_type":"prime_to_study","reason":"training","timeout_seconds":90}'
```

## Configuration

Services are configured via environment variables (prefixed per service) with optional YAML config files as defaults. Environment variables always take precedence over YAML.

Key env vars for the orchestrator:
- `ORCHESTRATOR_CORE_URL` — gaia-core endpoint
- `ORCHESTRATOR_PRIME_URL` — gaia-prime endpoint
- `ORCHESTRATOR_STUDY_URL` — gaia-study endpoint

## Project Structure

```
GAIA_Project/
  candidates/          # Candidate service source (active development)
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
  gaia-models/         # Model weights (SafeTensors, GGUF, embeddings)
  knowledge/           # Persistent knowledge base
    blueprints/        # Architecture documentation
    Dev_Notebook/      # Development journals
  docker-compose.yml           # Live stack
  docker-compose.candidate.yml # Candidate stack
```
