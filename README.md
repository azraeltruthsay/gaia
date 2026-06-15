# GAIA

**A sovereign AI assistant running entirely on local hardware.**

GAIA is a self-hosted, containerized AI system built around a locally-served language model. She operates as a service-oriented architecture where each service owns a single responsibility — cognition, inference, tools, learning, audio, and interface — coordinated by an orchestrator that manages GPU resources and the system-wide Consciousness Matrix.

## Hardware

- **GPU**: NVIDIA RTX 5080 (16 GB VRAM, Blackwell sm_120)
- **Inference Budget**: 70% VRAM (~11.2 GB) for the primary model, remainder for KV cache
- **CPU Fallback**: GGUF models via llama-cpp-python / llama-server when GPU is unavailable

## Models

### Two-Tier Local Inference (Sovereign Duality)

| Tier | Model | Base | Container | Backend | Context |
|------|-------|------|-----------|---------|---------|
| **Core/Operator** | Gemma4-E4B-GAIA-Core-v1 (V3) | google/gemma-4-E4B | gaia-core | GAIA Engine managed — GPU NF4 / CPU GGUF, embedded (:8092) | 8,192 |
| **Prime/Sovereign** | Qwen3-VL-8B-GAIA-Prime-v1 (abliterated) | Qwen3-VL-8B | gaia-prime | GAIA Engine — GPU safetensors / CPU GGUF, LoRA-enabled | 16,384 |

Two model families: Gemma 4 for Core (~8.8 GB on GPU NF4), Qwen3-VL for Prime (~4.6 GB on GPU). Core handles all requests directly (triage, intent, tools, vision, audio, chat); Prime is loaded on GPU only when deep reasoning is needed (FOCUSING gear). Both run **Q4_K_M** GGUF on CPU when off the GPU.

### Model Sourcing

GAIA's models are not included in the repository. Download base models from HuggingFace, then run identity-baking and quantization via gaia-study's QLoRA pipeline.

| Model | HuggingFace Source | Notes |
|-------|-------------------|-------|
| Gemma 4 E4B (base for Core) | [google/gemma-4-E4B](https://huggingface.co/google/gemma-4-E4B) | QLoRA identity-baked → `Gemma4-E4B-GAIA-Core-v1` (V3). Self-concept is weight-baked; volatile facts stay prompt-injected. GGUF quantized for CPU (Q4_K_M). |
| Qwen3-VL-8B (base for Prime) | [Qwen/Qwen3-VL-8B](https://huggingface.co/Qwen/Qwen3-VL-8B) | Azrael identity-aligned + self-abliterated → `Qwen3-VL-8B-GAIA-Prime-v1`. GGUF quantized for CPU (Q4_K_M). |
| all-MiniLM-L6-v2 | [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) | Embedding model for vector search (gaia-study) |
| Qwen3-ASR-0.6B | [Qwen/Qwen3-ASR-0.6B](https://huggingface.co/Qwen/Qwen3-ASR-0.6B) | Speech recognition (gaia-audio) |
| Qwen3-TTS-12Hz-0.6B-Base | [Qwen/Qwen3-TTS-12Hz-0.6B-Base](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-0.6B-Base) | Text-to-speech (gaia-audio) |

Place downloaded models in the `gaia-instance/gaia-models/` directory (adjacent to the source repo). The setup script (`scripts/setup_instance.sh`) creates the expected directory structure.

**Identity baking**: GAIA's QLoRA pipeline fine-tunes base models with identity curriculum, producing the `-GAIA-*` variants. Adapters are stored in `gaia-models/lora_adapters/` and loaded dynamically via the GAIA Engine's `/adapter/load` endpoint.

### Cloud Fallbacks

| Backend | Model | Purpose |
|---------|-------|---------|
| `groq_fallback` | llama-3.3-70b-versatile | Cloud escalation / external fallback (the only cloud backend) |

> OpenAI/Oracle (gpt-4o-mini) is **retired** — GAIA no longer uses OpenAI for any task. Groq is the sole cloud fallback.

### Embedding: all-MiniLM-L6-v2

Sentence-transformer model used by gaia-study for document embeddings and vector similarity search (RAG retrieval).

## Consciousness Matrix

GAIA's GPU lifecycle is a transmission ("the gearbox"): the orchestrator shifts gears via the Consciousness Matrix, moving each tier between GPU, CPU (GGUF), and unloaded to fit the 16 GB budget. The **clutch** is the Neural Handoff protocol — capture the prefix-cache text before a GPU unload, replay it into the CPU backend after load.

| Gear | State | Core | Prime | GPU VRAM |
|------|-------|------|-------|----------|
| **P** | PARKED | CPU (GGUF) | Unloaded | ~0 GB |
| **1** | AWAKE | GPU (NF4) | CPU (GGUF) | ~8.8 GB |
| **1+** | LISTENING | GPU (NF4) + Audio | CPU (GGUF) | ~8.8 GB |
| **2** | FOCUSING | CPU (GGUF) | GPU (Buffered) | ~4.6 GB |
| **S** | SLEEP | CPU (GGUF) | Unloaded | ~0 GB |
| **0** | DEEP_SLEEP | Unloaded | Unloaded | ~0 GB |
| **T** | MEDITATION | Unloaded | Unloaded | Study owns GPU |

Transition flow: `OFF → PARKED → AWAKE ↔ FOCUSING → PARKED`.

## Architecture

```
                   Discord / HTTP
                        |
                   [gaia-web]            Face — UI, API gateway, Discord bot, voice
                     /     \
              [gaia-wiki] [gaia-core]    Library & Brain — docs, cognition, session memory
                              |   \
                       [gaia-prime] [gaia-mcp]   Sovereign & Hands — GPU/CPU inference, tools
                                         |
                                    [gaia-study]   Subconscious — vector indexing, QLoRA training
                                         |
                          [gaia-orchestrator]       Coordinator — GPU scheduling, Consciousness Matrix

              [gaia-audio]               Ears & Mouth — STT (Qwen3-ASR), TTS (Coqui)
              [gaia-doctor]              Immune System — HA watchdog, cognitive battery
              [gaia-monkey]              Chaos Agent — adversarial testing, serenity
              [gaia-translate]           Tongue — multi-language translation (LibreTranslate)
```

### Services (12 Total)

| Service | Role | Port | Runtime |
|---------|------|------|---------|
| **gaia-orchestrator** | GPU scheduling, Consciousness Matrix, container lifecycle | 6410 | Python 3.11 |
| **gaia-prime** | GAIA Engine inference (Sovereign/Prime, GPU/CPU) | 7777 | gaia-engine-base |
| **gaia-core** | Cognitive pipeline, model pool, embedded Core GPU inference + triage | 6415 | Python 3.11 |
| **gaia-web** | Discord bot, HTTP API, voice manager, dashboard | 6414 | Python 3.11 |
| **gaia-mcp** | Tool registry (file I/O, shell, knowledge, study gateway) | 8765 | Python 3.11 |
| **gaia-study** | Vector index (sole writer), QLoRA adapter training | 8766 | CUDA 12.4 |
| **gaia-audio** | STT (Qwen3-ASR), TTS (Coqui), voice processing | 8080 | Python 3.11 + GPU |
| **gaia-doctor** | HA watchdog, cognitive test battery, auto-heal | 6419 | Python 3.12 |
| **gaia-monkey** | Adversarial chaos engine, serenity/meditation | 6420 | Python 3.12 + Node |
| **gaia-wiki** | Internal MkDocs Material documentation server | 8080* | Python 3.11 |
| **gaia-translate** | Multi-language translation (LibreTranslate) | 5100 | C++ / Python |
| **dozzle** | Real-time Docker log viewer | 9999 | Go |

**Infrastructure**: ELK stack (Elasticsearch, Logstash, Kibana, Filebeat) for centralized observability.

### GAIA Inference Engine

All inference tiers run the standalone **GAIA Engine** (separate repository). It provides:
- **5.3x Speedup**: Optimized generation at ~22 tok/s.
- **Hidden State Polygraph**: Real-time activation monitoring.
- **KV Cache Snapshots**: Thought preservation across turns.
- **Dynamic LoRA**: Hot-swapping adapters without reloading the base model.

## Features

### Neural Mind Map (Mindscape)
A 13-region anatomical brain map visualizing real-time neural activity. Each token generation fires lightning-neuron arcs across the tiers, reflecting the current state of consciousness.

### Self-Supervised Coding Skill Loop
GAIA can autonomously improve her own coding capabilities through a curriculum-based loop: generating challenges, evaluating solutions, and baking successful patterns back into her identity via QLoRA.

### Open Knowledge Ingestion
A structured pipeline for ingesting large-scale educational content (e.g., MIT OCW). GAIA classifies, chunks, and embeds documents for long-term epistemic retrieval.

### Routing
Core triages every request and handles intent, tools, vision, audio, and chat directly. When deep reasoning, architecture, code, or planning is needed, the orchestrator shifts to FOCUSING and Prime takes the GPU. Groq (llama-3.3-70b) is the cloud escalation path.

## Discord

GAIA runs a Discord bot (discord.py) with text and voice support.

- **Text**: DM/Mention support, newline-aware splitting, sleep-aware queuing.
- **Voice**: Qwen3-ASR STT -> Core Cognition -> Coqui TTS. Join via `!call`, leave via `!hangup`.

## GPU Handoff

The orchestrator coordinates the transition between Inference (Prime) and Training (Study).
- **Release**: Prime migrates to CPU (Subconscious) or Unloads; Study takes the GPU for QLoRA.
- **Reclaim**: Study releases CUDA; Prime restores from warm pool (/mnt/gaia_warm_pool).

## MCP Tools

Capabilities exposed to the cognitive pipeline through gaia-mcp:
- **File operations** (read, write, list, search)
- **Knowledge** (vector search, semantic codex)
- **Shell** (sandboxed command execution)
- **Study gateway** (training, adapter management)

## Running

```bash
# Live stack
docker compose up -d

# Start prime (GPU inference)
docker compose --profile prime up -d gaia-prime

# GPU handoff test
curl -X POST http://localhost:6410/handoff/prime-to-study \
  -H 'Content-Type: application/json' \
  -d '{"handoff_type":"prime_to_study","reason":"training","timeout_seconds":90}'
```
