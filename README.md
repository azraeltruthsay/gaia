# GAIA

**A sovereign AI assistant running entirely on local hardware.**

GAIA is a self-hosted, containerized AI system built around a locally-served language model. She operates as a service-oriented architecture where each service owns a single responsibility — cognition, inference, tools, learning, audio, and interface — coordinated by an orchestrator that manages GPU resources and the system-wide Consciousness Matrix.

## Hardware

- **GPU**: NVIDIA RTX 5080 (16 GB VRAM, Blackwell sm_120)
- **Inference Budget**: 70% VRAM (~11.2 GB) for the primary model, remainder for KV cache
- **CPU Fallback**: GGUF models via llama-cpp-python / llama-server when GPU is unavailable

## Models

### Three-Tier Local Inference

| Tier | Model | Base | Container | Backend | Context |
|------|-------|------|-----------|---------|---------|
| **Thinker/Prime** | Huihui-Qwen3-8B-GAIA-Prime-adaptive | Qwen3-8B | gaia-prime | GAIA Engine (GPU), LoRA-enabled | 16,384 |
| **Core/Operator** | Qwen3.5-2B-GAIA-Core-v3 | Qwen3.5-2B | gaia-core | embedded Core GPU (:8092) / GGUF (CPU) | 8,192 |
| **Nano/Reflex** | Qwen3.5-0.8B-Abliterated | Qwen3.5-0.8B | gaia-nano | GAIA Engine (GPU primary, GGUF fallback) | 2,048 |

Two model families: Qwen3.5 for Nano/Core, Qwen3 (Huihui abliterated) for Prime. Prime's subconscious mode uses **Q4_K_M** quantization for efficient CPU inference at ~15 tok/s.

### Model Sourcing

GAIA's models are not included in the repository. Download base models from HuggingFace, then run identity-baking and quantization via gaia-study's QLoRA pipeline.

| Model | HuggingFace Source | Notes |
|-------|-------------------|-------|
| Qwen3.5-0.8B-Abliterated | [huihui-ai/Qwen3.5-0.8B-abliterated](https://huggingface.co/huihui-ai/Qwen3.5-0.8B-abliterated) | Base for Nano tier. Few-shot prompted, no fine-tune needed. |
| Qwen3.5-2B (base for Core) | [Qwen/Qwen3.5-2B](https://huggingface.co/Qwen/Qwen3.5-2B) | QLoRA identity-baked → `Qwen3.5-2B-GAIA-Core-v3` |
| Huihui-Qwen3-8B-abliterated | [huihui-ai/Qwen3-8B-abliterated-v2](https://huggingface.co/huihui-ai/Qwen3-8B-abliterated-v2) | QLoRA identity-baked → `GAIA-Prime-adaptive`. GGUF quantized for CPU (Q4_K_M). |
| all-MiniLM-L6-v2 | [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) | Embedding model for vector search (gaia-study) |
| Qwen3-ASR-0.6B | [Qwen/Qwen3-ASR-0.6B](https://huggingface.co/Qwen/Qwen3-ASR-0.6B) | Speech recognition (gaia-audio) |
| Qwen3-TTS-12Hz-0.6B-Base | [Qwen/Qwen3-TTS-12Hz-0.6B-Base](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-0.6B-Base) | Text-to-speech (gaia-audio) |

Place downloaded models in the `gaia-instance/gaia-models/` directory (adjacent to the source repo). The setup script (`scripts/setup_instance.sh`) creates the expected directory structure.

**Identity baking**: GAIA's QLoRA pipeline fine-tunes base models with identity curriculum, producing the `-GAIA-*` variants. Adapters are stored in `gaia-models/lora_adapters/` and loaded dynamically via the GAIA Engine's `/adapter/load` endpoint.

### Cloud Fallbacks

| Backend | Model | Purpose |
|---------|-------|---------|
| `groq_fallback` | llama-3.3-70b-versatile | Free-tier API fallback |
| `oracle_openai` | gpt-4o-mini | High-quality reasoning oracle |

### Embedding: all-MiniLM-L6-v2

Sentence-transformer model used by gaia-study for document embeddings and vector similarity search (RAG retrieval).

## Consciousness Matrix

GAIA operates on a three-state consciousness model, allowing different parts of her brain to maintain independent states of awareness and resource allocation.

| State | Name | Resource | Description |
|-------|------|----------|-------------|
| **3** | Conscious | GPU | High-performance inference (SafeTensors/vLLM) |
| **2** | Subconscious | CPU | Efficient GGUF inference (llama-server) |
| **1** | Unconscious | Unloaded | Resource hibernation |

**Presets:**
- **AWAKE**: Core=3, Nano=3, Prime=2 (Prime observes on CPU)
- **FOCUSING**: Prime=3, Nano=3, Core=2 (Prime handles deep reasoning on GPU)
- **SLEEP**: Nano=2, Core=2, Prime=1
- **DEEP SLEEP**: All→1 (Nano stays 2 for wake detection)
- **TRAINING**: Target tier=1, others=2 (VRAM freed for QLoRA)

## Architecture

```
                   Discord / HTTP
                        |
                   [gaia-web]            Face — UI, API gateway, Discord bot, voice
                     /     \
              [gaia-wiki] [gaia-core]    Library & Brain — docs, cognition, session memory
                          /  |   \
             [gaia-prime] [gaia-nano] [gaia-mcp]   Voice, Reflex & Hands — GPU/CPU inference, tools
                                         |
                                    [gaia-study]   Subconscious — vector indexing, QLoRA training
                                         |
                          [gaia-orchestrator]       Coordinator — GPU scheduling, Consciousness Matrix

              [gaia-audio]               Ears & Mouth — STT (Qwen3-ASR), TTS (Coqui)
              [gaia-doctor]              Immune System — HA watchdog, cognitive battery
              [gaia-monkey]              Chaos Agent — adversarial testing, serenity
              [gaia-translate]           Tongue — multi-language translation (LibreTranslate)
```

### Services (13 Total)

| Service | Role | Port | Runtime |
|---------|------|------|---------|
| **gaia-orchestrator** | GPU scheduling, Consciousness Matrix, container lifecycle | 6410 | Python 3.11 |
| **gaia-prime** | GAIA Engine inference (Thinker/Prime, GPU/CPU) | 7777 | gaia-engine-base |
| **gaia-nano** | Nano/Reflex triage classifier (GAIA Engine managed mode) | 8090 | llama-server + GPU |
| **gaia-core** | Cognitive pipeline, model pool, embedded Core GPU inference | 6415 | Python 3.11 |
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

### Cascade Routing
Nano classifies queries (SIMPLE/COMPLEX) -> Core handles intent and tool selection -> Prime handles heavyweight reasoning and code generation.

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
