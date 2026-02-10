# Dev Journal Entry: 2026-02-07 - GAIA v0.3 Sovereign Sensory Architecture

**Date:** 2026-02-07
**Author:** Claude Code (Opus 4.6) via Happy

## Context and Motivation

GAIA v0.2 has matured into a stable containerized SOA with five services (orchestrator, core, web, mcp, study), a dual-track candidate/live SDLC, and functional GPU inference via vLLM embedded within gaia-core. However, the architecture has three structural limitations that block the next phase of capability:

1. **Inference is coupled to cognition.** gaia-core loads vLLM models directly via `_model_pool_impl.py`. If the cognitive loop restarts (crash, code update, config change), the model must reload — a 30-60s GPU-blocking operation. This makes iteration expensive and couples deployment cadence.

2. **No sensory modality.** GAIA can only read and write text. The RTX 5080 has 5.6GB of headroom beyond Prime's allocation that could run Whisper for hearing and Bark/Coqui for speech, but there's no service architecture for audio I/O.

3. **Observability is ad-hoc.** Logs go to Docker stdout. There's no centralized security monitoring, file integrity checking, or structured alerting. The StreamObserver exists but runs inline in gaia-core — it can't watch gaia-core itself.

v0.3 addresses all three with the "Sovereign Sensory Architecture": decoupled inference (gaia-prime), half-duplex audio (gaia-audio), and holistic observability (gaia-siem).

## Goal

Define a concrete, phased implementation plan for v0.3 that can be executed incrementally without disrupting the running candidate stack. Each phase must leave the system in a working state.

## Architectural Analysis

### What Changes

| Component | v0.2 (Current) | v0.3 (Target) |
|-----------|----------------|---------------|
| Inference | vLLM embedded in gaia-core | Standalone gaia-prime container (vllm/vllm-openai) |
| gaia-core | Loads models locally via ModelPool | HTTP client to gaia-prime's OpenAI-compatible API |
| Audio | None | gaia-audio: Whisper (STT) + Bark/Coqui (TTS), half-duplex model-swapping |
| Observer | StreamObserver inline in gaia-core | Bicameral: Lite model in gaia-core watches Prime's stream from gaia-prime |
| GPU split | 0.85 utilization, all to gaia-core | 0.65 gaia-prime + 0.35 gaia-audio |
| Observability | Docker logs only | gaia-siem (Wazuh): FIM, Docker socket, audit logs |
| Memory writes | gaia-study is sole writer (by convention) | Enforced via MemoryRequest relay protocol |
| Web search | Not implemented | Web-RAG via gaia-mcp tool, Tier 0 ephemeral context |

### What Stays the Same

- CognitionPacket as the central data structure
- gaia-web as the Unified Interface Gateway (Discord, HTTP)
- gaia-mcp as sandboxed tool execution
- gaia-study as exclusive vector store / LoRA writer
- gaia-orchestrator as container lifecycle manager
- Candidate/live dual-track SDLC
- gaia-common shared library

### Critical Path Dependencies

```
Phase 1 (gaia-prime) ──► Phase 2 (core refactor) ──► Phase 3 (observer)
                                                   ──► Phase 5 (audio, can start earlier)
Phase 4 (memory relay) is independent
Phase 6 (siem) is independent
```

## Key Design Decisions

### 1. gaia-prime as vllm/vllm-openai
Using the official `vllm/vllm-openai` Docker image means:
- Zero custom code for inference serving
- Standard OpenAI-compatible API (`/v1/chat/completions`, `/v1/models`)
- Streaming via SSE out of the box
- `--gpu-memory-utilization 0.65` as a launch flag
- Model path mounted as volume from `/gaia/GAIA_Project/gaia-models/`

Port: `7777` internally, exposed as `7777` on host.

### 2. gaia-core ModelPool becomes HTTP adapter
The current `_model_pool_impl.py` (662 lines) manages local vLLM, GGUF, and API models. For v0.3:
- `gpu_prime` and `prime` model types → replaced with `httpx.AsyncClient` calls to `http://gaia-prime:7777/v1`
- `lite` (GGUF via llama.cpp) → **kept locally in gaia-core** for the Bicameral Observer
- API fallbacks (groq, openai, gemini) → kept as-is
- GPU memory management code → removed from gaia-core (gaia-prime owns it)

### 3. Bicameral Observer stays in gaia-core
The spec says Lite validates Prime's output. Since gaia-core already has the StreamObserver and llama.cpp for the lite model, the simplest architecture is:
- gaia-core streams tokens from gaia-prime via SSE
- gaia-core feeds each chunk to the local Lite model for validation
- Kill switch remains in gaia-core (it controls the conversation, it can abort)

This avoids putting two models in gaia-prime and keeps the observer independent.

### 4. gaia-audio: half-duplex model swapping in 5.6GB
- Whisper-Large-v3-Turbo fits in ~3GB VRAM
- Bark (Small) / Coqui fits in ~2-3GB VRAM
- Never loaded simultaneously — half-duplex swap
- 10s capture chunks, 12s transcription window (2s overlap for seam dedup)
- Speech directives (`<voice tone="warm">`) parsed from gaia-core responses

### 5. VRAM budget (RTX 5080, 16GB)
```
gaia-prime:  10.4 GB (0.65 × 16GB)  — persistent, always loaded
gaia-audio:   5.6 GB (0.35 × 16GB)  — model-swapping (Whisper OR TTS)
gaia-core:    0 GB GPU               — CPU only (lite model on CPU, HTTP to prime)
gaia-study:   0 GB GPU               — CPU embeddings via sentence-transformers
```

## Files Modified / Created (Planned)

### Phase 1: gaia-prime
- `docker-compose.yml` — add gaia-prime service
- `docker-compose.candidate.yml` — add gaia-prime-candidate service
- `gaia-orchestrator/gaia_orchestrator/config.py` — add prime_url, vram quotas
- `gaia.sh` — add prime subcommand

### Phase 2: Core refactor
- `candidates/gaia-core/gaia_core/models/_model_pool_impl.py` — replace vLLM with HTTP client
- `candidates/gaia-core/gaia_core/gaia_constants.json` — update MODEL_CONFIGS
- `candidates/gaia-core/gaia_core/config.py` — add PRIME_ENDPOINT
- `docker-compose.candidate.yml` — remove GPU reservation from gaia-core-candidate

### Phase 3: Bicameral Observer
- `candidates/gaia-core/gaia_core/cognition/external_voice.py` — SSE streaming from prime
- `candidates/gaia-core/gaia_core/cognition/agent_core.py` — observer integration

### Phase 4: Memory relay
- `candidates/gaia-common/gaia_common/protocols/` — MemoryRequest schema
- `candidates/gaia-study/` — `/memory/request` endpoint
- `candidates/gaia-core/gaia_core/cognition/agent_core.py` — POST to study on "remember"

### Phase 5: gaia-audio
- New `gaia-audio/` service directory
- `docker-compose.yml` — add gaia-audio service
- Audio capture, Whisper transcription, TTS synthesis
- CognitionPacket extensions for audio routing

### Phase 6: gaia-siem
- `docker-compose.yml` — add Wazuh stack
- FIM rules for `/knowledge/system_reference/`
- Docker socket monitoring
- Alert → CognitionPacket integration

## Risk Assessment

- **Phase 1 is lowest risk** — adds a new service, doesn't change existing behavior
- **Phase 2 is highest risk** — rewrites the model pool, changes how every inference call works
- **Phase 3 is medium risk** — changes output validation, could cause false positives/negatives
- **Phases 4-6 are low risk** — additive capabilities, don't modify core inference path

## Testing Strategy

Each phase validates independently:
1. `curl http://localhost:7777/v1/models` — gaia-prime responds
2. Send Discord message → response comes from gaia-prime via HTTP (not local vLLM)
3. Observer catches a test violation in Prime's output
4. gaia-core "remembers" a fact → appears in gaia-study's vector store
5. Speak into microphone → transcription appears in logs
6. Modify a file in `/knowledge/system_reference/` → Wazuh alert fires
