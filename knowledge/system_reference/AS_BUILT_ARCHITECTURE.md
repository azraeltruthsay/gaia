# GAIA As-Built Architecture Reference

> **Last updated**: 2026-06-12 | **Era**: Sovereign Duality (two-tier) | partially refreshed
>
> ⚠️ **Service/model topology reference.** For the **cognitive architecture** (the developing mind —
> CFR, affect, Observer, Samvega, the loop audit), see
> `knowledge/blueprints/COGNITIVE_ARCHITECTURE.md`. Authoritative current model identity lives in
> `GAIA_Project/CLAUDE.md` — and **verify a tier against disk + `/health`, not env strings**
> (`PRIME_MODEL_PATH` has lied). Sections below the tables (SDH, VRAM, consciousness FSM) are
> structurally current but predate the Qwen Prime cutover — treat their "26B" figures as historical.

This document reflects GAIA's service/model topology in the **Sovereign Duality** era (Gemma4 Core +
Qwen3-VL-8B Prime; `gaia-nano` removed).

## Service Topology (Consolidated)

| Service | Role | Core Components |
|---------|------|-----------------|
| `gaia-core` | The Brain | **Neural Router**, Reasoning Loop, Embedded Inference |
| `gaia-prime` | The Voice / Sovereign | **Qwen3-VL-8B-GAIA-Prime** (abliterated, vision-language) — GAIA Engine (GPU safetensors / CPU GGUF) |
| ~~`gaia-nano`~~ | ~~The Reflex~~ | **DEPRECATED / removed** — Core handles all triage (Sovereign Duality) |
| `gaia-mcp` | The Hands | **Capability Engine (Limbs)**, Tool Execution |
| `gaia-study` | The Subconscious | **DocSentinel**, QLoRA, Vector Indexing |
| `gaia-common`| The Nervous System | **GaiaVitals**, Shared Protocols (Packet v0.5) |
| `gaia-web` | The Face | Unified UI (Discord/Web), Dashboard |

## Model Tiers (The Sovereign Duality)

| Tier | Model | Backend | VRAM | Role |
|------|-------|---------|------|------|
| **Sovereign / Prime** | **Qwen3-VL-8B-GAIA-Prime** (abliterated, vision-language) | GAIA Engine (GPU safetensors / CPU GGUF default) | ~4.6 GB | Deep reasoning, architecture, code, planning |
| **Operator / Core** | **Gemma4-E4B-GAIA-Core** (V3 identity-baked, multimodal) | GAIA Engine (GPU NF4 / CPU GGUF) | ~8.8 GB | Always-on triage, intent, tools, vision, audio, chat |

*Identity is both weight-baked (Core V3 self-concept) and prompt-anchored (`prompt_builder.py` arch facts).
Cloud fallback: Groq llama-3.3-70b. OpenAI is retired. Older "26B Prime" claims were wrong (corrected
2026-06-09) — Prime is Qwen3-VL-8B.*

## Consciousness Matrix (The Sovereign Duality)

GAIA manages her 16GB VRAM budget by maintaining the Operator (E4B) as Always-On Conscious, with the Sovereign (26B) loadable on demand.

| Preset | Operator (E4B) | Sovereign (26B) | Use Case |
|--------|----------------|-----------------|----------|
| **awake** | CONSCIOUS | SUBCONSCIOUS | Standard idle / daily chat |
| **focusing** | SUBCONSCIOUS | CONSCIOUS | Sovereign reasoning / coding |
| **parked** | SUBCONSCIOUS | UNCONSCIOUS | Zero-VRAM boot / Sentinel mode |

### VRAM Budgeting
- **Baseline (Desktop)**: ~1.5GB
- **Operator (E4B)**: 3.5GB (Always-On)
- **Free for Context**: ~2.1GB
- **Sovereign (26B)**: 8.9GB (Active)
- **Total GPU Ceiling**: 16.0GB (RTX 5080)

## The SDH Protocol (Neural Note)

To ensure seamless continuity during tier shifts (e.g. GPU Core -> CPU Sentinel), GAIA uses the **Sovereign Duality Handoff (SDH)** protocol.

*   **AAAK Handoff Buffer**: A sliding window of the last 3 turns stored in the `CognitionPacket.Reasoning.handoff_buffer`.
*   **Neural Note**: Each turn is distilled into an AAAK fragment (`U` intent, `L` limb result, `O` operator conclusion).
*   **Priming**: The CPU-Sentinel prompt is injected with the handoff buffer to bypass "restart amnesia" and maintain high-density context.

## Known Gaps
- DocSentinel full automation (sleep cycle integration) pending.
- Relational Autonomy (Phase 6) in design.

## Consciousness Matrix (GPU Lifecycle FSM)

Four consciousness states, managed by orchestrator at `/consciousness/*`:

| State | GPU | Inference | Use Case |
|-------|-----|-----------|----------|
| **Conscious** | Yes | Full speed | Active tier (Core default) |
| **Subconscious** | No | CPU/GGUF fallback | Background observer (Prime/Sentinel) |
| **Unconscious** | No | None | Fully unloaded, zero VRAM |

Lifecycle FSM states: AWAKE, FOCUSING, PARKED, SLEEP, DEEP_SLEEP, MEDITATION.
**PARKED (Idle GPU)**: System boots into PARKED state with Core on CPU.
**GEAR 1 (Engaging Clutch)**: Transition from PARKED to AWAKE loads Core Safetensors onto GPU with SAE sensors and LoRAs.

Tested transitions: AWAKE <-> FOCUSING (2026-03-25). Lifecycle FSM syncs with Consciousness Matrix states.

## Neural Brain Map (13 Regions)

The dashboard renders a sagittal brain visualization with 13 anatomical regions across 3 tiers:

**Prime tier (frontal cortex, 4 regions):**
- Prefrontal (layers 24-31) — reasoning, architecture
- Orbitofrontal (layers 16-24) — safety, emotion
- Broca's Area (layers 8-16) — creative
- Motor Cortex (layers 4-12) — code

**Core tier (mid-brain, 6 regions):**
- Somatosensory (layers 0-8) — identity
- Parietal (layers 8-16) — architecture, reasoning
- Wernicke's Area (layers 14-22) — identity
- Temporal (layers 0-10) — emotion, factual
- Occipital (layers 18-24) — factual
- Visual Cortex (layers 22-24)

**Nano tier (brainstem, 3 regions):**
- Thalamus (layers 8-16) — time, architecture
- Cerebellum (layers 16-23) — code
- Brain Stem (layers 0-8) — time, identity

### SAE Atlases and Causal Connectivity

SAE (Sparse Autoencoder) atlases have been trained for all 3 tiers. Each atlas maps hidden state features to interpretable concepts (neuron labels). Atlases feed the brain visualization with named feature activations rather than raw neuron indices.

**Lightning-bolt neurons**: Directed causal pathways are rendered as lightning-bolt arcs between regions. These show which region's activation causally drives another region's response, based on SAE atlas causal connectivity analysis. Arcs are animated during inference to show information flow direction.

## GAIA Inference Engine

Separate repo: `github.com/azraeltruthsay/gaia-engine` (Apache-2.0).
All three tiers use GAIA Engine managed mode.

### Recent Fixes (2026-03-25)
- **Streaming fix**: Removed `Transfer-Encoding: chunked` header that caused buffering issues with some proxies. SSE streaming now works cleanly.
- **ThreadingHTTPServer**: Manager uses `_ThreadingHTTPServer` (ThreadingMixIn + HTTPServer) so long-running requests (model load/unload) don't block health probes.
- **Dead worker detection**: Manager detects when worker process exits unexpectedly and resets state cleanly instead of leaving stale process references.

### Key Capabilities
- Hidden state polygraph (activation monitoring at every layer)
- KV prefix caching (sub-100ms on cache hits)
- LoRA hot-swap without model restart
- SAE atlas training for feature discovery
- ROME editing for surgical weight corrections
- GPU lifecycle state machine (AWAKE/FOCUSING/SLEEP/DEEP_SLEEP/MEDITATION)
- Subprocess isolation (zero-GPU standby, guaranteed VRAM release on unload)

## Known Gaps
- 9B training blocked on 16GB VRAM (RTX 5080). Needs cloud GPU or gptqmodel fix.
- CPU activation streaming not available for GGUF tiers (llama-server has no polygraph).
- Prime defaults to CPU/GGUF; GPU-mode Prime requires FOCUSING transition.
