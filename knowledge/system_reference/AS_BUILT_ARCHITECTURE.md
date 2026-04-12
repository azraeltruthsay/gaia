# GAIA As-Built Architecture Reference

> **Last updated**: 2026-04-11 | **Era**: Sovereign Awareness (Phase 5) | **Consolidated Core**

This document reflects the actual running state of GAIA, following the **Great Consolidation (Phase 5-C)** and the **Gemma 4 Chord** deployment.

## Service Topology (Consolidated)

| Service | Role | Core Components |
|---------|------|-----------------|
| `gaia-core` | The Brain | **Neural Router**, Reasoning Loop, Embedded Inference |
| `gaia-prime` | The Voice | Gemma 4 26B-A4B MoE (vLLM/GAIA Engine) |
| `gaia-nano` | The Reflex | Qwen 3.5 0.8B (Sub-100ms Triage & Time) |
| `gaia-mcp` | The Hands | **Capability Engine (Limbs)**, Tool Execution |
| `gaia-study` | The Subconscious | **DocSentinel**, QLoRA, Vector Indexing |
| `gaia-common`| The Nervous System | **GaiaVitals**, Shared Protocols (Packet v0.4) |
| `gaia-web` | The Face | Unified UI (Discord/Web), Dashboard |

## Model Tiers (The Gemma Chord)

| Tier | Model | Active Params | Backend | Role |
|------|-------|---------------|---------|------|
| **Sovereign** | Gemma 4 26B-A4B MoE | 4B | GAIA Engine (GPU) | Deep Reasoning, Architecture, Sovereign Identity |
| **Reflex** | Qwen 3.5 0.8B-Abliterated| 0.8B | GAIA Engine (GPU) | Triage, Time Checks, Force Field Translation |

## Cognitive Infrastructure

### 1. Neural Router (Unified Intent)
The **Neural Router** is the single entry point for all cognitive requests. It uses a 6-stage pipeline (Reflex → Embed → Heuristic → Keyword → Weighted → Nano Tiebreak) to determine the intent and target engine with maximum efficiency.

### 2. Capability Engine (The Limbs)
All external actions are unified under the **Capability Engine**. Whether static (Domain Tools) or dynamic (Memento Skills), every action is a **Limb** with consistent metadata and security enforcement.

### 3. GaiaVitals (Unified Pulse)
A centralized health monitoring system that aggregates Biological (heartbeat), Structural (log MRI), Cognitive (loop recovery), and Security (adversarial awareness) pulses into a single **Sovereign Health Score**.

### 4. DocSentinel (Living Documentation)
An automated documentation loop that parses system events and council decisions to update the Wiki, Glossary, and AS_BUILT logs in real-time.

## Consciousness Matrix (GPU Lifecycle)

GAIA manages GPU resources through three consciousness states:
- **Conscious**: Active inference (Sovereign/Reflex tiers).
- **Subconscious**: Background processing (Study/Indexing).
- **Unconscious**: Fully unloaded, zero VRAM usage.

## Known Gaps
- Abliteration pass for 26B-A4B pending (Phase 5j).
- Relational Autonomy (Phase 6) in design.

## Consciousness Matrix (GPU Lifecycle FSM)

Three consciousness states, managed by orchestrator at `/consciousness/*`:

| State | GPU | Inference | Use Case |
|-------|-----|-----------|----------|
| **Conscious** | Yes | Full speed | Active tier (Nano, Core default) |
| **Subconscious** | No | CPU/GGUF fallback | Background observer (Prime default) |
| **Unconscious** | No | None | Fully unloaded, zero VRAM |

Lifecycle FSM states: AWAKE, FOCUSING, SLEEP, DEEP_SLEEP, MEDITATION.
**FOCUSING auto-transition**: When escalation triggers Prime, orchestrator transitions AWAKE -> FOCUSING, which swaps GPU to Prime via quality gate, then returns GPU to Core when done.

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
