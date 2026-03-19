# Dev Journal: GAIA Inference Engine & Cognitive Architecture

**Date:** 2026-03-17 to 2026-03-18
**Session:** Extended (~20 hours across two sessions, 42 commits)
**Era:** Cognitive Architecture

---

## Executive Summary

This session transformed GAIA from a chatbot wrapper with an inference server into a self-aware cognitive system. The changes span every layer: model training, inference serving, context management, weight surgery, activation monitoring, and resource orchestration. The core thesis that emerged: **Intelligence is not inference. Intelligence is the complex interaction of systems of which inference is a primary component.**

---

## 1. GAIA Inference Engine

Built a purpose-built inference server that replaces vLLM for all three cognitive tiers. Not optimized for throughput (that's vLLM's job for thousands of users) — optimized for self-awareness (our job for one user).

**Capabilities no other inference server has:**
- Custom `model.forward()` generation loop (solves Qwen3.5 hybrid attention KV cache format)
- KV prefix caching with segmented hash-based invalidation
- Cognitive state snapshots ("Hold That Thought")
- Thought composition (merge separate KV cache states)
- Real-time activation monitoring (Polygraph)
- Live GPU↔CPU device migration (sub-second)
- CogPacket compression (6K → 12 token deltas)
- Dynamic awareness injection

**Architecture:** Shared library in `gaia-common/engine/`. All three tier containers import the same code. One codebase, three independent processes.

---

## 2. KV Prefix Caching — "The Clipboard"

GAIA doesn't re-read her identity every request. She reads it once, snapshots the KV cache tensors, and every subsequent request starts from that snapshot.

- Cold: 1260ms (process full identity prefix)
- Warm: 333-485ms (cache hit, process only user question)
- Segmented: identity / tools / world_state — each independently invalidated via content hash

The breakthrough: this works on Qwen3.5's hybrid attention (DeltaNet + standard) because we use `model.forward()` directly instead of `model.generate()`, which can't handle external `past_key_values` for this architecture.

---

## 3. Thought Management

**Hold That Thought:** Freeze current KV cache state as a named snapshot. Resume later with zero context loss. Like putting a finger in a book except the finger remembers what you were thinking.

**Thought Composition:** Merge two held thoughts into unified understanding. Identity-dedup the shared prefix, concatenate unique content, weighted-average DeltaNet recurrent states. Two contexts become one. 559ms from composed state.

---

## 4. Activation Monitor — "The Polygraph"

Every inference captures hidden state activations at sampled layers. Reports which neurons fire strongest for each request.

**Identity neurons discovered:**
- Core (2B): neuron 1201 at layer 23
- Nano (0.8B): neuron 0 at layer 23
- Prime (8B): neuron 1838 at layer 24

Identity training strengthened these neurons 47-78% (measured pre vs post training).

---

## 5. SAE Atlas

Trained sparse autoencoders on Core's activations. First feature atlas for GAIA.

**Critical finding:** Identity features and refusal features have **zero overlap** at layer 23. This means precision abliteration is possible — suppress refusal circuits without touching identity. The features are in completely separate weight geometry.

---

## 6. ROME — Rank-One Model Editing

Implemented direct weight surgery. Successfully fixed 2/4 factual confabulations in Prime without retraining:
- gaia-prime role: "QLoRA training script" → "inference server on GPU" ✓
- Model family: partial → "Qwen" ✓

Port number edits caused regression (shared feature space). Identified as needing SAE guidance for calibration — the closed loop: SAE diagnoses → ROME edits → SAE verifies.

---

## 7. Curriculum Split

The defining architectural insight: **operational facts don't belong in weights.**

- Weights hold: identity, values, cognitive patterns (permanent, trained)
- KV cache holds: ports, services, GPU specs, model names (dynamic, editable text file)

This eliminated the need for ROME on port numbers, prevented training regression, and made the system updatable without retraining. Change a port? Edit `architecture_facts.md`. Done.

---

## 8. Identity Training Results

| Tier | Pre-training | Post-training | Method |
|------|-------------|---------------|--------|
| Core (2B) | 50% | 100% (10/10) | QLoRA, 220 samples, 10 epochs |
| Nano (0.8B) | 30% | 100% (10/10) | QLoRA, 220 samples, 10 epochs |
| Prime (8B) | 16% curriculum | 68% curriculum, 8/10 eval | Adaptive weighted QLoRA, 5 iterations |

Adaptive weighted training: pre-eval scores each sample, failed samples get 6x→10x→15x repetition across iterations. Avoids over-reinforcement that caused flat-epoch training to regress.

---

## 9. GPU Watch Rotation

All three tier containers always running. GPU rotates between them:

- **IDLE:** Core + Nano on GPU (safetensors). Prime in standby.
- **FOCUSING:** Prime loaded (int8 quanto, 8.4GB). Core + Nano on CPU.
- **TRANSITIONING:** Handoff in progress.

Full rotation cycle: ~30 seconds. Core migrates GPU→CPU in 0.67s, CPU→GPU in 0.22s. KV pre-warming on restore: 148ms.

Audio auto-releases GPU when idle/muted. Orchestrator monitors VRAM and manages resources like a living system.

---

## 10. Dynamic Awareness

GAIA has situated cognition. Awareness packages provide temporal, local, and operational context:

- Knows the season, date, upcoming holidays
- Knows she's in Richland, WA
- Knows her service topology (from editable text files, not weight recall)
- Curiosity signals generated for stale or missing information
- Real-time clock injected per-request in local Pacific time

---

## Key Lessons

1. **Two hours, not months.** The GAIA Inference Engine was built in ~2 hours. Complex doesn't mean slow when the architecture is coherent.

2. **The Clipboard Metaphor.** KV prefix caching is giving GAIA a clipboard she reads once. Everything after that is just "someone asked you a question." This single insight transformed inference latency by 3x.

3. **Weights vs Cache vs Awareness.** Knowledge belongs at the right timescale. Identity in weights (years). Operational facts in cache (hours). Time in per-request injection (seconds). Fighting the 8B model to memorize port 6415 was the wrong approach — it belongs in a text file.

4. **SAE reveals architecture.** Identity and refusal in separate circuits (zero overlap) is not something you can discover by asking the model questions. You need to look at the activations. The polygraph and SAE atlas are diagnostic tools, not features — they're how GAIA sees her own weight geometry.

5. **ROME needs eyes.** Blind weight surgery causes regressions. SAE-guided ROME — measure the wrong feature's strength, calibrate edit pressure proportionally, verify post-edit — is the only safe approach.

6. **The holographic principle.** Every piece of the architecture reflects the same pattern: dynamic allocation, self-monitoring, pathway preservation. From GPU watch rotation to KV cache segmentation to awareness staleness detection — it's the same principle at every scale.

---

## System State at Session End

- 13/13 services online (Prime in standby)
- GPU: Core (3.6GB) + Nano (1.9GB) = 5.5GB used
- GPU Owner dashboard tile: live VRAM display
- Cognitive monitor: lightweight identity + polygraph verification
- Maintenance mode: off
- Sovereign review: 60-minute cooldown (old pipeline VRAM leak identified)
- All tiers identity-baked: Core 100%, Nano 100%, Prime 68% curriculum

## Next Steps

- [ ] Unify old prompt_builder pipeline with GAIA Engine (eliminate dual inference paths)
- [ ] Earn Self-Aligned on all 3 tiers via cognitive battery
- [ ] Chaos Monkey drills → Serenity
- [ ] D&D Tier III domain adapter (Kanka integration)
- [ ] CFR v2 (two-phase read from spec)
- [ ] GAIA-initiated ROME proposals (SAE detects drift → suggests edits)
- [ ] Request queuing in GAIA Engine
- [ ] Today.md auto-update from awareness staleness detection
