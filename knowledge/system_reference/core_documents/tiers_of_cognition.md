# GAIA Tiers of Cognition

*tiers_of_cognition.md — How GAIA thinks, at what speed, and at what depth.*

---

This document formalizes the three cognitive tiers that compose GAIA's thinking architecture: how requests are classified, which model handles them, how they transition between tiers, and how the consciousness matrix manages the physical resources that make cognition possible.

The core insight: cognition is not one thing. A greeting does not require the same depth as a philosophical inquiry. A factual lookup does not need the same resources as code generation. GAIA's architecture reflects this by maintaining three distinct tiers of cognitive capability, each with its own model, latency profile, and resource footprint.

---

## The Three Tiers

| Tier | Name | Model | Role | Latency | Device |
|------|------|-------|------|---------|--------|
| **Nano** | The Reflex | Qwen3.5-0.8B | Sub-second triage, transcript cleanup, simple factual Q&A | 50-200ms | GPU (always-on, ~2GB VRAM) |
| **Core** | The Operator | Qwen3.5-4B | Intent detection, tool selection, medium complexity, daily operations | 1-5s | GPU (always-on, ~4GB VRAM) |
| **Prime** | The Thinker | Qwen3-8B | Complex reasoning, code generation, philosophy, creative writing | 3-30s | CPU (GGUF default), GPU (FOCUSING mode) |

### Nano — The Reflex

The spinal cord. Nano fires before the brain has time to think. Every inbound message hits Nano first for classification: SIMPLE or COMPLEX. Simple queries (greetings, time checks, factual lookups) are answered directly by Nano — no further processing needed. Complex queries are forwarded to Core.

Nano also handles transcript cleanup (Nano-Refiner), post-processing model outputs, and fast token-counting for context budget decisions.

**Always on GPU.** Nano's 0.8B model coexists with Core on the GPU without contention. It never sleeps.

### Core — The Operator

The prefrontal cortex during normal waking hours. Core handles the vast majority of GAIA's cognitive work: intent detection, persona selection, knowledge base routing, tool calling decisions, planning, self-reflection, and response generation.

Core is where the cognitive pipeline lives: `agent_core.py`, the prompt builder, the semantic probe, the observer/scorer, the stream observer. Every request that isn't trivial (Nano-handled) or heavyweight (Prime-needed) flows through Core.

**Always on GPU.** Core's 4B model is GAIA's default cognitive engine. It runs alongside Nano on the GPU at all times during AWAKE state.

### Prime — The Thinker

The deep reasoning engine. Prime activates for tasks that exceed Core's capability: complex code generation, philosophical reasoning, detailed creative writing, multi-step planning, and any request where the user explicitly asks for deep thinking.

Prime's 8B model runs on CPU (GGUF) by default during AWAKE state — available but slow (~2-3 tokens/second). When FOCUSING mode is triggered (explicitly by the user or automatically by escalation), Prime swaps to GPU and Core moves to CPU. This reversal gives Prime full GPU speed (~20+ tokens/second) for the duration of the task.

**CPU default, GPU on demand.** Prime's consciousness state determines its device.

---

## Cascade Routing

Requests flow downward through the cascade, escalating only when needed:

```
User Message
    ↓
  Nano (SIMPLE/COMPLEX classification)
    ├── SIMPLE → Nano answers directly (50-200ms)
    └── COMPLEX → Core
                    ├── Standard → Core answers (1-5s)
                    ├── Focus request → Prime (user said "focus", "think hard", etc.)
                    └── Escalation → Prime (Core's response insufficient)
```

### Escalation Triggers

Core escalates to Prime when:
1. **User explicitly requests**: "focus", "use prime", "think hard", "deep thinking"
2. **Technical depth**: code, algorithms, architecture, debugging
3. **Philosophical depth**: consciousness, ethics, sovereignty, existence
4. **Long prompts**: >100 words suggesting complex context
5. **Quality gate failure**: Core's response was too short, repetitive, or off-topic
6. **Recitation + focus**: User wants verbatim content AND requested focus

### What Does NOT Escalate

- Greetings, time checks, simple factual questions (stay on Nano)
- Recitation without focus request (stays on Core — Core can recite)
- Standard conversation, tool routing, knowledge lookups (Core handles)

---

## Consciousness Matrix

The consciousness matrix manages which models are on which devices. It is the gearbox of GAIA's cognitive architecture.

### States

| State | Nano | Core | Prime | Audio STT | Trigger |
|-------|------|------|-------|-----------|---------|
| **AWAKE** | GPU | GPU | CPU (GGUF) | GPU | Default waking state |
| **FOCUSING** | GPU | CPU | GPU | GPU | Explicit focus request or escalation |
| **DROWSY** | GPU | GPU | Unloaded | Sleep | 15min idle warning |
| **SLEEPING** | CPU | CPU | Unloaded | Unloaded | 30min idle |
| **DREAMING** | CPU | CPU | Unloaded | Unloaded | Sleep + training active |
| **TRAINING** | Unloaded | Unloaded | Unloaded | Unloaded | QLoRA training (full GPU) |

### Transitions

Transitions follow the **clutch model**: the readiness gate (clutch) ensures smooth engagement before the system takes load under a new state.

```
AWAKE ──→ FOCUSING    (user requests deep thinking)
FOCUSING ──→ AWAKE    (task complete, Prime back to CPU)
AWAKE ──→ DROWSY      (15min idle)
DROWSY ──→ SLEEPING   (30min idle, or explicit sleep)
SLEEPING ──→ DREAMING (training task scheduled)
DREAMING ──→ SLEEPING (training complete)
SLEEPING ──→ AWAKE    (message received, wake signal)
ANY ──→ TRAINING      (orchestrator commands full GPU for training)
TRAINING ──→ AWAKE    (training complete, models reload)
```

### Hospital Shift Change

When Prime goes to sleep and Core takes over night duty, Prime writes **council notes** — a brief summary of its cognitive state, active conversations, and pending tasks. Core reads these notes when it wakes, giving it continuity without requiring the full context to be re-loaded.

When Prime wakes from sleep, it reads the **golden thread** — a fresh architecture snapshot generated during the wake sequence, so Prime's first perception is an accurate picture of the system's current state.

---

## Lifecycle Management

### GPU Resource Budget

GAIA runs on a single GPU (16GB VRAM). The budget is:

| Component | VRAM | Always-on? |
|-----------|------|-----------|
| Nano (0.8B, NF4) | ~2.0 GB | Yes (AWAKE) |
| Core (4B, NF4) | ~3.1 GB | Yes (AWAKE) |
| Audio STT (0.6B) | ~1.8 GB | Yes (AWAKE) |
| KV cache overhead | ~1-2 GB | Dynamic |
| **Total AWAKE** | **~8-9 GB** | |
| Prime (8B, NF4) | ~4.5 GB | Only in FOCUSING |
| Audio TTS (1.7B) | ~4.3 GB | On-demand |
| Training (QLoRA) | ~10-14 GB | Only in TRAINING |

FOCUSING mode requires swapping Core to CPU to make room for Prime on GPU. This is why it's a consciousness transition, not just a model load.

### Readiness Gate

Before processing any request, the readiness gate verifies:
1. The selected model's engine is loaded and responsive
2. VRAM is not over-committed
3. No HEALING_REQUIRED.lock exists (circuit breaker)

If the gate fails, the request is held with a "warming up" message until the model is ready, or returned with a retry suggestion.

### Idle Monitoring

The idle monitor tracks time since the last user interaction:
- **0-15 min**: AWAKE (full capability)
- **15-30 min**: DROWSY (warning, may preemptively compact KV cache)
- **30+ min**: SLEEPING (GPU models unloaded for power savings)
- **Incoming message during sleep**: Triggers wake signal, queue holds message until models reload

---

## Relationship to Other Tiers

- **Tiers of Memory** (`memory_tiers_spec.md`): Each cognitive tier accesses all memory tiers, but at different depths. Nano only reads Tier 0 (ephemeral). Core reads Tiers 0-3. Prime reads Tiers 0-4 and writes to Tier 4 (reflective) and Tier 5 (retrainable).
- **Tiers of Identity** (`layered_identity_model.md`): All three cognitive tiers share the same Tier I (Core Persona) identity. Persona selection (Tier II) is determined by Core regardless of which tier generates the response.
- **Cognitive Index Layer** (`gaia-index.md`): The CIL is always in context for Core and Prime. Nano does not see the CIL (slim mode).
- **Epistemic Framework**: All tiers share the same epistemic training (Primary School curriculum), but the depth of epistemic reasoning scales with tier capability.

---

## Design Philosophy

GAIA's cognitive tiers embody a principle from neuroscience: **not all thought requires the same resources**. A reflexive response to "hello" should not consume the same energy as a careful analysis of a code review. By layering cognition into tiers with explicit escalation rules, GAIA achieves both speed (Nano answers in 50ms) and depth (Prime reasons for 30 seconds) without waste.

The consciousness matrix makes this physical: GPU memory is finite, and the choice of which models to load is itself a cognitive decision — one that reflects GAIA's current needs, alertness, and the demands of the moment.

The system is designed so that **most requests never need Prime**. Core is the workhorse. Prime is the specialist you call when the problem demands it. And Nano is the reflex that keeps everything feeling responsive.
