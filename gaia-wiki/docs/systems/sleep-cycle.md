# Sleep Cycle & Consciousness Matrix

GAIA has a biologically-inspired sleep/wake cycle that manages resource allocation through a three-state **Consciousness Matrix**. This allows GAIA to conserve GPU resources while maintaining responsiveness and enabling background training.

## Consciousness Matrix

Managed by the **gaia-orchestrator**, the Consciousness Matrix tracks the target vs. actual state of each cognitive tier.

| State | Name | Resource | Description |
|-------|------|----------|-------------|
| **3** | Conscious | GPU | High-performance inference (SafeTensors via GAIA Engine) |
| **2** | Subconscious | CPU | Efficient GGUF inference (llama-server) |
| **1** | Unconscious | Unloaded | Resource hibernation |

### The Dolphin Metaphor
Like a dolphin's unihemispheric sleep, GAIA's tiers can maintain independent states. One tier can be "awake" (GPU) while another "rests" (CPU) or is completely "unconscious" (unloaded).

## System States & Presets

The system-wide sleep cycle translates user activity into matrix presets. Since
Sovereign Duality there are two tiers (Core, Prime — the Nano tier is deprecated), and
the full lifecycle is the **gearbox** (P/1/1+/2/S/0/T) defined in
`gaia-common/gaia_common/lifecycle/states.py`:

- **PARKED (P)**: Core=2 (CPU GGUF), Prime=1. (Pre-warmed sentinel standby, GPU empty).
- **AWAKE (1)**: Core=3 (GPU NF4, ~8.8 GB), Prime=2. (Core handles triage/intent, Prime observes on CPU).
- **LISTENING (1+)**: AWAKE + audio STT active.
- **FOCUSING (2)**: Prime=3 (GPU, ~4.6 GB), Core=2. (Deep reasoning on GPU, Core manages on CPU).
- **SLEEP (S)**: Core=2, Prime=1. (Low-power monitoring, ready to wake).
- **DEEP SLEEP (0)**: All → 1. (Full hibernation; Groq fallback only).
- **MEDITATION (T)**: All cognitive tiers → 1; Study owns the GPU for QLoRA training.

## State Machine

```
                ┌─────────┐
    user msg    │         │  idle timeout
   ┌───────────→│  AWAKE  ├──────────────┐
   │            │ (States)│              │
   │            └─────────┘              ▼
   │                              ┌───────────┐
   │            wake signal       │           │
   │           ┌─────────────────┤  SLEEP    │
   │           │                  │ (States)  │
   │           │                  └─────┬─────┘
   │           │                        │
   │     ┌─────┴─────┐                  ▼
   │     │           │           ┌───────────┐
   │     │           │           │ DEEP SLEEP│
   └─────┤  WAKING   │◀──wake───┤           │
         │           │           │ (States)  │
         └───────────┘           └───────────┘
```

## Components

- **gaia-orchestrator**: Tracks actual vs target states via 15s polling.
- **gaia-core**: Manages the high-level sleep/wake logic and triggers transitions.
- **GAIA Engine**: Handles the actual loading/unloading of models and KV cache offloading.

## Sleep Tasks

During **TRAINING** or **SLEEP** states, GAIA-Study performs background consolidation:
1. **Vector store indexing**
2. **QLoRA fine-tuning** (Identity baking)
3. **Open Knowledge Ingestion**

## Cognitive Checkpoints

Before entering lower consciousness states, GAIA-Core persists context:
- **prime.md** — introspective summary of current cognitive state.
- **Lite.md** — running journal from the Lite model's perspective.
- **KV Cache Slots** — serialized thought snapshots for instant resume.
