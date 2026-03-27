# Sleep Cycle & Consciousness Matrix

GAIA has a biologically-inspired sleep/wake cycle that manages resource allocation through a three-state **Consciousness Matrix**. This allows GAIA to conserve GPU resources while maintaining responsiveness and enabling background training.

## Consciousness Matrix

Managed by the **gaia-orchestrator**, the Consciousness Matrix tracks the target vs. actual state of each cognitive tier.

| State | Name | Resource | Description |
|-------|------|----------|-------------|
| **3** | Conscious | GPU | High-performance inference (SafeTensors/vLLM) |
| **2** | Subconscious | CPU | Efficient GGUF inference (llama-server) |
| **1** | Unconscious | Unloaded | Resource hibernation |

### The Dolphin Metaphor
Like a dolphin's unihemispheric sleep, GAIA's tiers can maintain independent states. One tier can be "awake" (GPU) while another "rests" (CPU) or is completely "unconscious" (unloaded).

## System States & Presets

The system-wide sleep cycle translates user activity into matrix presets:

- **AWAKE**: Core=3, Nano=3, Prime=2. (Fast reflex/intent, Prime observes on CPU).
- **FOCUSING**: Prime=3, Nano=3, Core=2. (Deep reasoning on GPU, Core manages on CPU).
- **SLEEP**: Nano=2, Core=2, Prime=1. (Low-power monitoring, ready to wake).
- **DEEP SLEEP**: All → 1. (Full hibernation, Nano stays 2 for wake detection if configured).
- **TRAINING**: Target Tier=1, Others=2. (Freeing VRAM for QLoRA training).

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
