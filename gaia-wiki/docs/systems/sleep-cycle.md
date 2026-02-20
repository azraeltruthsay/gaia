# Sleep Cycle

GAIA has a biological-inspired sleep/wake cycle that conserves GPU resources and enables background processing.

## State Machine

```
                ┌─────────┐
    user msg    │         │  idle timeout
   ┌───────────→│  ACTIVE ├──────────────┐
   │            │         │              │
   │            └─────────┘              ▼
   │                              ┌───────────┐
   │            wake signal       │           │
   │           ┌─────────────────┤  DROWSY   │
   │           │                  │           │
   │           │                  └─────┬─────┘
   │           │                        │ no activity
   │     ┌─────┴─────┐                  ▼
   │     │           │           ┌───────────┐
   └─────┤  WAKING   │◀──wake───┤           │
         │           │           │  ASLEEP   │
         └───────────┘           │           │
                                 └───────────┘
```

## Components

| Component | Role |
|-----------|------|
| `SleepCycleLoop` | Main loop — runs as asyncio task, manages state transitions |
| `SleepWakeManager` | State machine — tracks current state, transition timestamps |
| `IdleMonitor` | Activity tracking — marks active on user input, triggers drowsy on timeout |
| `Heartbeat` | Periodic tick (~20 min) — triggers Lite journal writes, temporal state updates |

## Sleep Tasks

When GAIA enters sleep, gaia-study can run background tasks:

1. **Vector store indexing** — update embeddings for new knowledge
2. **LoRA fine-tuning** — train on recent conversations
3. **Codebase analysis** — snapshot code evolution
4. **Knowledge summarization** — generate evolving conversation summaries

## Cognitive Checkpoints

Before sleeping, gaia-core writes:

- **prime.md** — introspective summary of current cognitive state (LLM-generated or static template)
- **Lite.md** — running journal entry from the Lite model's perspective

These are injected into the prompt on wake to restore context. See [prime.md over KV cache](../decisions/prime-md-over-kv-cache.md).

## GPU Handoff

The sleep cycle triggers a GPU handoff via the orchestrator:

1. gaia-core enters sleep → calls orchestrator to release GPU
2. Orchestrator puts gaia-prime to sleep (KV cache offloaded to CPU)
3. Orchestrator grants GPU lease to gaia-study
4. On wake → reverse handoff, gaia-prime wakes up
