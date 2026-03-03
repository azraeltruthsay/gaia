# GAIA Sleep Cycle System - Blueprint

**Date:** 2026-02-14 (original), 2026-03-02 (autonomous activation)
**Author:** Claude Sonnet 4.5 / Azrael
**Status:** Implemented — Active Maintenance
**Target System:** gaia-host (RTX 5080 16GB / Ryzen 9 / 32GB RAM)

---

## Review Notes (2026-03-02, Autonomous Activation)

Blueprint updated to reflect the activation of the autonomous thought and documentation cycle:

1. **Activated Initiative Loop** — `InitiativeEngine` registered in `SleepTaskScheduler` (priority 3). It now processes high-priority topics from `topic_cache.json` during sleep.
2. **Activated Golden Thread** — `auto_as_built_update` registered as a priority 1 task. It generates a fresh `AS_BUILT_LATEST.md` codebase snapshot at the start of every sleep cycle.
3. **Integrated Saṃvega** — Self-reflection now triggers Saṃvega analysis on low-confidence responses, persisting discernment artifacts for future behavior alignment.
4. **Auditory Environment Sensing** — Sleep and wake states now include auditory environment analysis (BPM, Key, Energy) via `gaia-audio/analyze`.

---

## Executive Summary

### Vision
Transform GAIA's idle time into productive autonomous operation through a biologically-inspired sleep cycle. When no users are actively engaged, GAIA enters a sleep state to perform maintenance, learning, and self-improvement tasks.

### Key Features
- **Autonomous Maintenance**: Background tasks execute during idle periods (As-Built updates, topic resolution).
- **Cognitive Continuity**: Prime model preserves context through `prime.md` checkpoint files.
- **Graceful Wake-Up**: CPU Lite handles first-response while Prime boots in background.
- **Self-Improvement**: Saṃvega discernment and autonomous initiative turns.

---

## Architecture Overview

### State Machine (6 public states)

```
              ┌──────────┐
              │  ACTIVE  │  Normal operation: process messages, stream responses
              └────┬─────┘
                   │ idle > 5 min
                   ▼
              ┌──────────┐
              │  DROWSY  │  Prime writes prime.md checkpoint
              └────┬─────┘
                   │ checkpoint written
                   ▼
              ┌──────────┐          ┌───────────┐
              │  ASLEEP  │─────────►│ DREAMING  │  GPU handed to Study
              └────┬─────┘ study    └─────┬─────┘  
                   │       handoff        │
                   │◄─────────────────────┘
                   │
                   │ CPU/GPU >25% for 5s
                   ▼
              ┌────────────┐
              │ DISTRACTED │  System under sustained load
              └────────────┘
```

### Sleep Task Registry (Priority Order)

| Priority | Task ID | Type | Description |
|----------|---------|------|-------------|
| 1 | `auto_as_built_update` | MAINTENANCE | **Golden Thread:** Fresh codebase snapshot for Prime wake-up. |
| 1 | `conversation_curation` | MAINTENANCE | Curation of notable session history for NotebookLM. |
| 3 | `initiative_cycle` | AUTONOMOUS | **Initiative Loop:** Self-prompting based on topic cache. |
| 5 | `wiki_doc_regen` | DOC_GENERATION | Re-generation of static wiki docs from updated knowledge. |
| 5 | `adversarial_resilience`| RESILIENCE | **Chaos Monkey:** Automated failover and health drills. |

---

## Data Flow: The Golden Thread

1. **Sleep Entry**: `SleepCycleLoop` triggers `initiate_drowsy()`.
2. **As-Built Sync**: `auto_as_built_update` runs first, calling `code_evolution.py`.
3. **Persistence**: `AS_BUILT_LATEST.md` is written to `/knowledge/system_reference/`.
4. **Wake Initiation**: On wake signal, `AgentCore` injects the As-Built report into the prompt.
5. **Real-time Awareness**: Prime model "wakes up" with full awareness of any manual or autonomous code changes made during its previous ACTIVE state.

## Meta
**Status**: live
**Blueprint Version**: 0.7 (Autonomous)
**Last Updated**: 2026-03-02
