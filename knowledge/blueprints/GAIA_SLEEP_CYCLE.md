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

### State Machine (6 public states) — legacy/local, NOT the authoritative one

`GaiaState` (`gaia_core/cognition/sleep_wake_manager.py`) is explicitly documented
in its own docstring as **legacy** — "preserved for backward compatibility. The
authoritative state machine is now `LifecycleState` in
`gaia_common.lifecycle.states`, running in the orchestrator." `SleepWakeManager`
uses `GaiaState` for local sleep/wake bookkeeping inside gaia-core; it is a
*different, narrower* state space than the orchestrator's Gearbox (P/1/1+/2/S/0/T
— see `CLAUDE.md`'s "The Gearbox" table), which governs actual GPU tier
placement for Core/Prime. The diagram below is accurate for `GaiaState` as-is,
but don't read it as the system's one true lifecycle state machine.

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

The 6th public state, **OFFLINE**, isn't in the diagram above — it's a terminal
state entered via graceful shutdown (`POST /shutdown`, below), not reached by
any of the transitions drawn. There are also three internal **sub-phases** of
waking, not part of the public state enum (`_TransientPhase`, same module):
`NONE` (default/no transient phase), `FINISHING_TASK` (letting an in-flight
response finish before fully waking), and `WAKING` (checkpoint restore in
progress).

When a request arrives while GAIA is in `DREAMING` or `DISTRACTED`, she
responds with a canned line rather than forwarding to the model —
`CANNED_DREAMING` ("I'm studying right now — give me a moment to wrap up and
I'll be right with you!") and `CANNED_DISTRACTED` ("I'm a little occupied at
the moment — give me a few minutes and I'll get back to you!"), both defined
alongside the state enums.

### Sleep/Wake HTTP API (`gaia_core/api/sleep_endpoints.py`)

Not previously documented here — the full surface `SleepWakeManager` and the
sleep cycle loop expose to gaia-web, the orchestrator, and the host-side
activity monitor:

| Endpoint | Purpose |
|----------|---------|
| `POST /wake` | Wake signal from gaia-web — called when the first message is queued during sleep. |
| `POST /voice-state` | Notify gaia-core when GAIA joins/leaves a Discord voice channel. |
| `GET /status` | Current sleep/wake state and task info, including drowsiness. |
| `POST /study-handoff` | Receive a GPU handoff signal from the orchestrator (`prime_to_study` / `study_to_prime`). |
| `GET /distracted-check` | Check whether GAIA is in a state that warrants a canned response (and which one). |
| `POST /toggle` | Enable/disable automatic sleep transitions. |
| `POST /hold` | Temporarily suppress auto-sleep for a long-form operation (e.g. a CFR ingest), for N minutes. |
| `POST /hold-release` | Release a sleep hold early. |
| `POST /force` | Immediately trigger a sleep transition via the lifecycle machine. |
| `POST /deep` | Deep sleep — unload ALL models from GPU via the lifecycle machine (transitions to DEEP_SLEEP). |
| `GET /config` | Current sleep configuration, for the dashboard. |
| `POST /shutdown` | Graceful shutdown — transitions to OFFLINE and stops the sleep loop. |
| `GET /wake-config` | Current PRIME_WAKE trigger settings. |
| `POST /wake-toggle` | Toggle a specific wake trigger on/off (e.g. `discord_typing`, `workstation_activity`). |
| `POST /wake-activity` | Wake signal from the host-side workstation activity monitor (only fires if the `workstation_activity` toggle is enabled and GAIA is asleep). |

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
**Blueprint Version**: 0.8
**Last Updated**: 2026-10-14

## Changelog

### 2026-10-14
Resolved the 21-item drift the `blueprint_validation` sleep task detected on
2026-09-17 (listed below) by folding each into the relevant section above
rather than leaving them as a bare auto-generated list: the `OFFLINE` state
and `_TransientPhase` sub-phases and the `CANNED_DREAMING`/`CANNED_DISTRACTED`
responses into **Architecture Overview**, and all 14 missing endpoints into a
new **Sleep/Wake HTTP API** section. Also added an explicit note that
`GaiaState` is the orchestrator's `LifecycleState`/Gearbox's legacy, narrower
local cousin, per its own docstring — not previously flagged by the validator
since it only diffs symbol presence, not doc-level framing accuracy.

Items originally detected 2026-09-17T22:21:31Z: `enum:GaiaState.OFFLINE`,
`enum:_TransientPhase.{NONE,FINISHING_TASK,WAKING}`, 14 `/wake*`, `/voice-state`,
`/status`, `/study-handoff`, `/distracted-check`, `/toggle`, `/hold*`,
`/force`, `/deep`, `/config`, `/shutdown` endpoints, `constant:CANNED_DREAMING`,
`constant:CANNED_DISTRACTED`.
