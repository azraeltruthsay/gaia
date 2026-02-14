# Dev Journal Entry: 2026-02-14 — Sleep Cycle Phase 1 Implementation

**Date:** 2026-02-14
**Author:** Claude Code (Opus 4.6) via Happy

## Context

The user (Seumas) proposed a comprehensive sleep/wake cycle for GAIA. When idle for 5+ minutes, GAIA should enter a sleep state to perform maintenance tasks (conversation cleanup, thought seed reflection, vector store maintenance, eventually QLoRA training). Messages received during sleep are queued in gaia-web; the first message triggers a graceful wake-up. The concept of "prime.md" — a natural-language cognitive state checkpoint — replaces KV cache persistence (which is impossible across vLLM container restarts due to CUDA context binding).

A full implementation plan was drafted in `GAIA_SLEEP_CYCLE_IMPLEMENTATION_PLAN.md`, reviewed, corrected (10 issues found including circular dependency, wrong checkpoint path, missing DROWSY state, missing CPU Lite parallel wake strategy), and approved. Phase 1 covers the core state machine, checkpoint system, API endpoints, message queue, and integration.

---

## Phase 1 Implementation

### New Files Created (6)

#### 1. `candidates/gaia-core/gaia_core/cognition/sleep_wake_manager.py`
Core 5-state machine: `AWAKE → DROWSY → SLEEPING → FINISHING_TASK → WAKING`

- `GaiaState` enum with 5 states
- `SleepWakeManager` class with full state transition logic
- DROWSY is cancellable — a wake signal during checkpoint writing aborts sleep
- WAKING uses parallel strategy: CPU Lite handles first message while Prime boots (~37-60s)
- `_format_checkpoint_as_review()` frames checkpoint as "[SLEEP RESTORATION CONTEXT — Internal Review Only]"
- ~210 lines

#### 2. `candidates/gaia-core/gaia_core/cognition/prime_checkpoint.py`
Natural-language KV cache replacement. Prime writes its cognitive state to `/shared/sleep_state/prime.md` before sleeping.

- `PrimeCheckpointManager` class
- `create_checkpoint()` — writes deterministic template (Phase 2+ will use LLM-generated summaries)
- `rotate_checkpoints()` — `prime.md → prime_previous.md` + `prime_history/<timestamp>-sleep.md`
- `load_latest()` — reads checkpoint for wake context restoration
- Uses existing `/shared` Docker named volume for persistence across container restarts
- ~170 lines

#### 3. `candidates/gaia-core/gaia_core/cognition/sleep_cycle_loop.py`
Daemon thread that polls `IdleMonitor` every 10 seconds and drives state transitions.

- `SleepCycleLoop` class with `start()`/`stop()` lifecycle
- Per-state handlers: `_handle_awake()`, `_handle_sleeping()`, `_handle_finishing_task()`, `_handle_waking()`
- Discord presence updates via injected `discord_connector` (e.g. "Drifting off...", "Sleeping", "Waking up...")
- Uses `gaia-common.IdleMonitor` for idle detection (low-level primitive), owns all orchestration in gaia-core (avoids circular dependency)
- ~130 lines

#### 4. `candidates/gaia-core/gaia_core/api/sleep_endpoints.py`
APIRouter following `gpu_endpoints.py` pattern.

- `POST /sleep/wake` — receive wake signal from gaia-web message queue
- `GET /sleep/status` — get current state machine status
- Accesses `SleepWakeManager` via `request.app.state.sleep_wake_manager`
- ~55 lines

#### 5. `candidates/gaia-web/gaia_web/queue/message_queue.py`
Async priority message queue for sleep/wake message buffering.

- `QueuedMessage` dataclass with priority, source, session_id, metadata
- `MessageQueue` class with async lock, priority-sorted dequeue
- Auto-sends wake signal via `httpx POST` to gaia-core `/sleep/wake` on first enqueue
- ~100 lines

#### 6. `candidates/gaia-core/gaia_core/cognition/tests/test_sleep_wake_manager.py`
26 unit tests covering the full state machine lifecycle.

- `TestInitialState` — starts AWAKE, no pending wake, prime unavailable
- `TestDrowsyThreshold` — idle threshold logic, state guards
- `TestInitiateDrowsy` — happy path, checkpoint written, rejects from wrong states, cancels on wake during checkpoint
- `TestReceiveWakeSignal` — wake while awake/sleeping/drowsy, non-interruptible task handling
- `TestCompleteWake` — full cycle test, no-checkpoint case, wrong-state guard
- `TestStatus` — field presence, reflects state changes
- `TestFormatCheckpoint` — empty checkpoint handling, review framing
- `TestTransitionToWaking` — valid and invalid source states

### Existing Files Modified (5)

#### 1. `candidates/gaia-common/.../background/processor.py`
Fixed broken import on line 14: `from app.cognition.initiative_handler import gil_check_and_generate` referenced a module never migrated from the monolith. Replaced with `None` stub + guarded call site on line 83.

#### 2. `candidates/gaia-core/gaia_core/main.py`
- Registered `sleep_router` via `app.include_router(sleep_router)`
- Added `SleepCycleLoop` start/stop in `lifespan()` context manager
- Stores `sleep_wake_manager` on `app.state` for endpoint access
- Added `/sleep/status` and `/sleep/wake` to root endpoint documentation
- Respects `SLEEP_ENABLED` config flag (graceful degradation if disabled)

#### 3. `candidates/gaia-core/gaia_core/config.py`
- Added `SLEEP_ENABLED`, `SLEEP_IDLE_THRESHOLD_MINUTES`, `SLEEP_CHECKPOINT_DIR`, `SLEEP_ENABLE_QLORA`, `SLEEP_ENABLE_DREAM`, `SLEEP_TASK_TIMEOUT` fields to `Config` dataclass
- Wired `_load_constants()` to populate from `gaia_constants.json` `SLEEP_CYCLE` section

#### 4. `gaia-common/gaia_common/constants/gaia_constants.json`
Added `SLEEP_CYCLE` config section with defaults:
```json
{
  "enabled": true,
  "idle_threshold_minutes": 5,
  "checkpoint_dir": "/shared/sleep_state",
  "enable_qlora": false,
  "enable_dream": false,
  "task_timeout_seconds": 600,
  "poll_interval_seconds": 10,
  "max_checkpoint_history": 10
}
```

#### 5. `candidates/gaia-core/gaia_core/utils/prompt_builder.py`
Injected sleep restoration context at Tier 1 (between summary_prompt and session_rag_prompt). Reads `/shared/sleep_state/prime.md` if it exists, formats via `SleepWakeManager._format_checkpoint_as_review()`, respects token budget.

---

## Test Results

- **26/26** new `test_sleep_wake_manager.py` tests pass (0.98s)
- **59/59** existing memory tests pass (6.90s) — no regressions

---

## Architecture Decisions

1. **Sleep logic in gaia-core, not gaia-common** — avoids circular dependency. gaia-common provides primitives (IdleMonitor, TaskQueue); gaia-core owns orchestration.
2. **DROWSY state is cancellable** — if a message arrives during checkpoint writing, sleep aborts and returns to AWAKE. This prevents unnecessary sleep/wake cycles for brief idle periods.
3. **Template checkpoint for Phase 1** — prime.md uses a deterministic template rather than LLM-generated content. Phase 2+ will replace this with an actual Prime meta-cognitive call.
4. **File-based checkpoint, not API** — `prime.md` is read directly from the filesystem in `prompt_builder.py` rather than through an API call. This is simpler and avoids circular HTTP calls during prompt assembly.

---

## What's Next (Phase 3)

- **LLM-generated checkpoints** — replace template with actual Prime meta-cognitive summary
- **QLoRA dream mode** — integrate gaia-study QLoRA training as a sleep task
- **Thought seed Observer integration** — add `THOUGHT_SEED` directive to Observer system prompt
- **Vector store maintenance** — add as a sleep task for pruning/optimizing session vectors

---

## Files Checklist — Phase 1

| Status | File | Action |
|--------|------|--------|
| NEW | `candidates/gaia-core/gaia_core/cognition/sleep_wake_manager.py` | 5-state machine |
| NEW | `candidates/gaia-core/gaia_core/cognition/prime_checkpoint.py` | Checkpoint manager |
| NEW | `candidates/gaia-core/gaia_core/cognition/sleep_cycle_loop.py` | Daemon thread |
| NEW | `candidates/gaia-core/gaia_core/api/sleep_endpoints.py` | HTTP endpoints |
| NEW | `candidates/gaia-web/gaia_web/queue/message_queue.py` | Message queue |
| NEW | `candidates/gaia-web/gaia_web/queue/__init__.py` | Package init |
| NEW | `candidates/gaia-core/gaia_core/cognition/tests/__init__.py` | Test package init |
| NEW | `candidates/gaia-core/gaia_core/cognition/tests/test_sleep_wake_manager.py` | 26 unit tests |
| MOD | `candidates/gaia-common/.../background/processor.py` | Fixed broken import |
| MOD | `candidates/gaia-core/gaia_core/main.py` | Router + lifespan integration |
| MOD | `candidates/gaia-core/gaia_core/config.py` | SLEEP_* config fields |
| MOD | `gaia-common/.../gaia_constants.json` | SLEEP_CYCLE section |
| MOD | `candidates/gaia-core/gaia_core/utils/prompt_builder.py` | Sleep context injection |

---

## Phase 2 Implementation — Sleep Task System

### New Files Created (4)

#### 1. `candidates/gaia-core/gaia_core/cognition/sleep_task_scheduler.py`
Central orchestrator for sleep-time autonomous maintenance tasks.

- `SleepTask` dataclass: task_id, task_type, priority, interruptible, handler, last_run, run_count
- `SleepTaskScheduler` class with priority + LRU scheduling
- Three default tasks registered at init:
  - P1: `conversation_curation` (60s, interruptible) — calls `ConversationCurator.curate()` on active sessions
  - P1: `thought_seed_review` (120s, interruptible) — calls `review_and_process_seeds()` with CPU Lite model
  - P2: `initiative_cycle` (180s, interruptible) — calls `InitiativeEngine.execute_turn()`
- `get_next_task()` — priority-first, then least-recently-run (nulls-first)
- `execute_task()` — runs handler in try/except, updates run metadata, never propagates
- ~165 lines

#### 2. `candidates/gaia-core/gaia_core/cognition/initiative_engine.py`
Port of archived `run_gil.py` adapted for v0.3 microservice architecture.

- `InitiativeEngine` class with `execute_turn()` method
- Selects top-priority topic via `prioritize_topics()` from existing `topic_manager.py`
- Builds self-prompt (ported from run_gil.py lines 74-85)
- Runs through `AgentCore.run_turn()` with `GIL_SESSION_ID`
- Returns `{"topic_id": ..., "status": "complete"|"error"}` or `None`
- Graceful fallback when `agent_core` is None
- ~90 lines

#### 3. `candidates/gaia-core/gaia_core/cognition/tests/test_sleep_task_scheduler.py`
13 unit tests covering registration, scheduling, execution, and status.

#### 4. `candidates/gaia-core/gaia_core/cognition/tests/test_initiative_engine.py`
8 unit tests covering no-topics, no-agent-core, successful turn, prompt construction, and error handling.

### Existing Files Modified (2)

#### 1. `candidates/gaia-core/gaia_core/cognition/sleep_cycle_loop.py`
- Added `model_pool` and `agent_core` parameters to `__init__`
- Created `SleepTaskScheduler` instance in constructor
- Replaced `_handle_sleeping()` `pass` with task execution loop:
  - Gets next task from scheduler
  - Registers current_task on SleepWakeManager for interruptibility checks
  - Updates Discord presence with task type
  - Executes task, clears current_task, checks wake signal
- Updated `_handle_finishing_task()` comment

#### 2. `candidates/gaia-core/gaia_core/main.py`
- Updated `SleepCycleLoop()` constructor in `lifespan()` to pass `model_pool` and `agent_core`
- Safely handles case where `_ai_manager` is None

### Test Results

- **47/47** cognition tests pass (Phase 1: 26, Scheduler: 13, Initiative: 8) — 0.97s
- **59/59** memory tests pass — zero regressions — 4.90s

### Architecture Notes

1. **Synchronous task execution** — all handlers are plain functions, matching the daemon thread model
2. **CPU Lite for sleep tasks** — GPU may be stopped; thought seed review and initiative use `model_pool.get_model_for_role("lite")`
3. **One task at a time, wake-check between** — worst-case wake latency = one task's duration
4. **Reuse existing code** — `ConversationCurator`, `review_and_process_seeds()`, `prioritize_topics()`, `AgentCore.run_turn()` called directly
5. **Initiative engine is a scheduler task** — not a standalone loop; idle check handled by the state machine
