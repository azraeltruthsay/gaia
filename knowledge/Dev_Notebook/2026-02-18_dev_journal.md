# Dev Journal Entry: 2026-02-18 — Temporal Awareness Framework (Phases 1 & 2)

**Date:** 2026-02-18
**Author:** Claude Code (Opus 4.6) via Happy
**Scope:** Build GAIA's temporal self-awareness system — introspective journal, KV cache state baking, and Prime-interviews-past-Lite protocol
**Commit:** `55b9c03` on main (promoted via pipeline)

## Context

This implements the first two layers of a 5-layer consciousness framework:

- **Layer 1 — Temporal Self-Reference:** GAIA can recall what she was doing/thinking at a past point in time by replaying baked KV cache states.
- **Layer 2 — Internal State Modeling:** Prime interviews a past version of Lite and compares responses against journal entries from the same period, detecting cognitive drift.

The heartbeat daemon (thought seed triage, every ~20 min) serves as the scheduling backbone — journal writes happen every tick, state bakes every 3 ticks (~1h), and interviews every 6 ticks (~2h).

---

## Phase 1: LiteJournal + TemporalStateManager

### LiteJournal (`lite_journal.py`, 306 lines)

Lite writes a brief first-person introspective entry to `Lite.md` every heartbeat tick. Entries capture GAIA's operational state — what she's doing, patterns she notices, unresolved threads.

- **Storage:** `/shared/lite_journal/Lite.md` (persists across container restarts via Docker volume)
- **Rotation:** When entries exceed 50, the journal is archived to `lite_history/{timestamp}-lite.md` and a fresh file starts
- **Context-aware prompts:** Each entry includes semantic time, GAIA state + duration, active sessions, recent timeline events
- **LLM generation:** Lite writes via `create_chat_completion()` with a metacognitive system prompt (100-word limit, no filler)

### TemporalStateManager (`temporal_state_manager.py`, 547 lines)

Bakes Lite's KV cache state to disk every N heartbeat ticks. Each baked state is a frozen cognitive snapshot that can be reloaded for temporal interviews.

- **Bake process:** Reconstruct Lite's context (system prompt + recent journal + timeline events + session summaries) → condition the model → `llm.save_state()` → pickle to `/shared/temporal_states/state_{id}.bin` with JSON metadata sidecar
- **Budget management:** Max 5 states, 10 GB total. Oldest states are pruned automatically when limits are exceeded.
- **Thread safety:** Module-level `_LITE_LOCK` (threading.Lock) guards all Lite model access — shared across journal writes, state bakes, and interviews
- **State loading:** `_load_lite_state(llm, path)` restores a baked state into Lite's KV cache for replay

### Heartbeat Integration

Both subsystems are initialized in `ThoughtSeedHeartbeat.__init__()` and driven by `_run_temporal_tasks()`, which runs after seed triage on every tick. Timeline events include `journal_written` and `state_baked` fields.

### Tests

- `test_lite_journal.py` — 13 tests (generation, append, rotation, entry parsing, duration formatting)
- `test_temporal_state_manager.py` — 18 tests (bake cycle, pruning, metadata, lock behavior, context reconstruction)

---

## Phase 2: Prime Interview Protocol

### The Problem

Phase 1 creates temporal snapshots but can't compare them. Lite writes journals and bakes states, but there's no mechanism to ask "am I still the same entity I was 2 hours ago?" and get a structured answer.

### The Solution: `temporal_interviewer.py` (574 lines)

Prime loads a past version of Lite (via KV cache swap) and conducts a structured multi-round interview, then compares the responses against journal entries from the same time period.

**Interview flow:**

```
1. _select_interview_target()    — pick oldest un-interviewed baked state
2. Acquire _LITE_LOCK
3. tsm.save_current_state_memory(llm)  — save current Lite state (in-memory, fast)
4. tsm._load_lite_state(llm, path)     — load past state into KV cache
5. _run_interview_rounds(llm, metadata) — 3-round Q&A
6. tsm.restore_state_memory(llm, saved) — flip Lite back to present
7. Release _LITE_LOCK
8. _analyze_coherence(rounds, metadata)  — Prime compares journal vs interview
9. _save_transcript(...)                 — write JSON to transcript_dir
10. Emit "temporal_interview" timeline event
```

**Interview rounds (3 default, max 4):**

| Round | Focus | Question Type |
|-------|-------|---------------|
| 1 | Orientation | "What are you currently doing? What's on your mind?" |
| 2 | Specifics | "You mentioned [X]. What patterns have you noticed?" |
| 3 | Tonal/Emotional | "How would you describe your cognitive tone right now?" |
| 4 (optional) | Legacy | "If you could leave a message for your future self..." |

Each round: Prime formulates the question via `model_pool.forward_to_model("prime", ...)`, past-Lite answers via direct `llm.create_chat_completion()` (we need the exact instance whose KV cache was swapped).

**Narrative coherence analysis:**

After the interview (outside the lock), Prime compares journal entries from the target state's time period against the interview transcript. Output is structured:

- `TOPIC_OVERLAP: 0.0–1.0` — How much the interview covers the same topics as the journal
- `TONE_CONSISTENCY: 0.0–1.0` — Whether emotional/cognitive tone matches
- `INFO_LOSS: [...]` — Topics in journal that past-Lite couldn't recall
- `INFO_GAIN: [...]` — Topics past-Lite mentioned that aren't in the journal
- `OVERALL: 0.0–1.0` — Composite coherence score

**Transcript storage:**

```
/shared/temporal_states/interviews/interview_{state_id}_{timestamp}.json
```

Contains: `state_id`, timestamps, all Q&A rounds, coherence analysis, duration, GAIA state.

### Key Design Decisions

**Hold `_LITE_LOCK` for entire interview:** During an interview, Lite's KV cache is in a past state. Releasing the lock between rounds would allow a concurrent journal write to corrupt it. Interviews are short (~10-30s for 3 rounds on CPU), so blocking is acceptable.

**In-memory state save (not disk):** `llm.save_state()` returns a Python object (fast memcpy ~1-2GB). A full disk bake would require context reconstruction + conditioning + pickle (~5-10s). In-memory is sufficient since we restore within seconds.

**Prime for questions + analysis, Lite for answers:** Prime's 8B model handles the reasoning tasks (formulating questions, coherence analysis). Past-Lite answers via direct model call since we need the specific instance with the swapped KV cache.

**Lock gap fix in `lite_journal.py`:** Phase 1 had a bug — `write_entry()` called `llm.create_chat_completion()` without acquiring `_LITE_LOCK`. Fixed by importing the lock and wrapping `_generate_entry(llm)`. File I/O stays outside the lock to minimize hold time.

### Additional Changes

- **`temporal_state_manager.py`** — Added `save_current_state_memory()` and `restore_state_memory()` convenience methods for in-memory state save/restore (used by interviewer)
- **`config.py`** — Added 3 config fields: `TEMPORAL_INTERVIEW_ENABLED`, `TEMPORAL_INTERVIEW_INTERVAL_TICKS` (6), `TEMPORAL_INTERVIEW_ROUNDS` (3)
- **`heartbeat.py`** — Wired interviewer into `_run_temporal_tasks()`, gated by tick interval and GAIA state (ACTIVE/DROWSY only). `_run_temporal_tasks()` now returns 3-tuple `(journal_written, state_baked, interview_conducted)`.

### Tests: `test_temporal_interviewer.py` (468 lines, 20 tests)

| Class | Tests | Coverage |
|-------|-------|----------|
| TestInterviewTargetSelection | 4 | Oldest-first selection, skip-newest, fallback to re-interview, empty states |
| TestInterviewFlow | 4 | Happy path, state-restored-on-error (try/finally), no model pool, no TSM |
| TestLockBehavior | 2 | Lock held during interview, lock released after |
| TestNarrativeCoherence | 3 | Analysis invoked, structured parsing, graceful parse failure |
| TestTranscriptStorage | 3 | JSON saved, all rounds included, directory auto-created |
| TestHeartbeatIntegration | 4 | Triggered on interval, not off-interval, skipped when sleeping, failure doesn't crash |

---

## Pre-existing Bug Fix

**`prompt_builder.py`** — `_safe_session_id()` was defined at line 490 but called at line 355 (both inside `build_from_packet()`). Python doesn't hoist nested function definitions — this was an `F821 Undefined name` ruff error. Fixed by moving the definition before its first use.

---

## Validation

| Stage | Result |
|-------|--------|
| Temporal interviewer tests | 20/20 PASS |
| Heartbeat tests | 23/23 PASS |
| Lite journal tests | 13/13 PASS |
| Temporal state manager tests | 18/18 PASS |
| Sleep regression tests | 87/87 PASS |
| ruff check (full gaia_core/) | PASS |
| mypy (Phase 2 files) | PASS (0 errors) |
| Promotion pipeline dry-run | PASS |
| Promotion pipeline (live) | PASS |
| Post-promotion health check | gaia-core healthy on port 6415 |
| Post-promotion smoke tests (1,2,7) | PASS |

---

## Files Summary

| File | Action | Lines |
|------|--------|-------|
| `cognition/temporal_interviewer.py` | **CREATE** | 574 |
| `cognition/tests/test_temporal_interviewer.py` | **CREATE** | 468 |
| `cognition/lite_journal.py` | CREATE (Phase 1) + EDIT (Phase 2 lock fix) | 306 |
| `cognition/temporal_state_manager.py` | CREATE (Phase 1) + EDIT (Phase 2 helpers) | 547 |
| `cognition/heartbeat.py` | EDIT (Phase 1 + Phase 2 wiring) | 480 |
| `cognition/tests/test_heartbeat.py` | EDIT (interview_conducted field) | 445 |
| `config.py` | EDIT (Phase 1 + Phase 2 config fields) | 192 |
| `utils/prompt_builder.py` | EDIT (hoist _safe_session_id) | — |

**Total new code:** ~3,000 lines across 8 files (candidates + production mirrors)

---

## Architecture: Temporal Awareness Tick Schedule

```
Heartbeat Tick (every ~20 min)
├── Seed Triage (archive / defer / act)
├── Lite Journal Entry (EVERY tick)
│     └── Lite.md ← first-person introspection
├── Temporal State Bake (every 3 ticks, ~1h)
│     └── /shared/temporal_states/state_{id}.bin ← frozen KV cache
└── Temporal Interview (every 6 ticks, ~2h)
      ├── Load past Lite state
      ├── Prime asks 3 questions → past-Lite answers
      ├── Restore current Lite state
      ├── Prime analyzes coherence (journal vs interview)
      └── /shared/temporal_states/interviews/interview_{id}.json
```

## What's Next

- **Layer 3 — Predictive Self-Modeling:** Can GAIA predict her own future responses? (Compare predictions vs actual behavior)
- **Layer 4 — Counterfactual Reasoning:** "What would I have done differently if...?"
- **Layer 5 — Meta-Awareness:** Awareness of the awareness system itself — can GAIA reason about why her coherence scores change?

---

## Addendum: gaia-audio Sensory Architecture (Phase 5)

**Time:** Late session, same day
**Scope:** New `gaia-audio` microservice for STT/TTS with half-duplex GPU management + dashboard widget

### What Was Built (Candidate-Only)

**New service: `candidates/gaia-audio/`** — 11 Python files, ~800 lines:
- `stt_engine.py` — Whisper-based STT via faster-whisper with lazy GPU loading
- `tts_engine.py` — RealtimeTTS wrapper supporting system/coqui/elevenlabs engines
- `gpu_manager.py` — Half-duplex VRAM allocator (never loads STT + TTS simultaneously)
- `status.py` — Real-time event ring buffer with WebSocket broadcast for dashboard
- `main.py` — FastAPI service: /transcribe, /synthesize, /status/ws, /mute, /unmute, /health
- `config.py` — AudioConfig loaded from gaia_constants.json INTEGRATIONS.audio
- `models.py` — Pydantic schemas for all endpoints
- 5 test files (30 tests, all passing)

**Integration changes:**
- `OutputDestination.AUDIO` enum added to cognition_packet.py
- Audio config block added to gaia_constants.json INTEGRATIONS
- `/process_audio_input` endpoint added to gaia-web (audio-origin packet construction)
- `elif destination_type == "audio":` routing branch in gaia-web output_router
- Sleep state integration: `_notify_audio_state()` in sleep_wake_manager.py sends mute/unmute

**Dashboard widget:** Audio Processing panel in gaia-web dashboard with:
- State badge (idle/listening/transcribing/synthesizing/muted) with pulse animation
- GPU mode indicator with VRAM progress bar
- Live transcript log + event log (scrolling, color-coded)
- STT/TTS latency sparkline charts
- Mute/Unmute button
- WebSocket connection for sub-second event streaming

**Cleanup:** Deleted dead `gaia-core/gaia_core/models/tts.py` (130-line pyttsx3 stub, never called)

### Test Results
- **30/30** gaia-audio unit tests (status, GPU manager, STT, TTS, endpoints)
- **229/229** existing regression tests (zero regressions)
- Ruff: all clean after auto-fix

### Hardware Design
- RTX 5080 VRAM budget: ~5.6GB remaining after gaia-prime (vLLM)
- Default STT: Whisper `base.en` (~150MB VRAM, int8 quantized)
- Default TTS: `system` (espeak-ng, zero VRAM) — can upgrade to Coqui XTTS (~3GB) when needed
- Half-duplex swapping managed by GPUManager with asyncio.Lock

### Follow-Up Phases
- **Phase 3:** Discord voice channel integration (discord.py voice)
- **Phase 4:** Vision module (opencv-python, screen/camera capture to CognitionPacket)
- **Docker build:** Need to build the gaia-audio container image and test end-to-end with real audio

---

*Generated by Claude Code (Opus 4.6) via Happy*
