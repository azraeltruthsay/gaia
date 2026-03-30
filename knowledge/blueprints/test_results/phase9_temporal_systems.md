# Phase 9: Temporal Awareness & Interview Systems — Test Results

**Date:** 2026-03-26
**Tester:** Claude Code
**Status:** Partially functional. Key subsystems working, critical bugs in state baking and idle heartbeat.

---

## System Inventory

GAIA has **6 temporal subsystems** spread across gaia-core and gaia-common:

| Subsystem | File | Status |
|-----------|------|--------|
| Heartbeat Time Check | `gaia-common/gaia_common/utils/heartbeat_time_check.py` | WORKING (with known drift issue) |
| Thought Seed Heartbeat | `gaia-core/gaia_core/cognition/heartbeat.py` | WORKING |
| Idle Heartbeat | `gaia-core/gaia_core/cognition/idle_heartbeat.py` | PARTIALLY BROKEN |
| Temporal Context Builder | `gaia-core/gaia_core/utils/temporal_context.py` | WORKING |
| Temporal State Manager (KV bake) | `gaia-core/gaia_core/cognition/temporal_state_manager.py` | BROKEN (since model migration) |
| Temporal Interviewer | `gaia-core/gaia_core/cognition/temporal_interviewer.py` | BLOCKED (depends on bake) |
| Lite Journal | `gaia-core/gaia_core/cognition/lite_journal.py` | WORKING |

---

## Detailed Findings

### 1. Heartbeat Time Check (WORKING, minor drift)

**Location:** `gaia-common/gaia_common/utils/heartbeat_time_check.py`
**Initialized in:** `gaia-core/gaia_core/main.py` line 183

**What it does:** Periodically asks Nano "What time is it?" and validates the response against actual time. Writes state to `/shared/heartbeat/time_check.json`. Adaptive interval based on cognitive state (faster when errors detected, slower during sleep).

**Current state (live):**
- Firing regularly every ~5 minutes (base 300s with jitter)
- Recent results: mix of PASS (drift 0-2min) and FAIL (drift 3min)
- Max drift tolerance: 2 minutes (configurable via `HEARTBEAT_MAX_DRIFT_MINUTES`)
- Stats from state file show it's been running consistently

**Issue: 3-minute drift consistently fails:**
The Nano model returns times that are 2-3 minutes behind actual time. This suggests the time context injected into Nano's prompt might be stale, or the model is rounding/truncating. The "FAIL" log entries like:
```
Heartbeat: FAIL -- claimed 9:03 PM, actual 9:06 PM (drift 3 min)
```
These appear at restart boundaries where the heartbeat fires immediately before the model has fresh time context. This is a **known, low-severity issue**.

---

### 2. Thought Seed Heartbeat (WORKING)

**Location:** `gaia-core/gaia_core/cognition/heartbeat.py`
**Initialized in:** `gaia-core/gaia_core/cognition/sleep_cycle_loop.py` line 76

**What it does:** Daemon thread (20-minute interval) that:
1. Promotes overdue pending seeds back to triage
2. Triages unreviewed seeds via Lite (ARCHIVE/PENDING/ACT)
3. Runs temporal tasks: journal write, state bake, interview

**Current state (live):**
- Thread running, interval 1200s
- Journal entries written on every tick (confirmed in logs)
- State bake attempted every 3 ticks but **failing** (see issue #4)
- Interview attempted every 6 ticks but **blocked** (see issue #5)
- Emits `heartbeat_tick` timeline events correctly

**No issues with the heartbeat daemon itself.** It correctly orchestrates all subtasks.

---

### 3. Idle Heartbeat (PARTIALLY BROKEN)

**Location:** `gaia-core/gaia_core/cognition/idle_heartbeat.py`
**Initialized in:** `gaia-core/gaia_core/main.py` line 266

**What it does:** Fires when GAIA has been idle for 5+ minutes. Generates brief reflective entries using the lightest available model.

**Current state (live):**
- Firing correctly every 10 minutes (confirmed: ticks #1-#7 visible in logs)
- Generating meaningful reflections via LLM (200+ char entries)
- Logs show reflections like: "In this quiet moment, I'm reflecting on the trajectory of my development..."

**Bug 1: Timeline write uses wrong API**
Line 233-238 in `idle_heartbeat.py`:
```python
self._timeline({...}, "idle_heartbeat", source="idle_heartbeat")
```
This calls `self._timeline(...)` as if the timeline_store is a callable, but `TimelineStore.append()` is the correct method. Should be:
```python
self._timeline.append("idle_heartbeat", {...})
```
This always silently fails (caught by the except on line 239).

**Bug 2: LiteJournal write_entry called with wrong signature**
Lines 222-226:
```python
self._lite_journal.write_entry(
    entry_type="idle_reflection",
    content=text.strip(),
    metadata={"idle_seconds": int(idle_seconds), "tick": self._tick_count},
)
```
But `LiteJournal.write_entry()` takes **zero parameters** (it generates content internally via LLM). This call always fails with `TypeError` and is silently caught.

**Bug 3: LiteJournal initialized without model_pool**
Line 80: `LiteJournal(config=config)` -- no model_pool passed, so even if the signature were correct, the journal would have no LLM to generate entries.

**Net effect:** Idle heartbeat reflections are logged to stdout but **never persisted** to journal or timeline.

---

### 4. Temporal State Manager / KV Bake (BROKEN)

**Location:** `gaia-core/gaia_core/cognition/temporal_state_manager.py`
**Initialized in:** `heartbeat.py` line 94 (within ThoughtSeedHeartbeat)

**What it does:** Bakes Lite's KV cache state to disk as pickled `.bin` files with JSON sidecar metadata. Captures the model's entire cognitive context at that moment for later restoration.

**Current state (live):**
- 5 existing baked states from March 8-9 (80MB-590MB each)
- **Baking has been broken since March 9** (17 days)
- Error from logs:
```
AttributeError: 'VLLMRemoteModel' object has no attribute 'save_state'
```

**Root cause:** The system was designed when Lite ran as a local `llama-cpp-python` instance (which has `save_state()/load_state()` methods). Since the migration to GAIA Engine managed mode, the model pool returns `VLLMRemoteModel` objects that access the model via HTTP API. These remote model objects don't have `save_state()` / `load_state()`.

**Fix required:** Either:
1. Add KV cache save/load endpoints to GAIA Engine and wrap in VLLMRemoteModel
2. Use the GAIA Engine's KV cache thought snapshot API (if it exists in gaia-engine)
3. Implement a different state capture mechanism for remote models (e.g., save the conditioning prompt + response rather than raw KV cache)

---

### 5. Temporal Interviewer (BLOCKED)

**Location:** `gaia-core/gaia_core/cognition/temporal_interviewer.py`
**Initialized in:** `heartbeat.py` line 112 (within ThoughtSeedHeartbeat)

**What it does:** Prime interviews past-Lite via KV cache state swapping. Multi-turn Q&A (3 rounds), then narrative coherence analysis comparing journal entries against interview responses.

**Current state (live):**
- 2 existing interview transcripts from February 20 and March 4
- No interviews since March 9 (when baking stopped)
- The interviewer itself is properly initialized but can't find new uninterviewed states

**Historical interviews show the system works:**
The March 4 transcript shows:
- 3-round interview with coherent Q&A
- Coherence analysis: topic_overlap=0.95, tone_consistency=0.90, overall=0.92
- Duration: 110 seconds

**Quality observation:** Past-Lite responses include `<think>` tags (Qwen3 thinking), which pollute the transcript. The interview answers sometimes show the model's chain-of-thought rather than clean answers. The coherence analysis appears to work despite this.

**Blocked on:** Temporal State Manager bake failure (issue #4). No new states to interview.

---

### 6. Temporal Context Builder (WORKING)

**Location:** `gaia-core/gaia_core/utils/temporal_context.py`
**Injected via:** `gaia-core/gaia_core/utils/prompt_builder.py` line 611

**What it does:** Assembles a `[Temporal Context]` block injected into every prompt, containing:
- Semantic time ("Tuesday 2026-03-25, 22:41 UTC (evening)")
- Wake cycle ("Awake for 2h 15m. Last sleep: 45m")
- Session summary ("This conversation: 45m old, 12 messages")
- Activity summary ("Since waking: 3 conversations")
- State summary ("State: ACTIVE for 2h 15m")
- Code evolution ("2 services have pending candidate changes")

**Current state:** Fully functional. Degrades gracefully when timeline_store or sleep manager data is unavailable.

---

### 7. Lite Journal (WORKING)

**Location:** `gaia-core/gaia_core/cognition/lite_journal.py`
**Data at:** `/shared/lite_journal/Lite.md`

**What it does:** Writes periodic first-person journal entries during each heartbeat tick. Entries include state, heartbeat number, and LLM-generated reflective text.

**Current state (live):**
- Writing entries on every heartbeat tick (confirmed 5 entries in recent Lite.md)
- Entry format: `## Entry: {timestamp}` with state/heartbeat metadata and 1-3 sentence reflection
- Content quality is reasonable but occasionally confabulates (e.g., "grab a coffee at the cafe nearby" -- GAIA has no physical presence)

---

## Unit Test Coverage

All temporal subsystems have unit tests:

| Test File | Tests | Status |
|-----------|-------|--------|
| `test_temporal_state_manager.py` | 12 tests (bake, load, rotate, context) | All pass (mock-based) |
| `test_temporal_interviewer.py` | 13 tests (selection, flow, lock, coherence, storage) | All pass (mock-based) |
| `test_heartbeat.py` | 12 tests (triage, act, lifecycle, temporal integration) | All pass (mock-based) |
| `test_temporal_context.py` | 11 tests (time, duration, session, state, code evolution) | All pass (mock-based) |

Tests are thorough but all mock-based -- they don't catch the `VLLMRemoteModel` incompatibility because they mock the LLM layer.

---

## Summary of Issues (Priority Order)

### Critical

1. **Temporal State Bake broken** (`temporal_state_manager.py:414`)
   - `VLLMRemoteModel` has no `save_state()` method
   - Broken since migration to GAIA Engine managed mode (~March 9)
   - Blocks both state baking AND temporal interviews

### Medium

2. **Idle Heartbeat timeline write uses wrong API** (`idle_heartbeat.py:233`)
   - Calls `self._timeline({...}, ...)` instead of `self._timeline.append("idle_heartbeat", {...})`
   - Idle reflections never written to timeline

3. **Idle Heartbeat LiteJournal write_entry wrong signature** (`idle_heartbeat.py:222`)
   - Calls `write_entry(entry_type=..., content=..., metadata=...)` but method takes 0 args
   - Idle reflections never persisted to journal

4. **Idle Heartbeat LiteJournal missing model_pool** (`idle_heartbeat.py:80`)
   - `LiteJournal(config=config)` without model_pool makes journal non-functional even if API were correct

### Low

5. **Heartbeat time check drift** (`heartbeat_time_check.py`)
   - 3-minute drift at restart boundaries, possibly due to stale time context in Nano's prompt
   - MAX_DRIFT_MINUTES=2 is tight; consider raising to 3 or fixing time injection

6. **Interview transcript includes `<think>` tags** (`temporal_interviewer.py`)
   - Past-Lite's answers contain Qwen3 chain-of-thought markup
   - Should strip `<think>...</think>` from answers before saving transcript

7. **No temporal API endpoints** on gaia-core
   - No `/temporal/status`, `/heartbeat/status`, or `/interview/history` routes
   - Dashboard has no visibility into temporal subsystem state

---

## Recommendations

1. **For the bake failure (critical):** The simplest fix is to change the bake approach. Instead of saving raw KV cache (which requires direct model access), save the conditioning prompt and Lite's response as a "cognitive snapshot" JSON. The interview protocol can then re-inject this context into Lite rather than swapping KV cache state. This approach works with any model backend.

2. **For idle heartbeat bugs:** Fix the three bugs in `idle_heartbeat.py`:
   - Change timeline call to `self._timeline.append("idle_heartbeat", {...})`
   - Remove the broken LiteJournal integration (idle reflections are already logged)
   - Or properly integrate by passing model_pool and using the correct write API

3. **For API visibility:** Add a `/temporal/status` endpoint that returns bake state, interview history, heartbeat stats, and journal entry count.
