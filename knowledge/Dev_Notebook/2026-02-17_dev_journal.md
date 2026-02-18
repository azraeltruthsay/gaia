# Dev Journal Entry: 2026-02-17 — prime.md Checkpoint Audit & Fix

**Date:** 2026-02-17
**Author:** Claude Code (Opus 4.6) via Happy
**Scope:** Audit and fix GAIA's sleep/wake cognitive checkpoint system (prime.md)
**Commit:** `d589a85` on main

## Context

User asked: "Is she using prime.md properly when going to sleep and waking?" This triggered a full audit of the sleep/wake checkpoint pipeline — from checkpoint creation through prompt injection on wake.

---

## Audit Findings (4 issues)

### 1. Checkpoint rotation order — BUG
**File:** `sleep_wake_manager.py:initiate_drowsy()`
**Was:** `create_checkpoint()` was called BEFORE `rotate_checkpoints()`, meaning the rotation backed up the *new* checkpoint instead of the *previous* one. The backup was always identical to the current file — useless as a safety net.
**Severity:** Medium — data loss risk if a checkpoint write corrupts.

### 2. Stale checkpoint injection — BUG
**File:** `prompt_builder.py` (sleep restoration block)
**Was:** `prime.md` was read and injected into EVERY prompt request with no consumed/stale check. Once GAIA woke up, every subsequent response included the sleep restoration context forever — wasting tokens and potentially confusing the model.
**Severity:** High — token waste + context pollution on every request after wake.

### 3. Static template checkpoint — DESIGN GAP
**File:** `prime_checkpoint.py:create_checkpoint()`
**Was:** Checkpoint was always a deterministic template with generic placeholders. No actual cognitive introspection — the checkpoint couldn't capture what GAIA was thinking about, emotional tone, or unresolved threads.
**Severity:** Medium — defeats the purpose of cognitive continuity across sleep cycles.

### 4. Evolving summary not captured — DESIGN GAP
**File:** `prime_checkpoint.py`
**Was:** Session-specific evolving summaries (`data/shared/summaries/{session_id}.summary`) existed but were never referenced during checkpoint generation. The richest source of conversation context was ignored.
**Severity:** Low — missed opportunity for richer checkpoints.

---

## Fixes Applied (5 files × 2 copies = 10 files)

### prime_checkpoint.py — Major rewrite (~169 → ~284 lines)
- **LLM-generated checkpoints (Phase 2):** When `model_pool` is provided, CPU Lite generates an introspective first-person checkpoint via metacognitive prompt. Covers: what she was thinking about, unresolved threads, emotional tone, and what to do first on wake.
- **Graceful fallback chain:** LLM available → use it; lite model returns None → template; LLM throws → template; no model_pool → template.
- **Consumed sentinel:** Added `.prime_consumed` sentinel file pattern. `mark_consumed()` creates the sentinel; `is_consumed()` checks it; `create_checkpoint()` clears it. Prevents stale injection.
- **Evolving summary integration:** `_load_evolving_summary(session_id)` reads session `.summary` file (truncated to 1000 chars) and includes it in the LLM prompt for richer context.
- **Context extraction:** New `_extract_context(packet)` pulls `session_id`, `last_prompt`, `persona` from `CognitionPacket`.

### sleep_wake_manager.py — 3 targeted fixes
- **Rotation order:** `rotate_checkpoints()` now called BEFORE `create_checkpoint()` in `initiate_drowsy()`.
- **model_pool threading:** Constructor accepts `model_pool=None`, passes it through to `create_checkpoint()`.
- **Consumed on wake:** `complete_wake()` calls `checkpoint_manager.mark_consumed()` after loading, preventing re-injection.

### sleep_cycle_loop.py — 1-line change
- Passes `model_pool=model_pool` to `SleepWakeManager()` constructor.

### prompt_builder.py — Sentinel check
- Sleep restoration block now checks for `.prime_consumed` sentinel:
  ```python
  consumed_path = os.path.join(checkpoint_dir, ".prime_consumed")
  if os.path.exists(checkpoint_path) and not os.path.exists(consumed_path):
  ```
- Stale checkpoints are silently skipped instead of injected.

### test_sleep_wake_manager.py — 12 new tests (50 → 62)
- `TestInitiateDrowsy`: rotation order, backup preservation, consumed sentinel cleared
- `TestCompleteWake`: consumed marking, no-mark when no checkpoint
- `TestLLMCheckpoint`: LLM generation, fallback on no lite model, fallback on exception, template without model_pool
- `TestCheckpointConsumed`: initial state, sentinel creation, sentinel clearing

---

## Validation

| Stage | Result |
|-------|--------|
| Candidate pytest (Docker) | 76/76 PASS (62 unit + 14 GPU integration) |
| Promotion (5 files) | PASS |
| Production pytest (Docker) | 76/76 PASS |
| gaia-core container restart | Healthy |
| New code verified loaded | Confirmed via docker exec import check |

## Process Notes

- Verified candidates == production before making changes (all 5 files identical)
- Built and tested in Docker containers (`localhost:5000/gaia-core:local`) since host lacks `regex` dependency
- `docker cp` to production container failed (read-only volume mount) — used `docker run` with PYTHONPATH override instead
- Promoted by direct file copy (candidates → production), then restarted container

## Architecture Notes

The checkpoint flow is now:

```
ACTIVE → idle threshold → DROWSY
  ├── rotate_checkpoints()     (backup previous prime.md)
  ├── create_checkpoint()      (LLM introspection or template fallback)
  │     └── clears .prime_consumed sentinel
  ├── wake signal? → cancel, return to ACTIVE
  └── success → ASLEEP (GPU released)

ASLEEP → wake signal → WAKING
  ├── reclaim GPU
  ├── load_latest()            (read prime.md)
  ├── mark_consumed()          (create .prime_consumed sentinel)
  ├── format as REVIEW context (Tier 1 injection)
  └── → ACTIVE

prompt_builder (every request):
  └── check prime.md exists AND .prime_consumed absent
      ├── yes → inject sleep restoration context
      └── no → skip (checkpoint already consumed or doesn't exist)
```

---

*Generated by Claude Code (Opus 4.6) via Happy*
