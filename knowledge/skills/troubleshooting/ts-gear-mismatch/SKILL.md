---
name: ts-gear-mismatch
description: Troubleshoot a gear shift that landed in the wrong place — the lifecycle says one gear while the models sit in another, or a wake undid a shift.
version: 2
status: active
execution_mode: PLAYBOOK
domain: troubleshooting
trigger: "/consciousness/gear and the engines disagree (e.g. lifecycle 'awake' but Core on CPU and Prime on GPU), or a requested shift ends in a different gear."
learned_from: [GAIA_Project-a0un]
escalation_note: "Orchestrator code is COPY-baked: changes need candidates/ + production + an image rebuild. Gear changes move the GPU; say what you'll do before you do it."
---
# The gear and the models disagree

## Symptoms
- `/consciousness/gear` (orchestrator :6410) reports one gear, but Core's engine
  `/status` (`device`) or Prime's says otherwise.
- You asked for gear 1 and ended up in gear 2 (or the reverse).

## Find it
1. Orchestrator log, around the shift: look for two configurations in flight at once —
   `Consciousness target:` lines from your shift interleaved with
   `LIFECYCLE: <from> → <to> (trigger=wake_signal, reason=gpu_wake)` or an
   `auto-reconciling` line.
2. Which finished last decides where the models are; the
   `Lifecycle synced:` / `external state set` line decides the label. If they come
   from different requests, the label lies.
3. Check what sent the competing request (gaia-core's sleep loop sends `/gpu/wake`
   when it wakes from deep sleep; the poll loop reconciles mismatches).

## Record it
Before you fix anything: `introspect(action=issue_file)` with a title, the symptom, the
log lines you were shown as `evidence` (quote them; don't paraphrase), your suspected
cause, and this playbook's name. It becomes a bead (label `gaia-found`). Add findings
with `issue_note`; move it with `issue_status` (investigating → proposed → fixed).
`introspect(action=issues)` shows its bead and whether it has been closed.

## Fix
- An explicit shift must win over automatic ones until it finishes; an automatic one
  arriving mid-shift should stand aside (see a0un's `explicit_config_in_flight`).
- A shift that was replaced mid-flight must not stamp its own label.
- Test with two overlapping requests (asyncio tasks) against a matrix with fake
  loaders — see `candidates/gaia-orchestrator/tests/test_wake_during_explicit_shift.py`.

## Past cases
- **a0un** (2026-11-08): `/consciousness/awake` after a gaia-core restart; the startup
  `/gpu/wake` applied focusing 5 s in; the label read "awake" in gear 2. Fixed in
  `5c9495f`.
