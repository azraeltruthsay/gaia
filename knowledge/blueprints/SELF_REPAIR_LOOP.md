# Self-Repair Loop — GAIA finds, fixes and tracks her own bugs

> Status: **PROPOSED, first run pending** (2026-11-08). Experiment bead: `mjcg`.

## Why

Azrael, 2026-11-08: *"If we just build in hardcoded keyword triggers and algorithmic
architecture to bolt onto the side of an LLM, that doesn't make the LLM better, it
tricks the user. … involve the LLM in the system directly — intent detection, response
validation, introspection, self-improvement."*

Both bugs fixed that day were keyword triggers misfiring (`"write "` → poem routed to a
tool; `"cat "` → haiku routed to a tool). A loop where she notices those from her own
logs, and the fix and its lesson both stay with her, is the point.

## The loop, on organs that already exist

| Step | Who / what | Exists? |
|---|---|---|
| **Notice** | her own turn review: `introspect_logs` (MCP) over gaia-core's log; the other mind's per-reply check (b9ts); samvega on dissonance; thought seeds | yes |
| **Match** | troubleshooting playbooks: `/knowledge/skills/ts-*/SKILL.md` (Unified Skills format, palace-indexed via `skill_palace`) | **started** (2 playbooks) |
| **Record** | an issue entry: symptom, evidence (log lines), suspected cause, status | **gap**: beads is the system of record, but no container has `bd`. Interim: CodeMind `detect_queue.jsonl` (`emit_detection`, source `user_request`/`code_review`), mirrored to a bead by the session that reviews it |
| **Propose** | CodeMind ANALYZE → PROPOSE (`build_fix_prompt`, `codemind_patch` search/replace blocks), candidate files only | yes, never closed on code |
| **Test** | candidate pytest for the touched module, plus a new regression test she writes | **gap**: `codemind_validator` does syntax + ruff + diff-safety only |
| **Validate** | the live symptom gone on `gaia-core-candidate` (replay the turn that showed it) | **gap**: no replay step |
| **Promote** | `promote_pipeline.sh`; vital organs (`agent_core.py`, `main.py`, …) need Azrael's approval — CodeMind's `VITAL_ORGANS` gate | yes (gated) |
| **Learn** | write or update the playbook that would have found this faster; record the outcome in the palace (`record_skill_outcome`); close the issue | partial (SkillCreator drafts from failure patterns) |

The **Learn** step is the recursion: every fix leaves a playbook, and the playbooks are
what the next Notice/Match step runs on.

## Honest baseline (2026-11-08)

CodeMind changelog since 2026-09-17: 492 cycles; 354 `no_detections`, 138 `complete`
— of whose 136 changes **none applied**: 134 were one recurring false-positive
heartbeat-drift detection (fixed upstream by `wlmf`), one multi-file, one drafted
playbook (`capgap_units_convert`). The machinery is live; it has never carried a real
bug from notice to promote.

## First run: `mjcg`

Symptom she can see from her side: *"Write a haiku about a cat watching snow"* →
log shows `Tool routing triggered for intent: brainstorming` on a creative turn.
Playbook `ts-misrouted-request` (written from `yn31`) covers this class. Success =
she, not Claude:
1. finds the misrouted turn in her logs,
2. names the cause (`"cat "` in `exec_indicators`, `AgentCore._should_use_tool_routing`),
3. proposes a candidate patch + regression test,
4. the test and the candidate suite pass, the haiku no longer routes on the candidate,
5. Azrael approves (vital organ), promoted,
6. she updates the playbook with what she learned.

Claude's role: watch, unblock infrastructure gaps (above), and record what broke —
not do her steps for her.

## Gaps to build (each its own bead)

1. Issue ledger she can write that lands in beads.
2. Test step in CodeMind VALIDATE (run the touched module's tests in the candidate).
3. Replay step: re-send the offending turn to `gaia-core-candidate` and compare.
4. A sleep task that runs the Notice step over the day's logs with the playbooks.

## Run log

- **2026-11-08, Notice probe** (before any gap work): asked to review the last hour
  for turns that took the wrong path, she chose `introspect` herself. The tool returned
  the last 50 lines (`start_time` ignored), which were her own current turn; she
  reported it as a past misroute and invented a second finding not in what she was
  shown. → `a0aw.6` (turn-level view over a window + findings must cite shown lines) is
  the first gap to close, before `mjcg` can be attempted fairly.
- **2026-11-08, Match**: `skill_palace.find_skills_for_topic` had always returned `[]`
  (wrong recall kwarg and result shape, mock encoded the same mistake). Fixed
  (`a0aw.5`); palace recall now surfaces `ts-gear-mismatch`; wording-sensitive.
