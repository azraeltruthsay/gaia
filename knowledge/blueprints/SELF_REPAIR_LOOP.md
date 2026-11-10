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
| **Notice** | `introspect(action=turns)` (a0aw.6): her turns over a window (default 24 h), one block each — asked, intent, routing fired / selector choice, tools run (routed and mid-reply), which mind answered, reply head; her current turn and earlier self-reviews left out; warnings that recur across turns dropped. `logs` honours since/until. Also: the other mind's per-reply check (b9ts), samvega, thought seeds | yes |
| **Match** | troubleshooting playbooks (`/knowledge/skills/ts-*/SKILL.md`, Unified Skills format, palace-indexed). Playbooks with `seen_in: [turns, …]` have their Symptoms listed under the turn view (ycx7): criteria read from her own files, no verdicts in code | **yes** (ycx7) |
| **Record** | the `self_review` sleep task (a0aw.4, `gaia_core/cognition/self_review.py`) does Notice → Match → Record in one pass: turn view (oldest first from a cursor) + playbook symptoms in, her `ISSUE/PLAYBOOK/EVIDENCE/SYMPTOM/CAUSE` blocks out, filed by code; every review kept with her full answer in `reviews.jsonl`. Also her issue ledger: `issue(action=file / note / status / list)` → `/shared/issues/ledger.jsonl` (`gaia_common/utils/issue_ledger.py`); evidence required, duplicates of open issues refused, 10/day cap. Host bridge `scripts/gaia_issue_bridge.py` makes each a bead (`gaia-found`) and writes bead id + status back (`beads.json`) | **yes** (a0aw.1); bridge timer in `scripts/systemd/`, installed by hand |
| **Propose** | CodeMind ANALYZE → PROPOSE (`build_fix_prompt`, `codemind_patch` search/replace blocks), candidate files only. Her open issues reach it once she names where (`issue(note/status, file=…)`, or a file/function in her cause or notes): `gaia_common/utils/codemind_issues.py` (a0aw.7) turns them into direct-fix detections, a large file is shown as the function she named, and the outcome goes back to her issue ('proposed' with the test result, or a note why not; 2 tries, a retry only after she adds something) | **yes** (a0aw.7); she can't yet find the place herself in sleep (a0aw.9) |
| **Test** | candidate pytest for the touched module, plus a new regression test she writes | **yes** (a0aw.2): CodeMind's VERIFY runs the tests named after or importing the touched module in `gaia-core-candidate` (orchestrator `POST /candidate/test`, `gaia_common/utils/codemind_tests.py`); failure restores the file and goes back to PROPOSE once with the report. No covering test = not validated. She can't write the regression test yet |
| **Validate** | the live symptom gone on `gaia-core-candidate` (replay the turn that showed it) | **gap**: no replay step |
| **Promote** | `promote_pipeline.sh`; vital organs (`agent_core.py`, `main.py`, …) need Azrael's approval: a CodeMind fix touching one is written to `/shared/doctor/architect_queue/`, never `restart_requests/` (the `is_vital_organ` gate existed but nothing called it until a0aw.7) | yes (gated) |
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

1. ~~Issue ledger she can write that lands in beads.~~ Done (a0aw.1).
2. ~~Test step in CodeMind VALIDATE (run the touched module's tests in the candidate).~~ Done (a0aw.2).
5. ~~Her issue becomes a CodeMind detection; a direct-fix path for a one-file behavioral bug.~~ Done (a0aw.7).
6. CodeMind writes a regression test with the fix (a0aw.8).
7. She investigates an open issue in sleep and names where the bug is (a0aw.9).
3. Replay step: re-send the offending turn to `gaia-core-candidate` and compare.
4. A sleep task that runs the Notice step over the day's logs with the playbooks. **Done** (`a0aw.4`): `self_review`.

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
- **2026-11-08, Record** (`a0aw.1`): issue ledger live. She appends events through the
  `introspect` domain; the host bridge turns them into beads and mirrors notes, status
  and closure back. Tested end to end on the candidate (bead `nilf`, closed as a test).
  "fixed" from her adds `gaia-fixed-unverified`; a bead closes only after promotion.
- **2026-11-08, Notice** (`a0aw.6`): turn view live; everything introspect shows her is
  kept 2 h (`shown.jsonl`) and an issue she files herself needs at least one evidence
  line that matches it. Probes in both gears: the selector needed the action described
  (it took `recall` for "look back"), a default window (it sees action names, not
  params), and her earlier reviews hidden (she repeated their "nothing wrong"). After
  that: `turns` chosen 3/3 in gear 1, she was shown the haiku misroute and judged it
  fine — no invented findings, but no criteria either → `ycx7` (Match). The view also
  surfaced two real bugs she can find herself: `bz6f` (GAIA-CORE-081, native tool call
  abandoned mid-stream) and `av4l` (claimed a check after an invalid `introspect.trace`).
- **2026-11-08, Match** (`ycx7`): playbook symptoms listed under the turn view. Gear 2
  (Prime), 6 clean probes: never invented a turn; named the haiku misroute (`mjcg`) by
  line with the right playbook once, and unprompted found `weco` ("Excalibur. ."),
  `afsf` (grounding on a poem), a missing vector index (3/6, filed `guga`), a checkpoint
  warning (`70am`); some reasoning confused. Two contamination fixes on the way: whole
  review conversations are left out of the view; playbooks opt in by `seen_in` (the
  gear playbook made her claim gear mismatches no turn showed). The ledger got its own
  `issue` domain (she reached for `issue(action=file)` before it existed).
- **2026-11-08, Record in chat**: not yet. Asked to file what she found, she guessed
  shapes, then repeated "the issue tool isn't implemented". Blocked by `bz6f`, by tool
  results not carrying into the next turn, and by echoing her own prior reply →
  `6839`: do Notice→Match→Record inside one task (`a0aw.4`).
- **2026-11-09, Record in sleep** (`a0aw.4`): `self_review` sleep task (P4, every 4 h,
  `SELF_REVIEW_ENABLED`). Trial over the 48 h backlog (Prime, 24 turns, 4 batches, scratch
  ledger): she filed the haiku misroute (`mjcg`'s symptom) by its line under
  `ts-misrouted-request`, unprompted. Also the poem routed to palace whose reply described
  the poem, a `context` 500, "Excalibur. ." (`weco`), stale 'nano' load calls; 2 of 7
  doubtful. Three guards came out of the trials: she cited the request itself as evidence
  (Capital of France, a turn where routing didn't fire) → evidence that is only the
  request is set aside; she named playbooks that don't exist → names checked against
  those shown; she joins a quote and a log line with a dash → parts checked separately.
  Log times are said to be UTC (she called 8:08 AM PST = 16:08 UTC wrong).
- **2026-11-09, first live self_review** (in sleep, 23 turns, 3 batches, 167 s; Prime when
  up, Core otherwise): filed 3 → beads `z18l` (correct, but the same turn as `yn31`,
  already fixed: closed as duplicate, and the closure reached her ledger), `pcsb`
  (`context` 500, new), `iv01` (routing on two chat turns, new). `bz6f` fixed the same day:
  inline tool turns now finish in the stream path and failed calls reach her next turn.
  Still open on the chat path: she claims actions she never took (`9nxd`).
- **2026-11-09, second live self_review** (4 turns): filed 2. `2d3j` was the haiku turn,
  the first time she noticed the `mjcg` bug herself (closed as its duplicate; `mjcg` stays
  hers to fix). `jcio` was a false positive she couldn't have avoided: the lines were a
  test's fake selector output, written into the production log by a test run (`ba6o`:
  no log files under pytest, candidates log as `<service>-candidate`).
- **`9nxd` fixed**: a reply that claims a finished action ("Noted in the issue ledger",
  "the note was added successfully") is checked against the writes the session actually
  made (outcome ledger, plus every MCP call made by code during the turn). With none
  behind it, a correction is shown and kept in her history. Live: both 9nxd turns
  corrected; a real told-fact save and a greeting were not; asked again, she made a call
  and reported its failure instead of claiming success.
- **2026-11-10, Test** (`a0aw.2`): live run of the real cycle against the real orchestrator,
  model answers scripted: a proposal that broke `action_claims.claims` failed 11 of 35
  candidate tests, the file was restored and the report went back to PROPOSE; the second
  proposal passed 35 and was kept. On the way: tests can't run inside gaia-core (they write
  to /shared, the knowledge graph and chat history, which there are production's), so they
  run in the candidate container. Two gaps found for `mjcg`: no path from her issue to a
  CodeMind detection, and behavioral issues only ever get a blueprint, not a fix.
- **2026-11-10, Propose** (`a0aw.7`): her issues reach CodeMind once she names a file or
  function. Checked with the real `agent_core.py`: `AgentCore._should_use_tool_routing`
  resolves, and the prompt shows that function (3.8k of 589k chars). On the way: CodeMind's
  vital-organ gate was never called (a passing fix to `agent_core.py` would have written a
  self-deploy request; only an absolute-path mismatch stopped it), and the issue ledger
  read `GAIA_ISSUES_DIR` once at import, so a test wrote into the candidate's real ledger
  (cleaned). Her two open issues name no place yet: nothing is attempted until she does.
