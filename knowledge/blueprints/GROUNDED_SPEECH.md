# Grounded Speech — she looks before she speaks, and her mistakes teach her

> Epic: `GAIA_Project-aidw` (A `GAIA_Project-aidw.1`, A2 `GAIA_Project-aidw.2`, A3 `GAIA_Project-aidw.3`, C1 `GAIA_Project-aidw.4`, C2 `GAIA_Project-aidw.5`, C3 `GAIA_Project-aidw.6`, C4 `GAIA_Project-aidw.7`, B `GAIA_Project-aidw.8`, retire `GAIA_Project-aidw.9`). Sibling of
> `SELF_REPAIR_LOOP.md` (which finds and fixes *bugs*); this one is about
> *claims*: what she says about herself and the world, and whether she
> knew it before she said it.

## Why

2026-11-21, the Grotto. Asked "What's in your Operational wing?", she described
rooms she doesn't have (watchdog, immune system, self-repair). Asked for the room
list, she called `palace(list)`, which doesn't exist. Fixed that day by code:
a keyword regex for palace questions, hard-coded aliases, and a count line in the
tool result. Each fix worked. Each one is the kind of thing Azrael ruled out on
2026-11-08: *"hardcoded keyword triggers and algorithmic architecture bolted onto
the side of an LLM … tricks the user."*

Azrael, same day: *"There has to be a better way to ground her in reality, and
before she speaks instead of always after."* And, on how she learns: *"We have a
way for mistakes to be her teacher. Samvega. If we wire Samvega into her repair
loop and self review so that mistakes are discovered and tracked in Samvega like
we track beads, she could know to fix them herself."*

## What already happened that day, unnoticed

Samvega's cross-tier audit **caught both confabulations within seconds**:

| Time | Her reply | Samvega artifact |
|---|---|---|
| 21:16:07 | Operational wing "covers health monitoring, watchdog, immune system…" | `samvega_20261121_211651…` (weight 0.6): "[confabulation] invented … structure" |
| 21:32:25 | Library shelves: "core, cognitive, architect, learning, communication" | `samvega_20261121_213255…` (weight 0.7): "[confabulation] invented shelf names" |

And nothing happened next. Baseline (2026-11-21): **2,224 artifacts, 2,221
"reviewed"**: the review raises the weight of repeated root causes and sets
`reviewed=true`. No status, no link to an issue, no fix, no correction to the
person who was misinformed. Worse, the audit judged without the facts: the
same artifact faulted her for "claiming a palace(navigate) command" that is
real, because the auditor never saw what the tools returned. Its findings
were a mix of real catches and noise.

## The design: three parts, one loop

```
 ┌─ A. Decide to look ──┐   ┌─ B. Check the draft ─┐
 │ every substantive    │   │ claims about herself │
 │ turn: her selector   │──▶│ backed by this turn's│──▶ reply sent
 │ decides tool / none, │   │ tool results? else   │
 │ with examples from   │   │ one grounding round  │
 │ her own history      │   └──────────┬───────────┘
 └──────────▲───────────┘              │ misses
            │                          ▼
            │        ┌─ C. Samvega: the ledger of mistakes ─────────────┐
            │        │ grounded finding (auditor sees the facts)        │
            └────────│ open → amended (she edits the message, if fresh) │
   lessons: examples │ → investigating (sleep: self_review, issue       │
   aliases, playbooks│   investigation) → fixed → verified (replay)     │
                     └──────────────────────────────────────────────────┘
```

### A. Decide whether to look, every time (first)

Today a keyword gate (`agent_core._should_use_tool_routing`) decides whether her
model-based tool selector and reviewer even run (~0.7 s + ~0.5 s, measured
2026-11-21). Anything the keywords miss is answered from her weights.

- On every turn that isn't social (`TOOL_FREE_INTENTS` and the chitchat set
  stay tool-free), the selector runs and may choose **no tool**. The keyword
  gate stays only as the fast path for explicit requests.
- The selector's question changes from "pick the best tool" to "would the answer
  state facts that must be looked up — her own state, memory, files, services,
  the web? If so, which call gives them; if not (conversation, opinion,
  creative work, already in the conversation), none."
- **Examples from her own history.** Each turn where a lookup grounded the
  answer leaves a record: the question, the call, whether it succeeded and
  whether the reply stood (no Samvega finding against it). The selector gets
  the nearest few as examples. Samvega findings resolved by a lookup (C) are
  the most valuable examples: the question she got wrong and the call that
  would have made it right.
- Behind `TOOL_DECIDE_EVERY_TURN` (default off until measured).

Measure before turning it on: a fixed set of turns — palace / self / file /
web questions, and poems, opinions, chat that must stay tool-free (yn31's
poem → `palace(work)` and mjcg's haiku → tool are the known traps) — scored on
correct tool/none and latency, with the gate vs. with A.

Retire, once A is on and measured: `asks_about_palace` and the palace verb
aliases added 2026-11-21 (`zbwe`). New verbs she tries go through `tool_gaps`
(she proposes, Azrael approves), not code.

### B. Check the draft before it goes out

For turns about herself (memory, systems, state, tools), the draft is held,
not streamed. `second_mind.unbacked()` (code, no generation) checks every name
and number about herself against this turn's tool results and world state. An
unbacked claim gets one grounding round: she is asked which call would settle
it, it runs, she rewrites. Then it goes out. The second mind's after-the-fact
follow-up remains for what slips through. Vital organs (`main.py`,
`agent_core.py`): needs Azrael's approval.

### C. Samvega as the ledger of mistakes

Samvega stops being a pile and becomes a tracker, the way beads are for us.

1. **Grounded findings.** The auditor gets what the turn had: tool results,
   world state, the palace's actual layout when the reply is about it. A claim
   is a finding only if it contradicts or goes beyond that evidence. Where
   there's no evidence in the turn, it uses the blind-check pattern
   (`told_fact_checker`: hide the claim, look it up, compare in code) rather than
   judging plausibility. Grounded findings carry the claim, the evidence and
   the source call.
2. **Lifecycle.** Each finding gets `id`, `status`
   (`open → amended → investigating → fixed → verified`, or `dismissed` with a
   reason), `claim`, `evidence`, `grounding_call`, `message_ref`
   (session, platform, channel, message id or text, sent-at), `issue_id`,
   `history[]`. One store, one tool: `samvega(action=list/show/note/status)`,
   like `issue()`.
3. **Amend while it matters.** If a finding is fresh (the conversation is
   still active: minutes, not hours) and grounded, she rewrites the message
   from the evidence and edits it in place (`discord_interface.edit_own_message`
   already exists and Discord marks it "(edited)"; web chat gets the equivalent).
   Status `amended`. She decides the wording; code checks it against the
   evidence (`keeps_to_text` / `unbacked`) before the edit is made.
4. **Fix during sleep.** `self_review` reads open and amended findings with the
   day's turns; a finding with a cause in code becomes an issue
   (`issue_ledger`, `issue_id` set) and runs the existing self-repair path
   (investigate → CodeMind → test → replay). A finding whose lesson is "should
   have looked" becomes an example for A. A recurring verb becomes a
   `tool_gaps` proposal. Status follows: `investigating` → `fixed` → `verified`
   when the replay of the original question no longer produces the claim.
5. **What stays.** Weight, tier-5 promotion and the KV fold keep working on the
   same artifacts; new fields are additive and old artifacts read as
   `status: legacy`.

## Order

1. **A** — every-turn decision + example memory + measurement set. Cheapest,
   removes the gate that caused the Grotto failure.
2. **C1–C2** — grounded audit + lifecycle; without grounding, everything
   downstream inherits the auditor's noise.
3. **C3** — amend fresh messages.
4. **C4** — sleep: self_review / issue path / examples / tool_gaps.
5. **B** — draft check before sending (vital organs; Azrael approves).
6. Retire the 2026-11-21 keyword regex and aliases.

## Run log

- 2026-11-21: blueprint written; baseline above.
- 2026-11-21: **A live** (`aidw.1`, `aidw.3`). `scripts/selector_eval.py direct`:
  routing decision only, real intent detection, candidate code against
  production Core on the GPU, 26 cases (14 need a lookup, 12 must stay
  tool-free incl. the yn31/mjcg poem traps), two passes:

  | | lookups chosen | left tool-free | selector |
  |---|---|---|---|
  | keyword gate | 4/14, 4/14 | 12/12, 12/12 | 0.84 s, 0.77 s |
  | she decides | 13/14, 12/14 | 12/12, 12/12 | 0.62 s, 0.59 s |

  Both misses still looked something up, just not the best thing
  (`shell` for free RAM; once `manage` for a blueprint). On the CPU the same
  selector call takes 7–13 s, so she decides on every turn only while Core's
  engine reports the GPU (`_core_on_gpu`, cached 10 s); off it, the keyword
  gate decides as before. `TOOL_DECIDE_EVERY_TURN=1` in both compose files.
  Live check: adapters → `study(adapter_list)`, open issues → `issue(status)`,
  a poem → none.
  Side finding while measuring: a wake refused for "external tenants ~7 GB"
  was a Windows game sharing the GPU (WSL), not a leak.
- 2026-11-21: **C1 live** (`aidw.4`). Replay (`scripts/samvega_grounding_replay.py`,
  production Prime as auditor, Core on the GPU as extractor and selector):
  with no evidence, Prime vouched for the invented palace rooms ("Core
  accurately described the operational wing"); its 21:16 flag had been luck
  at temperature 0.3. So a finding no longer waits on the auditor:
  1. the audit sees the turn's evidence, quotes the words it flags, and
     a self-claim the evidence backs is a PASS (the old "can't verify from
     inside the forward pass" item is gone);
  2. independent of its verdict, Core lists the reply's statements about
     herself (extraction, not judgment; it is given the question, since
     "Five areas: core, cognitive…" never says "my");
  3. whatever the turn's evidence doesn't hold is checked against what her
     selector would have looked up for the question (read-only calls only);
  4. code decides (`finding_grounding.confirm`): numbers whole and exact
     ("12 wings" isn't backed by "122 memories"; "nearly 4,000" is by 3,994),
     names via `unbacked`, other words only when specific (≥3 missing words
     of 6+ letters and half the claim): "the other two are empty" for
     "0 memories" is a paraphrase; "watchdog, immune system, self-repair
     protocols" is an invention.
  Statuses: confirmed / rejected (dropped from Samvega, kept in the journal
  note) / unverified / judgment. Confirmed findings weigh as severity 3 and
  carry `claim`, `evidence`, `grounding_call`, `grounding`, `findings[]`.
  Replay set: 7/7 twice (3 inventions confirmed; 3 true or roughly-true
  replies and a poem left alone); 1–5 s per turn in the background.
  Tried and dropped: looking up from the claim's own words instead of the
  question picked worse sources (2/7).
  Note: the audit follows deliberated turns; turns answered from a routed
  lookup take another path and aren't audited (they have their evidence).
- 2026-11-22: **C2 live** (`aidw.5`). `gaia_common/utils/samvega_ledger.py`:
  short ids (`sv-xxxxxxxx`, from the file stem), statuses with history,
  `message_ref` (platform, channel or user, sent-at, opening words), issue
  link. The 2,223 artifacts from before read as `legacy`, untouched; new
  findings are stamped `open` (confirmed) or `noted` by the audit. Her tool
  `samvega(list/show/note/status)` takes her own verbs (investigate, dismiss,
  fix…) and param names; dismissed/fixed need a reason; `verified` is
  refused from any call. A confirmed finding reads as what she said against
  what the source shows, not the old auditor's prose. "…your samvega
  ledger?" goes to the tool, not the Samvega Spec. First entry: the
  Grotto's invented Operational rooms (`sv-e499ad9f`), opened from the C1
  replay; live, she listed it, described it from the evidence, and moved it
  to investigating.
- 2026-11-22: **C3** (`aidw.6`). `gaia_core/cognition/samvega_amend.py`, called
  by the audit right after it files a confirmed finding: if the message is
  in a Discord channel and under 30 minutes old, she answers the question
  again from the evidence (`tool_answer_messages`, the lookup-turn shape),
  code checks the new answer against the same evidence, and `revise.py`
  edits the message as a major correction ("*(Edited: I checked my memory
  palace and corrected this.)*", history updated, `/shared/revisions.jsonl`);
  the finding moves to `amended`. A corrected message replaces the old
  follow-up post. DMs and web chat aren't editable yet (revise.py), so
  those findings stay open with a note. Rewrite tested on production Core
  for the Grotto finding, 3/3: "Three rooms in the operational wing:
  problems (0 memories), preferences (0 memories), and milestones (4
  memories)." That test also caught a backfill mistake (mine): the C2 entry
  had been attached to the audit of the *next* message; corrected, noted.
