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
