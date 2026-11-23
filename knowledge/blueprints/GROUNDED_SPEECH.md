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
  Live (Azrael's go-ahead, age limit lifted once): her 21:16:07 Grotto reply
  with the invented rooms now reads "Three rooms in the operational wing:
  problems (0 memories), preferences (0 memories), and milestones (4
  memories). … *(Edited: I checked my memory palace and corrected this.)*";
  `sv-e499ad9f` → amended. Her session history wasn't updated that time
  only because a day-old message has left the 8-message live window; within
  the 30-minute limit it is still there.
- 2026-11-22: **C4 live** (`aidw.7`). Sleep task `samvega_review` (every 2 h,
  `SAMVEGA_REVIEW_ENABLED`), `gaia_core/cognition/samvega_review.py`: up to 3
  active findings per run, oldest first. She is shown what she was asked,
  what she said, the checking call and what the source showed, and writes
  LESSON / ISSUE+SYMPTOM+CAUSE / DISMISS; code files an issue (source
  `samvega`, linked by `issue_id`, then the self-repair loop), notes a
  lesson, and records the question with the call that answers it in
  `/shared/selector_examples.jsonl` (`gaia_common/utils/selector_examples.py`,
  for A2). Verification is code's: the question is asked again as a private
  turn (`samvega-replay-*`, source `api`: no post, no wake, the audit skips
  it), the checking call runs again, `finding_grounding.confirm` compares;
  holds up → verified, doesn't → a note with what she said this time.
  Findings now record `question` and `turn_looked_up`.
  Fixed on the way (C1 bug): the world-state block counted as evidence, so
  the audit's own flags were "confirmed" against the clock without a lookup.
  Live, on the Grotto finding `sv-e499ad9f`: her lesson ("I should have
  checked the actual room list with palace(action=navigate, wing=operational)
  before describing them"), one selector example, and a replay answering
  "Three rooms in the Operational wing: problems (0 memories), preferences
  (0 memories), and milestones (4 memories)" → **verified**. The loop Azrael
  described, end to end: noticed, corrected in place, reviewed in sleep,
  verified.
- 2026-11-22: **A2 live** (`aidw.2`). Her selector sees the nearest of her own
  earlier questions with the call that answered them, found by **meaning**
  (all-MiniLM-L6-v2 cosine similarity, `selector_examples.nearest`), not by
  keywords; she decides whether one applies. Examples come from Samvega
  lessons (C4, ranked +0.05) and from her own lookups that worked (read-only
  calls). Threshold 0.40, measured: unrelated 0.08–0.28, same kind with a
  different subject 0.38–0.43, paraphrase 0.66+.
  Eval with examples (production Core, GPU, two passes): 14/14 lookups,
  12/12 left tool-free, 0.58 s; examples attached only where relevant.
  Paraphrase set: 5/5, and she adapts rather than copies (the Operational
  example → wing=technical / creative). One lesson-style example (test copy
  only) turned the blueprint miss (`manage(blueprint)` *generates* a
  blueprint) into `palace(read)` on the blueprint shelf for unseen
  services. Live: "Which rooms does your technical wing have?" retrieved the
  Grotto lesson (0.575), chose wing=technical, answered with the three real
  rooms, and left a new example.
  Limits: MiniLM misses some same-kind phrasings ("read me your blueprint
  for gaia-web" < 0.40); a wrong lookup that "works" can become an example
  until a Samvega finding corrects it.
- 2026-11-22: **Blind check** (`aidw.11`) replaces the word-overlap verdict
  wherever a model is at hand (`finding_grounding.confirm_blind`; switch:
  `GROUNDING_VERDICT=lexical`). No model sees a claim it is asked to approve:
  (1) from the claim alone, the questions it answers with its answers;
  (2) from the source alone, answers or NOT IN SOURCE, marked (complete)
  when whole; (3) short answers compared by code where they have a shape
  (yes/no, numbers, noun-phrase lists; disjoint lists = conflict), by the
  model otherwise (CONSISTENT / CONFLICT / ADDS / OFF-TOPIC).
  Measured (`scripts/confirm_eval.py`: a tuned set of 14 and three held-out
  sets of 10, each written before the next round of changes):

  | | tuned | held-out 1 | held-out 2 | held-out 3 | 7 real turns e2e |
  |---|---|---|---|---|---|
  | lexical | 11/14 | 9/10 | 8/10 | 6/10 | 7/7 |
  | blind (shipped) | 12/14 | 8/10 | 10/10 | 8/10 | 7/7 |

  Wins: negation, "loaded" vs "available", spoken numbers, paraphrase;
  "not in source" is unverified instead of a false alarm. Weak link: the
  comparison of two short answers. The 12B's holistic judgement is lenient
  (it called "rooms covering health monitoring…" consistent with "problems,
  preferences, milestones"), and the code rules keep needing patches (an
  earlier variant scored 41/44 on the sets but missed the Grotto rooms end
  to end). Stopped tuning there; next is an item-level model comparison
  (`aidw.12`). Known misses: hedges ("…among others"), spoken times ("half
  past three" vs 15:30).
- 2026-11-22: **Item-level comparison** (`aidw.12`) replaces step 3's holistic
  judgement and the list rules. Step 1 (claim alone) writes each answer's
  items separated by semicolons; step 3 shows the model those exact items
  next to the source's answer and asks, per item, HAS / CONTRADICTS / LACKS;
  code counts (a contradiction, or items a complete answer lacks, or nothing
  in common = conflict; all present = consistent; some lacking from a
  partial answer = adds). Code still settles plain yes/no, plain figures,
  and items whose figures all match the source's answer (whole numbers,
  2.5% for large ones). Bracketed detail and hedges ("among others",
  "roughly") are dropped from items. A list or figure answer from step 1
  must be in the claim's words (step 1 sometimes invents).
  Also fixed: a claim with no content word ("About 3 GB.") was treated as
  words she didn't say.

  | | tuned | h1 | h2 | h3 | h4 | h5 | h6 | real turns |
  |---|---|---|---|---|---|---|---|---|
  | lexical | 11/14 | 9 | 8 | 6 | 4 | 5 | 7 | 7/7 |
  | item-level | 13/14 | 10 | 10 | 10 | 8 | 9 | 8 | 7/7 (×2) |

  Honest reading: h1–h3 shaped aidw.11; h4 and h5 were traced during this
  work; **h6 is the only set never looked at before its one run: 8/10 vs
  7/10.** First runs of the fresh sets: h4 7 vs 4, h5 6 vs 5, h6 8 vs 7.
  Tried and reverted: a word-overlap guard on step 1's yes/no questions
  (it dropped legitimate rephrasings: "finished it" -> "finish reading").
  Known misses: spoken times against clock times ("twenty to eleven" vs
  22:40), an invented list item left unverified rather than confirmed,
  step 1 occasionally asking about what the claim doesn't say.
- 2026-11-22: **B live** (`aidw.8`, Azrael approved the vital-organ change).
  A deliberated reply is complete before it is sent, and the answer checks
  already run there, so no streaming change: `pre_send_check.py` runs after
  them on both deliberation paths in `agent_core.py`. Person-facing,
  non-social turns, Core on the GPU: up to 2 of her statements about
  herself are checked (C1 extraction, blind item-level verdict, lookup when
  the turn's evidence doesn't cover it). On the first contradicted one:
  does the source answer the question at all? (asked of the source alone;
  her tool list doesn't say which tools she uses most, and a rewrite from it
  had claimed all 17 were "the ones I reach for most"); if so, she answers
  again from it (lookup-turn shape), that answer is checked too, and it is
  what goes out; else the draft goes out for the audit. The question and the
  call become a selector example. The audit skips its own claim check when
  B found the reply fine or regrounded it. C3's rewrite got the same
  answerability gate.
  Live: her draft for "How many wings does your memory palace have?" was
  "Five — the Spire, the Citadel, the Foundry, the Arboretum, and the
  Observatory"; sent: "Six wings: technical, identity, operational, creative,
  samvega, and library." Cost: +0.2 s with nothing about herself, +1.7–2 s
  when statements check out, ~5 s when it regrounds (from 14 s: stop at the
  first contradicted statement, write a statement's questions once, run the
  same read-only call once a minute). Switch: `PRE_SEND_CHECK=0`.
- 2026-11-23: **Step 6, measured** (`aidw.9`). What retired and what stays:
  - `asks_about_palace` steps aside wherever her selector decides every turn
    (Core on the GPU): `selector_eval.py direct --examples`, palace cases ×2,
    keyword on 8/8, keyword off 8/8 (the blueprint→`manage` miss is the same
    either way). Off the GPU the selector doesn't run, so the keyword still
    routes there. `PALACE_KEYWORD_ON_GPU=1` brings it back. Live on
    production: Operational wing and palace overview via `[decide]`, both
    grounded; a poem stays tool-free.
  - **Verb aliases stay.** Asked 12 palace/Samvega questions twice, she
    still sent `palace.search` (1/2) and `samvega.dismiss` (2/2): the
    aliases still catch real calls. New verbs still go through tool_gaps.
  - `_SAMVEGA_LEDGER_RE` stays: it isn't a router, it keeps the founding-docs
    router from taking Samvega questions.
  - The number-word map stays: since aidw.12 it is part of the blind check's
    own code comparison (`_numerize`), not a patch.
