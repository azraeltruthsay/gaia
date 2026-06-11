# CFR-as-Virtual-Memory for Conversation — Implementation Plan

> **Status:** Draft (2026-06-11) · **Author:** Azrael + Claude · **Tracks:** bd CFR-conversation epic
> **Premise:** Treat conversation context as *virtual memory*. The token window is RAM;
> CFR's on-disk resolution tree + vector index is an arbitrarily large backing store.
> Relevance is the page-replacement policy; CFR_EXPAND is the page-fault handler.
> Realizes `project_cfr_universal` for chat and kills recency-bleed as a side effect.

---

## 1. The model

| Virtual memory | CFR primitive | Conversation meaning |
|----------------|---------------|----------------------|
| Physical RAM | the token window (Core 4k / Prime 16k) | what the model actually attends to this turn |
| Resident page | **FOCUS** | a turn/topic at full text |
| Swapped (thumbnail) | **SUMMARY** | a turn/topic compressed to its gist |
| Swapped out (~0 tok) | **BLUR** | dropped from the window, still on disk |
| `mlock` (pinned) | **ANCHOR** | never evicted: identity, the active goal, user-pinned facts |
| Backing store | CFR tree (full text on disk) + vector index | every turn ever — lossless, retrievable |
| Page-replacement policy | **relevance × recency-decay score** | which turns are resident this turn |
| Page fault handler | **CFR_EXPAND** | page a blurred/archived item back to full res, on demand |
| Eviction / aging | **CFR_PRUNE** | age the working set down: FOCUS→SUMMARY→BLUR→archive |

**Invariant:** at any instant, `sum(FOCUS tokens) + SUMMARY + ANCHOR ≤ window budget`. The *working
set* fits; the *corpus* need not. Full text is **always** on disk (lossless), so any eviction is recoverable.

---

## 2. What we REUSE (this is extension, not greenfield)

- **`cfr_manager.py`** — the resolution-tree primitives (FOCUS/SUMMARY/expand, on-disk persistence,
  full-text-always-retained). Adapt `CFRSection` (document chunk) → a conversational **turn/topic** node.
- **`session_history_indexer.py`** — already embeds turn-pairs + topic summaries (cosine 0.45/0.40).
  This IS the relevance scorer — **no new embedding infra needed.**
- **`context_compactor.py`** — the existing recent/middle/old 3-zone compactor is a *positional* mini-CFR.
  We replace its position-based zones with **relevance-based** FOCUS/SUMMARY/BLUR. Same output seam into the prompt.
- **`recency.decayed_relevance(confidence, age, fact_type)`** (KG kernel) — reuse for the
  `relevance × decay` blend so conversation and facts share one scoring model.
- **`session_manager.summarize_and_archive()`** — already writes turns + a vector index to disk at 20
  messages. That orphaned write IS the backing store; we wire a reader to it.

**Design refinement (latency):** the document-CFR spec uses an LLM "planning pass" to assign resolution.
For *conversation* that's too slow per turn. So: **the per-turn replacement policy is embedding-only
(cheap, already computed); LLM summarization runs at EVICTION time (async / sleep-cycle), never on the hot path.**

---

## 3. Phases

### Phase 0 — Scaffolding (no behavior change, flag-gated)
- New `ConversationCFR` (gaia-core/memory) wrapping the working set: nodes = `{turn_id, role, text,
  summary?, resolution, relevance, last_focus_ts, anchored}`.
- `score(node, current_msg) → relevance × decay(age)` reusing indexer embeddings + `recency.py`.
- `assign_resolution(nodes, budget) → FOCUS/SUMMARY/BLUR` (the replacement policy; pure function, testable).
- Behind `CFR_CONVERSATION_ENABLED` (default off). Unit-test the policy in isolation.

### Phase 1 — Replacement policy = the bleed fix (smallest valuable surface)
- **Replace** the recency sliding-window + greeting heuristic (`agent_core.py:606–627`) with
  `assign_resolution` over the candidate turns.
- Per turn: score every candidate by `relevance(current) × decay(age)`; **FOCUS** top-K above a floor,
  **SUMMARY** the mid band, **BLUR** (drop) below the floor. Keep a **1-turn recency anchor** (the
  immediately-prior turn) for "as I just said" continuity regardless of score.
- Run on **every** turn — kill the greeting bypass that skips relevance.
- Feed the result through the existing `context_compactor`/`relevant_history_snippet` seam → prompt.
- **Acceptance:** "Good morning" after a clock chat pulls **zero** clock turns into FOCUS; an on-topic
  follow-up still FOCUSes the relevant prior turn; working-set tokens stay under budget.

### Phase 2 — Fault handler (EXPAND on demand)
- Wire **CFR_EXPAND** as a page-fault: mid-generation GAIA can page a SUMMARY/BLUR/archived item back to
  FOCUS. Two triggers:
  - **Explicit signal** — she emits a `<cfr_expand turn="…">` tag, intercepted in the stream loop
    (same mechanism as existing CFR signals + the tool-call interceptor in `main.py:1376`).
  - **Auto-fault** — if her draft references an entity whose only record is BLUR/archived, auto-expand it
    and regenerate (a guarded continuation, like the native tool-call continuation).
- Recovery path for mis-eviction. **Acceptance:** ask about something dropped from the window 30 turns
  ago → she expands + answers correctly instead of confabulating.

### Phase 3 — Backing store + aggressive aging (the archive tier)
- `SessionArchiveRetriever` spanning **current + archived** vector indexes; results tagged with a
  **decay factor** (`[from 4 days ago, weak]`) so the model discounts them.
- Aging policy (config `RETENTION_POLICY`): active FOCUS → SUMMARY (soft) → BLUR/archive (evict) →
  hard-TTL purge. `CFR_PRUNE` sleep task does the aggressive eviction off the hot path.
- Full text always retained until hard-TTL → lossless, retrievable backing store.
- **Acceptance:** working set shrinks aggressively; aged turns are still answerable via expand/RAG; a
  metric logs what was evicted/archived (no silent loss).

### Phase 4 — Unify the relevance kernel
- One `relevance = confidence × decay(age, type)` scorer across **conversation turns + KG facts +
  documents**, so every context surface shares the foveation model and one budget. Retire the
  parallel positional compactor + ad-hoc sliding window.

---

## 4. Open design decisions (need your call)

1. **Relevance floor / aggressiveness** — how eagerly to BLUR. Start conservative (only drop clearly-
   unrelated turns) and tighten, or start aggressive? (Affects bleed-vs-continuity tradeoff.)
2. **Recency anchor size** — 1 turn (tight) or 2 (safer continuity)?
3. **Budget split** — what fraction of the window for FOCUS vs SUMMARY vs ANCHOR.
4. **Auto-fault vs explicit-only EXPAND** — auto is smoother but riskier (extra inferences); explicit is
   safe but depends on GAIA noticing she needs it.
5. **Summary freshness** — pre-summarize on eviction (sleep) vs lazily on first SUMMARY use.

## 5. Risks & guardrails
- **Mis-eviction (the core risk)** — bad relevance scoring drops needed context. Mitigations: the 1-turn
  recency anchor, ANCHOR for pins, EXPAND fault recovery, conservative initial floor, **log every
  eviction** (no silent truncation).
- **Lossy in-window summaries** — true gist may drift; EXPAND recovers lossless truth from disk.
- **Latency** — embedding-only hot path; summarization async; EXPAND amortized.
- **Live-cognition safety** — candidate-first, `CFR_CONVERSATION_ENABLED` flag, phase-gated, A/B against
  the current path before cutover.

## 6. File seams
- `agent_core.py:606–627` — sliding window / greeting heuristic → replacement policy (Phase 1)
- `context_compactor.py` — positional zones → relevance resolution (Phase 1)
- `session_history_indexer.py` — relevance scorer source (reuse)
- `gaia_common/utils/recency.py` — decay kernel (reuse)
- `cfr_manager.py` — node/resolution primitives (adapt for turns)
- `main.py:1376` stream loop — CFR_EXPAND signal interception (Phase 2)
- `session_manager.py:200–257` archival — backing store reader + retention (Phase 3)

