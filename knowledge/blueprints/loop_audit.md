# Cognitive Loop Audit — verified (2026-06-12)

> Two parallel code-trace audits (conscience/deliberation/salience + memory/incubation/affect/temporal),
> reconciled against **live container probes** (the agents' sandboxes blocked `docker exec`; the live
> check corrected two of their predictions). Status is now fact, not inference.

## Verified status

| Organ | Status | Live evidence |
|-------|--------|---------------|
| **Observer** | ✅ **LIVE + gates output** | 39 fires; `OBSERVER_USE_LLM=1`; BLOCK aborts the stream, CAUTION→`reflect_and_refine`→Samvega; bicameral cross-tier (non-generating tier observes). Soft-fails to "OK" on exception (silent-degrade risk). |
| **Samvega** | ✅ **LIVE + consumed** | 1291 artifacts on disk, 1761 log hits. Read by journal consolidation, sleep introspection, tier-5 training — via **disk glob** (the `samvega_path` return value is discarded; the file is consumed). Weighted by observer severity (block 1.5×). |
| **mempalace + AAAK** | ✅ **LIVE + integrated** | live read on grounding (`knowledge_router._query_mempalace`, MemPalace queried first/highest-trust) + write (`palace_store`); real palace in gaia-mcp over MCP; **AAAK compresses every store**. Prior silent param-schema bug fixed. |
| **Deliberation** (the "Council") | ✅ **LIVE — but a diarized single mind** | 6 fires; `DELIBERATION.enabled=true`. observer/recaller/responder/introspector are **regex tags on one LLM `<think>` trace**, not separate calls. Strict per-voice separation was *tried and abandoned* — "the cage was making things worse." |
| **Affect appraiser** | ⚠️ **LIVE-but-unsoaked + PARALLEL to Samvega** | `AFFECT_APPRAISAL_ENABLED=1` in the running container; only 2 fires (curiosity+competence from manual tests); the **coherence path never fired** (consistency checks not exercised in window). **Duplicates Samvega:** both consume the *same* `consistency_detector` result (`:491` Samvega + `:509` `note_coherence`), un-joined. |
| **Thought Seed** | 🔴 **write floods / germination drains nothing** | mount is **RW** (the read-only Errno-30 was *stale/historical* — agent was right). But: **11,374 pending / 0 archived.** DocsMaintenance plants a seed every ~30s; heartbeat triage runs but reports `archive=0, pending=N, act=0` every tick — it **re-defers everything, resolves nothing.** Unbounded landfill. |
| **Council-note escalation** (Lite→Prime mid-turn) | 🔴 **DORMANT (orphaned)** | `_escalate_to_prime`/`_assess_complexity` have **no live caller**. (Sleep-cycle note writes still fire; the *request-path* escalation never triggers.) |
| **Temporal Interviewer** | 🔴 **BROKEN now** | `ERROR: TemporalState: bake failed` at `_save_lite_state` (KV slot-save fails); heartbeat shows `bake=False interview=False`. The 16 states / 2 interviews on disk are **stale** (Feb–Mar 2026, old Lite era). Calls `get_model_for_role("lite")` → no such tier → **falls back to Prime** (interviews the wrong self); docstrings/prompts reference deprecated Lite/nano. |

## The integration findings (what this changes)

1. **The Observer already IS gate-2-as-an-agent — and it works.** It gates output today. Our runtime
   Voice Gate is a slice of what the Observer does. **Integration = route gate-2 into the Observer**, not
   build it. (And add a health probe — its silent soft-fail means a dead conscience passes output
   through unnoticed.)

2. **Samvega is the mature affect pipeline; my P0 affect duplicates it.** One detector
   (`consistency_detector`), two un-joined consumers (Samvega event + affect `coherence` drive). Samvega
   is already emitted, severity-weighted, and consumed into journal/sleep/training. **Affect should plug
   INTO the Samvega pipeline (read/derive from it), not run a parallel coherence drive.** This is the
   single clearest integration win — and it's a duplication *I introduced* this session.

3. **Thought Seed is a landfill, not an incubator.** Input ≫ drain; triage resolves nothing
   (`archive=0/act=0` every tick). Two fixes: throttle the DocsMaintenance planter, and make triage
   actually archive/act (germinate) instead of perpetual defer. Until then, incubation is write-only.

4. **Temporal/KV is broken now → validates the journal+AAAK pivot.** Bake actively fails; the path is
   tier-confused (Lite→Prime) and references deprecated tiers. KV-rehydration is *not* a safe foundation
   today. Rebuild temporal continuity on **journals + AAAK** (which mempalace already does well);
   keep KV-rehydration as an optional premium only after slot-save + model-pinning are fixed.

5. **Convergence worth noting:** deliberation abandoned strict per-voice structure because "the cage made
   it worse" — the *same* lesson we learned when in-prompt gate-2 backfired on Gemma4. The architecture
   independently discovered that these models resist imposed internal structure.

## Net

The nervous system is **more alive than the premise feared** — Observer, Samvega, mempalace, and
deliberation are all live and integrated. The damage is concentrated: **three broken/stuck loops**
(Thought-Seed backlog, Temporal/KV, orphaned Council escalation) and **one duplication** (affect↔Samvega,
self-inflicted). The integration plan now has concrete, prioritized targets instead of `❓`s.
