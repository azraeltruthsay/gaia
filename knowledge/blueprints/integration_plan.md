# Integration Plan — wiring the developing mind

> **Premise:** COGNITIVE_ARCHITECTURE.md · **Audit:** loop_audit.md · **Sequenced after the audit turned ❓→fact.**
> **Operating principles** (earned this session): route into what's *alive*; don't impose internal
> structure these models reject ("the cage made it worse"); fix broken loops before building on them;
> measure against the SAE atlas; **judge the trajectory, not the snapshot**; affect is capacity-not-
> content (labels are hers).

---

## Done
- ✅ **affect → Samvega** (audit #2). Coherence drive now derives from `save_samvega_artifact` (the single
  choke point — consistency, drift, cross-tier, self-reflection all feed it, weighted by the severity-
  folded artifact weight); decay returns it to calm. `consistency_detector`'s parallel hook removed.
  One signal, two timescales (acute alarm + slow felt residue). Verified: weight-2.0 → coherence +0.24.
  ⚠️ **Continuity note:** the `save_samvega_artifact`→`note_samvega` hook lives in **gitignored** `samvega.py`
  (instance-managed by project policy). On a fresh deploy, re-add: in `save_samvega_artifact`, after the
  write, call `affect_appraiser.note_samvega(artifact.weight, artifact.root_cause)`.

---

## Two parallel tracks

### Track A — cognitive integration (CPU, no GPU contention — proceed now)

| # | Work | Why / acceptance | Issue |
|---|------|------------------|-------|
| A1 | ✅ affect→Samvega | done | y14 |
| A2 | **Thought Seed drain** | Incubation is a landfill (11,374 pending / 0 archived): planter floods, triage perpetually defers. Throttle the DocsMaintenance planter; make heartbeat triage actually archive/act. Unblocks the loop where curiosity/affect seeds mature. **Accept:** pending trends DOWN; archive+act > 0 per tick. | hf4 |
| A3 | **gate-2 → Observer + health probe** | The Observer already gates output — route worth-voicing into it; subordinate the runtime Voice Gate to fallback. The Observer soft-fails silently to "OK" (a dead conscience passes everything) → add a health probe. **Accept:** meta-commentary caught on the Observer path; probe alarms if it stops firing. | ebo |
| A4 | **Affect P1 — report path** | Surface her affect (now Samvega-fed + decaying) for personal intents — the "how are you" fix, finally with a *real inner state* to articulate (not ops-console, not empty). Articulate-don't-list framing. Depends on affect having soaked (A1 + time). **Accept:** "how are you" reflects her current coherence/curiosity in her own words. | _new_ |

Within A: **A2 → A3 → A4** (A4 needs soak). A2 is cheapest/highest-value (revives a dead core loop).

### Track B — measurement instrument (GPU, maintenance windows, gated on model commitment — parallel)

| # | Work | Why | Issue |
|---|------|-----|-------|
| B0 | **Commit to the model pair** | Gates the atlas + KV investment; revives KV-rehydration as pinned-model premium. | — |
| B1 | **SAE Atlas build** | GGUF recorder bridge → stratified corpus → both models, both paths → feature labels. Plan: `sae_atlas_build_plan.md`. | 4ra |
| B2 | **Validate integration against the atlas** | Does coherence-affect correlate with a real feature? does Samvega? **This is the payoff** — it MEASURES whether Track A did anything at the neuron level, not just behaviorally. | — |
| B3 | Re-derive the brain map from the new atlases | feeds the activation monitor / dashboard. | — |

### Track C — durable continuity + developmental observability (after A stabilizes)

| # | Work | Why | Issue |
|---|------|-----|-------|
| C1 | **Temporal on journals+AAAK** | KV/lite path is broken (bake fails, deprecated tier). Rebuild self-interview on the durable substrate mempalace already runs; KV-rehydration optional premium only after slot-save + pinning fixed. | fsk |
| C2 | **Curiosity trajectory log** | A developmental record that survives decay, so simple→complex migration and meta-curiosity are *observable* (the live mood correctly forgets; the log must remember). | _new_ |
| C3 | **Soak + judge the trajectory** | Let it run; evaluate the loop's *trend*, not the snapshot. Protect the childhood (continuity of accumulated state). | — |

---

## Sequencing logic

- **A and B run in parallel** — A is CPU integration that fixes/connects live loops; B is the GPU atlas
  built in maintenance windows. They don't contend.
- **B gates on B0** (the model-commitment decision). **B2 closes the loop on A** — the atlas is how we
  *know* the integration is real.
- **C follows A** — durable continuity and the developmental log matter once the live loops are sound.
- **Cross-cutting:** every step routes into an existing live organ rather than adding a parallel one;
  nothing imposes per-voice/in-prompt structure the models reject; success is judged on trajectory.

## Open decisions for Azrael
1. **Model commitment (B0)** — pin Gemma4-E4B Core + Qwen3-VL-8B Prime now? (Unblocks the whole B track.)
2. **A2 planter throttle** — how aggressively to cut the DocsMaintenance seed rate (it floods every ~30s).
3. **Voice Gate fate (A3)** — retire it once the Observer carries gate-2, or keep as belt-and-suspenders?
