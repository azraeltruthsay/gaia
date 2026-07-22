# GAIA Self-Restart & Deployment Roles — Design Standard

> **Status**: **v1 IMPLEMENTED** (2026-07-16) · **Beads**: `GAIA_Project-kmcb` (implemented), `GAIA_Project-r67d` (candidate parity, done)
> **Scope**: How GAIA autonomously deploys her own code changes that require a container restart — with no negative impact and no outside influence.

## v1 Implementation Notes (2026-07-16)

Implementation revised the division of labor: the **orchestrator is the hands**
(it already had git, `/candidate/snapshot`, `/containers/*` and docker manager;
kmcb fixed its silently-broken rollback — repo was mounted `:ro` — and added
`POST /deploy/rollback` + `POST /deploy/commit`), while the **doctor is the
deadman supervisor** (manifest watcher in `doctor.py`, gates + deadman +
rollback trigger). A process still never restarts itself; the supervisor never
supervises its own restart (manifests targeting `doctor` are rejected).

- Manifest schema: `contracts/schemas/restart_manifest.yaml`; drop-box
  `/shared/doctor/restart_requests/`.
- **Auto-commit on success** (Azrael decision 2026-07-16): orchestrator
  `/deploy/commit` as `GAIA Self-Deploy <gaia@localhost>`, local only, never
  pushed. Each success rotates `/shared/doctor/lkg.json` — the rollback anchor.
- Rollback = `git checkout <LKG SHA> -- <service dirs>` in **both** trees
  (candidate parity preserved on rollback), then container restarts.
- Deadman v1 = deep `/health` for every deployed service within
  `DEADMAN_TIMEOUT` (300s); the cognitive-battery slice is opt-in
  (`DEADMAN_RUN_BATTERY=1`) because CPU-gear inference would false-fail
  GPU-calibrated expectations.
- Kill switch: `SELF_DEPLOY_ENABLED` (compose sets 1 for production doctor).
- **One judge at a time (drill 3 lesson)**: the doctor's pre-existing reflexive
  immune machinery (structural audit → Tier-2 HA Surgery) raced the first
  failure drill — it healed the deliberately broken file mid-deploy, so the
  deadman blessed a deploy whose content no longer matched its manifest and
  auto-committed a half-poisoned tree. Fixes: (a) `_deploy_guard` — containers
  under supervised deploy are exempt from reflexive remediation and code-change
  audits; the deadman is the only judge until it rules; (b) **content survival
  check** — the deadman records a source digest at execution start and fails
  the deploy if the running source was mutated during verification, even when
  health passes. Health alone is necessary but not sufficient.
- **Phase 2 SHIPPED (nfi3, 2026-07-16)** — the loop is closed: CodeMind's
  validated candidate fixes become restart manifests automatically
  (`codemind_manifest.build_manifest_for_files`, rendered via the s4r2
  `restart_manifest` scaffold, `CODEMIND.self_deploy` default true — Azrael's
  "live now" decision). Manifests may carry `promote_files`; the doctor calls
  orchestrator `POST /deploy/promote` (candidates→prod copy) before restart,
  and a promote failure aborts pre-restart. Full autonomous chain:
  **detect → patch (s4r2) → validate → manifest → promote → deadman → commit/rollback.**
- Web handoff during core restarts needs no new machinery: gaia-web already
  falls back to `gaia-core-candidate` via `CORE_FALLBACK_ENDPOINT`
  (docker-compose.yml). First autonomous *core* self-deploy should still be
  Architect-supervised.
- Image-tier manifests are rejected from the autonomous path but staged to
  `/shared/doctor/architect_queue/` — the Architect's review inbox.
- Deadman battery is CPU-aware and ON (identity section,
  `DEADMAN_BATTERY_QUERY_TIMEOUT=180`s per query).
- Remaining future work: scaffold-aware skill_llm_body prompts; image-tier
  approval UX beyond the queue dir.

## Premise

GAIA's self-improvement loop (CodeMind proposals during sleep → candidate test bed →
promotion gates) ends at a wall: applying a change to a live container requires a restart,
and today a human performs it. This document defines the standard for closing that loop
safely. It is the [autonomic awareness principle](COGNITIVE_ARCHITECTURE.md) applied to
deployment: **don't ask the model to judge whether a deploy went well — build a mechanism
that knows.**

## Cardinal Rules

1. **A process never restarts itself.** The service being updated *requests*; a different
   service that survives the restart *executes*. If the new code can't boot, somebody must
   still be home.
2. **Failover runs older, proven code — never newer, unproven code.** The candidate is
   therefore never the user-traffic failover (it runs code *newer* than what's being
   deployed). Failover is the *previous* production container, retained.
3. **Rollback is a mechanism, not a decision.** A deadman timer reverts automatically.
   GAIA wakes up improved or exactly as she was — never broken.
4. **Autonomy is tiered by blast radius.** Cheaply-revertible changes are autonomous;
   expensive-to-revert changes require the Architect.

## The Three Roles (not three maintained stacks)

| Role | What it is | Upkeep cost |
|------|-----------|-------------|
| **Live prod** | The serving container | Maintained (as today) |
| **Last-known-good** | The *previous* prod container + image + git SHA, **stopped but retained** after each successful deploy | Zero — it is not a copy of what works, it *is* what was working before the bounce |
| **Candidate** | Dev/test bed only (CodeMind experiments, promotion validation) | Maintained to prod-grade parity (`r67d`) |

A standing third stack was considered and rejected: it doubles the drift surface
(candidates have silently rotted before — see 2026-07-02 findings), while the retained
previous-prod container provides the same role with no drift possibility. The existing
gaia-web → candidate-core fallback remains as a last-ditch net only.

## Choreography (supervised restart)

1. **Request** — gaia-core writes a *restart manifest* to `/shared`:
   what changed, originating bead, tests run and results, deploy SHA, last-known-good SHA.
   The manifest is the audit trail.
2. **Gate** — gaia-doctor honors the request only in SLEEP/PARKED gear with no active
   session. Restart is a gear operation — the most invasive one, same shape as the rest.
3. **Drain & checkpoint** — `/inference/drain` (r2kn machinery); sessions already persist
   to `sessions.json`; Neural Handoff / KV checkpoint (the clutch) captures thought-state.
4. **Rotate** — doctor stops (does not delete) current prod → it becomes last-known-good;
   starts the updated container. During the gap the orchestrator flips gaia-web's upstream
   endpoint to the retained container if traffic arrives.
5. **Deadman** — new code has T seconds to pass `/health` plus a cognitive smoke-battery
   slice. **Pass**: manifest marked deployed, bead closed, last-known-good rotated.
   **Fail**: doctor reverts volume-mounted source to the recorded SHA, restarts, files a
   bug bead with captured logs. The failed attempt becomes CodeMind training material.

## Autonomy Tiers

| Change class | Revert cost | Autonomy |
|--------------|-------------|----------|
| Volume-mounted source (`git checkout <sha>`) | Seconds | **Fully autonomous** |
| Dockerfile / image rebuild, compose changes, engine changes | Rebuild + risk (cf. the 2026-07-15 Dockerfile regression that would have detached the vision tower) | **Staged + flagged for the Architect** |

She restarts her own mind; rebuilding her skull stays a two-signature operation.

## New Components Required

The loop is ~80% built (CodeMind, candidate test bed, `promotion_readiness` sleep task,
`promote_pipeline.sh`, doctor auto-restart, web fallback, drain protocol, the clutch).
Missing:

1. Restart-manifest contract/schema (`contracts/schemas/`)
2. gaia-doctor: supervised restart-with-rollback endpoint + deadman timer
   (`contracts/services/gaia-doctor.yaml`)
3. gaia-orchestrator: upstream-handoff endpoint for gaia-web
   (`contracts/services/gaia-orchestrator.yaml`)
4. Last-known-good rotation (container/image/SHA bookkeeping)

## Prerequisite: Candidate Parity (`r67d`)

GAIA must not self-deploy on the verdict of a test bed that can't be trusted to match
production. Before `kmcb` is implemented: automated candidate/prod drift detection (a
verification-shaped sleep task — no GPU needed), doctor health-watching of candidates at
prod seriousness, mirror-back discipline for prod-first hotfixes, and an exempt-by-design
carve-out for `gaia-prime-candidate` (deliberately a different stack: raw vLLM/LMCache).

## Sidebar: Single-GPU-Holder Doctrine (`85mb`, 2026-07-19)

Of the four inference systems — Core, Prime, Core-Candidate, Prime-Candidate — **exactly
one may hold GPU VRAM at any time**. The gearbox is the judge: its GPU-tenant registry
(`GPU_TENANTS` in `lifecycle_machine.py`) probes non-tier GPU-capable containers and the
snapshot reports `vram_tenants` / `gpu_holders` / `gpu_single_holder_ok`. Enforcement:

- **VRAM preflight** negotiates away *every* running GPU-holding tenant before a GPU gear
  (guard file v2 lists all stopped containers; the doctor stands down while it's live).
- **`gaia-prime-candidate` is default-stopped** (`/shared/doctor/tenant_policy.json`,
  durable, no TTL): its vLLM preallocates ~9.4 GB the moment it starts, so down IS its
  healthy state — the doctor neither revives nor alarms on it, and downshift restore
  leaves it down. Engage for testing via `POST /lifecycle/tenant/{name}/start` (refused
  while a GPU gear is active, unless forced) or the dashboard GPU Tenants row.
- **`gaia-core-candidate` is CPU-pinned** (`CANDIDATE_CORE_DEVICE=cpu`, `N_GPU_LAYERS=0`):
  the HA mirror never spontaneously grabs VRAM; GPU-path candidate testing is a
  deliberate env override + engage.

This refines the r67d carve-out: prime-candidate remains a different stack (raw
vLLM/LMCache), and is additionally not a resident — it is a *guest* of the GPU, admitted
only when the gearbox has parked.

## Related

- Sleep cognition contract: dream-work never wakes the GPU (`2l9`), heartbeats are waking
  organs (silenced in sleep), CPU-gear inference gets CPU-scaled timeouts (`q5ab`,
  `SLEEP_CPU_INFERENCE_TIMEOUT`).
- `.claude/rules/candidate-first.md`, `.claude/rules/promotion.md`, `.claude/rules/safety.md`
