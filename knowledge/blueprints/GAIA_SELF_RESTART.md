# GAIA Self-Restart & Deployment Roles — Design Standard

> **Status**: Design (agreed 2026-07-16, Azrael + Claude) · **Beads**: `GAIA_Project-kmcb` (design), `GAIA_Project-r67d` (candidate parity, blocks kmcb)
> **Scope**: How GAIA autonomously deploys her own code changes that require a container restart — with no negative impact and no outside influence.

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

## Related

- Sleep cognition contract: dream-work never wakes the GPU (`2l9`), heartbeats are waking
  organs (silenced in sleep), CPU-gear inference gets CPU-scaled timeouts (`q5ab`,
  `SLEEP_CPU_INFERENCE_TIMEOUT`).
- `.claude/rules/candidate-first.md`, `.claude/rules/promotion.md`, `.claude/rules/safety.md`
