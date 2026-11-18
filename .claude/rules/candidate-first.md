---
paths:
  - gaia-core/**
  - gaia-web/**
  - gaia-mcp/**
  - gaia-common/**
  - gaia-study/**
  - gaia-audio/**
  - gaia-orchestrator/**
  - gaia-prime/**
  - gaia-doctor/**
---

# Candidate-First Development — MANDATORY

These services have candidate mirrors in `candidates/`. NEVER edit production directly.

ALL code changes MUST go to `candidates/` first:
- `candidates/gaia-core/` instead of `gaia-core/`
- `candidates/gaia-web/` instead of `gaia-web/`
- `candidates/gaia-common/` instead of `gaia-common/`
- `candidates/gaia-mcp/` instead of `gaia-mcp/`
- `candidates/gaia-study/` instead of `gaia-study/`
- `candidates/gaia-audio/` instead of `gaia-audio/`
- `candidates/gaia-orchestrator/` instead of `gaia-orchestrator/`
- `candidates/gaia-doctor/` instead of `gaia-doctor/`

**Special case**: `candidates/gaia-prime/` exists but is NOT a mirror of `gaia-prime/` — it is a separate experimental build (raw vLLM vs production's GAIA Engine). Do not blind-sync it either direction.

After testing in candidates, use `/project:sync` or the promotion pipeline to sync to production.

**Exceptions** (no candidate mirrors — edit directly, rebuild required):
- `gaia-monkey/` — chaos agent, COPY-based
- `gaia-wiki/` — MkDocs documentation

## Candidate Parity is Enforced (r67d)

gaia-doctor continuously audits candidate/prod parity: file hashing (`.py`,
`entrypoint.sh`, `Dockerfile`, `*.yaml`, `constants/*.json`) across all mirrored
services, compose build-context sanity, and health-watching of ALL candidate
containers with auto-restart. Violations land in `/shared/doctor/drift_alerts.jsonl`
and `status.json → candidate_parity` — check both at session start.

Consequences:
- **Mirror-back rule**: if a hotfix lands in production first, mirror it to
  `candidates/` in the SAME session — the doctor will flag it within one poll
  cycle and alert daily until parity is restored.
- **Deliberately stopping a candidate**: the doctor WILL resurrect it —
  enable maintenance mode first (`is_maintenance_active()` suspends doctor
  restarts). EXCEPTION (85mb, 2026-07-19): `gaia-prime-candidate` is now a
  **default-stopped GPU tenant** (`/shared/doctor/tenant_policy.json`) —
  down IS its healthy state; the doctor never revives it and no maintenance
  mode is needed. Engage it for testing via
  `POST /lifecycle/tenant/gaia-prime-candidate/start` (or the dashboard
  GPU Tenants row); it returns to stopped on the next gear negotiation.
  Single-GPU-holder doctrine: exactly one of Core/Prime/Core-Candidate/
  Prime-Candidate may hold VRAM (snapshot `gpu_holders` reports it).
  `gaia-core-candidate` is CPU-pinned (`CANDIDATE_CORE_DEVICE`, default cpu).
- `gaia-orchestrator` has candidate source (parity-diffed) but deliberately no
  candidate container — only one gearbox controller may own the GPU lifecycle.
- gaia-doctor's own image (production AND candidate) builds from
  `candidates/gaia-doctor/` sources — doctor changes always need both trees in
  sync plus an image rebuild (`docker compose build gaia-doctor`).
