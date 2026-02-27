# Specialist Agent Mesh v2 — Phase 1 Implementation — 2026-02-25

## Summary

Implemented the full Phase 1 foundation of the Specialist Agent Mesh v2 — GAIA's external development toolchain for purpose-built Claude Code review agents. This delivers two fully-populated specialist agents (CodeMind, Sentinel), shared infrastructure, five placeholder agents, and a slash-command orchestration layer. The mesh is a bootstrap-phase scaffolding system that will be retired as GAIA develops internal equivalents.

## What Was Built

### Shared Infrastructure (`.claude/agents/shared/`)
- **architectural-overview.md** — distilled service topology, communication patterns, key invariants
- **cognition-packet-v03.md** — CognitionPacket schema reference (fields, enums, lifecycle)
- **container-topology.md** — volume mounts, environment variables, network config, health checks
- **context-maintenance.md** — co-review rule: PRs that change interfaces must update context files (addresses context staleness concern before gaia-study drift detection exists)
- **models/review.py** — AgentReviewResult + Finding Pydantic models (toolchain-only, not in gaia-common)

### CodeMind Agent (`.claude/agents/codemind/`) — Fully Populated
Structural code review specialist. Reviews against 5 dimensions: contract compliance, dependency correctness, failure mode coverage, intent alignment, idiom fidelity. Includes:
- CLAUDE.md with identity, cognitive mode, authority boundaries, severity calibration, output contract
- 4 context files: coding-idioms, interface-contracts, known-drift-patterns, blueprint-schema
- 2 calibration examples (good-review.json, bad-review.json) for output quality anchoring

### Sentinel Agent (`.claude/agents/sentinel/`) — Fully Populated
Security review specialist. Adversarial cognitive mode — thinks about what's actually exploitable, not generic OWASP. Includes:
- CLAUDE.md with trust boundary map, severity calibration, "what NOT to flag" guardrails
- 4 context files: security-patterns (Feb 2026 audit distillation), mcp-threat-model (capability constraints, approval flow), injection-history (known vectors and fixes), container-boundaries (per-service permissions)

### Placeholder Agents — Directory + CLAUDE.md Only
- **AlignmentAgent** — service contract alignment review
- **BlueprintAgent** — blueprint YAML validation
- **StudyAgent** — QLoRA training suitability review
- **UX Designer** — creative design agent (template authority clause pre-written)
- **Service Scaffold** — service generation agent

### Slash Commands (`.claude/slash-commands/`)
- **codemind-review.sh** — single-agent CodeMind invocation
- **sentinel-review.sh** — single-agent Sentinel invocation
- **mesh-review.sh** — sequential orchestration (CodeMind → Sentinel → verdict summary)

## Key Design Decisions

### 1. AgentReviewResult in `.claude/`, not `gaia-common`
The v2 proposal placed Pydantic models for the review schema in gaia-common. Review feedback correctly identified this as scope bleed — toolchain models don't belong in production containers. Moved to `.claude/agents/shared/models/review.py`.

### 2. Priority: Two agents, not seven
Rather than spreading effort across all 7 agents on a fixed 4-week schedule, focused on the two highest-frequency review agents (CodeMind, Sentinel). Remaining agents get scaffolding only and are populated on-demand.

### 3. Context co-review rule
Context staleness is worse than no context. Until gaia-study drift detection exists (Phase 2), the manual discipline is: any PR that changes a service interface must also update the relevant context/ files. Documented in `shared/context-maintenance.md`.

### 4. Sequential orchestration, human integration
Multi-agent review is sequential (CodeMind → Sentinel), not parallel. Each agent produces an independent AgentReviewResult. Any `reject` blocks promotion. No automated consensus — human reads all verdicts and makes the final call.

### 5. Adversarial, not paranoid (Sentinel)
Sentinel's cognitive mode is explicitly "adversarial, not paranoid" — it assesses what's actually exploitable given GAIA's architecture, not what's theoretically risky. Internal service-to-service calls on the Docker network are trusted by design; missing auth between them is not a finding.

## Architecture

```
.claude/
  agents/
    codemind/       ← FULL: CLAUDE.md + 4 context + 2 examples
    sentinel/       ← FULL: CLAUDE.md + 4 context
    alignment/      ← PLACEHOLDER: CLAUDE.md only
    blueprint/      ← PLACEHOLDER: CLAUDE.md only
    study/          ← PLACEHOLDER: CLAUDE.md only
    ux-designer/    ← PLACEHOLDER: CLAUDE.md only
    service-scaffold/ ← PLACEHOLDER: CLAUDE.md only
  shared/
    architectural-overview.md
    cognition-packet-v03.md
    container-topology.md
    context-maintenance.md
    models/review.py
  slash-commands/
    codemind-review.sh
    sentinel-review.sh
    mesh-review.sh
```

## What's Next

- **Phase 2**: UX Designer agent — seed templates/ from existing console pages, extract design tokens
- **Phase 3+**: Remaining agents populated on-demand as their review domains become active
- **Future**: gaia-study drift detection (sleep-cycle flagging of stale context files)

## Files Created

25 new files total across `.claude/agents/`, `.claude/agents/shared/`, and `.claude/slash-commands/`.

## Relationship to v2 Proposal

The implementation follows `GAIA_Specialist_Agent_Mesh_v2.md` with four amendments captured in the plan file (`/home/azrael/.claude/plans/agent-mesh-v2-implementation.md`): AgentReviewResult location, context co-review rule, agent prioritization, and orchestration specification. The proposal's core architecture (two-layer knowledge model, promotion pipeline, template authority clause, retirement criteria) is preserved unchanged.
