# Context File Maintenance Rules

> Until automated drift detection exists (gaia-study Phase 2), context freshness depends on manual discipline.

## The Co-Review Rule

**Any PR that modifies a service interface, schema, or security boundary MUST include updates to the relevant `context/` files.**

Stale context is worse than no context — an agent confidently applying outdated interface contracts produces subtle, hard-to-catch errors.

## What Triggers a Context Update

| Change Type | Context Files Affected |
|-------------|----------------------|
| New/changed HTTP endpoint | `shared/architectural-overview.md`, relevant agent's `interface-contracts.md` |
| CognitionPacket field added/changed | `shared/cognition-packet-v03.md` |
| New container or volume mount | `shared/container-topology.md` |
| New MCP tool or permission change | `sentinel/context/mcp-threat-model.md` |
| Security pattern added/fixed | `sentinel/context/security-patterns.md`, `sentinel/context/injection-history.md` |
| Naming convention change | `codemind/context/coding-idioms.md` |
| Blueprint schema change | `codemind/context/blueprint-schema.md` |
| New service boundary | `codemind/context/interface-contracts.md` |

## Review Checklist

When reviewing a PR, ask: **"Does this change make any agent context file inaccurate?"**

If yes, the context update is part of the PR — not a follow-up task.

## Staleness Signals

Watch for these indicators that context may have drifted:
- Agent findings that reference interfaces/patterns that no longer exist
- Agent missing issues that updated context would have caught
- Promotion pipeline failures traceable to outdated agent assumptions

When staleness is detected, fix the context file immediately and note the drift in the relevant agent's `known-drift-patterns.md` (if it has one).
