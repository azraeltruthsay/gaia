# 2026-03-16: Workflow Improvements — Claude Certified Architect Best Practices

## Summary

Applied 4 best practices from the Claude Certified Architect Foundations exam guide to the GAIA development workflow: project-scoped commands, path-scoped rules, structured MCP error responses, and a candidate-first development guard.

## Changes

### 1. `.claude/commands/` — Project-Scoped Slash Commands (6 files)

Wrapped existing scripts and patterns as Claude Code native slash commands:

| Command | Purpose |
|---------|---------|
| `/project:sync <service>` | Candidate↔production sync with restart |
| `/project:test <service>` | pytest in Docker (never host) |
| `/project:deploy [opts]` | `promote_pipeline.sh` wrapper |
| `/project:health [service]` | curl all health endpoints |
| `/project:validate [service]` | `validate.sh` wrapper (ruff/mypy/pytest) |
| `/project:logs <service>` | `docker logs --tail` |

### 2. Path-Scoped `.claude/rules/` with `paths:` Frontmatter (4 modified)

Added YAML frontmatter to 4 of 5 rule files so they only load when touching relevant files:

- `testing.md` → `tests/**`, `**/tests/**`
- `docker.md` → `**/Dockerfile`, `docker-compose*.yml`
- `promotion.md` → `scripts/promote*`, `candidates/**`
- `safety.md` → `gaia-mcp/**`, `**/security/**`
- `workflow.md` → unchanged (always loaded, general guidance)

### 3. Structured MCP Error Responses (3 files modified)

**`gaia-common/gaia_common/errors.py`**:
- Added `is_retryable: bool = False` to `GaiaErrorDef` dataclass
- Updated `register()` to accept `is_retryable` parameter
- Marked retryable: GAIA-MCP-020 (tool crash), GAIA-CORE-015 (model forward fail), GAIA-CORE-055 (Prime unreachable), GAIA-CORE-150 (MCP connection), GAIA-CORE-155 (tool timeout)
- Non-retryable: GAIA-MCP-010 (blast shield), 015 (approval), 025 (syntax), 030 (sandbox)

**`gaia-mcp/gaia_mcp/server.py`**:
- Both `PermissionError` and generic `Exception` handlers now include `errorCategory` and `isRetryable` in JSON-RPC error `data` dict

**`gaia-mcp/gaia_mcp/tools.py`**:
- Gateway error returns (study, adapter list/load/unload/delete/info, introspect logs) enriched with `errorCategory` + `isRetryable`

### 4. Candidate-First Claude Code Guard (1 new file)

**`.claude/rules/candidate-first.md`**:
- Path-scoped to all production service directories
- Loads when touching `gaia-core/**`, `gaia-web/**`, `gaia-mcp/**`, etc.
- Instructs Claude Code to redirect edits to `candidates/` instead
- Defense-in-depth: Layer 1 (Claude Code rule) + Layer 2 (MCP Production Lock)
- Exceptions listed: gaia-doctor, gaia-monkey, gaia-wiki (no candidate mirrors)

## Testing

- All 8 `test_errors.py` tests pass in Docker
- `is_retryable` field verified post-restart in both gaia-core and gaia-mcp containers
- All 4 key services healthy after restart (gaia-core, gaia-web, gaia-mcp, gaia-doctor)
- Production↔candidate sync verified via md5sum comparison

## File Inventory

| File | Action |
|------|--------|
| `.claude/commands/sync.md` | NEW |
| `.claude/commands/test.md` | NEW |
| `.claude/commands/deploy.md` | NEW |
| `.claude/commands/health.md` | NEW |
| `.claude/commands/validate.md` | NEW |
| `.claude/commands/logs.md` | NEW |
| `.claude/rules/testing.md` | MODIFIED (frontmatter) |
| `.claude/rules/docker.md` | MODIFIED (frontmatter) |
| `.claude/rules/promotion.md` | MODIFIED (frontmatter) |
| `.claude/rules/safety.md` | MODIFIED (frontmatter) |
| `.claude/rules/candidate-first.md` | NEW |
| `gaia-common/gaia_common/errors.py` | MODIFIED |
| `gaia-mcp/gaia_mcp/server.py` | MODIFIED |
| `gaia-mcp/gaia_mcp/tools.py` | MODIFIED |
| `candidates/gaia-common/gaia_common/errors.py` | SYNCED |
| `candidates/gaia-mcp/gaia_mcp/server.py` | SYNCED |
| `candidates/gaia-mcp/gaia_mcp/tools.py` | SYNCED |
