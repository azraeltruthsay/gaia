---
paths:
  - scripts/promote*
  - candidates/**
---

# Promotion Process

**Flow**: Candidate-only development → Pre-flight checks → Grammar tests (ruff/mypy/pytest in Docker) → 16-test cognitive smoke battery → Promote (dependency order) → Post-promotion health → Session sanitization → Dev journal → Flatten SOA → Git commit/push

**Master script**: `./scripts/promote_pipeline.sh` (dry-run with `--dry-run`)

**Dependency order** (script default): gaia-common → gaia-mcp → gaia-core → gaia-study. Other services (e.g. gaia-web) only when passed via `--services`.

**Vital Organs** (require extra scrutiny): `main.py`, `agent_core.py`, `mcp_client.py`, `tools.py`, `immune_system.py`
