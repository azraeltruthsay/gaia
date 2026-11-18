---
paths:
  - "**/Dockerfile"
  - docker-compose*.yml
---

# Docker & Container Rules

## Volume Mounts
- `gaia-common` is volume-mounted (`:ro`) into all containers — file changes are on disk immediately
- Production containers mount `./gaia-core:/app:rw`, etc.
- Candidate containers mount `./candidates/gaia-core:/app:rw`

## When to Restart vs Rebuild
- **Restart only** (`docker restart <service>`): Source file changes (.py), config changes
- **Rebuild required** (`docker compose build <service>`): Dockerfile changes, new dependencies, new COPY directives
- **Always restart** after any code change — Python bytecache needs clearing
- `gaia-doctor` and `gaia-monkey` use COPY (not volume mounts) — require rebuild for code changes
- `gaia-orchestrator` is volume-mounted (`./gaia-orchestrator:/app:rw`) — restart is enough for code changes
  — **BUT its gaia_common is COPY-baked** (`COPY candidates/gaia-common /gaia-common` + pip install -e);
  gaia-common changes reaching the orchestrator need `docker compose build gaia-orchestrator` (9zrx lesson)
- **gaia-engine is ONE directory mount** (2026-10-28): `./gaia-engine/gaia_engine` →
  site-packages `gaia_engine` (ro) in gaia-prime, gaia-core, gaia-web, gaia-study,
  gaia-mcp, gaia-audio, gaia-core-candidate. A restart picks up engine edits (directory
  mounts see replaced inodes; the old single-file mounts needed `--force-recreate`,
  3tch). The gaia_cpp native build lives in the same directory, git-ignored
  (`gaia_engine/cpp/*.so`, `cpp/lib/`): after changing `gaia_cpp.cpp` or the llama.cpp
  pin, rebuild the gaia-prime image and run `gaia-engine/scripts/build_native.sh`.
  Before this, Core/Web/Study/MCP/Audio silently lacked 12 engine modules and gaia_cpp
  (Core's CPU gear fell back to llama-server: no LoRA, ChatML template).

## Shared Volumes
- `gaia-shared` → `/shared/` — session state, packets, locks
- `gaia-sandbox` → `/sandbox/` — MCP tool execution isolation
- `/models/` — model weights (read-only for most services)
- `/knowledge/` — knowledge bases, blueprints, curricula (read-only)
- `/vector_store/` — semantic index (read-only except gaia-study)

## After any code change, always sync both production AND candidate, then restart/rebuild both.
