# Candidate Pipeline

All development happens in the `candidates/` directory. Code is promoted to production via the promotion pipeline after passing validation.

## Workflow

```
1. Edit candidate code    candidates/gaia-core/...
2. Test in isolation      docker compose -f docker-compose.candidate.yml up
3. Run validation         ./scripts/promote_pipeline.sh --dry-run
4. Promote to production  ./scripts/promote_pipeline.sh
5. Rebuild + restart      docker compose up -d --build gaia-core
```

## Directory Layout

```
candidates/
├── gaia-core/          # Candidate Brain
├── gaia-web/           # Candidate Face
├── gaia-mcp/           # Candidate Hands
├── gaia-study/         # Candidate Subconscious
├── gaia-orchestrator/  # Candidate Coordinator
├── gaia-prime/         # Candidate Voice
├── gaia-audio/         # Candidate Ears & Mouth
└── gaia-common/        # Candidate shared library
```

Each candidate mirrors the production service structure exactly. Containers mount from `candidates/` instead of the production paths.

## Promotion Pipeline

The `promote_pipeline.sh` script:

1. **Pre-flight** — verify candidate files exist, containers healthy
2. **Grammar tests** — ruff, mypy, pytest in Docker
3. **Cognitive smoke battery** — 16-test validation suite
4. **Promote** — copy candidate → production (dependency order: common → core → web → orchestrator)
5. **Post-promotion health** — verify services restart cleanly
6. **Session sanitization** — clean up test artifacts
7. **Dev journal** — log what changed
8. **Git commit/push** — version the changes

## Candidate vs Production Volumes

| Volume | Live | Candidate |
|--------|------|-----------|
| Shared state | `gaia-shared` | `gaia-candidate-shared` |
| Sandbox | `gaia-sandbox` | `gaia-candidate-sandbox` |
| Knowledge | `./knowledge` (shared, RO for candidates) | Same mount |
| Models | `./gaia-models` (shared, RO) | Same mount |

## HA Mode

After promotion, candidates can run as hot standbys:

```bash
./scripts/ha_start.sh   # Start HA standby
./scripts/ha_stop.sh    # Stop HA standby
```

See [Network Layout](network-layout.md) for HA failover details.
