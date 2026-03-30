# gaia-wiki — The Library

Internal developer documentation, served by MkDocs with the Material theme.

## Design

gaia-wiki is intentionally simple:

- **No runtime dependencies** — purely static content served by MkDocs
- **No GPU** — zero resource contention with cognitive services
- **No external port** — internal to `gaia-net` only
- **Live reload** — `mkdocs serve` watches for file changes without rebuild
- **Read-only mounts** — docs and knowledge are mounted read-only

## Access

- **Internal:** `http://gaia-wiki:8080` (from any service on `gaia-net`)
- **Via gaia-web proxy:** `/wiki/*` (if `WIKI_ENDPOINT` is configured)
- **Local dev:** `docker compose port gaia-wiki 8080` or add `ports: ["8080:8080"]` temporarily

## Future: Self-Maintaining Docs

The planned integration with gaia-study's sleep tasks would auto-generate pages in `docs/dev/auto/`:

- Function reference from codebase analysis
- Per-service summaries from code evolution snapshots
- Blueprint-derived architecture diagrams

This closes the docs-rot loop — documentation stays current because the system maintains it during sleep cycles.
