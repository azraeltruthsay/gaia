# Container Topology

> Volume mounts, network config, and environment for all GAIA containers.

## Volume Mount Architecture

| Volume | Live Container | Candidate Container | Access | Purpose |
|--------|---------------|--------------------| -------|---------|
| `./gaia-core` | `/app` (RW) | — | source | Core service code |
| `./candidates/gaia-core` | — | `/app` (RW) | source | Candidate core code |
| `./gaia-common` | `/gaia-common` (RO) | — | shared lib | Common protocols, utils |
| `./candidates/gaia-common` | — | `/app/gaia-common` (RO) | shared lib | Candidate common code |
| `./knowledge` | `/knowledge` (RW) | `/knowledge` (RO) | shared | Knowledge base, blueprints |
| `./knowledge/vector_store` | `/vector_store` (RO) | `/vector_store` (RO) | gaia-study writes | Vector index |
| `./gaia-models` | `/models` (RO) | `/models` (RO) | gaia-study writes | Model files, LoRA adapters |
| `./logs` | `/logs` (RW) | `/logs` (RW) | shared | Consolidated service logs |
| `gaia-shared` (docker vol) | `/shared` (RW) | separate volume | state | Sessions, packets, HA flags |
| `gaia-sandbox` (docker vol) | `/sandbox` (RW) | separate volume | state | MCP tool workspace |

## Key Environment Variables

```bash
# Service discovery (Docker network hostnames)
CORE_ENDPOINT=http://gaia-core:6415
CORE_FALLBACK_ENDPOINT=http://gaia-core-candidate:6415
MCP_ENDPOINT=http://gaia-mcp:8765/jsonrpc
MCP_FALLBACK_ENDPOINT=http://gaia-mcp-candidate:8765/jsonrpc
STUDY_ENDPOINT=http://gaia-study:8766
PRIME_ENDPOINT=http://gaia-prime:7777

# Service identity
GAIA_SERVICE=core|web|study|mcp|orchestrator|audio
GAIA_ENV=development|production
PYTHONPATH=/app:/gaia-common

# Feature flags
MCP_APPROVAL_REQUIRED=true
MCP_APPROVAL_TTL=900         # seconds
GAIA_AUTOLOAD_MODELS=0       # defer model loading
ENABLE_DISCORD=1
```

## Network

- Single Docker network: `gaia-network` (bridge, 172.28.0.0/16)
- All services on same network — hostname resolution by container name
- No ports exposed externally except gaia-web (6414) and dozzle (9999)

## Health Checks

All services expose `GET /health` → `{"status": "healthy"}`. Orchestrator polls at 30s intervals. Docker compose uses `depends_on` with health conditions for startup ordering.

## Restart vs Rebuild Rules

| Change Type | Action Needed |
|-------------|---------------|
| Source file (.py) edit | `docker restart <service>` (bytecache invalidation) |
| Config file change | `docker restart <service>` |
| Dockerfile change | `docker compose build <service> && docker compose up -d <service>` |
| New dependency | Rebuild required |
| gaia-common change | On disk via mount, but restart needed for Python to reload |

**Always sync both production AND candidate** after any code change.
