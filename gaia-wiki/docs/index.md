# GAIA Wiki

Internal developer documentation for the GAIA project — a service-oriented AI system with cognitive loops, sleep cycles, and self-maintaining knowledge.

## Quick Links

| Section | What's Here |
|---------|------------|
| [Architecture](architecture/overview.md) | Service topology, data flow, and container layout |
| [Systems](systems/blueprint-system.md) | Deep dives into cross-cutting systems (sleep cycle, blueprints, cognition packets) |
| [Operations](operations/deployment.md) | Deployment, promotion pipeline, GPU management |
| [Decisions](decisions/gateway-principle.md) | Architectural decision records — the "why" layer |
| [Dev](dev/getting-started.md) | Getting started, adding services, code quality |

## Service Map

```
gaia-prime ─── The Voice    (GPU, vLLM inference)
gaia-core  ─── The Brain    (CPU, cognitive loop)
gaia-web   ─── The Face     (Web UI + Discord gateway)
gaia-study ─── The Subconscious (GPU, background processing)
gaia-mcp   ─── The Hands    (Sandboxed tool execution)
gaia-orchestrator ── The Coordinator (Container + GPU lifecycle)
gaia-wiki  ─── The Library   (This documentation)
```

## Network Layout

All services communicate on `gaia-net` (Docker bridge, `172.28.0.0/16`). External access:

| Service | Internal Port | External Port |
|---------|--------------|---------------|
| gaia-prime | 7777 | 7777 |
| gaia-core | 6415 | 6415 |
| gaia-web | 6414 | 6414 |
| gaia-study | 8766 | 8766 |
| gaia-mcp | 8765 | 8765 |
| gaia-orchestrator | 6410 | 6410 |
| gaia-wiki | 8080 | (internal only) |
