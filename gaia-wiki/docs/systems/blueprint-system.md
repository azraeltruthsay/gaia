# Blueprint System

Blueprints are YAML service specifications stored in `knowledge/blueprints/`. They capture each service's contract, dependencies, failure modes, and architectural intent in a machine-readable, self-describing format.

## Purpose

1. **Service discovery** — any service can read blueprints to understand the system topology
2. **Reflection cycles** — gaia-study can compare blueprints against actual code to detect drift
3. **Onboarding** — new developers (or new Claude Code sessions) get a complete system map
4. **Dashboard integration** — the Mission Control blueprint graph is generated from these files

## Schema

Each blueprint contains:

| Section | Contents |
|---------|----------|
| Identity | `id`, `version`, `role`, `service_status` |
| Runtime | Port, base image, GPU requirement, Dockerfile path, health check |
| Interfaces | Inbound/outbound endpoints with transport details |
| Dependencies | Services, volumes, external APIs with required/fallback |
| Source files | Key implementation files with roles |
| Failure modes | Documented failure scenarios with severity and recovery |
| Intent | Purpose, cognitive role, design decisions, open questions |
| Architecture | Internal component decomposition with data flow edges |
| Meta | Confidence scores, divergence tracking, reflection notes |

## Location

```
knowledge/blueprints/
├── gaia-core.yaml
├── gaia-prime.yaml
├── gaia-web.yaml
├── gaia-study.yaml
├── gaia-mcp.yaml
├── gaia-orchestrator.yaml
└── gaia-wiki.yaml
```

## API

gaia-web serves blueprint data via `/api/blueprints`:

- `GET /api/blueprints` — list all blueprints
- `GET /api/blueprints/{id}` — single blueprint
- `GET /api/blueprints/graph` — topology graph for visualization
