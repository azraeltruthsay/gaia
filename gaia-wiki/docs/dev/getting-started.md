# Getting Started

## Prerequisites

- Docker + Docker Compose v2
- NVIDIA GPU with drivers installed
- NVIDIA Container Toolkit
- ~32 GB RAM (for model warm pool + containers)
- Git

## Clone and Start

```bash
git clone <repo-url> /gaia/GAIA_Project
cd /gaia/GAIA_Project

# Copy environment template
cp .env.example .env
cp .env.discord.example .env.discord

# Edit .env with your configuration
# Edit .env.discord with your Discord bot token

# Build all images
docker compose build

# Start the stack
docker compose up -d

# Verify health
./gaia.sh status
```

## Development Workflow

1. **Edit candidates** — all changes go in `candidates/` first
2. **Test locally** — `docker compose -f docker-compose.candidate.yml up -d`
3. **Run tests** — always in Docker containers:
   ```bash
   docker compose exec -T gaia-core python -m pytest /app/tests/ -v
   docker compose exec -T gaia-web python -m pytest /app/tests/ -v
   ```
4. **Promote** — `./scripts/promote_pipeline.sh`

## Key Paths

| Path | Purpose |
|------|---------|
| `gaia-core/` | Production Brain service |
| `gaia-web/` | Production Face service |
| `gaia-common/` | Shared library (protocols, utils) |
| `candidates/` | Development copies of all services |
| `knowledge/` | Knowledge base, blueprints, dev notebooks |
| `gaia-models/` | Model files and LoRA adapters |
| `scripts/` | Operational scripts (promote, HA, sync) |
| `logs/` | Consolidated service logs |

## Testing

**Never run pytest on the host** — the host Python lacks project dependencies. Always use Docker:

```bash
# gaia-core tests
docker compose exec -T gaia-core python -m pytest /app/tests/ -v --tb=short

# gaia-common tests (via gaia-core container)
docker compose exec -T gaia-core python -m pytest /gaia-common/tests/ -v --tb=short

# gaia-web tests
docker compose exec -T gaia-web python -m pytest /app/tests/ -v --tb=short
```
