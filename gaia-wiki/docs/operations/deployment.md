# Deployment

GAIA runs as a Docker Compose stack on a single machine with one NVIDIA GPU.

## Quick Start

```bash
# Start the full live stack
./gaia.sh live start

# Or manually
docker compose up -d

# Check status
./gaia.sh status

# View logs
docker compose logs -f gaia-core
```

## gaia.sh

The master control script provides commands for all service operations:

```bash
./gaia.sh live start|stop|restart|build|logs|status
./gaia.sh candidate start|stop|build
./gaia.sh swap core|mcp|study|web
./gaia.sh gpu status|release|reclaim
./gaia.sh wiki start|stop|build|logs|status
./gaia.sh status
```

## Compose Files

| File | Purpose |
|------|---------|
| `docker-compose.yml` | Live production stack |
| `docker-compose.candidate.yml` | Candidate testing stack |
| `docker-compose.ha.yml` | HA hot standby override |

## Build

```bash
# Build all services
docker compose build

# Build a single service
docker compose build gaia-core

# Build candidates
docker compose -f docker-compose.candidate.yml build
```

## Environment

Key environment variables (set in `.env` or shell):

| Variable | Purpose |
|----------|---------|
| `GAIA_ENV` | `development` or `production` |
| `LOG_LEVEL` | Logging level (default: `INFO`) |
| `PRIME_MODEL_PATH` | Active model path in warm pool |
| `GROQ_API_KEY` | Cloud fallback API key |
| `ENABLE_DISCORD` | Enable Discord bot (`0` or `1`) |
