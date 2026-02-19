#!/bin/bash
# Stop HA hot standby services
#
# See: knowledge/Dev_Notebook/2026-02-19_ha_failover_plan.md

set -euo pipefail

COMPOSE_DIR="$(cd "$(dirname "$0")/.." && pwd)"

echo "Stopping HA hot standby services..."

docker compose \
    -f "$COMPOSE_DIR/docker-compose.candidate.yml" \
    -f "$COMPOSE_DIR/docker-compose.ha.yml" \
    --profile ha \
    down

echo "HA standby stopped."
