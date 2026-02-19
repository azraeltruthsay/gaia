#!/bin/bash
# Start HA hot standby services (gaia-core-candidate + gaia-mcp-candidate)
#
# These run as warm standbys pointing at live gaia-prime for GPU inference.
# When live gaia-core fails, gaia-web automatically routes to candidate-core.
#
# Requires: docker-compose.candidate.yml + docker-compose.ha.yml
# See: knowledge/Dev_Notebook/2026-02-19_ha_failover_plan.md

set -euo pipefail

COMPOSE_DIR="$(cd "$(dirname "$0")/.." && pwd)"

echo "Starting HA hot standby services..."

docker compose \
    -f "$COMPOSE_DIR/docker-compose.candidate.yml" \
    -f "$COMPOSE_DIR/docker-compose.ha.yml" \
    --profile ha \
    up -d

echo ""
echo "HA standby active:"
echo "  gaia-core-candidate → http://gaia-core-candidate:6415 (fallback for gaia-core)"
echo "  gaia-mcp-candidate  → http://gaia-mcp-candidate:8765  (fallback for gaia-mcp)"
echo ""
echo "Both point at LIVE gaia-prime for inference."
echo "Disable with: ./scripts/ha_stop.sh"
