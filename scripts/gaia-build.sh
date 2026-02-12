#!/usr/bin/env bash
# gaia-build.sh — Build candidate services then auto-prune dangling images.
#
# Usage:
#   ./scripts/gaia-build.sh [service...]
#
# Examples:
#   ./scripts/gaia-build.sh                          # build all candidate services
#   ./scripts/gaia-build.sh gaia-core-candidate      # build just core
#   ./scripts/gaia-build.sh gaia-core-candidate gaia-study-candidate
#
# After a successful build, dangling (untagged) images are pruned automatically.
# This prevents the Docker data directory from growing unbounded.

set -euo pipefail

COMPOSE_FILE="${GAIA_COMPOSE_FILE:-docker-compose.candidate.yml}"
PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

cd "$PROJECT_DIR"

echo "==> Building with: docker compose -f $COMPOSE_FILE build $*"
docker compose -f "$COMPOSE_FILE" build "$@"

echo "==> Pruning dangling images..."
RECLAIMED=$(docker image prune -f 2>&1 | tail -1)
echo "    $RECLAIMED"

echo "==> Done."
