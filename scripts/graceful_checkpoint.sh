#!/bin/bash
# Write cognitive checkpoints before container shutdown.
#
# Triggers prime.md and lite.md writes via gaia-core's checkpoint endpoint.
# Called by promote_pipeline.sh before restarting live containers.
#
# See: knowledge/Dev_Notebook/2026-02-19_ha_failover_plan.md (Phase 4.5)

set -euo pipefail

CORE_URL="${CORE_ENDPOINT:-http://localhost:6415}"

echo "Writing cognitive checkpoints..."

response=$(curl -s -w "\n%{http_code}" -X POST "$CORE_URL/cognition/checkpoint" \
    --connect-timeout 5 \
    --max-time 15 \
    2>/dev/null) || {
    echo "WARN: gaia-core checkpoint request failed (may already be down)"
    exit 0
}

http_code=$(echo "$response" | tail -1)
body=$(echo "$response" | head -1)

if [ "$http_code" = "200" ]; then
    echo "Checkpoints written successfully: $body"
else
    echo "WARN: gaia-core returned HTTP $http_code: $body"
fi
