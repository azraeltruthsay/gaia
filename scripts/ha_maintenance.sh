#!/bin/bash
# Toggle HA maintenance mode
#
# Maintenance mode disables FAILOVER ROUTING — when live gaia-core is down,
# requests will fail instead of routing to the candidate.
#
# Maintenance mode does NOT disable direct inter-service calls. Candidate-core
# can still call live gaia-prime for inference during development (hybrid mode).
#
# See: knowledge/Dev_Notebook/2026-02-19_ha_failover_plan.md

set -euo pipefail

# The flag file lives on the shared Docker volume, visible to all services.
# On the host, this maps to the gaia-shared named volume.
# For host-side access, we also maintain a copy at the project root.
SHARED_DIR="${GAIA_SHARED_DIR:-/gaia/GAIA_Project/shared}"
MAINTENANCE_FILE="$SHARED_DIR/ha_maintenance"

mkdir -p "$SHARED_DIR"

case "${1:-status}" in
    on)
        touch "$MAINTENANCE_FILE"
        echo "Maintenance mode ON — failover routing disabled"
        echo "  Candidate services can still call live prime (hybrid mode)"
        ;;
    off)
        rm -f "$MAINTENANCE_FILE"
        echo "Maintenance mode OFF — failover routing enabled"
        ;;
    status)
        if [ -f "$MAINTENANCE_FILE" ]; then
            echo "Maintenance mode: ON (failover routing disabled)"
        else
            echo "Maintenance mode: OFF (failover routing enabled)"
        fi
        ;;
    *)
        echo "Usage: $0 {on|off|status}"
        exit 1
        ;;
esac
