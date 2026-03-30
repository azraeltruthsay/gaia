#!/bin/bash
# ═════════════════════════════════════════════════════════════════════════════
# GPU Maintenance / Training Mode
# ═════════════════════════════════════════════════════════════════════════════
#
# Uses GAIA's built-in maintenance mode flag (/shared/maintenance_mode.json)
# which doctor and orchestrator both respect — no restarts during training.
#
# Usage:
#   ./scripts/gpu_maintenance.sh enter    # Enter training mode, free GPU
#   ./scripts/gpu_maintenance.sh exit     # Exit training mode, restore GAIA
#   ./scripts/gpu_maintenance.sh status   # Show current state
# ═════════════════════════════════════════════════════════════════════════════

set -euo pipefail

SHARED_DIR="/gaia/gaia-instance/shared"
FLAG_FILE="$SHARED_DIR/maintenance_mode.json"
LEGACY_FLAG="$SHARED_DIR/ha_maintenance"
GPU_CONTAINERS=(gaia-core gaia-nano gaia-prime gaia-audio)

enter_training() {
    echo "Entering GPU training mode..."

    # 1. Set maintenance flag FIRST (doctor/orchestrator will see it immediately)
    mkdir -p "$SHARED_DIR"
    cat > "$FLAG_FILE" <<ENDJSON
{
  "active": true,
  "entered_at": "$(date -u +%Y-%m-%dT%H:%M:%S+00:00)",
  "entered_by": "gpu_maintenance.sh",
  "reason": "training"
}
ENDJSON
    touch "$LEGACY_FLAG"
    echo "  Maintenance flag set — doctor will stop restarts"

    # 2. Also write the flag inside the shared Docker volume
    docker exec gaia-doctor sh -c "
        mkdir -p /shared &&
        echo '{\"active\":true,\"entered_at\":\"$(date -u +%Y-%m-%dT%H:%M:%S+00:00)\",\"entered_by\":\"gpu_maintenance.sh\",\"reason\":\"training\"}' > /shared/maintenance_mode.json &&
        touch /shared/ha_maintenance
    " 2>/dev/null || echo "  (doctor not running — flag set on host only)"

    # 3. Unload GPU models gracefully via API
    for endpoint in "localhost:8090/model/unload" "localhost:7777/model/unload" "localhost:8080/sleep"; do
        curl -s -X POST "http://$endpoint" >/dev/null 2>&1
    done
    docker exec gaia-core curl -s -X POST http://localhost:8092/model/unload >/dev/null 2>&1 || true
    echo "  Sent model unload requests"
    sleep 3

    # 4. Stop GPU containers (they won't restart — maintenance flag is set)
    for c in "${GPU_CONTAINERS[@]}"; do
        docker stop "$c" 2>/dev/null || true
    done
    # Also stop candidates
    docker stop gaia-core-candidate gaia-mcp-candidate 2>/dev/null || true
    sleep 3

    # 5. Verify VRAM is free
    echo ""
    echo "  GPU VRAM free: $(nvidia-smi --query-gpu=memory.free --format=csv,noheader 2>/dev/null)"
    echo ""
    echo "Training mode active. Doctor and orchestrator will NOT restart GPU services."
    echo "Run './scripts/gpu_maintenance.sh exit' when done."
}

exit_training() {
    echo "Exiting GPU training mode..."

    # 1. Remove maintenance flags
    rm -f "$FLAG_FILE" "$LEGACY_FLAG" 2>/dev/null
    docker exec gaia-doctor sh -c "rm -f /shared/maintenance_mode.json /shared/ha_maintenance" 2>/dev/null || true
    echo "  Maintenance flags removed"

    # 2. Restart GPU containers
    for c in "${GPU_CONTAINERS[@]}"; do
        docker compose up -d "$c" 2>/dev/null || true
    done
    docker compose up -d gaia-orchestrator 2>/dev/null || true
    echo "  GPU containers starting..."

    sleep 20
    echo ""
    echo "Container status:"
    docker ps --format "  {{.Names}}: {{.Status}}" | grep -E "$(IFS='|'; echo "${GPU_CONTAINERS[*]}")" | sort
    echo ""
    echo "VRAM: $(nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader 2>/dev/null)"
}

show_status() {
    echo "=== Maintenance Mode ==="
    if [ -f "$FLAG_FILE" ]; then
        echo "  Status: ACTIVE (training mode)"
        cat "$FLAG_FILE" | sed 's/^/  /'
    else
        echo "  Status: inactive"
    fi

    echo ""
    echo "=== GPU Containers ==="
    for c in "${GPU_CONTAINERS[@]}"; do
        state=$(docker inspect --format '{{.State.Status}}' "$c" 2>/dev/null || echo "not found")
        printf "  %-20s %s\n" "$c" "$state"
    done

    echo ""
    echo "=== GPU ==="
    nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader 2>/dev/null | sed 's/^/  /'
    echo ""
    nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader 2>/dev/null | sed 's/^/  /' || echo "  No compute processes"
}

case "${1:-status}" in
    enter)  enter_training ;;
    exit)   exit_training ;;
    status) show_status ;;
    *)      echo "Usage: $0 {enter|exit|status}"; exit 1 ;;
esac
