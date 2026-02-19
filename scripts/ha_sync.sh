#!/bin/bash
# HA Session State Sync: Live → Candidate (one-way)
#
# Copies session state from the live gaia-shared volume to the candidate
# gaia-candidate-shared volume. One-way only — never candidate → live.
#
# Modes:
#   --incremental  (default) Copy only files newer than candidate's copies
#   --full         Wipe candidate state, then copy everything fresh
#
# Called by:
#   - health_watchdog.py (incremental, every 30s when HA active + not maintenance)
#   - promote_pipeline.sh (full, after promotion to reset candidate state)
#
# See: knowledge/Dev_Notebook/2026-02-19_ha_failover_plan.md

set -euo pipefail

MODE="${1:---incremental}"

# Source and destination directories
# These map to Docker named volumes. On host, we access via docker cp
# or by running commands inside the orchestrator container.
LIVE_SHARED="/gaia/GAIA_Project/shared"
CANDIDATE_SHARED="/gaia/GAIA_Project/candidate-shared"

# If running inside the orchestrator container, use volume paths directly
if [ -d "/shared" ] && [ -d "/candidate-shared" ]; then
    LIVE_SHARED="/shared"
    CANDIDATE_SHARED="/candidate-shared"
fi

# Files to sync (session state only, not archives)
SYNC_ITEMS=(
    "sessions.json"
    "session_vectors"
    "sleep_state/prime.md"
    "sleep_state/prime_previous.md"
    "lite_journal/Lite.md"
)

case "$MODE" in
    --full)
        echo "Full sync: wiping candidate state and copying from live..."
        # Wipe candidate session state (preserve other shared files)
        for item in "${SYNC_ITEMS[@]}"; do
            rm -rf "$CANDIDATE_SHARED/$item" 2>/dev/null || true
        done

        # Ensure target directories exist
        mkdir -p "$CANDIDATE_SHARED/session_vectors"
        mkdir -p "$CANDIDATE_SHARED/sleep_state"
        mkdir -p "$CANDIDATE_SHARED/lite_journal"

        # Copy from live
        for item in "${SYNC_ITEMS[@]}"; do
            if [ -e "$LIVE_SHARED/$item" ]; then
                cp -a "$LIVE_SHARED/$item" "$CANDIDATE_SHARED/$item"
            fi
        done
        echo "Full sync complete."
        ;;

    --incremental)
        # Incremental: copy only files newer than destination
        mkdir -p "$CANDIDATE_SHARED/session_vectors"
        mkdir -p "$CANDIDATE_SHARED/sleep_state"
        mkdir -p "$CANDIDATE_SHARED/lite_journal"

        for item in "${SYNC_ITEMS[@]}"; do
            if [ -e "$LIVE_SHARED/$item" ]; then
                if [ -d "$LIVE_SHARED/$item" ]; then
                    # Directory: use rsync-like behavior with cp -u
                    cp -au "$LIVE_SHARED/$item/." "$CANDIDATE_SHARED/$item/" 2>/dev/null || true
                else
                    # File: copy if newer
                    cp -u "$LIVE_SHARED/$item" "$CANDIDATE_SHARED/$item" 2>/dev/null || true
                fi
            fi
        done
        ;;

    *)
        echo "Usage: $0 [--incremental|--full]"
        exit 1
        ;;
esac
