#!/usr/bin/env bash
# prime_wake_activity.sh — Workstation activity monitor for GAIA Prime wake
#
# Polls xprintidle to detect keyboard/mouse activity after idle.
# When user transitions from idle (>60s) to active, sends a wake signal
# to gaia-web which proxies to gaia-core's /sleep/wake-activity endpoint.
#
# Requirements: xprintidle (X11), curl
# Install: sudo pacman -S xprintidle  (Arch) or apt install xprintidle

set -euo pipefail

POLL_INTERVAL="${PRIME_WAKE_POLL_INTERVAL:-5}"
IDLE_THRESHOLD_MS="${PRIME_WAKE_IDLE_THRESHOLD_MS:-60000}"  # 60 seconds
WAKE_ENDPOINT="${PRIME_WAKE_ENDPOINT:-http://localhost:6414/api/hooks/sleep/wake-activity}"
DEBOUNCE_SECONDS="${PRIME_WAKE_DEBOUNCE:-60}"

was_idle=false
last_wake=0

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

if ! command -v xprintidle &>/dev/null; then
    log "ERROR: xprintidle not found. Install it: sudo pacman -S xprintidle"
    exit 1
fi

log "Prime wake activity monitor started (poll=${POLL_INTERVAL}s, idle_threshold=${IDLE_THRESHOLD_MS}ms)"

while true; do
    idle_ms=$(xprintidle 2>/dev/null || echo "0")

    if (( idle_ms > IDLE_THRESHOLD_MS )); then
        was_idle=true
    elif $was_idle; then
        # Transition from idle to active
        now=$(date +%s)
        if (( now - last_wake >= DEBOUNCE_SECONDS )); then
            log "Activity detected after idle — sending wake signal"
            if curl -sf -X POST "$WAKE_ENDPOINT" -o /dev/null 2>/dev/null; then
                log "Wake signal sent successfully"
            else
                log "Wake signal failed (endpoint may be down)"
            fi
            last_wake=$now
        fi
        was_idle=false
    fi

    sleep "$POLL_INTERVAL"
done
