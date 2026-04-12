#!/bin/bash
# ═════════════════════════════════════════════════════════════════════════════
# GAIA Self-Pulse — Internal temporal anchor for agent panes
# ═════════════════════════════════════════════════════════════════════════════
# Usage: bash scripts/gaia_self_pulse.sh [interval_seconds]

INTERVAL=${1:-1800} # Default to 30 minutes to avoid context noise
PANE_ID=$TMUX_PANE

if [ -z "$PANE_ID" ]; then
    echo "Error: Not running inside a tmux pane."
    exit 1
fi

echo "Internal Pulse started for pane $PANE_ID (Interval: ${INTERVAL}s)"

while true; do
    sleep "$INTERVAL"
    # Send keys back to the current pane's input buffer
    tmux send-keys -t "$PANE_ID" "[PULSE] $(date '+%Y-%m-%d %H:%M') PST" C-m
done
