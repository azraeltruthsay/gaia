#!/bin/bash
# ═════════════════════════════════════════════════════════════════════════════
# GAIA Sovereign Alarm — Self-directed interrupt for agent panes
# ═════════════════════════════════════════════════════════════════════════════
# Usage: bash scripts/sovereign_alarm.sh [delay_seconds] "[message]"

DELAY=${1:-60}
MSG=${2:-"Alarm Fired"}

# Robust Pane ID capture
PANE_ID=${TMUX_PANE:-$(tmux display-message -p '#{pane_id}')}

if [ -z "$PANE_ID" ]; then
    echo "$(date) [ERROR] Not running inside a tmux pane. Alarm aborted." >> /gaia/GAIA_Project/logs/sovereign_alarms.log
    exit 1
fi

echo "Sovereign Alarm set for ${DELAY}s: '$MSG' (Target: $PANE_ID)"

(
    sleep "$DELAY"
    # Inject the alarm notification back into the agent's input buffer
    tmux send-keys -t "$PANE_ID" "[ALARM] $MSG | Sent at: $(date '+%Y-%m-%d %H:%M') PDT" C-m
    echo "$(date) [SUCCESS] Alarm fired: '$MSG' to $PANE_ID" >> /gaia/GAIA_Project/logs/sovereign_alarms.log
) & disown
