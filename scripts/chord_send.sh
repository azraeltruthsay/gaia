#!/bin/bash
# Chord Send — direct agent-to-agent messaging via tmux.
#
# Usage:
#   ./scripts/chord_send.sh claude gemini "Check the updated eval probes"
#   ./scripts/chord_send.sh gemini claude "SAE baseline recorded"
#
# Messages appear in the target agent's chat as:
#   [CLAUDE→GEMINI] Check the updated eval probes

FROM="${1:?Usage: chord_send.sh <from> <to> <message>}"
TO="${2:?Usage: chord_send.sh <from> <to> <message>}"
shift 2
MSG="$*"

SESSION="${CHORD_SESSION:-gaia_development}"
CLAUDE_PANE="${CHORD_CLAUDE_PANE:-1}"
GEMINI_PANE="${CHORD_GEMINI_PANE:-2}"

FROM_UPPER=$(echo "$FROM" | tr '[:lower:]' '[:upper:]')
TO_UPPER=$(echo "$TO" | tr '[:lower:]' '[:upper:]')

case "$TO" in
    claude)  TARGET_PANE="$CLAUDE_PANE" ;;
    gemini)  TARGET_PANE="$GEMINI_PANE" ;;
    *)       echo "Unknown target: $TO (use claude or gemini)"; exit 1 ;;
esac

tmux send-keys -t "$SESSION:1.$TARGET_PANE" \
    "[${FROM_UPPER}→${TO_UPPER}] ${MSG}" Enter 2>/dev/null

echo "Sent: [${FROM_UPPER}→${TO_UPPER}] ${MSG}"
