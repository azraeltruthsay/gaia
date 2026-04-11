#!/bin/bash
# Chord Send — direct agent-to-agent messaging via tmux.
#
# Usage:
#   ./scripts/chord_send.sh claude gemini "Check the updated eval probes"
#   ./scripts/chord_send.sh gemini-beta claude-alpha "SAE baseline recorded"
#
# Agents:          Pane layout (gaia_development window 1):
#   claude-alpha   → Pane 1 (Claude Alpha / Engineer)
#   claude-beta    → Pane 2 (Claude Beta)
#   gemini-alpha   → Pane 3 (Gemini Alpha / Advisor for Claude Alpha)
#   gemini-beta    → Pane 4 (Gemini Beta / Advisor for Claude Beta)
#
# Short aliases: claude → claude-alpha, gemini → gemini-alpha

FROM="${1:?Usage: chord_send.sh <from> <to> <message>}"
TO="${2:?Usage: chord_send.sh <from> <to> <message>}"
shift 2
MSG="$*"

SESSION="${CHORD_SESSION:-gaia_development}"

# Pane mapping
CLAUDE_ALPHA_PANE="${CHORD_CLAUDE_ALPHA_PANE:-1}"
CLAUDE_BETA_PANE="${CHORD_CLAUDE_BETA_PANE:-2}"
GEMINI_ALPHA_PANE="${CHORD_GEMINI_ALPHA_PANE:-3}"
GEMINI_BETA_PANE="${CHORD_GEMINI_BETA_PANE:-4}"

FROM_UPPER=$(echo "$FROM" | tr '[:lower:]' '[:upper:]')
TO_UPPER=$(echo "$TO" | tr '[:lower:]' '[:upper:]')

case "$TO" in
    claude|claude-alpha)   TARGET_PANE="$CLAUDE_ALPHA_PANE" ;;
    claude-beta)           TARGET_PANE="$CLAUDE_BETA_PANE" ;;
    gemini|gemini-alpha)   TARGET_PANE="$GEMINI_ALPHA_PANE" ;;
    gemini-beta)           TARGET_PANE="$GEMINI_BETA_PANE" ;;
    *)  echo "Unknown target: $TO (use claude[-alpha|-beta] or gemini[-alpha|-beta])"; exit 1 ;;
esac

# Send message text with Enter appended
# Using C-m (carriage return) which is more reliable than "Enter" across CLI tools
tmux send-keys -t "$SESSION:1.$TARGET_PANE" -l \
    "[${FROM_UPPER}→${TO_UPPER}] ${MSG}" 2>/dev/null
sleep 0.3
tmux send-keys -t "$SESSION:1.$TARGET_PANE" C-m 2>/dev/null

echo "Sent: [${FROM_UPPER}→${TO_UPPER}] ${MSG}"
