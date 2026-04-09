#!/bin/bash
# Chord Watcher — monitors key files for changes and notifies both agents.
#
# Uses inotifywait to watch COUNCIL_CHAMBER.md, TODO.md, and the manifest.
# When a change is detected:
#   1. Regenerates the AAAK manifest
#   2. Sends a notification to both tmux panes
#
# Usage:
#   ./scripts/chord_watcher.sh [tmux_session] [claude_pane] [gemini_pane]
#
# Defaults: gaia_development, pane 1, pane 2

SESSION="${1:-gaia_development}"
CLAUDE_PANE="${2:-1}"
GEMINI_PANE="${3:-2}"
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

WATCH_FILES=(
    "$PROJECT_ROOT/COUNCIL_CHAMBER.md"
    "$PROJECT_ROOT/knowledge/Dev_Notebook/TODO.md"
    "$PROJECT_ROOT/GAIA_CHORD_MANIFEST.aaak"
)

notify_agents() {
    local changed_file="$1"
    local basename=$(basename "$changed_file")
    local timestamp=$(date +"%H:%M:%S")

    # Regenerate manifest
    python3 "$PROJECT_ROOT/scripts/chord_sync.py" > /dev/null 2>&1

    # Read the fresh manifest
    local manifest=""
    if [ -f "$PROJECT_ROOT/GAIA_CHORD_MANIFEST.aaak" ]; then
        manifest=$(cat "$PROJECT_ROOT/GAIA_CHORD_MANIFEST.aaak" | head -1)
    fi

    local msg="[CHORD $timestamp] $basename updated. $manifest"

    # Notify Claude (send as a comment so it doesn't execute as input)
    # We use display-message which shows in the status bar without interfering
    tmux display-message -t "$SESSION:1.$CLAUDE_PANE" "$msg" 2>/dev/null

    # Notify Gemini
    tmux display-message -t "$SESSION:1.$GEMINI_PANE" "$msg" 2>/dev/null

    echo "[$timestamp] Notified both agents: $basename changed"
}

echo "Chord Watcher started"
echo "  Session: $SESSION"
echo "  Claude: pane $CLAUDE_PANE"
echo "  Gemini: pane $GEMINI_PANE"
echo "  Watching: ${WATCH_FILES[*]}"
echo ""

while true; do
    # Watch for modifications, creating files, or moves
    changed=$(inotifywait -q -e modify,create,moved_to \
        "${WATCH_FILES[@]}" 2>/dev/null)

    if [ $? -eq 0 ]; then
        # inotifywait returns "path event filename"
        changed_path=$(echo "$changed" | awk '{print $1}')
        notify_agents "$changed_path"
    fi

    # Brief cooldown to prevent rapid-fire on multiple writes
    sleep 2
done
