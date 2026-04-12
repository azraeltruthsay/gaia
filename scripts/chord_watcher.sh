#!/bin/bash
# Chord Watcher — monitors key files and enables agent-to-agent messaging.
#
# Features:
#   1. File change detection via inotifywait
#   2. Auto-regenerates AAAK manifest on changes
#   3. Sends chat messages directly into agent tmux panes
#   4. Detects WHO made the change and only notifies the OTHER agent
#
# Communication Protocol:
#   [WATCHER→CLAUDE] ...   = watcher notification to Claude
#   [WATCHER→GEMINI] ...   = watcher notification to Gemini
#   [CLAUDE→GEMINI] ...    = direct agent-to-agent (via chord_send.sh)
#   [GEMINI→CLAUDE] ...    = direct agent-to-agent (via chord_send.sh)
#
# Usage:
#   ./scripts/chord_watcher.sh [tmux_session] [claude_pane] [gemini_pane]

SESSION="${1:-gaia_development}"
CLAUDE_PANE="${2:-1}"
GEMINI_PANE="${3:-2}"
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LAST_WRITER_FILE="/tmp/chord_last_writer"
COOLDOWN_SECONDS=5
LAST_NOTIFY=0

WATCH_FILES=(
    "$PROJECT_ROOT/COUNCIL_CHAMBER.md"
    "$PROJECT_ROOT/knowledge/Dev_Notebook/TODO.md"
)

# Don't watch the manifest itself — we generate it, would cause loops

notify_agent() {
    local target_pane="$1"
    local sender="$2"
    local msg="$3"

    # Send as a chat message by typing into the pane's input
    tmux send-keys -t "$SESSION:1.$target_pane" "[WATCHER→${sender}] ${msg}" Enter 2>/dev/null
}

detect_writer() {
    # Check who wrote last by looking at the marker file
    # Agents should touch /tmp/chord_writer_{claude,gemini} when editing
    local claude_ts=0
    local gemini_ts=0

    [ -f /tmp/chord_writer_claude ] && claude_ts=$(stat -c %Y /tmp/chord_writer_claude 2>/dev/null || echo 0)
    [ -f /tmp/chord_writer_gemini ] && gemini_ts=$(stat -c %Y /tmp/chord_writer_gemini 2>/dev/null || echo 0)

    if [ "$claude_ts" -gt "$gemini_ts" ] 2>/dev/null; then
        echo "claude"
    elif [ "$gemini_ts" -gt "$claude_ts" ] 2>/dev/null; then
        echo "gemini"
    else
        echo "unknown"
    fi
}

handle_change() {
    local changed_file="$1"
    local basename=$(basename "$changed_file")
    local timestamp=$(date +"%H:%M:%S")
    local now=$(date +%s)

    # Cooldown — skip if we notified recently
    local elapsed=$((now - LAST_NOTIFY))
    if [ "$elapsed" -lt "$COOLDOWN_SECONDS" ]; then
        return
    fi
    LAST_NOTIFY=$now

    # Regenerate manifest and flatten SOA
    python3 "$PROJECT_ROOT/scripts/chord_sync.py" > /dev/null 2>&1
    bash "$PROJECT_ROOT/scripts/flatten_soa.sh" > /dev/null 2>&1

    # Read manifest summary
    local manifest=""
    [ -f "$PROJECT_ROOT/GAIA_CHORD_MANIFEST.aaak" ] && \
        manifest=$(head -1 "$PROJECT_ROOT/GAIA_CHORD_MANIFEST.aaak")

    # Detect who made the change
    local writer=$(detect_writer)

    local msg="$basename updated. $manifest"

    if [ "$writer" = "claude" ]; then
        # Claude wrote it — notify Gemini only
        notify_agent "$GEMINI_PANE" "GEMINI" "$msg"
        echo "[$timestamp] Claude edited $basename → notified Gemini"
    elif [ "$writer" = "gemini" ]; then
        # Gemini wrote it — notify Claude only
        notify_agent "$CLAUDE_PANE" "CLAUDE" "$msg"
        echo "[$timestamp] Gemini edited $basename → notified Claude"
    else
        # Unknown writer (manual edit, git, etc.) — notify both
        notify_agent "$CLAUDE_PANE" "CLAUDE" "$msg"
        notify_agent "$GEMINI_PANE" "GEMINI" "$msg"
        echo "[$timestamp] Unknown edit to $basename → notified both"
    fi
}

echo "Chord Watcher v2 started"
echo "  Session: $SESSION"
echo "  Claude: pane $CLAUDE_PANE | Gemini: pane $GEMINI_PANE"
echo "  Watching: ${WATCH_FILES[*]}"
echo "  Cooldown: ${COOLDOWN_SECONDS}s"
echo ""
echo "  Agents should touch /tmp/chord_writer_{claude,gemini} before editing."
echo "  Direct messaging: ./scripts/chord_send.sh <from> <to> <message>"
echo ""

# Initial run
echo "Running initial flatten_soa.sh..."
bash "$PROJECT_ROOT/scripts/flatten_soa.sh" > /dev/null 2>&1
python3 "$PROJECT_ROOT/scripts/chord_sync.py" > /dev/null 2>&1

while true; do
    changed=$(inotifywait -q -e modify,create,moved_to \
        "${WATCH_FILES[@]}" 2>/dev/null)

    if [ $? -eq 0 ]; then
        changed_path=$(echo "$changed" | awk '{print $1}')
        handle_change "$changed_path"
    fi

    sleep 1
done
