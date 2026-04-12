#!/bin/bash
# ═════════════════════════════════════════════════════════════════════════════
# GAIA Restore Services — Re-establishes the gaia_services tmux session
# ═════════════════════════════════════════════════════════════════════════════

SESSION="gaia_services"
PROJECT_ROOT="/gaia/GAIA_Project"

# Kill existing session if it somehow exists
tmux kill-session -t "$SESSION" 2>/dev/null || true

# 1. Create the session and Window 1 (Training)
tmux new-session -d -s "$SESSION" -n "training"
# (Placeholder for Claude's active training command if needed)
# tmux send-keys -t "$SESSION:1" "cd $PROJECT_ROOT && ..." Enter

# 2. Window 2 (Health)
tmux new-window -t "$SESSION:2" -n "health"
tmux send-keys -t "$SESSION:2" "watch -n 5 'docker compose ps'" Enter

# 3. Window 3 (Sync)
tmux new-window -t "$SESSION:3" -n "sync"
tmux send-keys -t "$SESSION:3" "cd $PROJECT_ROOT && bash ./scripts/start_notebooklm_sync.sh" Enter

# 4. Window 4 (Watchdog)
tmux new-window -t "$SESSION:4" -n "watchdog"
tmux send-keys -t "$SESSION:4" "cd $PROJECT_ROOT && bash ./scripts/chord_watcher.sh" Enter

# Set default window
tmux select-window -t "$SESSION:4"

echo "GAIA Services Tmux session '$SESSION' has been restored."
