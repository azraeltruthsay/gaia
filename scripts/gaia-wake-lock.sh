#!/usr/bin/env bash
# gaia-wake-lock.sh — Prevent system sleep while GAIA containers are active.
# Uses systemd-inhibit via a file descriptor lock.
#
# Install as a systemd user service:
#   cp scripts/gaia-wake-lock.service ~/.config/systemd/user/
#   systemctl --user enable --now gaia-wake-lock.service

POLL_INTERVAL=60  # seconds between checks
MIN_CONTAINERS=3  # minimum running GAIA containers to hold the lock

inhibit_fd=""
inhibited=false

acquire_lock() {
    if [ "$inhibited" = false ]; then
        exec {inhibit_fd}< <(systemd-inhibit --what=idle:sleep --who="GAIA" --why="GAIA services active" --mode=block cat)
        inhibited=true
        echo "$(date '+%H:%M:%S') Wake lock ACQUIRED"
    fi
}

release_lock() {
    if [ "$inhibited" = true ] && [ -n "$inhibit_fd" ]; then
        exec {inhibit_fd}<&-
        inhibited=false
        echo "$(date '+%H:%M:%S') Wake lock RELEASED"
    fi
}

trap release_lock EXIT

while true; do
    count=$(docker ps --filter "name=gaia-" --format "{{.Names}}" 2>/dev/null | wc -l)

    if [ "$count" -ge "$MIN_CONTAINERS" ]; then
        acquire_lock
    else
        release_lock
    fi

    sleep "$POLL_INTERVAL"
done
