# Promotion Journal — 2026-02-17

## Services Promoted
- **gaia-web** (sleep/wake message queue integration)
- **gaia-common** (bugfix: missing `concurrent.futures` import)

## Changes Summary

### gaia-web: Sleep-Aware Message Queuing
Three files changed (+123 lines):

1. **`discord_interface.py`** — When GAIA is asleep and a Discord message arrives, the handler now enqueues the message, shows a typing indicator, sends a wake signal to gaia-core, and polls until active before processing. Timeout after 120s with a friendly "trouble waking up" response. `DiscordInterface` and `start_discord_bot` now accept an optional `message_queue` parameter.

2. **`main.py`** — `MessageQueue` initialized on startup and passed to the Discord bot. New `/queue/status` endpoint exposes queue depth, wake signal state, and oldest message age. The `/process_user_input` endpoint also checks sleep state before processing, enqueuing and waiting for wake if needed.

3. **`queue/message_queue.py`** — Default `core_url` changed from candidate port (`http://gaia-core-candidate:6416`) to live port (`http://gaia-core:6415`). New `wait_for_active()` method polls `/sleep/status` at 5s intervals up to 120s timeout, returning `False` for unresolvable states (offline, dreaming).

### gaia-common: Bugfix
- **`discord_connector.py`** — Added missing `import concurrent.futures` (ruff F821 undefined name `concurrent`).

## Validation Results

| Stage | Result |
|-------|--------|
| Ruff (lint) | PASS (after gaia-common fix) |
| MyPy (types) | PASS (warning only: gaia-common package name) |
| Pytest (36 tests) | PASS |
| File promotion (rsync) | PASS |
| Docker image rebuild (--no-cache) | PASS |
| Health: gaia-web /health | healthy |
| Health: gaia-web /queue/status | operational (0 queued) |
| Health: gaia-core | healthy |
| Health: gaia-mcp | healthy |

## Process Notes
- Used `promote_candidate.sh gaia-web --validate` for single-service promotion
- Followed with `docker compose build --no-cache gaia-web` per promotion process docs
- `docker compose up -d gaia-web` recreated dependent containers (gaia-mcp, gaia-core) automatically
- All services came back healthy
- gaia-common import fix applied to both candidate and live directories
