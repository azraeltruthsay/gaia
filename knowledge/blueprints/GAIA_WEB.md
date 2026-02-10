# GAIA Service Blueprint: `gaia-web` (The Face)

## Role and Overview

`gaia-web` is the unified interface gateway for the GAIA system. It provides HTTP REST API endpoints and Discord bot integration, serving as the primary entry point for all user interactions. It converts user input into CognitionPackets, forwards them to `gaia-core` for processing, and routes completed responses back to their origin (Discord channel, HTTP response, or log).

## Container Configuration

**Base Image**: `python:3.11-slim`

**Port**: 6414

**Health Check**: `curl -f http://localhost:6414/health` (30s interval, 30s start_period)

**Startup**: `uvicorn gaia_web.main:app --host 0.0.0.0 --port 6414`

**Dependencies**: Waits for `gaia-core` (healthy) before starting.

### Key Environment Variables

| Variable | Value | Purpose |
|----------|-------|---------|
| `PYTHONPATH` | `/app:/gaia-common` | Module resolution fix for volume mounts |
| `CORE_ENDPOINT` | `http://gaia-core:6415` | Cognitive engine address |
| `ENABLE_DISCORD` | `1` | Enable Discord bot integration |
| `DISCORD_BOT_TOKEN` | from `.env.discord` | Discord authentication |
| `GAIA_SERVICE` | `web` | Service identifier |

### PYTHONPATH Fix (v0.3)

The Dockerfile installs gaia-common via `pip install -e /app/gaia-common/`, but the compose volume mount `./gaia-web:/app:rw` overwrites `/app` entirely, removing the pip-installed reference. The separate mount `./gaia-common:/gaia-common:ro` provides the files at `/gaia-common`. Setting `PYTHONPATH=/app:/gaia-common` in the compose environment ensures Python can find both packages at runtime.

### Volume Mounts

- `./gaia-web:/app:rw` — Source code (editable in dev)
- `./gaia-common:/gaia-common:ro` — Shared library
- `./knowledge:/knowledge:ro` — Knowledge base
- `./gaia-web/static:/app/static:ro` — Static assets

## Source Structure

```
gaia-web/
├── Dockerfile              # python:3.11-slim, installs gaia-common + gaia-web
├── requirements.txt        # fastapi, uvicorn, httpx, discord.py, sse-starlette, flask
├── pyproject.toml          # Project metadata (hatchling build)
├── setup.py                # Setuptools configuration
├── gaia_web/
│   ├── __init__.py         # Package metadata (version 0.1.0)
│   ├── main.py             # FastAPI application (262 lines)
│   └── discord_interface.py # Discord bot integration (410 lines)
├── static/                 # Static assets directory
└── tests/
    ├── conftest.py         # ENABLE_DISCORD=0, TestClient fixture
    ├── test_health.py      # /health and / endpoint tests
    └── test_discord_utils.py # Message splitting tests
```

## FastAPI Endpoints

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/health` | GET | Returns `{status: "healthy", service: "gaia-web"}` |
| `/` | GET | Root info, lists available endpoints |
| `/process_user_input` | POST | Main input handler — creates CognitionPacket, sends to gaia-core |
| `/output_router` | POST | Receives completed packets from gaia-core, routes to destination |

### Request Flow

1. User input arrives via HTTP POST to `/process_user_input` or Discord message
2. `gaia-web` creates a CognitionPacket (v0.3) with header, intent, context, content
3. POST to `{CORE_ENDPOINT}/process_packet` (300s timeout)
4. `gaia-core` processes and returns completed packet
5. `/output_router` routes response based on `packet.header.output_routing.primary.destination`:
   - `DISCORD` — Send to Discord channel/DM
   - `WEB` — HTTP response (not yet implemented, returns 501)
   - `LOG` — Log only, no output

## Discord Bot Integration

**File**: `discord_interface.py` (410 lines)

**Architecture**:
- `DiscordInterface` class wrapping discord.py
- Runs in a background daemon thread with its own event loop
- Started on FastAPI startup if `ENABLE_DISCORD=1` and token provided

**Intents**: `message_content`, `guild_messages`, `dm_messages`, `members`

**Message Flow**:
1. Bot listens for mentions and DMs (`on_message`)
2. Strips bot mention from content
3. Creates CognitionPacket with Discord metadata (channel_id, user_id, guild_id)
4. POSTs to `{CORE_ENDPOINT}/process_packet`
5. Sends response back to Discord (splits at 2000 char limit, respecting newlines > spaces)

**Key Functions**:
- `start_discord_bot(token, core_endpoint)` — Start in background thread
- `stop_discord_bot()` — Graceful shutdown
- `is_bot_ready()` — Connection status check
- `send_to_channel()`, `send_to_user()` — Async message routing
- `_split_message(content, max_length=2000)` — Respects Discord's char limit

## Dependencies

**Runtime**: fastapi >=0.115.0, uvicorn >=0.29.0, httpx >=0.25.0, discord.py >=2.3.0, sse-starlette >=1.6.0, python-multipart, flask >=2.3.0
**Shared**: gaia-common (CognitionPacket, health check filter)
**Dev**: pytest, pytest-asyncio, ruff, mypy

## Interaction with Other Services

- **`gaia-core`** (callee): Sends CognitionPackets via HTTP POST, receives completed packets
- **`gaia-common`** (library): Imports CognitionPacket protocol, health check filter utility
- **Discord API** (external): Bot authentication, message send/receive via discord.py
