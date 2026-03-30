# GAIA Service Blueprint: `gaia-web` (The Face)

## Role and Overview

`gaia-web` is the unified interface gateway for the GAIA system. It provides HTTP REST API endpoints, a web dashboard, Discord bot integration (text + voice), and wiki proxy. It serves as the primary entry point for all user interactions, converting user input into CognitionPackets, forwarding them to `gaia-core` for processing, and routing completed responses back to their origin (Discord channel, HTTP response, or log).

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
| `CORE_FALLBACK_ENDPOINT` | (optional) | HA failover target for gaia-core |
| `AUDIO_ENDPOINT` | `http://gaia-audio:8080` | STT/TTS sensory service |
| `WIKI_ENDPOINT` | `http://gaia-wiki:8080` | Internal wiki proxy target |
| `ENABLE_DISCORD` | `1` | Enable Discord bot integration |
| `DISCORD_BOT_TOKEN` | from `.env.discord` | Discord authentication |
| `VOICE_DATA_DIR` | `/app/data` | Voice whitelist persistence |
| `GAIA_SERVICE` | `web` | Service identifier |

### PYTHONPATH Fix (v0.3)

The Dockerfile installs gaia-common via `pip install -e /app/gaia-common/`, but the compose volume mount `./gaia-web:/app:rw` overwrites `/app` entirely, removing the pip-installed reference. The separate mount `./gaia-common:/gaia-common:ro` provides the files at `/gaia-common`. Setting `PYTHONPATH=/app:/gaia-common` in the compose environment ensures Python can find both packages at runtime.

### Volume Mounts

- `./gaia-web:/app:rw` — Source code (editable in dev)
- `./gaia-common:/gaia-common:ro` — Shared library
- `./knowledge:/knowledge:ro` — Knowledge base
- `./gaia-web/static:/app/static:ro` — Static assets
- `./gaia-web/data:/app/data:rw` — Persistent data (voice whitelist, etc.)

## Source Structure

```
gaia-web/
├── Dockerfile              # python:3.11-slim, installs gaia-common + gaia-web
├── requirements.txt        # fastapi, uvicorn, httpx, discord.py, py-cord, webrtcvad, numpy
├── pyproject.toml          # Project metadata (hatchling build)
├── setup.py                # Setuptools configuration
├── gaia_web/
│   ├── __init__.py         # Package metadata
│   ├── main.py             # FastAPI application (~605 lines)
│   ├── discord_interface.py # Discord bot + voice commands (~583 lines)
│   ├── voice_manager.py    # Voice channel orchestration + VAD + audio pipeline (~878 lines)
│   ├── routes/
│   │   ├── __init__.py
│   │   ├── blueprints.py   # Blueprint browsing API
│   │   ├── files.py        # File browser API
│   │   ├── hooks.py        # Webhook/hook endpoints
│   │   ├── terminal.py     # Terminal/shell API
│   │   ├── voice.py        # Voice whitelist + status API
│   │   └── wiki.py         # Wiki proxy endpoints
│   ├── queue/              # Sleep-aware message queueing
│   └── utils/
│       └── retry.py        # HTTP retry with HA fallback
├── static/                 # Dashboard static assets
├── data/                   # Persistent runtime data (voice_whitelist.json)
└── tests/
    ├── conftest.py
    ├── test_health.py
    ├── test_discord_utils.py
    ├── test_voice.py
    ├── test_blueprints.py
    ├── test_files.py
    ├── test_hooks.py
    ├── test_terminal.py
    ├── test_message_queue_persistence.py
    └── test_retry_failover.py
```

## FastAPI Endpoints

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/health` | GET | Returns `{status: "healthy", service: "gaia-web"}` |
| `/` | GET | Root info, lists available endpoints |
| `/process_user_input` | POST | Main input handler — creates CognitionPacket, sends to gaia-core |
| `/output_router` | POST | Receives completed packets from gaia-core, routes to destination |
| `/api/voice/users` | GET | All seen Discord users with whitelist status |
| `/api/voice/whitelist` | GET/POST | Get or add to voice whitelist |
| `/api/voice/whitelist/{user_id}` | DELETE | Remove user from voice whitelist |
| `/api/voice/status` | GET | Current voice connection status |
| `/api/voice/disconnect` | POST | Force disconnect from voice |
| `/wiki/*` | GET | Proxy to internal gaia-wiki MkDocs server |

### Request Flow

1. User input arrives via HTTP POST to `/process_user_input` or Discord message
2. `gaia-web` creates a CognitionPacket (v0.3) with header, intent, context, content
3. POST to `{CORE_ENDPOINT}/process_packet` with HA fallback via `post_with_retry`
4. `gaia-core` processes and returns completed packet
5. `/output_router` routes response based on `packet.header.output_routing.primary.destination`:
   - `DISCORD` — Send to Discord channel/DM
   - `AUDIO` — Play through Discord voice (synthesize via gaia-audio)
   - `WEB` — HTTP response
   - `LOG` — Log only, no output

## Discord Bot Integration

**File**: `discord_interface.py` (~583 lines)

**Architecture**:
- `DiscordInterface` class wrapping discord.py
- Runs in a background daemon thread with its own event loop
- Started on FastAPI startup if `ENABLE_DISCORD=1` and token provided
- Thread-safe cross-loop communication via `_run_on_bot_loop()`

**Intents**: `message_content`, `guild_messages`, `dm_messages`, `members`, `voice_states`

**Text Message Flow**:
1. Bot listens for mentions and DMs (`on_message`)
2. Strips bot mention from content
3. Checks sleep state via gaia-core `/sleep/distracted-check`
4. If asleep: queues message, sends wake signal, waits for active state
5. Creates CognitionPacket with Discord metadata (channel_id, user_id, guild_id)
6. POSTs to `{CORE_ENDPOINT}/process_packet` (with HA fallback)
7. Sends response back to Discord (splits at 2000 char limit)

**Bot Commands**:
- `!call` — Join the caller's voice channel and start listening
- `!hangup` — Disconnect from the current voice channel

**Key Functions**:
- `start_discord_bot(token, core_endpoint, message_queue, voice_manager)` — Start in background thread
- `stop_discord_bot()` — Graceful shutdown
- `is_bot_ready()` — Connection status check
- `get_discord_status()` — Dashboard connectivity info
- `send_to_channel()`, `send_to_user()` — Async message routing (thread-safe)
- `change_presence_from_external()` — Thread-safe presence updates

## Voice System

**File**: `voice_manager.py` (~878 lines)

### Components

**VoiceWhitelist**: Persistent JSON store of whitelisted user IDs. Also tracks all "seen" Discord users for dashboard selection. File: `/app/data/voice_whitelist.json`.

**SimpleVAD**: Hybrid voice activity detection (webrtcvad primary, energy-based RMS fallback). Segments 48kHz Discord audio into utterances based on silence detection (800ms threshold, 300ms minimum speech, 30s max utterance).

**GaiaVoiceSink**: py-cord voice sink that streams decoded PCM from Discord's voice receiver thread into an asyncio.Queue for processing.

**VoiceManager**: Main orchestrator class.

### Voice Connection Triggers

1. **`!call` command**: User types `!call` in any text channel while in a voice channel
2. **Auto-answer**: Whitelisted user joins a voice channel → GAIA auto-joins
3. **Auto-disconnect**: All whitelisted users leave → GAIA disconnects

### Voice Pipeline

```
Discord Voice (48kHz stereo PCM)
    ↓
GaiaVoiceSink → asyncio.Queue (~10s buffer)
    ↓
numpy fast downsample → 16kHz mono
    ↓
SimpleVAD (20ms frames) → utterance detection
    ↓
FFmpeg high-quality resample → WAV base64
    ↓
gaia-audio /transcribe (Whisper STT)
    ↓
[pause audio capture — echo prevention]
    ↓
[check core sleep state → Lite stalling response if not active]
    ↓
gaia-core /process_packet (Prime cognition)
    ↓
gaia-audio /synthesize (Coqui TTS)
    ↓
FFmpegPCMAudio → Discord voice playback
    ↓
[drain echo audio, resume capture]
```

### State Machine

```
disconnected → listening → transcribing → responding → speaking → listening
```

### Sleep-Aware Voice

When a voice call is established while gaia-core is not in `active` state:
- VoiceManager notifies core of voice connection (`/sleep/voice-state`)
- Sends wake signal (`/sleep/wake`)
- Uses Lite model for quick stalling responses while Prime boots
- Switches to Prime once core reports `active`

## High Availability

`gaia-web` supports HA failover via `CORE_FALLBACK_ENDPOINT`. The `post_with_retry` utility in `utils/retry.py` handles:
- Retry with backoff on transient failures
- Automatic failover to the candidate gaia-core if the primary is unreachable
- Transparent to Discord and HTTP callers

## Dependencies

**Runtime**: fastapi, uvicorn, httpx, discord.py (with voice extras), webrtcvad, numpy, sse-starlette, python-multipart
**Shared**: gaia-common (CognitionPacket, health check filter)
**Dev**: pytest, pytest-asyncio, ruff, mypy

## Interaction with Other Services

- **`gaia-core`** (callee): Sends CognitionPackets via HTTP POST, receives completed packets. HA fallback to candidate-core.
- **`gaia-audio`** (callee): Sends audio for STT (`/transcribe`) and text for TTS (`/synthesize`) during voice sessions.
- **`gaia-wiki`** (proxy): Forwards `/wiki/*` requests to internal MkDocs server.
- **`gaia-common`** (library): Imports CognitionPacket protocol, health check filter utility.
- **Discord API** (external): Bot authentication, text message send/receive, voice channel connect/record/play via discord.py.
