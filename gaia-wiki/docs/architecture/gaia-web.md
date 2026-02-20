# gaia-web — The Face

Web UI and Discord gateway. Translates user input from any interface into CognitionPackets and routes responses back.

## Responsibilities

- Serve the Mission Control dashboard (static HTML/JS/CSS)
- Accept user input from web UI and Discord
- Convert input to CognitionPackets and forward to gaia-core
- Route gaia-core responses back to the originating interface
- Manage Discord bot lifecycle (presence, voice, message handling)
- Proxy system status endpoints (orchestrator, sleep, services)

## Interface-Agnostic Design

gaia-web implements the **Gateway Principle**: the core cognitive engine never knows whether a message came from Discord, the web UI, or an API call. All input is normalized into CognitionPackets with `OutputRouting` that specifies where the response should go.

See [Gateway Principle](../decisions/gateway-principle.md) for the design rationale.

## Key Components

| Component | Path | Role |
|-----------|------|------|
| FastAPI app | `gaia_web/main.py` | HTTP endpoints, packet construction, routing |
| DiscordInterface | `gaia_web/discord_interface.py` | Discord.py bot lifecycle, message handling |
| MessageQueue | `gaia_web/queue/message_queue.py` | Sleep-aware message queueing with persistence |
| post_with_retry | `gaia_web/utils/retry.py` | HTTP retry with HA failover support |
| VoiceManager | `gaia_web/voice_manager.py` | Discord voice channel auto-answer |

## Endpoints

| Path | Method | Purpose |
|------|--------|---------|
| `/health` | GET | Container health check |
| `/dashboard` | GET | Redirect to Mission Control UI |
| `/process_user_input` | POST | Web UI text input |
| `/process_audio_input` | POST | Audio transcription input |
| `/output_router` | POST | Route gaia-core responses to destinations |
| `/presence` | POST | Update Discord bot presence |
| `/api/system/services` | GET | Health check all services |
| `/api/system/status` | GET | Proxy orchestrator status |
| `/api/system/sleep` | GET | Proxy sleep cycle status |
| `/api/blueprints/*` | GET | Blueprint API |
| `/wiki/*` | GET | Proxy to gaia-wiki (if configured) |
