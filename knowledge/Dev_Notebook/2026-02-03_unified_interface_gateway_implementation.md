**Date:** 2026-02-03
**Title:** Unified Interface Gateway - Initial Implementation

## Summary

Implemented the first phase of the Unified Interface Gateway pattern, moving Discord bot functionality into `gaia-web-candidate`. This establishes `gaia-web` as the single entry/exit point for all external communications, with `gaia-core` becoming interface-agnostic.

## Architecture Overview

```
                    ┌─────────────────┐
   Discord ────────►│                 │
                    │   gaia-web      │──────► /process_message ──────►┌─────────────┐
   Web UI ─────────►│  (The Face)     │                                │  gaia-core  │
                    │                 │◄────── /output_router ◄────────│ (The Brain) │
   API ────────────►│                 │                                └─────────────┘
                    └─────────────────┘
```

**Key Principle:** `gaia-core` no longer knows or cares where messages come from. It receives a message, processes it, and returns a response. `gaia-web` handles all interface-specific logic.

## Files Created/Modified

### New Files

**`candidates/gaia-web/gaia_web/discord_interface.py`**
- `DiscordInterface` class managing the Discord bot lifecycle
- `start_discord_bot(token, core_endpoint)` - Starts bot in background thread
- `stop_discord_bot()` - Graceful shutdown
- `send_to_channel(channel_id, content)` - For autonomous messages
- `send_to_user(user_id, content)` - For DM responses
- `is_bot_ready()` - Status check

The Discord interface:
1. Receives messages via discord.py event handlers
2. Strips mentions and cleans content
3. Sends to `gaia-core` via HTTP POST to `/process_message`
4. Receives response and sends back to Discord

### Modified Files

**`candidates/gaia-web/gaia_web/main.py`**
```python
# Key additions:
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Start Discord bot on startup if enabled
    if ENABLE_DISCORD and DISCORD_BOT_TOKEN:
        start_discord_bot(DISCORD_BOT_TOKEN, CORE_ENDPOINT)
    yield
    # Stop on shutdown
    stop_discord_bot()

@app.post("/output_router")
async def output_router(request: OutputRouteRequest):
    # Routes responses from gaia-core to appropriate destination
    # Supports: discord, web (future), log

@app.get("/discord/status")
async def discord_status():
    # Returns bot connection status
```

**`candidates/gaia-core/gaia_core/main.py`**
```python
@app.post("/process_message")
async def process_message(request: MessageRequest):
    # Receives: user_message, source metadata, session_id
    # Returns: response text, session_id, source
    # Currently returns placeholder - AgentCore integration pending
```

**`candidates/gaia-web/requirements.txt`**
```
discord.py>=2.3.0
python-dotenv>=1.0.0
```

**`docker-compose.candidate.yml`**
```yaml
gaia-web-candidate:
  environment:
    - DISCORD_BOT_TOKEN=${DISCORD_BOT_TOKEN:-}
    - ENABLE_DISCORD=${ENABLE_DISCORD:-0}
```

## API Contracts

### gaia-web → gaia-core: `/process_message`

Request:
```json
{
  "user_message": "Hello GAIA!",
  "source": {
    "type": "discord",
    "channel_id": "123456789",
    "user_id": "987654321",
    "author_name": "TestUser",
    "is_dm": false
  },
  "session_id": "discord_channel_123456789"
}
```

Response:
```json
{
  "response": "Hello! This is GAIA responding.",
  "session_id": "discord_channel_123456789",
  "source": { ... }
}
```

### gaia-core → gaia-web: `/output_router`

Request:
```json
{
  "content": "Autonomous message from GAIA",
  "destination_type": "discord",
  "channel_id": "123456789",
  "user_id": "987654321",
  "is_dm": false
}
```

Response:
```json
{
  "success": true,
  "message": "Message sent"
}
```

## Testing Results

```bash
# Health check shows Discord status
curl http://localhost:6417/health
{"status":"healthy","service":"gaia-web","discord":"disabled","core_endpoint":"http://gaia-core-candidate:6415"}

# Process message endpoint works
curl -X POST http://localhost:6416/process_message \
  -H "Content-Type: application/json" \
  -d '{"user_message":"Hello!","source":{"type":"discord"},"session_id":"test"}'
{"response":"Hello user! I received your message via discord...","session_id":"test","source":{"type":"discord"}}

# Output router correctly reports Discord not connected
curl -X POST http://localhost:6417/output_router \
  -H "Content-Type: application/json" \
  -d '{"content":"Test","destination_type":"discord","channel_id":"123"}'
{"success":false,"message":"Discord bot not connected"}
```

## What's Working

1. **Web gateway starts and is healthy**
2. **Discord bot lifecycle management** (start/stop)
3. **Message forwarding pipeline** (web → core → web)
4. **Output routing endpoint** ready for autonomous messages
5. **Configuration via environment variables**

## What's Pending

1. **Full AgentCore integration** in `gaia-core/main.py` - currently returns placeholder
2. **Live Discord testing** - requires `DISCORD_BOT_TOKEN`
3. **CognitionPacket integration** - currently using simplified message format
4. **Streaming support** - for long responses
5. **Error recovery** - retry logic, circuit breakers

## Migration Path

Once validated:

1. Set `ENABLE_DISCORD=1` and provide `DISCORD_BOT_TOKEN` to test
2. Integrate actual AgentCore processing in `/process_message`
3. Promote candidate code to live
4. Remove `gaia-discord-bot` service from `docker-compose.yml`
5. Update documentation

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `DISCORD_BOT_TOKEN` | (none) | Discord bot token |
| `ENABLE_DISCORD` | `0` | Set to `1` to enable Discord |
| `CORE_ENDPOINT` | `http://gaia-core:6415` | gaia-core service URL |

## Commands

```bash
# Start candidates with Discord enabled
DISCORD_BOT_TOKEN=your_token ENABLE_DISCORD=1 \
  docker compose -f docker-compose.candidate.yml --profile web --profile core up -d

# Check Discord status
curl http://localhost:6417/discord/status

# View logs
docker logs gaia-web-candidate -f
```
