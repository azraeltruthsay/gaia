# GAIA Interface Contracts

> What each service exposes, accepts, and returns. Use this to verify contract compliance.

## gaia-core (The Brain) — Port 6415

**Primary endpoint:**
- `POST /process_packet` — accepts serialized `CognitionPacket` dict, returns finalized response + metadata
- Flow: deserialize → `AgentCore.run_turn()` → intent detection → knowledge enhancement → tool routing → generation → output routing

**Internal pipeline (agent_core.py):**
```
Intent Detection → Semantic Probe → Knowledge Enhancement →
Cognitive Dispatch → [Tool Routing → MCP execution] →
[Reflection] → Generation (Prime/Lite) → Output Routing
```

## gaia-web (The Face) — Port 6414

| Route File | Key Endpoints | Purpose |
|------------|---------------|---------|
| `blueprints.py` | `GET /api/blueprints`, `GET /api/blueprints/{id}`, `GET /api/blueprints/{id}/markdown`, `GET /api/blueprints/{id}/components` | Blueprint graph & service detail |
| `logs.py` | `GET /api/logs/services`, `GET /api/logs/stream?service=...`, `GET /api/logs/search?service=...&q=...` | SSE log streaming, search |
| `generation.py` | `POST /api/generate`, `GET /api/generation/stream` (SSE) | Generation dispatch, token stream |
| `discord.py` | `GET/POST /api/discord/blocklist`, `GET /api/discord/dm-users` | DM blocklist management |
| `voice.py` | `GET/POST /api/voice/...` | Voice settings |
| `terminal.py` | `POST /api/terminal/execute` | Shell execution (sidecar) |
| `files.py` | `GET/POST /api/files/...` | File browser & editor |
| `audio.py` | `POST /api/audio/transcribe`, `GET /api/audio/status` | Audio transcription |
| `consent.py` | `GET/POST /api/consent/...` | User consent tracking |

**Pattern**: All routes are async FastAPI, use `request.app.state` for shared state, return JSON or SSE.

## gaia-mcp (The Hands) — Port 8765

**Protocol**: JSON-RPC 2.0 at `POST /jsonrpc`

**Tool categories:**
- **File I/O**: read_file, write_file, list_dir, list_files, list_tree, find_files
- **Shell**: run_shell (SENSITIVE, whitelist-gated)
- **Knowledge**: query_knowledge, embed_documents, add_document, find_relevant_documents
- **Web**: web_search, web_fetch (domain allowlist)
- **Kanka**: kanka_list_campaigns, kanka_search, kanka_list_entities, kanka_get_entity, kanka_create_entity (SENSITIVE), kanka_update_entity (SENSITIVE)
- **NotebookLM**: notebooklm_list_notebooks, notebooklm_chat, notebooklm_create_note (SENSITIVE)
- **Audio**: audio_listen_start (SENSITIVE), audio_listen_stop, audio_listen_status
- **Memory**: memory_query, memory_status, memory_rebuild_index (SENSITIVE)
- **Promotion**: generate_blueprint, assess_promotion, promotion_create_request (SENSITIVE)
- **Fragmentation**: fragment_write, fragment_read, fragment_assemble
- **Introspection**: introspect_logs

**Rate limiting**: Kanka enforces 25 req/min client-side (hard cap 30/min), TTL cache 300s.

## gaia-study (The Subconscious) — Port 8766

- `POST /study/start` — trigger sleep cycle reflection or QLoRA training
- `GET /study/status` — check progress
- Sole writer to `/knowledge/vector_store` and `/gaia-models` (LoRA adapters)
- Called by gaia-core via `ServiceClient`

## gaia-prime (The Voice) — Port 7777

- OpenAI-compatible vLLM API: `/v1/completions`, `/v1/chat/completions`
- `GET /health`
- `POST /sleep`, `POST /wake_up` — GPU handoff
- LoRA support via `X-LoRA-Adapter` request header

## gaia-audio (The Ears & Mouth) — Port 8080

- `POST /transcribe` — STT (Whisper, m4a/AAC support via ffmpeg transcode)
- `POST /synthesize` — TTS (Coqui)
- `GET /health`

## Inter-Service Client Pattern

```python
from gaia_common.clients import ServiceClient

client = ServiceClient(base_url="http://gaia-core:6415", service_name="gaia-core")
response = await client.post("/process_packet", packet_dict)
# With fallback:
response = await client.post_with_retry(
    "/process_packet", packet_dict,
    fallback_url="http://gaia-core-candidate:6415"
)
```

Behavior: 3x retry with exponential backoff (2s, 4s), then fallback URL. Catches `ConnectionError`, `Timeout`.
