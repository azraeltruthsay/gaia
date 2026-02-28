# Dev Journal — 2026-02-24: Audio Pipeline + Observability

## Change Tier Classification

| Change | Tier | Rationale |
|--------|------|-----------|
| NotebookLM MCP tools | **Tier 1 — Candidate-first** | New MCP tools with external API integration |
| System audio listener + MCP tools | **Tier 1 — Candidate-first** | New MCP tools + host daemon |
| gaia-audio promotion (candidate → prod) | **Tier 1 — Candidate-first** | Service promotion, port change, compose file edit |
| m4a/AAC transcription via ffmpeg | **Tier 1 — Candidate-first** | Audio pipeline code change |
| Observer false-positive fix | **Tier 1 — Candidate-first** | Cognition safety check behavior change |
| Candidate presence spam fix | **Tier 3 — Dual-write** | Env var `GAIA_IS_CANDIDATE=1` in both compose files |
| Log infrastructure improvements | **Tier 3 — Dual-write** | Web + core changes, shared endpoints |

---

## NotebookLM MCP Tools + System Audio Listener (`bc03675`)

### NotebookLM Integration (8 tools)
- `list_notebooks`, `get_notebook`, `list_sources`, `list_notes`, `list_artifacts` — read access
- `chat` — ask questions to notebooks with follow-up support
- `download_audio` — download + transcribe audio overviews via gaia-audio
- `create_note` — sensitive, requires approval

### System Audio Capture (3 tools + host daemon)
- `audio_listen_start` / `audio_listen_stop` / `audio_listen_status` MCP tools for GAIA-controlled capture
- `gaia_listener.py` host-side daemon — PipeWire/PulseAudio monitor source
- Pipeline: chunks audio → gaia-audio `/transcribe` → gaia-core `/process_packet`
- systemd user service for autonomous startup at login

29 unit tests (20 NotebookLM + 9 listener), all passing.

## gaia-audio Promotion (`1469f95`)

- Promoted gaia-audio from candidate to production (port 8080)
- Removed opt-in profile gate from `docker-compose.yml`
- Added `scripts/gaia_transcribe.py` — CLI tool for transcribing full audio files via gaia-audio (ffmpeg chunking, overlap, optional gaia-core ingestion as knowledge)
- Extended `gaia_listener.py` with `--save-audio` (48kHz stereo WAV recording via separate `pw-cat` stream) and `--compress` (WAV → MP3 via ffmpeg on stop)
- Control file now accepts `save_audio` / `compress` fields for MCP toggling
- Updated systemd service to point at production audio port (8080)

## m4a/AAC Audio Transcription (`f0a307d`)

`audio_bytes_to_array` now falls back to ffmpeg for formats soundfile can't handle (m4a, AAC, opus, webm, etc.). Transcodes to WAV before passing to Whisper. ffmpeg was already in the container image.

## Observer False-Positive Fix (`3336ec9`)

- Observer `fast_check` no longer blocks on the word "error" in conversational responses. Now only triggers on Python tracebacks and HTTP 500 errors.
- Candidates skip Discord presence updates (`GAIA_IS_CANDIDATE=1` env var) to prevent overriding production core's status.

## Log Infrastructure (`29a792a`)

- Log stream now sends last 200 lines on connect (no more empty Logs tab)
- New `GET /api/logs/search` endpoint with substring/regex, level filter, tail
- Discord logs routed to dedicated `discord_bot.log` (was orphaned since Feb 2)
- Dozzle container added to `docker-compose.yml` (port 9999)
- `rpc.discover` handled gracefully in gaia-mcp (was throwing ValueError)
