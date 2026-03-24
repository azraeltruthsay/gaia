# GAIA Connectivity Matrix

Every inter-service call in the GAIA mesh. Read as: **Consumer calls Provider**.

## Service-to-Service Calls

| Consumer | Provider | Protocol | Endpoint | Port | Purpose |
|----------|----------|----------|----------|------|---------|
| gaia-web | gaia-core | HTTP POST (streaming) | `/process_packet` | 6415 | Primary cognition entry: sends CognitionPacket, receives NDJSON token stream |
| gaia-web | gaia-core | HTTP POST | `/presence` | 6415 | Update Discord bot presence from sleep cycle |
| gaia-web | gaia-core | HTTP GET | `/sleep/status` | 6415 | Poll sleep state for dashboard |
| gaia-web | gaia-core | HTTP POST | `/sleep/wake` | 6415 | Send wake signal on first message during sleep |
| gaia-web | gaia-core | HTTP POST | `/sleep/force` | 6415 | Force sleep from dashboard |
| gaia-web | gaia-core | HTTP POST | `/sleep/deep` | 6415 | Deep sleep (unload all models) |
| gaia-web | gaia-core | HTTP POST | `/sleep/toggle` | 6415 | Toggle auto-sleep from dashboard |
| gaia-web | gaia-core | HTTP POST | `/sleep/hold` | 6415 | Suppress auto-sleep temporarily |
| gaia-web | gaia-core | HTTP POST | `/sleep/hold-release` | 6415 | Release sleep hold early |
| gaia-web | gaia-core | HTTP GET | `/sleep/config` | 6415 | Get sleep configuration |
| gaia-web | gaia-core | HTTP GET/POST | `/sleep/wake-config`, `/sleep/wake-toggle` | 6415 | Prime wake trigger config |
| gaia-web | gaia-core | HTTP GET | `/health` | 6415 | Health + sleep state check |
| gaia-web | gaia-doctor | HTTP GET | `/status` | 6419 | Aggregate service health for dashboard |
| gaia-web | gaia-doctor | HTTP GET | `/irritations` | 6419 | Full irritation list |
| gaia-web | gaia-doctor | HTTP GET | `/dissonance` | 6419 | Prod vs candidate dissonance report |
| gaia-web | gaia-doctor | HTTP GET | `/serenity` | 6419 | Serenity state (fallback if monkey unavailable) |
| gaia-web | gaia-doctor | HTTP GET | `/cognitive/status` | 6419 | Cognitive battery status + alignment |
| gaia-web | gaia-doctor | HTTP GET | `/cognitive/results` | 6419 | Full cognitive test results |
| gaia-web | gaia-doctor | HTTP GET | `/cognitive/tests` | 6419 | List registered cognitive tests |
| gaia-web | gaia-doctor | HTTP POST | `/cognitive/run` | 6419 | Trigger cognitive battery run |
| gaia-web | gaia-doctor | HTTP GET | `/cognitive/monitor` | 6419 | Cognitive heartbeat monitor |
| gaia-web | gaia-doctor | HTTP GET | `/pipeline` | 6419 | Training pipeline status |
| gaia-web | gaia-doctor | HTTP POST | `/pipeline/run` | 6419 | Trigger pipeline run |
| gaia-web | gaia-doctor | HTTP GET/POST | `/maintenance/*` | 6419 | Maintenance mode enter/exit/status |
| gaia-web | gaia-doctor | HTTP GET/POST | `/surgeon/*` | 6419 | Surgeon approval queue (config, queue, approve, reject, history) |
| gaia-web | gaia-doctor | HTTP GET | `/oom/history` | 6419 | OOM resolution history |
| gaia-web | gaia-doctor | HTTP GET | `/registry` | 6419 | Service registry validation |
| gaia-web | gaia-orchestrator | HTTP GET | `/status` | 6410 | GPU owner, general health |
| gaia-web | gaia-orchestrator | HTTP GET/POST | `/lifecycle/*` | 6410 | Lifecycle state, transitions, history, reconcile |
| gaia-web | gaia-monkey | HTTP GET | `/serenity` | 6420 | Serenity state (primary) |
| gaia-web | gaia-monkey | HTTP * | `/chaos/*`, `/meditation/*` | 6420 | Chaos injection, meditation control (proxy) |
| gaia-web | gaia-study | HTTP GET | `/study/training/status` | 8766 | Training progress |
| gaia-web | gaia-wiki | HTTP * | `/*` | 8080 | Wiki proxy |
| gaia-core | gaia-prime | HTTP POST | `/v1/chat/completions` | 7777 | Thinker GPU inference (OpenAI-compatible) |
| gaia-core | gaia-prime | HTTP GET | `/health` | 7777 | Prime health check (GPU reclaim flow) |
| gaia-core | gaia-prime | HTTP POST | `/model/load` | 7777 | Load model on GPU (GAIA Engine managed mode) |
| gaia-core | gaia-prime | HTTP POST | `/model/unload` | 7777 | Unload model from GPU |
| gaia-core | gaia-prime | HTTP POST | `/adapter/load` | 7777 | Load LoRA adapter |
| gaia-core | gaia-prime | HTTP POST | `/adapter/set` | 7777 | Set active LoRA adapter |
| gaia-core | gaia-nano | HTTP POST | `/v1/chat/completions` | 8080 | Nano triage / reflex inference (OpenAI-compatible) |
| gaia-core | gaia-nano | HTTP GET | `/health` | 8080 | Nano health check |
| gaia-core | gaia-mcp | JSON-RPC 2.0 POST | `/jsonrpc` | 8765 | Tool execution (sandboxed) |
| gaia-core | gaia-orchestrator | HTTP POST | `/gpu/sleep` | 6410 | Release GPU for sleep cycle |
| gaia-core | gaia-orchestrator | HTTP POST | `/gpu/wake` | 6410 | Reclaim GPU after wake |
| gaia-core | gaia-orchestrator | HTTP POST | `/tier/unload-all` | 6410 | Deep sleep: unload all tiers |
| gaia-core | gaia-orchestrator | HTTP GET | `/lifecycle/state` | 6410 | Query lifecycle state |
| gaia-core | gaia-audio | HTTP POST | `/mute` | 8080 | Mute audio on sleep |
| gaia-core | gaia-audio | HTTP POST | `/unmute` | 8080 | Unmute audio on wake |
| gaia-core | gaia-audio | HTTP POST | `/sleep` | 8080 | Deep sleep audio (unload GPU) |
| gaia-core | gaia-audio | HTTP POST | `/wake` | 8080 | Wake audio (reload GPU) |
| gaia-core | localhost:8092 | HTTP POST | `/v1/chat/completions` | 8092 | Embedded Core CPU inference (llama-server or GAIA Engine) |
| gaia-doctor | gaia-core | HTTP GET | `/health` | 6415 | Health polling |
| gaia-doctor | gaia-core | HTTP POST | `/api/repair/structural` | 6415 | Cognitive repair for structural errors |
| gaia-doctor | gaia-core | HTTP POST | `/api/doctor/diagnose` | 6415 | Doctor-initiated diagnostic turn |
| gaia-doctor | gaia-core | HTTP POST | `/api/doctor/review` | 6415 | Sovereign promotion review |
| gaia-doctor | gaia-core | HTTP POST | `/api/cognitive/query` | 6415 | Cognitive test battery queries |
| gaia-doctor | gaia-core | HTTP POST | `/api/cognitive/similarity` | 6415 | Semantic similarity for test validation |
| gaia-doctor | gaia-prime | HTTP GET | `/health` | 7777 | Health polling |
| gaia-doctor | gaia-web | HTTP GET | `/health` | 6414 | Health polling |
| gaia-doctor | gaia-mcp | HTTP GET | `/health` | 8765 | Health polling |
| gaia-doctor | gaia-audio | HTTP GET | `/health` | 8080 | Health polling |
| gaia-doctor | gaia-monkey | HTTP POST | `/serenity/record_recovery` | 6420 | Report recovery events for serenity scoring |
| gaia-doctor | gaia-monkey | HTTP GET | `/serenity` | 6420 | Read serenity state |
| gaia-orchestrator | gaia-core | HTTP POST | `/gpu/release` | 6415 | Update model pool after GPU release |
| gaia-orchestrator | gaia-core | HTTP POST | `/gpu/reclaim` | 6415 | Restore model pool after GPU reclaim |
| gaia-orchestrator | gaia-core | HTTP GET | `/health` | 6415 | Health watchdog polling |
| gaia-orchestrator | gaia-core | HTTP POST | `/sleep/study-handoff` | 6415 | Notify study handoff direction |
| gaia-orchestrator | gaia-prime | HTTP GET | `/health` | 7777 | Health watchdog polling |
| gaia-orchestrator | gaia-prime | HTTP POST | `/model/load` | 7777 | Load model on Prime GPU |
| gaia-orchestrator | gaia-prime | HTTP POST | `/model/unload` | 7777 | Unload model from Prime |
| gaia-orchestrator | gaia-study | HTTP POST | `/study/gpu-ready` | 8766 | Signal GPU available for training |
| gaia-orchestrator | gaia-study | HTTP POST | `/study/gpu-release` | 8766 | Request GPU release from training |
| gaia-orchestrator | gaia-study | HTTP GET | `/study/training/status` | 8766 | Monitor training subprocess |
| gaia-orchestrator | gaia-study | HTTP POST | `/study/training/kill` | 8766 | Force-kill training subprocess |
| gaia-orchestrator | gaia-audio | HTTP GET | `/gpu/status` | 8080 | Check audio GPU usage for idle detection |
| gaia-study | gaia-core | HTTP POST | `/model/adapters/notify` | 6415 | Notify adapter load/unload |
| gaia-study | gaia-core | HTTP POST | `/model/release` | 6415 | Release embedded model for GGUF overwrite |
| gaia-study | gaia-core | HTTP POST | `/model/reload` | 6415 | Reload embedded model after deploy |
| gaia-audio | gaia-core | HTTP POST | `/sleep/wake` | 6415 | Signal voice activity (wake trigger) |
| gaia-audio | gaia-nano | HTTP POST | `/v1/chat/completions` | 8080 | Nano-Refiner transcript cleanup |
| gaia-audio | gaia-orchestrator | HTTP POST | `/register` | 6410 | Register capabilities on startup |
| gaia-monkey | gaia-core | HTTP POST | `/api/cognitive/query` | 6415 | Linguistic chaos evaluation |

## Shared State (File-Based)

| Writer | Reader | Path | Purpose |
|--------|--------|------|---------|
| gaia-monkey | gaia-doctor | `/shared/doctor/serenity.json` | Serenity score state |
| gaia-monkey | gaia-doctor | `/shared/doctor/defensive_meditation.json` | Meditation active flag |
| gaia-orchestrator | all | `/shared/orchestrator/` | GPU state persistence |
| gaia-study | gaia-core | `/shared/pipeline/self_awareness_state.json` | Training pipeline state |
| gaia-study | gaia-core | `/vector_store/` | Vector indexes (SOLE WRITER) |
| gaia-doctor | all | `/shared/maintenance_mode.json` | Maintenance mode flag |

## Library Dependencies (gaia-engine)

`gaia-engine` is a **Python library** (separate repo: github.com/azraeltruthsay/gaia-engine),
not a containerized service. It provides the inference engine, lifecycle state machine,
and model surgery tools used by tier containers.

| Consumer | Import | Purpose |
|----------|--------|---------|
| gaia-prime | `gaia_engine.serve_managed` | Runs GAIA Engine on port 7777 (GPU, 8B Thinker) |
| gaia-nano | `gaia_engine.serve_managed` | Runs GAIA Engine on port 8080 (GPU/GGUF, 0.8B Reflex) |
| gaia-core | `gaia_engine.serve_managed` | Embedded Core inference on localhost:8092 (CPU, 2B) |
| gaia-orchestrator | `gaia_engine.lifecycle` | Lifecycle state machine types (LifecycleState, transitions) |
| gaia-study | `gaia_engine.weighted_trainer`, `gaia_engine.sae_trainer` | QLoRA training, SAE atlas |
| gaia-common (shim) | `gaia_engine.*` | Backward-compat re-export for `from gaia_common.engine import ...` |

## Port Summary

| Service | Internal Port | Host Port | Protocol |
|---------|--------------|-----------|----------|
| gaia-orchestrator | 6410 | 6410 | HTTP (FastAPI) |
| gaia-prime | 7777 | 7777 | HTTP (GAIA Engine) |
| gaia-nano | 8080 | 8090 | HTTP (GAIA Engine) |
| gaia-core | 6415 | 6415 | HTTP (FastAPI) |
| gaia-core (embedded) | 8092 | -- | HTTP (llama-server/GAIA Engine, localhost only) |
| gaia-web | 6414 | 6414 | HTTP (FastAPI) |
| gaia-study | 8766 | 8766 | HTTP (FastAPI) |
| gaia-mcp | 8765 | 8765 | HTTP (FastAPI + JSON-RPC) |
| gaia-audio | 8080 | 8080 | HTTP (FastAPI) |
| gaia-doctor | 6419 | 6419 | HTTP (stdlib http.server) |
| gaia-monkey | 6420 | 6420 | HTTP (FastAPI) |
| gaia-wiki | 8080 | -- | HTTP (MkDocs) |
| gaia-translate | 5000 | -- | HTTP (LibreTranslate) |
| dozzle | 8080 | 9999 | HTTP (Web UI) |
