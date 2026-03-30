# GAIA System Test Plan — Master Feature Inventory

> **Generated**: 2026-03-25
> **Source**: All 11 service contracts, connectivity matrix, blueprints, sprint backlog
> **Structure**: Center-outward rings (Ring 0 = Cognitive Core, Ring 6 = Support Services)

---

## Summary

| Metric | Count |
|--------|-------|
| **Total testable features** | 178 |
| Ring 0 — Cognitive Core (gaia-core) | 42 |
| Ring 1 — Inference Engine (gaia-engine) | 28 |
| Ring 2 — Orchestrator (gaia-orchestrator) | 38 |
| Ring 3 — Routing & Triage (gaia-nano, gaia-prime) | 14 |
| Ring 4 — Tools & Execution (gaia-mcp) | 26 |
| Ring 5 — Web & Interface (gaia-web) | 18 |
| Ring 6 — Support Services | 12 |
| **By test type** | |
| health_check | 13 |
| smoke_test | 48 |
| functional_test | 72 |
| integration_test | 37 |
| stress_test | 8 |

### Recommended Execution Order

1. **Phase 1 — Health checks** (all services, 2 min): Verify every container responds on its health endpoint.
2. **Phase 2 — Ring 1 smoke** (engine): Model load/unload on Prime, Nano, Core embedded.
3. **Phase 3 — Ring 0 smoke** (core): `/process_packet` with a trivial message, verify streaming.
4. **Phase 4 — Ring 2 smoke** (orchestrator): Lifecycle state, tier status, consciousness probe.
5. **Phase 5 — Ring 3 smoke** (nano/prime): Triage classification, Prime generation.
6. **Phase 6 — Ring 4 smoke** (mcp): JSON-RPC tool listing, `ai_read`, `web_search`.
7. **Phase 7 — Ring 5 smoke** (web): Dashboard serves, `/process_user_input` end-to-end.
8. **Phase 8 — Ring 6 smoke** (doctor, monkey, study, audio): Basic status endpoints.
9. **Phase 9 — Functional tests** (per-ring, deep feature coverage).
10. **Phase 10 — Integration tests** (cross-ring flows: full cognitive loop, sleep cycle, training handoff).
11. **Phase 11 — Stress tests** (concurrent requests, VRAM pressure, chaos drills).

---

## Ring 0 — Cognitive Core (`gaia-core`, port 6415)

### 0.1 Packet Processing

| # | Feature | Endpoint/Method | Test Type | How to Test | Dependencies | Status | Priority |
|---|---------|----------------|-----------|-------------|--------------|--------|----------|
| 0.1.1 | Process packet (streaming) | `POST /process_packet` | smoke_test | `curl -X POST http://localhost:6415/process_packet -H 'Content-Type: application/json' -d '{"user_input":"hello","source":"test","session_id":"test-1"}'` | gaia-core running, at least one inference tier available | working | P0 |
| 0.1.2 | Process packet — NDJSON token stream | `POST /process_packet` | functional_test | Parse NDJSON lines from response: verify `{"type":"token"}`, `{"type":"flush"}`, `{"type":"packet"}` chunks | gaia-core + inference tier | working | P0 |
| 0.1.3 | Process packet — error handling | `POST /process_packet` | functional_test | Send malformed packet, verify `{"type":"error"}` response | gaia-core | unknown | P1 |
| 0.1.4 | Empty response quality gate | `POST /process_packet` | functional_test | Trigger scenario where Core returns empty, verify escalation to Prime | gaia-core + nano + prime | working | P0 |

### 0.2 Health & Status

| # | Feature | Endpoint/Method | Test Type | How to Test | Dependencies | Status | Priority |
|---|---------|----------------|-----------|-------------|--------------|--------|----------|
| 0.2.1 | Health check | `GET /health` | health_check | `curl http://localhost:6415/health` — verify `{"status":"healthy","inference_ok":true/false}` | gaia-core | working | P0 |
| 0.2.2 | Root API directory | `GET /` | smoke_test | `curl http://localhost:6415/` | gaia-core | working | P2 |
| 0.2.3 | Cognitive status | `GET /status` | smoke_test | `curl http://localhost:6415/status` — verify initialization state, loaded models | gaia-core | working | P1 |

### 0.3 GPU Management

| # | Feature | Endpoint/Method | Test Type | How to Test | Dependencies | Status | Priority |
|---|---------|----------------|-----------|-------------|--------------|--------|----------|
| 0.3.1 | GPU status | `GET /gpu/status` | smoke_test | `curl http://localhost:6415/gpu/status` | gaia-core | working | P1 |
| 0.3.2 | GPU release | `POST /gpu/release` | functional_test | `curl -X POST http://localhost:6415/gpu/release -d '{"reason":"test"}'` — verify fallback chain activates | gaia-core + orchestrator | working | P1 |
| 0.3.3 | GPU reclaim | `POST /gpu/reclaim` | functional_test | `curl -X POST http://localhost:6415/gpu/reclaim` — verify gpu_prime restored in model pool | gaia-core + orchestrator + prime | working | P1 |

### 0.4 Sleep Cycle

| # | Feature | Endpoint/Method | Test Type | How to Test | Dependencies | Status | Priority |
|---|---------|----------------|-----------|-------------|--------------|--------|----------|
| 0.4.1 | Sleep status | `GET /sleep/status` | smoke_test | `curl http://localhost:6415/sleep/status` | gaia-core | working | P0 |
| 0.4.2 | Wake signal | `POST /sleep/wake` | functional_test | `curl -X POST http://localhost:6415/sleep/wake` — verify state transitions from ASLEEP to WAKING | gaia-core (must be sleeping) | working | P0 |
| 0.4.3 | Force sleep | `POST /sleep/force` | functional_test | `curl -X POST http://localhost:6415/sleep/force` — verify transition to ASLEEP | gaia-core + orchestrator | working | P1 |
| 0.4.4 | Deep sleep (unload all) | `POST /sleep/deep` | functional_test | `curl -X POST http://localhost:6415/sleep/deep` — verify all models unloaded via orchestrator | gaia-core + orchestrator + all tiers | working | P1 |
| 0.4.5 | Toggle auto-sleep | `POST /sleep/toggle` | smoke_test | `curl -X POST http://localhost:6415/sleep/toggle -d '{"enabled":false}'` | gaia-core | working | P2 |
| 0.4.6 | Sleep hold | `POST /sleep/hold` | functional_test | `curl -X POST http://localhost:6415/sleep/hold -d '{"minutes":5,"reason":"test"}'` | gaia-core | working | P2 |
| 0.4.7 | Sleep hold release | `POST /sleep/hold-release` | smoke_test | `curl -X POST http://localhost:6415/sleep/hold-release` | gaia-core | working | P2 |
| 0.4.8 | Sleep config | `GET /sleep/config` | smoke_test | `curl http://localhost:6415/sleep/config` | gaia-core | working | P2 |
| 0.4.9 | Distracted check | `GET /sleep/distracted-check` | smoke_test | `curl http://localhost:6415/sleep/distracted-check` | gaia-core | working | P2 |
| 0.4.10 | Shutdown | `POST /sleep/shutdown` | functional_test | `curl -X POST http://localhost:6415/sleep/shutdown` — verify OFFLINE transition | gaia-core | unknown | P2 |
| 0.4.11 | Voice state notify | `POST /sleep/voice-state` | smoke_test | `curl -X POST http://localhost:6415/sleep/voice-state -d '{"connected":true}'` | gaia-core | unknown | P2 |
| 0.4.12 | Study handoff | `POST /sleep/study-handoff` | integration_test | `curl -X POST http://localhost:6415/sleep/study-handoff -d '{"direction":"prime_to_study","handoff_id":"test-1"}'` | gaia-core + orchestrator + study | working | P1 |
| 0.4.13 | Wake config | `GET /sleep/wake-config` | smoke_test | `curl http://localhost:6415/sleep/wake-config` | gaia-core | working | P2 |
| 0.4.14 | Wake toggle | `POST /sleep/wake-toggle` | smoke_test | `curl -X POST http://localhost:6415/sleep/wake-toggle -d '{"trigger":"discord_typing","enabled":true}'` | gaia-core | working | P2 |
| 0.4.15 | Wake activity | `POST /sleep/wake-activity` | smoke_test | `curl -X POST http://localhost:6415/sleep/wake-activity` | gaia-core | unknown | P2 |

### 0.5 Model/Adapter Management

| # | Feature | Endpoint/Method | Test Type | How to Test | Dependencies | Status | Priority |
|---|---------|----------------|-----------|-------------|--------------|--------|----------|
| 0.5.1 | Adapter notify | `POST /model/adapters/notify` | functional_test | `curl -X POST http://localhost:6415/model/adapters/notify -d '{"adapter_name":"test","action":"load","tier":3}'` | gaia-core | working | P1 |
| 0.5.2 | Embedded model release | `POST /model/release` | functional_test | `curl -X POST http://localhost:6415/model/release` — verify embedded llama-server stops | gaia-core | working | P1 |
| 0.5.3 | Embedded model reload | `POST /model/reload` | functional_test | `curl -X POST http://localhost:6415/model/reload` — verify embedded llama-server restarts | gaia-core | working | P1 |
| 0.5.4 | Model status | `GET /model/status` | smoke_test | `curl http://localhost:6415/model/status` | gaia-core | working | P2 |

### 0.6 Cognitive Endpoints

| # | Feature | Endpoint/Method | Test Type | How to Test | Dependencies | Status | Priority |
|---|---------|----------------|-----------|-------------|--------------|--------|----------|
| 0.6.1 | Cognitive query (core) | `POST /api/cognitive/query` | functional_test | `curl -X POST http://localhost:6415/api/cognitive/query -d '{"prompt":"What is GAIA?","target":"core"}'` | gaia-core + embedded model | working | P0 |
| 0.6.2 | Cognitive query (nano) | `POST /api/cognitive/query` | functional_test | Same with `"target":"nano"` | gaia-core + nano | working | P1 |
| 0.6.3 | Cognitive query (prime) | `POST /api/cognitive/query` | functional_test | Same with `"target":"prime"` | gaia-core + prime loaded | working | P1 |
| 0.6.4 | Semantic similarity | `POST /api/cognitive/similarity` | functional_test | `curl -X POST http://localhost:6415/api/cognitive/similarity -d '{"text":"hello world","reference":"hi there"}'` | gaia-core + nano | working | P1 |
| 0.6.5 | Structural repair | `POST /api/repair/structural` | functional_test | `curl -X POST http://localhost:6415/api/repair/structural -d '{"service":"gaia-core","broken_code":"def x(","error_msg":"SyntaxError"}'` | gaia-core + inference tier | unknown | P2 |
| 0.6.6 | Doctor diagnose | `POST /api/doctor/diagnose` | functional_test | `curl -X POST http://localhost:6415/api/doctor/diagnose -d '{"service":"gaia-core","logs":"Error: test"}'` | gaia-core + inference tier | unknown | P2 |
| 0.6.7 | Sovereign review | `POST /api/doctor/review` | functional_test | `curl -X POST http://localhost:6415/api/doctor/review -d '{"diffs":[],"source":"test","file_count":0}'` — verify approved/denied | gaia-core + inference tier | unknown | P2 |
| 0.6.8 | Cognition checkpoint | `POST /cognition/checkpoint` | functional_test | `curl -X POST http://localhost:6415/cognition/checkpoint` | gaia-core | unknown | P2 |

### 0.7 KV Cache

| # | Feature | Endpoint/Method | Test Type | How to Test | Dependencies | Status | Priority |
|---|---------|----------------|-----------|-------------|--------------|--------|----------|
| 0.7.1 | KV cache save | `POST /api/kv-cache/save` | functional_test | `curl -X POST http://localhost:6415/api/kv-cache/save` | gaia-core + loaded model | working | P1 |
| 0.7.2 | KV cache restore | `POST /api/kv-cache/restore/core` | functional_test | `curl -X POST http://localhost:6415/api/kv-cache/restore/core` | gaia-core + saved cache | working | P1 |
| 0.7.3 | KV cache pressure | `GET /api/kv-cache/pressure` | smoke_test | `curl http://localhost:6415/api/kv-cache/pressure` | gaia-core | working | P2 |
| 0.7.4 | KV cache compact | `POST /api/kv-cache/compact/core` | functional_test | `curl -X POST http://localhost:6415/api/kv-cache/compact/core` | gaia-core + loaded model | unknown | P2 |

### 0.8 Audio Context

| # | Feature | Endpoint/Method | Test Type | How to Test | Dependencies | Status | Priority |
|---|---------|----------------|-----------|-------------|--------------|--------|----------|
| 0.8.1 | Audio ingest | `POST /audio/ingest` | functional_test | `curl -X POST http://localhost:6415/audio/ingest -d '{"transcript":"test audio","mode":"passive"}'` | gaia-core | working | P2 |
| 0.8.2 | Audio listen toggle | `POST /audio/listen` | smoke_test | `curl -X POST http://localhost:6415/audio/listen` | gaia-core | working | P2 |
| 0.8.3 | Audio context | `GET /audio/context` | smoke_test | `curl http://localhost:6415/audio/context` | gaia-core | working | P2 |

### 0.9 Presence

| # | Feature | Endpoint/Method | Test Type | How to Test | Dependencies | Status | Priority |
|---|---------|----------------|-----------|-------------|--------------|--------|----------|
| 0.9.1 | Update presence | `POST /presence` | smoke_test | `curl -X POST http://localhost:6415/presence -d '{"activity":"testing","status":"online"}'` | gaia-core | working | P2 |

---

## Ring 1 — Inference Engine (`gaia-engine`, library used by nano/core/prime)

### 1.1 Model Lifecycle (Managed Mode HTTP API)

| # | Feature | Endpoint/Method | Test Type | How to Test | Dependencies | Status | Priority |
|---|---------|----------------|-----------|-------------|--------------|--------|----------|
| 1.1.1 | Model load (Prime) | `POST gaia-prime:7777/model/load` | smoke_test | `curl -X POST http://localhost:7777/model/load -d '{"model":"/models/Huihui-Qwen3-8B-GAIA-Prime-adaptive-GPTQ","device":"cuda"}'` | gaia-prime running, GPU free | working | P0 |
| 1.1.2 | Model unload (Prime) | `POST gaia-prime:7777/model/unload` | smoke_test | `curl -X POST http://localhost:7777/model/unload` | gaia-prime with model loaded | working | P0 |
| 1.1.3 | Model swap (Prime) | `POST gaia-prime:7777/model/swap` | functional_test | `curl -X POST http://localhost:7777/model/swap -d '{"model":"/models/new-model","device":"cuda"}'` | gaia-prime + two model paths | unknown | P1 |
| 1.1.4 | Model info | `GET gaia-prime:7777/model/info` | smoke_test | `curl http://localhost:7777/model/info` | gaia-prime | working | P1 |
| 1.1.5 | Model load (Nano) | `POST localhost:8090/model/load` | smoke_test | `curl -X POST http://localhost:8090/model/load -d '{"model":"/models/Qwen3.5-0.8B-Abliterated","device":"cuda"}'` | gaia-nano running | working | P0 |
| 1.1.6 | Model unload (Nano) | `POST localhost:8090/model/unload` | smoke_test | `curl -X POST http://localhost:8090/model/unload` | gaia-nano with model loaded | working | P0 |
| 1.1.7 | Model load (Core embedded) | `POST localhost:8092/model/load` (from inside gaia-core container) | smoke_test | `docker exec gaia-core curl -X POST http://localhost:8092/model/load -d '{"model":"/models/Qwen3.5-2B-GAIA-Core-v3","device":"cpu"}'` | gaia-core running | working | P0 |

### 1.2 Health & Status

| # | Feature | Endpoint/Method | Test Type | How to Test | Dependencies | Status | Priority |
|---|---------|----------------|-----------|-------------|--------------|--------|----------|
| 1.2.1 | Health (Prime) | `GET gaia-prime:7777/health` | health_check | `curl http://localhost:7777/health` — verify `model_loaded`, `mode` fields | gaia-prime | working | P0 |
| 1.2.2 | Health (Nano) | `GET localhost:8090/health` | health_check | `curl http://localhost:8090/health` | gaia-nano | working | P0 |
| 1.2.3 | Status (Prime) | `GET gaia-prime:7777/status` | smoke_test | `curl http://localhost:7777/status` | gaia-prime | working | P1 |
| 1.2.4 | Status (Nano) | `GET localhost:8090/status` | smoke_test | `curl http://localhost:8090/status` | gaia-nano | working | P1 |

### 1.3 Inference

| # | Feature | Endpoint/Method | Test Type | How to Test | Dependencies | Status | Priority |
|---|---------|----------------|-----------|-------------|--------------|--------|----------|
| 1.3.1 | Non-streaming generation (Prime) | `POST gaia-prime:7777/v1/chat/completions` | functional_test | `curl -X POST http://localhost:7777/v1/chat/completions -d '{"messages":[{"role":"user","content":"hello"}],"max_tokens":50,"stream":false}'` | gaia-prime + model loaded | working | P0 |
| 1.3.2 | Streaming generation (Prime) | `POST gaia-prime:7777/v1/chat/completions` | functional_test | Same with `"stream":true` — verify SSE chunks | gaia-prime + model loaded | working | P0 |
| 1.3.3 | Non-streaming generation (Nano) | `POST localhost:8090/v1/chat/completions` | functional_test | Same pattern on port 8090 | gaia-nano + model loaded | working | P0 |
| 1.3.4 | Streaming generation (Nano) | `POST localhost:8090/v1/chat/completions` | functional_test | Same with `"stream":true` on port 8090 | gaia-nano + model loaded | working | P0 |
| 1.3.5 | Embedded Core inference | `POST localhost:8092/v1/chat/completions` (from inside gaia-core) | functional_test | `docker exec gaia-core curl -X POST http://localhost:8092/v1/chat/completions -d '{"messages":[{"role":"user","content":"hello"}],"max_tokens":50}'` | gaia-core + embedded model loaded | working | P0 |
| 1.3.6 | Text completion (Prime) | `POST gaia-prime:7777/v1/completions` | functional_test | `curl -X POST http://localhost:7777/v1/completions -d '{"prompt":"The capital of France","max_tokens":20}'` | gaia-prime + model loaded | unknown | P2 |
| 1.3.7 | List models | `GET gaia-prime:7777/v1/models` | smoke_test | `curl http://localhost:7777/v1/models` | gaia-prime | working | P2 |

### 1.4 Polygraph & Activations

| # | Feature | Endpoint/Method | Test Type | How to Test | Dependencies | Status | Priority |
|---|---------|----------------|-----------|-------------|--------------|--------|----------|
| 1.4.1 | Enable polygraph | `POST gaia-prime:7777/polygraph/enable` | functional_test | `curl -X POST http://localhost:7777/polygraph/enable` | gaia-prime + model loaded (safetensors only) | working | P1 |
| 1.4.2 | Disable polygraph | `POST gaia-prime:7777/polygraph/disable` | functional_test | `curl -X POST http://localhost:7777/polygraph/disable` | gaia-prime | working | P1 |
| 1.4.3 | Get activations | `GET gaia-prime:7777/polygraph/activations` | smoke_test | `curl http://localhost:7777/polygraph/activations` | gaia-prime + polygraph enabled + generation done | working | P1 |

### 1.5 LoRA Adapter Management

| # | Feature | Endpoint/Method | Test Type | How to Test | Dependencies | Status | Priority |
|---|---------|----------------|-----------|-------------|--------------|--------|----------|
| 1.5.1 | Load adapter | `POST gaia-prime:7777/adapter/load` | functional_test | `curl -X POST http://localhost:7777/adapter/load -d '{"name":"test-adapter","path":"/models/adapters/test"}'` | gaia-prime + model loaded + adapter exists | working | P1 |
| 1.5.2 | Unload adapter | `POST gaia-prime:7777/adapter/unload` | functional_test | `curl -X POST http://localhost:7777/adapter/unload` | gaia-prime + adapter loaded | working | P1 |
| 1.5.3 | Set active adapter | `POST gaia-prime:7777/adapter/set` | functional_test | `curl -X POST http://localhost:7777/adapter/set -d '{"name":"test-adapter"}'` | gaia-prime + adapter loaded | working | P1 |
| 1.5.4 | Adapter status | `GET gaia-prime:7777/adapter/status` | smoke_test | `curl http://localhost:7777/adapter/status` | gaia-prime | working | P2 |
| 1.5.5 | List adapters | `GET gaia-prime:7777/adapter/list` | smoke_test | `curl http://localhost:7777/adapter/list` | gaia-prime | working | P2 |

### 1.6 KV Cache / Thought Snapshots

| # | Feature | Endpoint/Method | Test Type | How to Test | Dependencies | Status | Priority |
|---|---------|----------------|-----------|-------------|--------------|--------|----------|
| 1.6.1 | Hold thought | `POST gaia-prime:7777/thought/hold` | functional_test | `curl -X POST http://localhost:7777/thought/hold` | gaia-prime + model loaded + generation done | unknown | P1 |
| 1.6.2 | Resume thought | `POST gaia-prime:7777/thought/resume` | functional_test | `curl -X POST http://localhost:7777/thought/resume` | gaia-prime + held thought | unknown | P1 |
| 1.6.3 | Drop thought | `POST gaia-prime:7777/thought/drop` | smoke_test | `curl -X POST http://localhost:7777/thought/drop` | gaia-prime + held thought | unknown | P2 |
| 1.6.4 | Compose thoughts | `POST gaia-prime:7777/thought/compose` | functional_test | `curl -X POST http://localhost:7777/thought/compose` | gaia-prime + multiple held thoughts | unknown | P2 |
| 1.6.5 | List thoughts | `GET gaia-prime:7777/thought/list` | smoke_test | `curl http://localhost:7777/thought/list` | gaia-prime | unknown | P2 |

### 1.7 Device Migration

| # | Feature | Endpoint/Method | Test Type | How to Test | Dependencies | Status | Priority |
|---|---------|----------------|-----------|-------------|--------------|--------|----------|
| 1.7.1 | Migrate to GPU (Nano) | `POST localhost:8090/device/gpu` | functional_test | `curl -X POST http://localhost:8090/device/gpu` | gaia-nano + model on CPU | working | P1 |
| 1.7.2 | Migrate to CPU (Nano) | `POST localhost:8090/device/cpu` | functional_test | `curl -X POST http://localhost:8090/device/cpu` | gaia-nano + model on GPU | working | P1 |

---

## Ring 2 — Orchestrator (`gaia-orchestrator`, port 6410)

### 2.1 Health & Status

| # | Feature | Endpoint/Method | Test Type | How to Test | Dependencies | Status | Priority |
|---|---------|----------------|-----------|-------------|--------------|--------|----------|
| 2.1.1 | Health check | `GET /health` | health_check | `curl http://localhost:6410/health` | gaia-orchestrator | working | P0 |
| 2.1.2 | Root API directory | `GET /` | smoke_test | `curl http://localhost:6410/` | gaia-orchestrator | working | P2 |
| 2.1.3 | Full status | `GET /status` | smoke_test | `curl http://localhost:6410/status` — verify GPU owner, watch state, containers | gaia-orchestrator | working | P0 |

### 2.2 GPU Management

| # | Feature | Endpoint/Method | Test Type | How to Test | Dependencies | Status | Priority |
|---|---------|----------------|-----------|-------------|--------------|--------|----------|
| 2.2.1 | GPU status | `GET /gpu/status` | smoke_test | `curl http://localhost:6410/gpu/status` | gaia-orchestrator | working | P0 |
| 2.2.2 | GPU acquire | `POST /gpu/acquire` | functional_test | `curl -X POST http://localhost:6410/gpu/acquire -d '{"requester":"core","reason":"test"}'` | gaia-orchestrator | working | P1 |
| 2.2.3 | GPU release | `POST /gpu/release` | functional_test | `curl -X POST http://localhost:6410/gpu/release` | gaia-orchestrator + active lease | working | P1 |
| 2.2.4 | GPU wait | `POST /gpu/wait` | functional_test | `curl -X POST http://localhost:6410/gpu/wait -d '{"requester":"study","reason":"training"}'` | gaia-orchestrator | unknown | P2 |
| 2.2.5 | GPU sleep | `POST /gpu/sleep` | integration_test | `curl -X POST http://localhost:6410/gpu/sleep` — verify Prime unloaded, Core notified | gaia-orchestrator + prime + core | working | P1 |
| 2.2.6 | GPU wake | `POST /gpu/wake` | integration_test | `curl -X POST http://localhost:6410/gpu/wake` — verify Prime reloaded, Core notified | gaia-orchestrator + prime + core | working | P1 |

### 2.3 Container Lifecycle

| # | Feature | Endpoint/Method | Test Type | How to Test | Dependencies | Status | Priority |
|---|---------|----------------|-----------|-------------|--------------|--------|----------|
| 2.3.1 | Container status | `GET /containers/status` | smoke_test | `curl http://localhost:6410/containers/status` | gaia-orchestrator + docker socket | working | P1 |
| 2.3.2 | Stop live stack | `POST /containers/live/stop` | functional_test | `curl -X POST http://localhost:6410/containers/live/stop` | gaia-orchestrator + docker socket | unknown | P2 |
| 2.3.3 | Start live stack | `POST /containers/live/start` | functional_test | `curl -X POST http://localhost:6410/containers/live/start -d '{"gpu_enabled":true}'` | gaia-orchestrator + docker socket | unknown | P2 |
| 2.3.4 | Stop candidate stack | `POST /containers/candidate/stop` | functional_test | `curl -X POST http://localhost:6410/containers/candidate/stop` | gaia-orchestrator + docker socket | unknown | P2 |
| 2.3.5 | Start candidate stack | `POST /containers/candidate/start` | functional_test | `curl -X POST http://localhost:6410/containers/candidate/start` | gaia-orchestrator + docker socket | unknown | P2 |
| 2.3.6 | Swap service | `POST /containers/swap` | functional_test | `curl -X POST http://localhost:6410/containers/swap -d '{"service":"gaia-core","target":"candidate"}'` | gaia-orchestrator + docker socket | unknown | P2 |
| 2.3.7 | Restart container | `POST /containers/{name}/restart` | functional_test | `curl -X POST http://localhost:6410/containers/gaia-core/restart` | gaia-orchestrator + docker socket | working | P1 |

### 2.4 Handoff Protocol

| # | Feature | Endpoint/Method | Test Type | How to Test | Dependencies | Status | Priority |
|---|---------|----------------|-----------|-------------|--------------|--------|----------|
| 2.4.1 | Prime to study handoff | `POST /handoff/prime-to-study` | integration_test | `curl -X POST http://localhost:6410/handoff/prime-to-study` — verify Prime unloads, Study gets GPU signal | gaia-orchestrator + prime + study | working | P1 |
| 2.4.2 | Study to prime handoff | `POST /handoff/study-to-prime` | integration_test | `curl -X POST http://localhost:6410/handoff/study-to-prime` — verify Study releases, Prime loads | gaia-orchestrator + prime + study | working | P1 |
| 2.4.3 | Handoff status | `GET /handoff/{id}/status` | smoke_test | `curl http://localhost:6410/handoff/test-1/status` | gaia-orchestrator | working | P2 |

### 2.5 Tier Router

| # | Feature | Endpoint/Method | Test Type | How to Test | Dependencies | Status | Priority |
|---|---------|----------------|-----------|-------------|--------------|--------|----------|
| 2.5.1 | Tier infer | `POST /tier/infer` | functional_test | `curl -X POST http://localhost:6410/tier/infer -d '{"tier":"prime","messages":[{"role":"user","content":"hello"}]}'` | gaia-orchestrator + target tier | unknown | P1 |
| 2.5.2 | Tier ensure | `POST /tier/ensure` | functional_test | `curl -X POST http://localhost:6410/tier/ensure -d '{"tier":"prime"}'` — verify model loaded | gaia-orchestrator + target tier | working | P1 |
| 2.5.3 | Tier status | `GET /tier/status` | smoke_test | `curl http://localhost:6410/tier/status` | gaia-orchestrator | working | P0 |
| 2.5.4 | Tier unload all | `POST /tier/unload-all` | functional_test | `curl -X POST http://localhost:6410/tier/unload-all` — verify zero GPU memory | gaia-orchestrator + all tiers | working | P1 |
| 2.5.5 | Tier SAE record | `POST /tier/sae-record` | functional_test | `curl -X POST http://localhost:6410/tier/sae-record -d '{"tier":"prime"}'` | gaia-orchestrator + loaded tier | unknown | P2 |

### 2.6 Lifecycle State Machine (Consciousness Matrix)

| # | Feature | Endpoint/Method | Test Type | How to Test | Dependencies | Status | Priority |
|---|---------|----------------|-----------|-------------|--------------|--------|----------|
| 2.6.1 | Lifecycle state | `GET /lifecycle/state` | smoke_test | `curl http://localhost:6410/lifecycle/state` — verify LifecycleSnapshot JSON | gaia-orchestrator | working | P0 |
| 2.6.2 | Lifecycle transition (AWAKE→FOCUSING) | `POST /lifecycle/transition` | integration_test | `curl -X POST http://localhost:6410/lifecycle/transition -d '{"trigger":"COMPLEX_QUERY","reason":"test"}'` — verify Prime loads on GPU | gaia-orchestrator + prime | working | P0 |
| 2.6.3 | Lifecycle transition (FOCUSING→AWAKE) | `POST /lifecycle/transition` | integration_test | `curl -X POST http://localhost:6410/lifecycle/transition -d '{"trigger":"IDLE_TIMEOUT","reason":"test"}'` — verify Prime unloads | gaia-orchestrator + prime | working | P0 |
| 2.6.4 | Available transitions | `GET /lifecycle/transitions` | smoke_test | `curl http://localhost:6410/lifecycle/transitions` | gaia-orchestrator | working | P1 |
| 2.6.5 | Transition history | `GET /lifecycle/history` | smoke_test | `curl http://localhost:6410/lifecycle/history` | gaia-orchestrator | working | P1 |
| 2.6.6 | Reconcile | `POST /lifecycle/reconcile` | functional_test | `curl -X POST http://localhost:6410/lifecycle/reconcile` — verify probes all tiers, infers actual state | gaia-orchestrator + all tiers | working | P1 |
| 2.6.7 | Live tier status | `GET /lifecycle/tiers` | smoke_test | `curl http://localhost:6410/lifecycle/tiers` | gaia-orchestrator | working | P1 |

### 2.7 Training Monitoring

| # | Feature | Endpoint/Method | Test Type | How to Test | Dependencies | Status | Priority |
|---|---------|----------------|-----------|-------------|--------------|--------|----------|
| 2.7.1 | Training status | `GET /training/status` | smoke_test | `curl http://localhost:6410/training/status` | gaia-orchestrator + study | working | P1 |
| 2.7.2 | Training validate | `POST /training/validate` | functional_test | `curl -X POST http://localhost:6410/training/validate` | gaia-orchestrator + completed training | unknown | P2 |
| 2.7.3 | Training kill | `POST /training/kill` | functional_test | `curl -X POST http://localhost:6410/training/kill` | gaia-orchestrator + active training | unknown | P2 |

### 2.8 Nano Management

| # | Feature | Endpoint/Method | Test Type | How to Test | Dependencies | Status | Priority |
|---|---------|----------------|-----------|-------------|--------------|--------|----------|
| 2.8.1 | Nano status | `GET /nano/status` | smoke_test | `curl http://localhost:6410/nano/status` | gaia-orchestrator | working | P2 |
| 2.8.2 | Nano backoff (GPU→CPU) | `POST /nano/backoff` | functional_test | `curl -X POST http://localhost:6410/nano/backoff` — verify Nano migrates to CPU | gaia-orchestrator + nano | working | P1 |
| 2.8.3 | Nano restore (CPU→GPU) | `POST /nano/restore` | functional_test | `curl -X POST http://localhost:6410/nano/restore` — verify Nano migrates to GPU | gaia-orchestrator + nano | working | P1 |

### 2.9 Watch Rotation

| # | Feature | Endpoint/Method | Test Type | How to Test | Dependencies | Status | Priority |
|---|---------|----------------|-----------|-------------|--------------|--------|----------|
| 2.9.1 | Watch state | `GET /watch/state` | smoke_test | `curl http://localhost:6410/watch/state` | gaia-orchestrator | working | P1 |
| 2.9.2 | Watch focus | `POST /watch/focus` | integration_test | `curl -X POST http://localhost:6410/watch/focus` — verify Prime loaded, Core+Nano unloaded | gaia-orchestrator + all tiers | working | P1 |
| 2.9.3 | Watch idle | `POST /watch/idle` | integration_test | `curl -X POST http://localhost:6410/watch/idle` — verify Core+Nano loaded, Prime unloaded | gaia-orchestrator + all tiers | working | P1 |

### 2.10 Other

| # | Feature | Endpoint/Method | Test Type | How to Test | Dependencies | Status | Priority |
|---|---------|----------------|-----------|-------------|--------------|--------|----------|
| 2.10.1 | Candidate snapshot | `GET /candidate/snapshot` | smoke_test | `curl http://localhost:6410/candidate/snapshot` | gaia-orchestrator | working | P2 |
| 2.10.2 | Candidate rollback | `POST /candidate/rollback` | functional_test | `curl -X POST http://localhost:6410/candidate/rollback -d '{"sha":"abc123","services":["gaia-core"]}'` | gaia-orchestrator + git | unknown | P2 |
| 2.10.3 | Warm pool sync | `POST /warm-pool/sync` | functional_test | `curl -X POST http://localhost:6410/warm-pool/sync -d '{"model":"test"}'` | gaia-orchestrator + model dir | unknown | P2 |
| 2.10.4 | Notify oracle fallback | `POST /notify/oracle-fallback` | smoke_test | `curl -X POST http://localhost:6410/notify/oracle-fallback` | gaia-orchestrator | unknown | P2 |
| 2.10.5 | WebSocket notifications | `WS /ws/notifications` | functional_test | `wscat -c ws://localhost:6410/ws/notifications` | gaia-orchestrator | unknown | P2 |

---

## Ring 3 — Routing & Triage (`gaia-nano` port 8090, `gaia-prime` port 7777)

### 3.1 Nano Triage

| # | Feature | Endpoint/Method | Test Type | How to Test | Dependencies | Status | Priority |
|---|---------|----------------|-----------|-------------|--------------|--------|----------|
| 3.1.1 | SIMPLE/COMPLEX classification | `POST localhost:8090/v1/chat/completions` | functional_test | Send triage prompt with few-shot examples, verify SIMPLE or COMPLEX output | gaia-nano + model loaded | working | P0 |
| 3.1.2 | Nano reflex response | `POST localhost:8090/v1/chat/completions` | functional_test | Send simple query, verify sub-second response | gaia-nano + model loaded | working | P0 |
| 3.1.3 | Nano GGUF fallback | `POST localhost:8090/v1/chat/completions` | functional_test | Unload safetensors model, verify GGUF backend responds | gaia-nano (GPU model unloaded, GGUF path set) | working | P1 |

### 3.2 Prime Inference

| # | Feature | Endpoint/Method | Test Type | How to Test | Dependencies | Status | Priority |
|---|---------|----------------|-----------|-------------|--------------|--------|----------|
| 3.2.1 | Prime GPU generation | `POST localhost:7777/v1/chat/completions` | functional_test | Send complex query, verify quality output with identity | gaia-prime + model on GPU | working | P0 |
| 3.2.2 | Prime CPU/GGUF generation | `POST localhost:7777/v1/chat/completions` | functional_test | Load GGUF model on CPU, verify generation works (slower) | gaia-prime + GGUF model on CPU | working | P1 |
| 3.2.3 | Prime with enable_thinking=false | `POST localhost:7777/v1/chat/completions` | functional_test | Send with `"chat_template_kwargs":{"enable_thinking":false}`, verify no `<think>` block | gaia-prime + model loaded | working | P1 |

### 3.3 FOCUSING Auto-Transition

| # | Feature | Endpoint/Method | Test Type | How to Test | Dependencies | Status | Priority |
|---|---------|----------------|-----------|-------------|--------------|--------|----------|
| 3.3.1 | Quality gate escalation triggers FOCUSING | Integration: gaia-core → orchestrator → prime | integration_test | Send complex query to `/process_packet` in AWAKE state. Verify orchestrator transitions to FOCUSING, Prime loads, response uses Prime, transitions back to AWAKE. | Full stack: core + orchestrator + prime + nano | untested | P0 |

### 3.4 Vision

| # | Feature | Endpoint/Method | Test Type | How to Test | Dependencies | Status | Priority |
|---|---------|----------------|-----------|-------------|--------------|--------|----------|
| 3.4.1 | Vision status (Prime) | `GET localhost:7777/vision/status` | smoke_test | `curl http://localhost:7777/vision/status` | gaia-prime | unknown | P2 |

### 3.5 Self-Awareness

| # | Feature | Endpoint/Method | Test Type | How to Test | Dependencies | Status | Priority |
|---|---------|----------------|-----------|-------------|--------------|--------|----------|
| 3.5.1 | Awareness status (Prime) | `GET localhost:7777/awareness/status` | smoke_test | `curl http://localhost:7777/awareness/status` | gaia-prime | unknown | P2 |
| 3.5.2 | Curiosity metrics | `GET localhost:7777/awareness/curiosity` | smoke_test | `curl http://localhost:7777/awareness/curiosity` | gaia-prime | unknown | P2 |

### 3.6 Compression

| # | Feature | Endpoint/Method | Test Type | How to Test | Dependencies | Status | Priority |
|---|---------|----------------|-----------|-------------|--------------|--------|----------|
| 3.6.1 | Compression stats (Prime) | `GET localhost:7777/compression/stats` | smoke_test | `curl http://localhost:7777/compression/stats` | gaia-prime | unknown | P2 |

---

## Ring 4 — Tools & Execution (`gaia-mcp`, port 8765)

### 4.1 HTTP Endpoints

| # | Feature | Endpoint/Method | Test Type | How to Test | Dependencies | Status | Priority |
|---|---------|----------------|-----------|-------------|--------------|--------|----------|
| 4.1.1 | Health check | `GET /health` | health_check | `curl http://localhost:8765/health` | gaia-mcp | working | P0 |
| 4.1.2 | Root status | `GET /` | smoke_test | `curl http://localhost:8765/` | gaia-mcp | working | P2 |
| 4.1.3 | Request approval | `POST /request_approval` | functional_test | `curl -X POST http://localhost:8765/request_approval -d '{"method":"ai_write","params":{"path":"/test","content":"x"}}'` | gaia-mcp | working | P1 |
| 4.1.4 | Approve action | `POST /approve_action` | functional_test | Get action_id from request_approval, then `curl -X POST http://localhost:8765/approve_action -d '{"action_id":"...","approval":"..."}'` | gaia-mcp + pending action | working | P1 |

### 4.2 JSON-RPC Tool Execution

| # | Feature | Endpoint/Method | Test Type | How to Test | Dependencies | Status | Priority |
|---|---------|----------------|-----------|-------------|--------------|--------|----------|
| 4.2.1 | ai_read | `POST /jsonrpc` | functional_test | `curl -X POST http://localhost:8765/jsonrpc -d '{"jsonrpc":"2.0","method":"ai_read","params":{"path":"/app/README.md"},"id":1}'` | gaia-mcp | working | P0 |
| 4.2.2 | ai_write | `POST /jsonrpc` | functional_test | Same pattern with `"method":"ai_write"` — requires approval if GAIA_MCP_BYPASS=false | gaia-mcp + approval | working | P1 |
| 4.2.3 | write_file | `POST /jsonrpc` | functional_test | `"method":"write_file","params":{"path":"/sandbox/test.txt","content":"hello"}` | gaia-mcp + approval | working | P1 |
| 4.2.4 | run_shell | `POST /jsonrpc` | functional_test | `"method":"run_shell","params":{"command":"echo hello"}` | gaia-mcp + approval | working | P1 |
| 4.2.5 | web_search | `POST /jsonrpc` | functional_test | `"method":"web_search","params":{"query":"test"}` | gaia-mcp + internet access | working | P1 |
| 4.2.6 | web_fetch | `POST /jsonrpc` | functional_test | `"method":"web_fetch","params":{"url":"https://example.com"}` | gaia-mcp + internet access | working | P1 |
| 4.2.7 | memory_search | `POST /jsonrpc` | functional_test | `"method":"memory_search","params":{"query":"GAIA"}` | gaia-mcp + vector index | working | P1 |
| 4.2.8 | memory_rebuild_index | `POST /jsonrpc` | functional_test | `"method":"memory_rebuild_index"` — requires approval | gaia-mcp + approval | unknown | P2 |
| 4.2.9 | introspect_logs | `POST /jsonrpc` | functional_test | `"method":"introspect_logs","params":{"service":"gaia-core"}` | gaia-mcp + log access | working | P1 |
| 4.2.10 | world_state | `POST /jsonrpc` | smoke_test | `"method":"world_state"` | gaia-mcp | working | P1 |
| 4.2.11 | cfr_search | `POST /jsonrpc` | functional_test | `"method":"cfr_search","params":{"query":"test"}` | gaia-mcp | unknown | P2 |
| 4.2.12 | kanka_list_campaigns | `POST /jsonrpc` | functional_test | `"method":"kanka_list_campaigns"` | gaia-mcp + KANKA_API_KEY | unknown | P2 |
| 4.2.13 | kanka_search | `POST /jsonrpc` | functional_test | `"method":"kanka_search","params":{"query":"test"}` | gaia-mcp + KANKA_API_KEY | unknown | P2 |
| 4.2.14 | kanka_list_entities | `POST /jsonrpc` | functional_test | `"method":"kanka_list_entities"` | gaia-mcp + KANKA_API_KEY | unknown | P2 |
| 4.2.15 | kanka_get_entity | `POST /jsonrpc` | functional_test | `"method":"kanka_get_entity","params":{"id":1}` | gaia-mcp + KANKA_API_KEY | unknown | P2 |
| 4.2.16 | kanka_create_entity | `POST /jsonrpc` | functional_test | Requires approval | gaia-mcp + KANKA_API_KEY + approval | unknown | P2 |
| 4.2.17 | kanka_update_entity | `POST /jsonrpc` | functional_test | Requires approval | gaia-mcp + KANKA_API_KEY + approval | unknown | P2 |
| 4.2.18 | notebooklm_list_notebooks | `POST /jsonrpc` | functional_test | `"method":"notebooklm_list_notebooks"` | gaia-mcp | unknown | P2 |
| 4.2.19 | notebooklm_chat | `POST /jsonrpc` | functional_test | `"method":"notebooklm_chat"` | gaia-mcp + notebook | unknown | P2 |
| 4.2.20 | notebooklm_create_note | `POST /jsonrpc` | functional_test | Requires approval | gaia-mcp + approval | unknown | P2 |
| 4.2.21 | audio_listen_start | `POST /jsonrpc` | functional_test | Requires approval | gaia-mcp + approval + audio service | unknown | P2 |
| 4.2.22 | audio_listen_stop | `POST /jsonrpc` | smoke_test | `"method":"audio_listen_stop"` | gaia-mcp | unknown | P2 |

---

## Ring 5 — Web & Interface (`gaia-web`, port 6414)

### 5.1 Core Endpoints

| # | Feature | Endpoint/Method | Test Type | How to Test | Dependencies | Status | Priority |
|---|---------|----------------|-----------|-------------|--------------|--------|----------|
| 5.1.1 | Health check | `GET /health` | health_check | `curl http://localhost:6414/health` | gaia-web | working | P0 |
| 5.1.2 | Chat input proxy | `POST /process_user_input?user_input=hello` | integration_test | `curl -X POST 'http://localhost:6414/process_user_input?user_input=hello'` — verify NDJSON stream proxied from core | gaia-web + gaia-core | working | P0 |
| 5.1.3 | Dashboard static files | `GET /` | smoke_test | `curl http://localhost:6414/` — verify HTML dashboard loads | gaia-web | working | P0 |
| 5.1.4 | Presence update | `POST /presence` | smoke_test | `curl -X POST http://localhost:6414/presence -d '{"activity":"test","status":"online"}'` | gaia-web | working | P2 |

### 5.2 System API Proxies

| # | Feature | Endpoint/Method | Test Type | How to Test | Dependencies | Status | Priority |
|---|---------|----------------|-----------|-------------|--------------|--------|----------|
| 5.2.1 | Service status | `GET /api/system/services` | integration_test | `curl http://localhost:6414/api/system/services` — verify aggregated health from doctor | gaia-web + gaia-doctor | working | P0 |
| 5.2.2 | Sleep state | `GET /api/system/sleep` | integration_test | `curl http://localhost:6414/api/system/sleep` — verify core sleep + orchestrator GPU | gaia-web + gaia-core + orchestrator | working | P1 |
| 5.2.3 | System status | `GET /api/system/status` | integration_test | `curl http://localhost:6414/api/system/status` | gaia-web + orchestrator + monkey | working | P1 |
| 5.2.4 | Irritations | `GET /api/system/irritations` | integration_test | `curl http://localhost:6414/api/system/irritations` | gaia-web + gaia-doctor | working | P1 |
| 5.2.5 | Dissonance report | `GET /api/system/dissonance` | integration_test | `curl http://localhost:6414/api/system/dissonance` | gaia-web + gaia-doctor | working | P2 |
| 5.2.6 | Lifecycle proxy | `GET /api/system/lifecycle/state` | integration_test | `curl http://localhost:6414/api/system/lifecycle/state` | gaia-web + orchestrator | working | P1 |
| 5.2.7 | Cognitive battery | `GET /api/system/cognitive/status` | integration_test | `curl http://localhost:6414/api/system/cognitive/status` | gaia-web + gaia-doctor | working | P1 |
| 5.2.8 | Training progress | `GET /api/system/training/progress` | integration_test | `curl http://localhost:6414/api/system/training/progress` | gaia-web + gaia-study | working | P2 |
| 5.2.9 | Registry validation | `GET /api/system/registry/validation` | integration_test | `curl http://localhost:6414/api/system/registry/validation` | gaia-web + gaia-doctor | working | P2 |
| 5.2.10 | Registry paths | `GET /api/system/registry/paths` | integration_test | `curl http://localhost:6414/api/system/registry/paths` | gaia-web + all services | working | P2 |

### 5.3 Other Proxies

| # | Feature | Endpoint/Method | Test Type | How to Test | Dependencies | Status | Priority |
|---|---------|----------------|-----------|-------------|--------------|--------|----------|
| 5.3.1 | Chaos proxy | `POST /api/chaos/inject` | integration_test | `curl -X POST http://localhost:6414/api/chaos/inject` | gaia-web + gaia-monkey | working | P2 |
| 5.3.2 | Changelog | `GET /api/changelog/` | smoke_test | `curl http://localhost:6414/api/changelog/` | gaia-web | working | P2 |
| 5.3.3 | Wiki proxy | `GET /wiki/` | smoke_test | `curl http://localhost:6414/wiki/` | gaia-web + gaia-wiki | unknown | P2 |
| 5.3.4 | CodeMind proxy | `GET /api/codemind/status` | smoke_test | `curl http://localhost:6414/api/codemind/status` | gaia-web | unknown | P2 |

---

## Ring 6 — Support Services

### 6.1 gaia-doctor (port 6419)

| # | Feature | Endpoint/Method | Test Type | How to Test | Dependencies | Status | Priority |
|---|---------|----------------|-----------|-------------|--------------|--------|----------|
| 6.1.1 | Health check | `GET /health` | health_check | `curl http://localhost:6419/health` | gaia-doctor | working | P0 |
| 6.1.2 | Full status | `GET /status` | smoke_test | `curl http://localhost:6419/status` — verify service health map, alarms, irritations | gaia-doctor + polled services | working | P0 |
| 6.1.3 | Alarms | `GET /alarms` | smoke_test | `curl http://localhost:6419/alarms` | gaia-doctor | working | P1 |
| 6.1.4 | Irritations | `GET /irritations` | smoke_test | `curl http://localhost:6419/irritations` | gaia-doctor | working | P1 |
| 6.1.5 | Dissonance | `GET /dissonance` | functional_test | `curl http://localhost:6419/dissonance` — verify prod vs candidate diff | gaia-doctor + project root | working | P1 |
| 6.1.6 | Serenity | `GET /serenity` | smoke_test | `curl http://localhost:6419/serenity` | gaia-doctor + shared state file | working | P1 |
| 6.1.7 | Cognitive battery status | `GET /cognitive/status` | smoke_test | `curl http://localhost:6419/cognitive/status` | gaia-doctor | working | P1 |
| 6.1.8 | Cognitive battery results | `GET /cognitive/results` | smoke_test | `curl http://localhost:6419/cognitive/results` | gaia-doctor + battery run | working | P1 |
| 6.1.9 | List cognitive tests | `GET /cognitive/tests` | smoke_test | `curl http://localhost:6419/cognitive/tests` | gaia-doctor | working | P2 |
| 6.1.10 | Run cognitive battery | `POST /cognitive/run` | integration_test | `curl -X POST http://localhost:6419/cognitive/run` — verify async thread triggers, queries gaia-core | gaia-doctor + gaia-core | working | P1 |
| 6.1.11 | Cognitive monitor | `GET /cognitive/monitor` | smoke_test | `curl http://localhost:6419/cognitive/monitor` | gaia-doctor | working | P2 |
| 6.1.12 | Maintenance enter | `POST /maintenance/enter` | functional_test | `curl -X POST http://localhost:6419/maintenance/enter -d '{"reason":"test","entered_by":"test"}'` | gaia-doctor | working | P1 |
| 6.1.13 | Maintenance exit | `POST /maintenance/exit` | functional_test | `curl -X POST http://localhost:6419/maintenance/exit` | gaia-doctor | working | P1 |
| 6.1.14 | Maintenance status | `GET /maintenance/status` | smoke_test | `curl http://localhost:6419/maintenance/status` | gaia-doctor | working | P1 |
| 6.1.15 | Surgeon config | `GET /surgeon/config` | smoke_test | `curl http://localhost:6419/surgeon/config` | gaia-doctor | working | P2 |
| 6.1.16 | Surgeon queue | `GET /surgeon/queue` | smoke_test | `curl http://localhost:6419/surgeon/queue` | gaia-doctor | working | P2 |
| 6.1.17 | Surgeon approve | `POST /surgeon/approve` | functional_test | Requires pending repair in queue | gaia-doctor + pending repair | unknown | P2 |
| 6.1.18 | Surgeon reject | `POST /surgeon/reject` | functional_test | Requires pending repair in queue | gaia-doctor + pending repair | unknown | P2 |
| 6.1.19 | Surgeon history | `GET /surgeon/history` | smoke_test | `curl http://localhost:6419/surgeon/history` | gaia-doctor | working | P2 |
| 6.1.20 | OOM history | `GET /oom/history` | smoke_test | `curl http://localhost:6419/oom/history` | gaia-doctor | working | P2 |
| 6.1.21 | Errors | `GET /errors` | smoke_test | `curl http://localhost:6419/errors` | gaia-doctor | working | P2 |
| 6.1.22 | Registry | `GET /registry` | smoke_test | `curl http://localhost:6419/registry` | gaia-doctor | working | P2 |
| 6.1.23 | Logs | `GET /logs` | smoke_test | `curl http://localhost:6419/logs` | gaia-doctor | working | P2 |
| 6.1.24 | Log health | `GET /logs/health` | smoke_test | `curl http://localhost:6419/logs/health` | gaia-doctor | working | P2 |
| 6.1.25 | Pipeline status | `GET /pipeline` | smoke_test | `curl http://localhost:6419/pipeline` | gaia-doctor | working | P1 |
| 6.1.26 | Pipeline run | `POST /pipeline/run` | integration_test | `curl -X POST http://localhost:6419/pipeline/run` | gaia-doctor + gaia-study | unknown | P2 |
| 6.1.27 | GPU info | `GET /gpu` | smoke_test | `curl http://localhost:6419/gpu` | gaia-doctor | working | P2 |
| 6.1.28 | Model info | `GET /model` | smoke_test | `curl http://localhost:6419/model` | gaia-doctor | working | P2 |
| 6.1.29 | KV cache status | `GET /kv-cache` | smoke_test | `curl http://localhost:6419/kv-cache` | gaia-doctor | working | P2 |
| 6.1.30 | Chaos notification | `POST /notify/chaos_injection` | smoke_test | `curl -X POST http://localhost:6419/notify/chaos_injection -d '{"type":"test"}'` | gaia-doctor | unknown | P2 |
| 6.1.31 | Health polling (auto-restart) | Background polling loop | stress_test | Kill a service container, verify doctor detects failure within POLL_INTERVAL and triggers restart after FAILURE_THRESHOLD | gaia-doctor + docker socket + target service | working | P0 |

### 6.2 gaia-monkey (port 6420)

| # | Feature | Endpoint/Method | Test Type | How to Test | Dependencies | Status | Priority |
|---|---------|----------------|-----------|-------------|--------------|--------|----------|
| 6.2.1 | Health check | `GET /health` | health_check | `curl http://localhost:6420/health` | gaia-monkey | working | P0 |
| 6.2.2 | Full status | `GET /status` | smoke_test | `curl http://localhost:6420/status` | gaia-monkey | working | P1 |
| 6.2.3 | Config get | `GET /config` | smoke_test | `curl http://localhost:6420/config` | gaia-monkey | working | P2 |
| 6.2.4 | Config update | `POST /config` | functional_test | `curl -X POST http://localhost:6420/config -d '{"enabled":false}'` | gaia-monkey | working | P2 |
| 6.2.5 | Chaos inject | `POST /chaos/inject` | integration_test | `curl -X POST http://localhost:6420/chaos/inject` | gaia-monkey + target services | working | P1 |
| 6.2.6 | Container drill | `POST /chaos/drill` | integration_test | `curl -X POST http://localhost:6420/chaos/drill -d '{"targets":["gaia-mcp"]}'` | gaia-monkey + docker socket | working | P1 |
| 6.2.7 | Code fault | `POST /chaos/code` | integration_test | `curl -X POST http://localhost:6420/chaos/code` | gaia-monkey + project root | unknown | P2 |
| 6.2.8 | Linguistic eval | `POST /chaos/linguistic` | integration_test | `curl -X POST http://localhost:6420/chaos/linguistic -d '{"suite":"persona"}'` | gaia-monkey + gaia-core | unknown | P2 |
| 6.2.9 | Drill history | `GET /chaos/history` | smoke_test | `curl http://localhost:6420/chaos/history` | gaia-monkey | working | P2 |
| 6.2.10 | Meditation enter | `POST /meditation/enter` | functional_test | `curl -X POST http://localhost:6420/meditation/enter` | gaia-monkey | working | P1 |
| 6.2.11 | Meditation exit | `POST /meditation/exit` | functional_test | `curl -X POST http://localhost:6420/meditation/exit` | gaia-monkey | working | P1 |
| 6.2.12 | Serenity | `GET /serenity` | smoke_test | `curl http://localhost:6420/serenity` | gaia-monkey | working | P0 |
| 6.2.13 | Serenity break | `POST /serenity/break` | functional_test | `curl -X POST http://localhost:6420/serenity/break -d '{"reason":"test"}'` | gaia-monkey | working | P1 |
| 6.2.14 | Recovery record | `POST /serenity/record_recovery` | functional_test | `curl -X POST http://localhost:6420/serenity/record_recovery -d '{"category":"test","detail":"test recovery"}'` | gaia-monkey | working | P1 |
| 6.2.15 | Serenity reset | `POST /serenity/reset` | functional_test | `curl -X POST http://localhost:6420/serenity/reset` | gaia-monkey | working | P2 |

### 6.3 gaia-study (port 8766)

| # | Feature | Endpoint/Method | Test Type | How to Test | Dependencies | Status | Priority |
|---|---------|----------------|-----------|-------------|--------------|--------|----------|
| 6.3.1 | Health check | `GET /health` | health_check | `curl http://localhost:8766/health` | gaia-study | working | P0 |
| 6.3.2 | Status | `GET /status` | smoke_test | `curl http://localhost:8766/status` | gaia-study | working | P1 |
| 6.3.3 | Build vector index | `POST /index/build` | functional_test | `curl -X POST http://localhost:8766/index/build -d '{"knowledge_base_name":"test"}'` | gaia-study + knowledge dir | working | P1 |
| 6.3.4 | Add to index | `POST /index/add` | functional_test | `curl -X POST http://localhost:8766/index/add -d '{"knowledge_base_name":"test","file_path":"/knowledge/test.md"}'` | gaia-study + existing index | working | P1 |
| 6.3.5 | Query index | `POST /index/query` | functional_test | `curl -X POST http://localhost:8766/index/query -d '{"knowledge_base_name":"test","query":"GAIA"}'` | gaia-study + built index | working | P1 |
| 6.3.6 | Index status | `GET /index/{name}/status` | smoke_test | `curl http://localhost:8766/index/test/status` | gaia-study | working | P2 |
| 6.3.7 | Index refresh | `POST /index/{name}/refresh` | functional_test | `curl -X POST http://localhost:8766/index/test/refresh` | gaia-study + existing index | unknown | P2 |
| 6.3.8 | GPU ready signal | `POST /study/gpu-ready` | functional_test | `curl -X POST http://localhost:8766/study/gpu-ready` | gaia-study | working | P1 |
| 6.3.9 | GPU release signal | `POST /study/gpu-release` | functional_test | `curl -X POST http://localhost:8766/study/gpu-release` | gaia-study | working | P1 |
| 6.3.10 | Training status | `GET /study/training/status` | smoke_test | `curl http://localhost:8766/study/training/status` | gaia-study | working | P0 |
| 6.3.11 | Training kill | `POST /study/training/kill` | functional_test | `curl -X POST http://localhost:8766/study/training/kill` | gaia-study + active training | unknown | P2 |
| 6.3.12 | Start study/training | `POST /study/start` | integration_test | `curl -X POST http://localhost:8766/study/start -d '{"adapter_name":"test","documents":["/knowledge/test.md"],"tier":3,"max_steps":10}'` | gaia-study + GPU + base model | working | P1 |
| 6.3.13 | Study status | `GET /study/status` | smoke_test | `curl http://localhost:8766/study/status` | gaia-study | working | P1 |
| 6.3.14 | Study cancel | `POST /study/cancel` | functional_test | `curl -X POST http://localhost:8766/study/cancel` | gaia-study + active training | unknown | P2 |
| 6.3.15 | List adapters | `GET /adapters` | smoke_test | `curl http://localhost:8766/adapters` | gaia-study | working | P1 |
| 6.3.16 | Load adapter | `POST /adapters/load` | integration_test | `curl -X POST http://localhost:8766/adapters/load -d '{"adapter_name":"test","tier":3}'` | gaia-study + gaia-core + adapter exists | working | P1 |
| 6.3.17 | Unload adapter | `POST /adapters/unload` | integration_test | `curl -X POST http://localhost:8766/adapters/unload -d '{"adapter_name":"test","tier":3}'` | gaia-study + gaia-core | working | P1 |
| 6.3.18 | Adapter info | `GET /adapters/{name}` | smoke_test | `curl http://localhost:8766/adapters/test` | gaia-study | working | P2 |
| 6.3.19 | Delete adapter | `DELETE /adapters/{name}` | functional_test | `curl -X DELETE http://localhost:8766/adapters/test` | gaia-study + adapter exists | unknown | P2 |
| 6.3.20 | Pipeline run | `POST /pipeline/run` | integration_test | `curl -X POST http://localhost:8766/pipeline/run -d '{"dry_run":true}'` | gaia-study + GPU | unknown | P2 |
| 6.3.21 | Pipeline status | `GET /pipeline/status` | smoke_test | `curl http://localhost:8766/pipeline/status` | gaia-study | working | P1 |

### 6.4 gaia-audio (port 8080)

| # | Feature | Endpoint/Method | Test Type | How to Test | Dependencies | Status | Priority |
|---|---------|----------------|-----------|-------------|--------------|--------|----------|
| 6.4.1 | Health check | `GET /health` | health_check | `curl http://localhost:8080/health` | gaia-audio | working | P0 |
| 6.4.2 | Status | `GET /status` | smoke_test | `curl http://localhost:8080/status` | gaia-audio | working | P1 |
| 6.4.3 | Transcribe (STT) | `POST /transcribe` | functional_test | `curl -X POST http://localhost:8080/transcribe -d '{"audio_base64":"...","sample_rate":16000}'` | gaia-audio + Qwen3-ASR model | unknown | P1 |
| 6.4.4 | Analyze audio | `POST /analyze` | functional_test | `curl -X POST http://localhost:8080/analyze -d '{"audio_base64":"...","sample_rate":16000}'` | gaia-audio | unknown | P2 |
| 6.4.5 | Refine transcript | `POST /refine` | integration_test | `curl -X POST http://localhost:8080/refine -d '{"text":"unformatted transcript","max_tokens":200}'` | gaia-audio + gaia-nano | unknown | P1 |
| 6.4.6 | Synthesize (TTS) | `POST /synthesize` | functional_test | `curl -X POST http://localhost:8080/synthesize -d '{"text":"hello world","tier":"auto"}'` | gaia-audio + TTS model | unknown | P1 |
| 6.4.7 | List voices | `GET /voices` | smoke_test | `curl http://localhost:8080/voices` | gaia-audio | unknown | P2 |
| 6.4.8 | Config | `GET /config` | smoke_test | `curl http://localhost:8080/config` | gaia-audio | working | P2 |
| 6.4.9 | Mute | `POST /mute` | functional_test | `curl -X POST http://localhost:8080/mute` | gaia-audio | working | P1 |
| 6.4.10 | Unmute | `POST /unmute` | functional_test | `curl -X POST http://localhost:8080/unmute` | gaia-audio | working | P1 |
| 6.4.11 | Deep sleep | `POST /sleep` | functional_test | `curl -X POST http://localhost:8080/sleep` | gaia-audio | working | P1 |
| 6.4.12 | Wake | `POST /wake` | functional_test | `curl -X POST http://localhost:8080/wake` | gaia-audio | working | P1 |
| 6.4.13 | GPU release | `POST /gpu/release` | functional_test | `curl -X POST http://localhost:8080/gpu/release` | gaia-audio | working | P1 |
| 6.4.14 | GPU reclaim | `POST /gpu/reclaim` | functional_test | `curl -X POST http://localhost:8080/gpu/reclaim` | gaia-audio | working | P1 |
| 6.4.15 | GPU status | `GET /gpu/status` | smoke_test | `curl http://localhost:8080/gpu/status` | gaia-audio | working | P1 |
| 6.4.16 | WebSocket status | `WS /status/ws` | functional_test | `wscat -c ws://localhost:8080/status/ws` | gaia-audio | unknown | P2 |

### 6.5 gaia-translate (port 5000, internal only)

| # | Feature | Endpoint/Method | Test Type | How to Test | Dependencies | Status | Priority |
|---|---------|----------------|-----------|-------------|--------------|--------|----------|
| 6.5.1 | Translation | LibreTranslate API | smoke_test | `docker exec gaia-web curl http://gaia-translate:5000/translate -d '{"q":"hello","source":"en","target":"es"}'` | gaia-translate | unknown | P2 |

### 6.6 gaia-wiki (port 8080, internal only)

| # | Feature | Endpoint/Method | Test Type | How to Test | Dependencies | Status | Priority |
|---|---------|----------------|-----------|-------------|--------------|--------|----------|
| 6.6.1 | MkDocs serving | `GET /` | smoke_test | `docker exec gaia-web curl http://gaia-wiki:8080/` | gaia-wiki | unknown | P2 |

---

## Cross-Ring Integration Tests

These test full flows that traverse multiple rings.

| # | Flow | Rings | Test Type | How to Test | Dependencies | Status | Priority |
|---|------|-------|-----------|-------------|--------------|--------|----------|
| X.1 | End-to-end chat: user input → web → core → triage → core → prime → web | 0,1,3,5 | integration_test | `curl -X POST 'http://localhost:6414/process_user_input?user_input=Write+a+haiku'` — verify streaming tokens arrive | Full stack | working | P0 |
| X.2 | Sleep cycle: force sleep → verify checkpoint → wake → verify continuity | 0,2 | integration_test | Force sleep via `/sleep/force`, check checkpoint file, send wake signal, verify response has context | core + orchestrator | working | P0 |
| X.3 | FOCUSING flow: AWAKE → complex query → FOCUSING → Prime loads → response → AWAKE | 0,1,2,3 | integration_test | Send complex query while in AWAKE state, monitor lifecycle transitions | Full stack | untested | P0 |
| X.4 | Training handoff: Prime → Study → Train → Study → Prime | 1,2,6.3 | integration_test | Trigger `/handoff/prime-to-study`, start training, complete, trigger `/handoff/study-to-prime` | orchestrator + prime + study + GPU | working | P1 |
| X.5 | Doctor auto-restart: kill service → doctor detects → restart | 6.1 | stress_test | `docker stop gaia-mcp && sleep 45 && curl http://localhost:8765/health` — verify MCP restarted | doctor + docker socket | working | P0 |
| X.6 | Chaos drill full cycle: meditation → inject → recover → serenity check | 6.2,6.1 | stress_test | Enter meditation, inject chaos, exit meditation, check serenity | monkey + doctor + target | working | P1 |
| X.7 | Cognitive battery: trigger → queries → results → alignment | 0,6.1 | integration_test | `POST /cognitive/run` on doctor, wait, check `/cognitive/results` | doctor + core | working | P1 |
| X.8 | Tool execution with approval: core → MCP request_approval → dashboard approve → execute | 0,4,5 | integration_test | Trigger tool call requiring approval, verify pending action, approve via dashboard | core + mcp + web | working | P1 |
| X.9 | Audio wake trigger: voice detected → audio → core wake | 0,6.4 | integration_test | Send audio to `/transcribe`, verify wake signal sent to core | audio + core | unknown | P2 |
| X.10 | Nano-refiner transcript cleanup: audio → nano → refined text | 3,6.4 | integration_test | `POST /refine` on audio, verify nano called for cleanup | audio + nano | unknown | P2 |

---

## Stress Tests

| # | Scenario | Test Type | How to Test | Dependencies | Status | Priority |
|---|----------|-----------|-------------|--------------|--------|----------|
| S.1 | Concurrent chat requests (10 simultaneous) | stress_test | Use `ab` or parallel curl: 10 concurrent `/process_user_input` | Full stack | untested | P1 |
| S.2 | VRAM pressure: load all tiers simultaneously | stress_test | Load Nano+Core+Prime on GPU, verify OOM handling or graceful degradation | All tiers + orchestrator | unknown | P1 |
| S.3 | Rapid lifecycle transitions | stress_test | Script 20 rapid AWAKE→FOCUSING→AWAKE transitions, verify no stuck states | orchestrator + prime | untested | P1 |
| S.4 | Model hot-swap under load | stress_test | Send generation request, simultaneously trigger model swap | prime | untested | P2 |
| S.5 | Long-running training + inference contention | stress_test | Start training, then attempt Prime inference, verify handoff works | study + prime + orchestrator | untested | P1 |
| S.6 | Doctor watchdog stress | stress_test | Kill 3 services simultaneously, verify doctor restarts all correctly | doctor + 3 target services | untested | P1 |

---

## Known Issues from Sprint Backlog

| Issue | Ring | Impact | Status |
|-------|------|--------|--------|
| Consciousness Matrix probe tuning (container stop on unload) | 2 | Lifecycle transitions may leave stale containers | P1 — next session |
| Lifecycle machine vs Consciousness Matrix conflict | 2 | Potential state disagreements between old/new systems | P1 — next session |
| Orchestrator auto-transition on escalation (AWAKE→FOCUSING) | 2,3 | Quality gate escalation does not auto-trigger FOCUSING | P2 — soon |
| CPU activation streaming for GGUF tiers | 1 | No polygraph/activations when running GGUF backend | P2 — soon |
| 9B training blocked | 6.3 | gptqmodel segfault on RTX 5080 16GB VRAM | P3 — needs cloud GPU |
| Architecture.md outdated | docs | Model tiers section does not reflect current state | P2 — soon |
