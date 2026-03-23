# 2026-03-23 — Lifecycle State Machine, Episodic Memory, HTTPS

## Session Summary

Continuation of 2026-03-22 subprocess isolation work. Built the unified GPU lifecycle state machine (Phases 1-3), Mission Control dashboard, episodic memory buffer, GPTQ loading fix, and HTTPS remote access.

## Commits (11 total across both days)

1. **Subprocess isolation, tier router, GPTQ Prime** — 16GB→5.8GB VRAM
2. **Deep sleep button, log-scale graph, model path fixes**
3. **Lifecycle Phase 1** — states, snapshot models, orchestrator endpoints
4. **Mission Control dashboard** — state badge, VRAM bar, tier cards, transition buttons
5. **Automated penpal script** — E11 review + E12 request generation
6. **Phase 2** — orchestrator convergence (gpu/sleep, gpu/wake, watch/focus, watch/idle delegate to lifecycle)
7. **Phase 3** — Core convergence (LifecycleClient, sleep endpoints delegate to lifecycle)
8. **agent_core refactor** — 9 `_gpu_released` checks → `_is_prime_available` lifecycle property
9. **Episodic memory** — event buffer + world state injection
10. **recall_events MCP tool** — callable episodic memory with CFR support
11. **Streaming fix** — vllm_remote_model handles non-SSE responses from GAIA Engine

## Lifecycle State Machine

Seven states: AWAKE, LISTENING, FOCUSING, MEDITATION, SLEEP, DEEP_SLEEP, TRANSITIONING.
Orchestrator is the authority. gaia-core queries via LifecycleClient.
All GPU operations flow through validated transitions with rollback on failure.
Reconciliation on startup probes all tier engines and infers actual state.

Key endpoints: GET /lifecycle/state, POST /lifecycle/transition, GET /lifecycle/transitions, POST /lifecycle/reconcile.

## Episodic Memory

`gaia_common.event_buffer` — rolling JSONL log at `/shared/event_buffer.jsonl`.
- Lifecycle transitions, conversations, sleep/wake events auto-logged
- Recent events injected into world state section of system prompt
- `recall_events` MCP tool for explicit recall (recent mode + CFR deep analysis mode)
- Prevents hallucinated memories — GAIA answers from real event data

## Streaming Bug Found & Fixed

The GAIA Engine's managed proxy returns plain JSON, not SSE. `vllm_remote_model._stream_chat()` expected SSE `data:` lines, got nothing, yielded empty response. Fixed by detecting Content-Type and parsing both formats.

## Discord Bot Issue

StatReload watcher in uvicorn killed the Discord bot's asyncio task when static files changed. Multiple rapid restarts hit Discord's IDENTIFY rate limit. Fixed by force-recreating container for clean session.

## HTTPS Remote Access

Caddy reverse proxy with self-signed cert for WireGuard access:
- :8443 → localhost:6414 (Mission Control)
- :8410 → localhost:6410 (Orchestrator API)
- :9443 → localhost:9999 (Dozzle logs)

Config at `/etc/caddy/conf.d/gaia.caddyfile`. Manual cert at `/etc/caddy/gaia.crt`.

## Extended Sprint (25 commits total)

After the initial 12 commits, continued pushing through the backlog:

13. **Periodic lifecycle reconciliation** — 60s loop auto-detects and reloads missing models
14. **Knowledge contamination fix** — excluded dnd_campaign from semantic probe
15. **World state fix** — correct file used, immune system truncated, Recent Events visible
16. **Reflection cap** — OPERATOR capped to 1 iteration (was 3, caused over-thinking)
17. **Engine queueing** — semaphore limits concurrent inference to 1
18. **Event buffer expansion** — sleep transitions + model routing decisions logged
19. **Warm pool config** — GPTQ Prime + boot seeding
20. **Worker crash tracing** — traceback + event buffer logging for silent unloads
21. **Discord bot fix** — root cause: docker-compose.override.yml had --reload on gaia-web
22. **LISTENING state** — voice join/leave triggers lifecycle transitions
23. **MEDITATION state** — training handoffs delegate to lifecycle machine
24. **SSE streaming** — EngineHandler returns text/event-stream for stream=true
25. **Legacy deprecation** — GaiaState + _gpu_released marked as legacy with migration notes

## Remaining for Next Session

1. **GAIA Engine standalone repo** — extract to own GitHub repo with semver
2. **True per-token SSE streaming** — current SSE sends full response as single chunk
3. **Nano silent unloading root cause** — instrumented but not yet observed/diagnosed
4. **Full Phase 5 cleanup** — remove GaiaState, GPUState, GPUOwner entirely (needs test updates)
