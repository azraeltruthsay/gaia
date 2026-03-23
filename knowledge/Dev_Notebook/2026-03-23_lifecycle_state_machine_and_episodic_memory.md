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

## Known Issues for Next Session

1. **No periodic reconciliation** — lifecycle machine reconciles on startup but not continuously. Models can drift without detection.
2. **GAIA Engine needs SSE streaming** — managed proxy buffers full response. Not a blocker but adds latency for long responses.
3. **Engine queueing** — GAIA Engine has no request queue. Multiple rapid requests can overwhelm it.
4. **Discord rate limiting** — rapid container restarts exhaust Discord's IDENTIFY quota. Need a reconnect backoff strategy.
5. **Phase 5 cleanup** — old state enums (GaiaState, GPUState, GPUOwner) still exist as dead code.
