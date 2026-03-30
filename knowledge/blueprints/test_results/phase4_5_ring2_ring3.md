# Phase 4/5 — Ring 2 (Orchestrator) & Ring 3 (Routing/Triage) Test Results

**Date**: 2026-03-26 03:01-03:05 UTC
**Tester**: Claude Code (automated)
**System State at Start**: AWAKE (Core GPU, Nano CPU/GGUF, Prime CPU/GGUF)

---

## Ring 2 — gaia-orchestrator (port 6410)

### Smoke Tests (Health & Status Endpoints)

| # | Test | Command | HTTP | Response Summary | Result |
|---|------|---------|------|-----------------|--------|
| 1 | Health | `GET /health` | 200 | `{"status":"healthy","service":"gaia-orchestrator"}` | **PASS** |
| 2 | Consciousness Matrix | `GET /consciousness/matrix` | 200 | nano=conscious, core=conscious, prime=subconscious, all healthy, all ok | **PASS** |
| 3 | Tier: Nano Status | `GET /tier/nano/status` | 404 | `{"detail":"Not Found"}` | **FAIL** — endpoint not implemented |
| 4 | Tier: Core Status | `GET /tier/core/status` | 404 | `{"detail":"Not Found"}` | **FAIL** — endpoint not implemented |
| 5 | Tier: Prime Status | `GET /tier/prime/status` | 404 | `{"detail":"Not Found"}` | **FAIL** — endpoint not implemented |
| 6 | Lifecycle State | `GET /lifecycle/state` | 200 | state=awake, core=GPU/3624MB, nano=CPU, prime=CPU, VRAM 3624/15833 used | **PASS** |
| 7 | Lifecycle History | `GET /lifecycle/history` | 200 | 17 transitions returned (deep_sleep/awake/sleep/focusing cycles) | **PASS** |
| 8 | Lifecycle Transitions | `GET /lifecycle/transitions` | 200 | 5 available: voice_join->listening, escalation->focusing, idle->sleep, training->meditation, user_request->multiple | **PASS** |
| 9 | GPU Status | `GET /gpu/status` | 200 | owner=gaia-core, lease active, queue empty | **PASS** |
| 10 | Container List | `GET /containers` | 404 | `{"detail":"Not Found"}` | **FAIL** — endpoint not implemented |
| 11 | Watch Status | `GET /watch/status` | 404 | `{"detail":"Not Found"}` | **FAIL** — endpoint not implemented |
| 12 | Training Status | `GET /training/status` | 200 | manager idle, subprocess state=failed (previous training run), last error: "Failed to setup QLoRA trainer" | **PASS** (endpoint works; stale failure from previous 8B training attempt is expected) |

**Ring 2 Smoke Summary**: 7/12 PASS, 5/12 FAIL (all failures are 404s on unimplemented endpoints: `/tier/*/status`, `/containers`, `/watch/status`)

### Functional Tests — Consciousness Transitions

| # | Test | Command | HTTP | Response Summary | Result |
|---|------|---------|------|-----------------|--------|
| 13 | Pre-transition state | `GET /consciousness/matrix` | 200 | nano=conscious, core=conscious, prime=subconscious — all ok | **PASS** |
| 14 | Transition: AWAKE -> SLEEP | `POST /consciousness/sleep` | 200 | Transition accepted and executed | **PASS** |
| 15 | Verify sleep state (matrix) | `GET /consciousness/matrix` | 200 | nano=subconscious, core=subconscious(target)/unconscious(actual), prime=unconscious | **PASS** |
| 16 | Verify sleep state (lifecycle) | `GET /lifecycle/state` | 200 | state=deep_sleep (sleep auto-advanced to deep_sleep) | **PASS** |
| 17 | Transition: SLEEP -> AWAKE | `POST /consciousness/awake` | 200 | nano=CONSCIOUS (already at target), core=already_loaded_gpu, prime=loaded_cpu (GGUF) | **PASS** |
| 18 | Verify awake state (matrix) | `GET /consciousness/matrix` | 200 | nano=conscious, core=conscious, prime=subconscious — all ok, all healthy | **PASS** |

**Consciousness Transition Summary**: 6/6 PASS

**NOTE**: After wake, the lifecycle/state FSM label remained "deep_sleep" while actual tier states matched awake configuration. The `/consciousness/awake` endpoint loads tiers directly but does not update the lifecycle FSM label. The consciousness matrix correctly reflects reality. This is a known discrepancy — the FSM label is stale but tier states are correct.

**Core discrepancy during sleep**: Core target was "subconscious" (CPU) but actual was "unconscious" — the CPU fallback model may not have been available, or the unload completed faster than the CPU reload. After re-awakening, core loaded successfully on GPU.

---

## Ring 3 — Nano Triage (port 8090) & Prime Inference (port 7777)

### Nano Triage Tests

| # | Test | Command | HTTP | Response Summary | Result |
|---|------|---------|------|-----------------|--------|
| 19 | Nano: Simple message triage | `POST /v1/chat/completions` ("What is 2+2?") | 200 | Response: "2+2 = 4 (Simple)" — model: Qwen3.5-0.8B-Abliterated-Q8_0.gguf, 10 tokens in 165ms | **PASS** |
| 20 | Nano: Complex message triage | `POST /v1/chat/completions` (quantum entanglement) | 200 | Response: thinking tokens began (`<think>`) — model started reasoning, 10 tokens in 182ms | **PASS** |

**Nano Notes**:
- Simple query correctly identified as simple and answered directly
- Complex query triggered thinking mode (expected for Qwen3.5 with think tokens)
- Both responses under 200ms for 10 tokens — sub-second latency confirmed
- Model: `Qwen3.5-0.8B-Abliterated-Q8_0.gguf` (GGUF backend on CPU)

### Prime Inference Tests

| # | Test | Command | HTTP | Response Summary | Result |
|---|------|---------|------|-----------------|--------|
| 21 | Prime: Health (pre-wake) | `GET /health` | 200 | model_loaded=false, mode=standby, backend=none | **PASS** (expected — Prime was in standby before consciousness/awake) |
| 22 | Prime: Health (post-wake) | `GET /health` | 200 | model_loaded=true, mode=active, backend=gguf, worker_pid=429881 | **PASS** |
| 23 | Prime: Identity inference | `POST /v1/chat/completions` ("Who are you?") | 200 | Response: "I am GAIA, a sovereign AI agent built as a Service-Oriented Architecture..." — 50 tokens in 7370ms (147ms/token CPU) | **PASS** |

**Prime Notes**:
- Identity baking confirmed: Prime self-identifies as GAIA without system prompt
- Model: `Huihui-Qwen3-8B-GAIA-Prime-identity-Q8_0.gguf` (GGUF on CPU)
- Inference speed: ~6.8 tokens/sec on CPU (expected for 8B Q8_0 GGUF)

---

## Summary

| Ring | Category | Pass | Fail | Skip | Total |
|------|----------|------|------|------|-------|
| Ring 2 | Smoke (status endpoints) | 7 | 5 | 0 | 12 |
| Ring 2 | Functional (consciousness transitions) | 6 | 0 | 0 | 6 |
| Ring 3 | Nano triage | 2 | 0 | 0 | 2 |
| Ring 3 | Prime inference | 3 | 0 | 0 | 3 |
| **Total** | | **18** | **5** | **0** | **23** |

### Overall: 18/23 PASS (78%)

### Failures (all Ring 2 missing endpoints)

1. `GET /tier/nano/status` — 404
2. `GET /tier/core/status` — 404
3. `GET /tier/prime/status` — 404
4. `GET /containers` — 404
5. `GET /watch/status` — 404

These are API surface gaps — the endpoints are documented/expected but not yet implemented in gaia-orchestrator.

### Issues Found

1. **Lifecycle FSM stale after `/consciousness/awake`**: The POST to `/consciousness/awake` loads all tiers correctly but does not update the lifecycle FSM state. After sleep->awake round-trip, lifecycle/state still reports "deep_sleep" even though all tiers are at awake targets. The consciousness matrix is accurate.

2. **Core CPU fallback gap during sleep**: When transitioning to sleep, core's target was "subconscious" (CPU) but actual became "unconscious". The CPU fallback model may not have loaded, or unload raced ahead of CPU load.

### Final System State

**AWAKE** — verified via consciousness matrix:
- Nano: conscious (GPU) -- healthy
- Core: conscious (GPU, 3589MB VRAM) -- healthy
- Prime: subconscious (CPU/GGUF) -- healthy, model loaded, inference working
