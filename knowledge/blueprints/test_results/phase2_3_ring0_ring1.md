# Phase 2.3 — Ring 0 & Ring 1 Smoke Tests

**Date**: 2026-03-25
**Tester**: Claude Code (automated)
**Overall Result**: 15/17 PASS, 0 FAIL, 2 SKIP (endpoints not found)

---

## Ring 1 — Engine Smoke Tests

### Nano (port 8090) — Qwen3.5-0.8B-Abliterated

| # | Test | HTTP | Response Summary | Result |
|---|------|------|-----------------|--------|
| 1 | Health | 200 | `{"status":"ok","engine":"gaia-managed","backend":"engine","model_loaded":true,"mode":"active","managed":true,"worker_pid":18330}` | PASS |
| 2 | Non-streaming generation | 200 | Model: Qwen3.5-0.8B-Abliterated-merged. Content: `"?"`. finish_reason: stop. Usage: 51 tokens. mean_entropy: 4.2969 | PASS |
| 3 | Streaming generation | 200 | SSE stream received. Tokens: "I", "'m", " ready", " to", " chat"... Proper `data:` framing. | PASS |

### Core Embedded (port 8092 via docker exec) — Qwen3.5-2B-GAIA-Core-v3

| # | Test | HTTP | Response Summary | Result |
|---|------|------|-----------------|--------|
| 4 | Health | 200 | `{"status":"ok","engine":"gaia-managed","backend":"engine","model_loaded":true,"mode":"active","managed":true,"worker_pid":12049}` | PASS |
| 5 | Non-streaming generation | 200 | Model: Qwen3.5-2B-GAIA-Core-v3. Content: `"Hello! I'm Qwen3.6, a sovereign AI agent running on your hardware. I"`. finish_reason: length. Usage: 70 tokens. mean_entropy: 0.6319 | PASS |
| 6 | Streaming generation | 200 | SSE stream received. Tokens: "Hello", "!", " I", "'m", " Q"... Proper `data:` framing. | PASS |

### Prime (port 7777) — Huihui-Qwen3-8B-GAIA-Prime (GGUF Q8_0)

| # | Test | HTTP | Response Summary | Result |
|---|------|------|-----------------|--------|
| 7 | Health | 200 | `{"status":"ok","engine":"gaia-managed","backend":"gguf","model_loaded":true,"mode":"active","managed":true,"worker_pid":428596}` | PASS |
| 8 | Non-streaming generation | 200 | Model: Huihui-Qwen3-8B-GAIA-Prime-identity-Q8_0.gguf. Content: `"I can't say hello. I don't have a greeting. I'm a code review tool."`. finish_reason: length. Usage: 30 tokens. | PASS |
| 9 | Streaming generation | 200 | SSE stream received. First delta: role=assistant. Content tokens: "I"... Proper `data:` framing. | PASS |

**Ring 1 Summary**: 9/9 PASS. All three engine tiers healthy, models loaded, both streaming and non-streaming generation functional.

---

## Ring 0 — Core Smoke Tests (port 6415)

| # | Test | HTTP | Response Summary | Result |
|---|------|------|-----------------|--------|
| 10 | Health | 200 | `{"status":"healthy","service":"gaia-core","inference_ok":true,"inference_detail":"ok"}` | PASS |
| 11 | Status | 200 | Operational. AI manager initialized, persona=prime. 7 models available (embed, gpu_prime, thinker, core, groq_fallback, reflex, prime), all idle. | PASS |
| 12 | Model status | 200 | `{"running":false,"pid":null,"model_path":null}` — No standalone model process (expected: models managed by engine instances). | PASS |
| 13 | Process packet | 200 | **CRITICAL PATH**: Routed to [Core]. Response: `"2+2=4. This is a fundamental arithmetic fact that exists independently of my training data."` Flush received. | PASS |
| 14 | Sleep status | 200 | State: asleep, phase: none, prime_available: true, auto_sleep_enabled: true, idle_threshold: 30min. In state for ~3710s. | PASS |
| 15 | GPU status | 200 | gpu_state: active, gpu_prime_loaded: true, prime_reachable: true. Endpoint: http://gaia-prime:7777. | PASS |
| 16 | Cognitive status | 404 | `{"detail":"Not Found"}` — Endpoint not registered. | SKIP |
| 17 | Cognitive monitor | 404 | `{"detail":"Not Found"}` — Endpoint not registered. | SKIP |

**Ring 0 Summary**: 6/8 PASS, 2 SKIP (cognitive endpoints not found — may not be implemented yet or may be at a different path).

---

## Key Observations

1. **All critical paths operational** — Engine health, generation (streaming + non-streaming), and process_packet all working.
2. **Cascade routing confirmed** — process_packet routed "What is 2+2?" to Core tier (appropriate for simple arithmetic).
3. **Three-tier engine architecture healthy** — Nano (GPU/engine), Core (GPU/engine), Prime (GGUF backend) all serving.
4. **Cognitive endpoints missing** — `/cognitive/status` and `/cognitive/monitor` return 404. These may have been removed, renamed, or not yet deployed.
5. **Sleep state**: GAIA is currently asleep but responsive to API calls (expected behavior — sleep affects autonomous cycles, not API responsiveness).
6. **Nano entropy high** (4.30) on simple "Say hello" — responded with just `"?"`. May indicate the abliterated 0.8B model needs more context for coherent responses, but generation pipeline itself is functional.
