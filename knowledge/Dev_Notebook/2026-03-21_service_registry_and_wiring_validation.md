# Dev Journal: Service Registry, Wiring Validation & Live Path Interconnectivity

**Date:** 2026-03-21
**Session:** Extended session spanning March 20-21
**Era:** Sovereign Autonomy — Structural Visibility

---

## Executive Summary

Built a complete automated service registry and wiring validation system. Starting from the plan to fill 3 missing blueprints, the work expanded into a full structural automation layer: blueprint auto-discovery from live OpenAPI schemas, compiled JSON registry consumed by doctor and dashboard, wiring validation (do outbound calls have matching inbound endpoints?), and live path interconnectivity validation (does the target service *actually* expose the path the caller expects at runtime?). The system immediately justified itself by discovering two real integration bugs that had been silently failing in production.

---

## 1. Blueprint Completion (12/12 Services)

### New Blueprints Created
- **gaia-doctor.yaml** — 35 interfaces (all GET/POST endpoints from the stdlib HTTP handler), 4 outbound calls (orchestrator, core, monkey health probes)
- **gaia-nano.yaml** — 4 interfaces (health, chat completions, completions, slots). Port corrected: internal 8080, not host-mapped 8090
- **dozzle.yaml** — 1 interface (web UI). Infrastructure-only, no GAIA application code

### gaia-core.yaml Updated
Added ~25 missing inbound endpoints discovered by comparing the blueprint against actual code:
- Sleep lifecycle: `/sleep/toggle`, `/sleep/hold`, `/sleep/hold-release`, `/sleep/force`, `/sleep/config`, `/sleep/voice-state`, `/sleep/wake-config`, `/sleep/wake-toggle`, `/sleep/wake-activity`
- Model management: `/model/release`, `/model/reload`, `/model/status`, `/model/adapters/notify`
- Repair/diagnostics: `/api/repair/structural`, `/api/doctor/diagnose`, `/api/doctor/review`
- KV cache: `/api/kv-cache/save`, `/api/kv-cache/restore/{role}`, `/api/kv-cache/pressure`, `/api/kv-cache/compact/{role}`
- Cognitive: `/api/cognitive/similarity`, `/api/cognitive/query`
- Audio: `/audio/ingest`, `/audio/listen`, `/audio/context`
- Added outbound calls to gaia-nano and gaia-audio (missing from blueprint)

### Pre-existing Blueprint Fixes
- **gaia-audio.yaml**: Missing `path` on elevenlabs outbound transport, invalid `generated_by: gemini_cli`
- **gaia-monkey.yaml**: String `security` field (should be null), invalid `subprocess` transport type (removed — docker/promptfoo are subprocess calls, not HTTP), invalid severity enums (`critical`/`minor` → `fatal`/`degraded`), invalid `generated_by: manual` → `manual_seed`

---

## 2. Compiled Service Registry

### scripts/compile_registry.py
Loads all 12 live blueprints via `blueprint_io.load_all_live_blueprints()`, derives graph topology (edges from interface matching), validates wiring, writes stdlib-readable JSON to `/shared/registry/service_registry.json`.

**Output:** 12 services, 73 edges, wiring validation baked in.

### Doctor Integration
`_load_service_registry()` reads the compiled JSON at startup to build its `SERVICES` dict (health URLs + remediation strategies). Falls back to hardcoded dict if file missing. Eliminated the redundant `_REMEDIATION_MAP` — remediation is read from the existing `_HARDCODED_SERVICES` tuple.

Doctor also runs `_check_wiring_health()` every 5 minutes, caching results at the new `/registry` GET endpoint. Thread safety fix: `_wiring_last_check` timestamp is updated *after* the check completes, not before.

### Dashboard Widget
"Service Registry" card in the System State panel shows services covered, edges, orphaned count. Polls `/api/system/registry/validation` every 10s (proxied through doctor, matching the pattern of all other system.py endpoints).

---

## 3. Wiring Validation

### scripts/validate_wiring.py
Reads the compiled JSON (stdlib only — no pydantic). Reports orphaned outbound (no matching inbound) and uncalled inbound (no outbound points to them). Exit codes: 0=clean, 1=warnings, 2=errors.

### Promotion Pipeline Integration (Stage 3.5)
`compile_registry.py` + `validate_wiring.py` + `validate_paths.py` run between lint (Stage 3) and cognitive smoke tests (Stage 4). Non-blocking (warns, doesn't fail). `--skip-wiring` to bypass.

---

## 4. Blueprint Auto-Discovery

### scripts/discover_blueprint.py
Connects to a running service's `/openapi.json` (FastAPI default), extracts all routes, generates blueprint YAML. Two modes:

- **Discovery**: Auto-generates candidate blueprint for a new service
- **Refresh**: Diffs live endpoints against existing blueprint, reports added/removed. `--update` auto-merges new endpoints.

Ran `--all --refresh` against all services. Results:
- gaia-web: 83 endpoints not in blueprint (massive dashboard growth)
- gaia-orchestrator: 13 missing (watch, training, nano, candidate management)
- gaia-audio: 11 missing (GPU lifecycle, sleep/wake, mute/unmute)
- gaia-study: 4 missing (pipeline, training control)
- gaia-monkey: 1 missing (serenity/record_recovery)
- gaia-doctor and gaia-nano: Skip (no OpenAPI — stdlib HTTP server and llama-server)

All 5 updatable blueprints auto-refreshed with `--update`.

### scripts/refresh_blueprints.py
Single command for the full lifecycle: discover → compile → validate. Guarded sibling imports with try/except.

---

## 5. Live Path Interconnectivity Validation

### scripts/validate_paths.py
Goes beyond wiring (blueprint-level) to verify that every edge actually exists at runtime. For each outbound→inbound edge, fetches the target service's `/openapi.json` and checks that the expected path+method is registered.

### Dashboard Endpoint
`GET /api/system/registry/paths` — live cross-check from inside the Docker network. Returns verified count, mismatched details, unreachable services.

### What It Found
**3 mismatches** (all real bugs):

1. **`gaia-core → gaia-web POST /presence`**: Dead integration. The handler function `change_presence_from_external()` existed in `discord_interface.py` but no FastAPI route was ever created. gaia-core's sleep cycle was sending presence updates that silently 404'd.

2. **`gaia-core → gaia-core POST /model/adapters/notify`**: Dead code. `model_endpoints.py` had a router with prefix `/models` but it was never `include_router`'d into the app. gaia-study was calling `/models/adapters/notify` on every adapter load/unload — silently failing.

3. **`gaia-mcp → gaia-web GET /`** (2x): Bogus blueprint edges. MCP's `web_research` and `discord_webhook` outbound interfaces used `path: /` which matched every service's root endpoint, creating 10 false edges. These were external API calls (DuckDuckGo, Discord webhooks), not GAIA inter-service calls.

**After fixes: 0 mismatches, 55 verified, 7 unreachable (gaia-prime in GPU standby).**

---

## 6. Bug Fixes

### /presence Endpoint Created
`gaia-web/gaia_web/main.py`: New `POST /presence` route that calls `change_presence_from_external(activity, status)`. Handles bot-not-ready gracefully. gaia-core's sleep cycle can now update Discord presence in SOA mode.

### /model/adapters/notify Router Mounted
- `gaia-core/gaia_core/api/model_endpoints.py`: Changed prefix `/models` → `/model`, removed duplicate `/status` endpoint
- `gaia-core/gaia_core/main.py`: Added `app.include_router(model_router)`
- `gaia-study/gaia_study/server.py`: Fixed URL from `/models/adapters/notify` to `/model/adapters/notify`

gaia-study can now notify gaia-core when LoRA adapters are loaded/unloaded, enabling live adapter refresh.

### MCP Blueprint Cleaned
Removed `web_research` and `discord_webhook` outbound interfaces (external APIs, not GAIA services). External API dependencies preserved in `external_apis` section. Edge count dropped from 83 to 73.

---

## 7. Code Quality Pass

Three parallel review agents (reuse, quality, efficiency) audited all new code. 23 findings, 20 fixed:

- **compile_registry.py**: Extracted `_enum_val()` + `_transport_endpoint()` helpers (eliminated 4x repeated pattern). Added logging module. I/O error handling. Full type annotations. Removed unused import.
- **validate_wiring.py**: Removed unused variable. Status dict lookup instead of ternary chain. Defensive `.get()` in all print loops. TOCTOU fix (removed `.exists()` pre-check, catch `FileNotFoundError` directly).
- **discover_blueprint.py**: Fixed state mutation of `refresh` param. Removed dead code. Added try/except around YAML ops. Extracted `_SKIP_METHODS` constant.
- **refresh_blueprints.py**: Try/except guards on sibling imports. Propagates actual exit code.
- **doctor.py**: Consolidated redundant `_REMEDIATION_MAP`. Thread-safe `_wiring_last_check` update. `isinstance` check on loaded registry. Explicit `is not None` in `/registry` endpoint.
- **system.py**: Replaced inline file-reading with doctor proxy (matches all other endpoints). Consistent error response shape.
- **app.js**: Explicit error status handler. `throw` on `!resp.ok`.
- **index.html**: `state-extra` instead of `state-latency` for registry detail.

---

## 8. Wiki & GitHub Pages

### Wiki Updates
- 4 new architecture pages: gaia-doctor, gaia-monkey, gaia-audio, gaia-nano
- New systems page: Service Registry & Wiring Validation
- mkdocs.yml nav updated

### GitHub Pages Deployment
- `.github/workflows/deploy-blog.yml` expanded to build blog + wiki + dev journal
- `scripts/generate_journal_site.py`: Generates static HTML index from dev notebook markdown (63 entries)
- Blog nav updated with Wiki and Journal links

---

## 9. Final State

| Metric | Value |
|--------|-------|
| Blueprints | 12/12 services covered |
| Registry edges | 73 (derived from interface matching) |
| Live path mismatches | 0 |
| Orphaned outbound | 3 (all expected: MCP JSON-RPC, Discord direct_call, ElevenLabs external API) |
| Dead integrations fixed | 2 (/presence, /model/adapters/notify) |
| Bogus edges removed | 10 (MCP's fake `/` outbound) |
| Scripts created | 6 (compile_registry, validate_wiring, validate_paths, discover_blueprint, refresh_blueprints, generate_journal_site) |

### Validation Stack

```
Blueprint YAMLs (12)
    ↓ compile_registry.py
Compiled Registry JSON (/shared/registry/service_registry.json)
    ↓ read by doctor, dashboard
    ↓ validate_wiring.py (offline, CI)
    ↓ validate_paths.py (live, hits /openapi.json)
    ↓ discover_blueprint.py --refresh (drift detection)
    ↓ promotion pipeline Stage 3.5
All verified → 0 mismatches
```

---

## 10. Key Insight

The most valuable thing the system found wasn't in the blueprints — it was the **delta between intent and reality**. The blueprint said `/model/adapters/notify` exists. The OpenAPI said it doesn't. That gap was a real bug that had been silently failing for weeks. The `/presence` endpoint was the same pattern — designed, handler written, route never created.

Blueprint-driven validation catches a class of bugs that unit tests can't: **integration seams where two services believe they're connected but aren't.** The path validator is GAIA's structural MRI — it sees the wiring, not just the components.
