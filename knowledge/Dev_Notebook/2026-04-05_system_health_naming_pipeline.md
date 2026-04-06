# Dev Journal: System Health, Model Naming, and Recitation Pipeline

**Date**: 2026-04-05 to 2026-04-06
**Scope**: Full system audit, model naming normalization, prompt optimization, recitation pipeline, tool consolidation

## Starting State

Host restarted. 11 services running, 6 stale Created containers. Three active issues:
- Audio STT flapping (30-second wake/sleep loop)
- NVML missing in orchestrator
- Maintenance mode active, all models unloaded

## Fixes Applied

### Infrastructure Fixes

| Fix | Root Cause | Solution |
|-----|-----------|----------|
| Audio STT flapping | `wake()` didn't reset idle timer; doctor force-waking sleeping audio | `touch_activity()` in wake; doctor respects `gpu_mode=sleeping` |
| NVML in orchestrator | No GPU runtime in docker-compose | Added `deploy.resources.reservations.devices` block |
| Engine readiness gate | `/model/load` returned before weights loaded | `_wait_for_health` now checks `model_loaded=true`, not just HTTP 200 |
| Core health startup grace | Doctor restarted Core during model loading | 90-second grace period reports `healthy` during engine startup |
| Doctor restart grace | Doctor restart loop during lifecycle transitions | 90-second post-restart grace suppresses failure counting |
| Failure threshold | 2 failures at 15s = 30s too aggressive | Bumped to 4 (60s window) |
| Core autoload | Engine started in standby, waited for orchestrator | Enabled `GAIA_AUTOLOAD_MODEL=1` in docker-compose |
| Streaming dedup | Tool call parser AND raw pass-through both emitted tokens | Only emit through parser when parser is active |

### Model Naming Normalization

Canonical names established: **nano** / **core** / **prime**

| Old Name | New Name | Tier |
|----------|----------|------|
| reflex | nano | Qwen3.5-0.8B triage |
| lite, operator | core | Qwen3.5-4B operator |
| thinker, gpu_prime, cpu_prime | prime | Qwen3-8B reasoning |

Changed across 20+ files. Role resolution inverted: legacy names map TO canonical names for backward compat. Constants.json updated: Core is 4B (not 2B as previously stated).

### Prompt Optimization

| Metric | Before | After |
|--------|--------|-------|
| System prompt tokens | ~7,500 | ~3,000 |
| max_tokens (recitation) | 8,192 | 1,024 |
| Tool list in prompt | 70 names (~300 tokens) | 13 domains (~150 tokens) |
| E2E latency (Jabberwocky) | 248s | 6s |

Key changes:
- `kv_prefix_active` mode in prompt_builder skips static foundation when KV cache is warmed
- Intent-based token caps prevent 8192-token generation for simple requests
- Web grounding skipped for intents where training data suffices
- Domain tool catalog (`build_prompt_catalog()`) replaces 70-tool dump

### Recitation Pipeline

New direct-stream pipeline for verbatim text recitation:

1. Intent detection catches `recitation`
2. `_fetch_recitation_source()`: web_search (content_type=poem) -> pick best URL (Poetry Foundation priority) -> web_fetch -> clean text
3. Direct stream to user -- bypasses full cognitive pipeline entirely
4. Saves locally to `/knowledge/research/` for future RAG

Result: 6-second, word-perfect Jabberwocky from Poetry Foundation.

### Tool Execution

- `execution_allowed` changed from hardcoded `False` to source-based (`True` for web/discord/voice/api)
- Tool call JSON repair added for common model malformations (`"tool":"name":"web_search"` -> `"tool_name":"web_search"`)
- Task instructions added: `recitation` (verbatim reproduction) and `recitation_validation` (observer fidelity check)

### Tool Consolidation

Domain tools (already implemented in `domain_tools.py`) wired into the cognitive pipeline:
- `available_mcp_tools` now returns 13 domain names instead of 70 legacy names
- World state uses `build_prompt_catalog()` for compact tool descriptions
- Dispatcher routes domain calls to legacy implementations transparently

## Key Discoveries

1. **Doctor was the primary destabilizer** -- 15s poll interval with 2-failure threshold caused restart loops during any transient state (model loading, lifecycle transitions, long inference)
2. **Engine manager reported ready before model loaded** -- `_wait_for_health` only checked HTTP 200, not `model_loaded: true`
3. **Prompt builder and engine KV prefix double-injected** static content (identity, rules, tools) -- ~1,800 tokens of redundancy per request
4. **4B model can't faithfully copy text** -- even with retrieved document in context and temperature=0, it improvises. Direct streaming bypasses this limitation.
5. **Tool call parser and raw pass-through both emitted** -- every token was yielded twice to the client

## Files Modified (Summary)

- `gaia-common`: gaia_constants.json, config.py, init_registry.py, cognition_packet.py, packet_factory.py, world_state.py, tool_call_parser.py
- `gaia-core`: agent_core.py, main.py, _model_pool_impl.py, vllm_remote_model.py, prompt_builder.py, kv_cache_manager.py, idle_heartbeat.py, knowledge_ingestion.py, gaia_direct_response.py, gaia_rescue.py, stream_observer.py
- `gaia-orchestrator`: schemas.py
- `gaia-doctor`: doctor.py
- `gaia-web`: hooks.py, voice_manager.py
- `gaia-engine`: manager.py
- `docker-compose.yml`: NVML, autoload, failure threshold
