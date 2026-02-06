# GAIA Orchestrator - Development Journal

## 2026-02-06: Orchestrator Complete, Candidate Testing Issues

### Session Summary
Recovered from two context losses and successfully implemented the GAIA Orchestrator service. The orchestrator is now running and functional.

### What Was Built

**New Service: gaia-orchestrator (Port 6410)**
- Central coordinator for GPU resources and container lifecycle
- Manages handoffs between Core (Prime) and Study for training
- Tracks GPU ownership with lease-based system
- Broadcasts Oracle fallback notifications when API models are used

**Files Created:**
```
gaia-orchestrator/
├── Dockerfile
├── IMPLEMENTATION_PLAN.md
├── requirements.txt
├── pyproject.toml
├── config/orchestrator.yaml
└── gaia_orchestrator/
    ├── __init__.py
    ├── main.py
    ├── config.py
    ├── state.py
    ├── docker_manager.py
    ├── gpu_manager.py
    ├── handoff_manager.py
    ├── notification_manager.py
    └── models/schemas.py
```

**CLI Commands Added:**
```bash
./gaia.sh orchestrator [build|start|stop|status|logs]
./gaia.sh gpu [status|release]
./gaia.sh handoff [prime-to-study|study-to-prime|status]
```

**Integration Points:**
- Added to `docker-compose.yml`
- Oracle fallback hook in `_model_pool_impl.py`
- Status appears in `./gaia.sh status`

### Verified Working
- Health endpoint: ✓
- GPU acquire/release: ✓
- Oracle fallback notifications: ✓
- Service status in gaia.sh: ✓

---

## Current Issue: Live Core Repetition/Hallucination

### Root Causes Found (from logs)

1. **Empty Assistant Responses in History**
   - Many assistant turns in conversation history are empty
   - Shows as `<|im_start|>assistant\n<|im_end|>` in prompt
   - Pollutes context and confuses the model

2. **Thinking Tags Leaking into Summaries**
   - History summaries contain raw `<|start_thinking|>` tags
   - Example: `summary='<|start_thinking|>\n<|end_thinking|>\n\n<|start_thinking|>\nThe user is testing RAG capabilities by aski'`
   - Summaries are truncated mid-sentence

3. **Config Error**
   - `'Config' object has no attribute 'HISTORY_DIR'`
   - Archiver is failing, may affect session management

4. **Runaway Generation**
   - Last response was 10,962 characters
   - Indicates possible repetition loop

### Fixes Needed

1. **Session History Cleanup**
   - Filter out empty assistant responses before building prompt
   - Or prevent empty responses from being saved to history

2. **Summary Sanitization**
   - Strip `<|start_thinking|>` and `<|end_thinking|>` tags from summaries
   - Ensure summaries aren't truncated mid-word

3. **Config Fix**
   - Add `HISTORY_DIR` to Config class

4. **Repetition Prevention**
   - Check vLLM `repetition_penalty` setting
   - Consider adding stop sequences for repetitive patterns

### Files to Investigate
- `gaia-core/gaia_core/memory/session_manager.py` - history management
- `gaia-core/gaia_core/cognition/prompt_builder.py` - prompt assembly
- `gaia-core/gaia_core/config.py` - add HISTORY_DIR ✅ FIXED
- `gaia-core/gaia_core/models/vllm_model.py` - check repetition_penalty
- `gaia-core/gaia_core/utils/output_router.py` - `_strip_think_tags_robust` needs to handle `<|start_thinking|>` format

### Key Finding
The `_strip_think_tags_robust()` function in `output_router.py` handles `<think>` and `<thinking>` tags but NOT the `<|start_thinking|>` and `<|end_thinking|>` format that the model is actually producing. This causes thinking content to leak into:
1. Session history
2. History summaries
3. Possibly user-facing output

### Fixes Applied

1. **HISTORY_DIR Config** ✅
   - Added `HISTORY_DIR: str = "/logs/history"` to `gaia-core/gaia_core/config.py`

2. **Qwen-style Thinking Tags** ✅
   - Added handling for `<|start_thinking|>` and `<|end_thinking|>` in `_strip_think_tags_robust()`
   - File: `gaia-core/gaia_core/utils/output_router.py`

### Restart Required
After these fixes, gaia-core needs to be rebuilt and restarted:
```bash
docker compose build gaia-core
docker compose up -d gaia-core
```

---

## Remaining Work

### Phase 7: Bicameral Mind Support
- [ ] Enable two-model loading in candidate
- [ ] GPU memory validation for dual models
- [ ] Test simultaneous Prime + Observer inference

### Future Enhancements
- [ ] WebSocket integration with gaia-web for real-time notifications
- [ ] Prometheus metrics export
- [ ] Automatic GPU reclaim on container crash
- [ ] Priority queue for GPU access
