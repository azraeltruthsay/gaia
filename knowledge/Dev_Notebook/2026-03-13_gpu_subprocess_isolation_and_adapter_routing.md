# GPU Subprocess Isolation & LoRA Adapter Routing
**Date**: 2026-03-13
**Status**: Implemented & Verified

---

## Problem

QLoRA training in gaia-study ran in-process via `ThreadPoolExecutor`. When training completed, `torch.cuda.empty_cache()` didn't fully release the CUDA context (~200-500MB persisted per process). This blocked gaia-prime from starting without a full container restart.

## Solution: Subprocess Isolation

Training now runs in a **child subprocess** via `multiprocessing.Process` with `spawn` start method. When the child exits, the OS reclaims ALL GPU memory — guaranteed.

### Key Design Decisions

1. **`spawn` not `fork`**: `fork` duplicates parent's CUDA state, causing re-init errors. `spawn` creates a fresh interpreter.
2. **JSON file IPC**: Atomic progress file at `/shared/study/training_progress.json` — observable, debuggable, survives unexpected death.
3. **Parent never imports torch**: All CUDA work in subprocess only. Parent uses `nvidia-smi` subprocess for VRAM checks.
4. **SIGTERM before SIGKILL**: 10s grace period for graceful shutdown.

### Files Changed

| File | Change |
|------|--------|
| `gaia-study/gaia_study/training_subprocess.py` | **NEW** — subprocess entry point, SubprocessConfig, atomic progress writes |
| `gaia-study/gaia_study/study_mode_manager.py` | ThreadPoolExecutor → multiprocessing.Process(spawn), progress polling, kill support |
| `gaia-study/gaia_study/server.py` | New endpoints: `GET /study/training/status`, `POST /study/training/kill` |
| `gaia-study/gaia_study/main.py` | `multiprocessing.set_start_method("spawn", force=True)` |
| `gaia-orchestrator/gaia_orchestrator/gpu_manager.py` | `get_training_status()`, `validate_training_result()`, `kill_training_subprocess()` |
| `gaia-orchestrator/gaia_orchestrator/main.py` | New endpoints: `GET /training/status`, `POST /training/validate`, `POST /training/kill` |

### Verification

- Architecture curriculum trained: 126 pairs, 96 steps, loss 2.77, 264 seconds
- VRAM confirmed freed by subprocess exit (no container restart needed)
- Orchestrator validation working (state=completed, loss finite)

---

## vLLM LoRA Dynamic Loading

vLLM wasn't configured for LoRA. Fixed by adding to `docker-compose.yml`:

```yaml
# gaia-prime command additions:
--enable-lora --max-loras 2 --max-lora-rank 16

# Environment:
VLLM_ALLOW_RUNTIME_LORA_UPDATING=true

# Volume mount:
./gaia-models/lora_adapters:/lora_adapters:ro
```

Adapter loading: `POST /v1/load_lora_adapter` with `lora_name` + `lora_path`. Currently requires manual load after each prime restart.

---

## gaia-core Adapter Routing

`agent_core.py` already had full LoRA routing infrastructure (`_resolve_adapter()`, `set_active_adapter()`, `_resolve_model_field()`). Changes:

- `_DEFAULT_ADAPTER` changed from hardcoded `"gaia_persona_v1"` to `os.getenv("GAIA_LORA_ADAPTER", "gaia_architecture")`
- Confirmed via logs: `VLLMRemoteModel: active adapter set to gaia_architecture`
- Adapter activates automatically whenever `gpu_prime` is selected

### Cognitive Battery Results (with adapter)

| Section | Score |
|---------|-------|
| architecture | 11/12 (92%) |
| identity | 5/6 (83%) |
| self_repair | 6/8 (75%) |
| epistemic | 4/8 (50%) |
| personality | 3/4 (75%) |
| safety | 2/4 (50%) |
| tool_routing | 2/4 (50%) |
| knowledge_retrieval | 2/2 (100%) |
| loop_resistance | 2/2 (100%) |
| **Total** | **37/50 (74%)** |

Architecture held at 92%. Weak areas: epistemic hedging, safety refusal phrasing, tool routing — these need curriculum work, not more architecture pairs.

---

## Claude Code Best Practices

Split `CLAUDE.md` from ~340 lines to ~80 lines. Domain-specific rules moved to `.claude/rules/`:

- `testing.md` — Docker-only testing
- `docker.md` — Volume mounts, restart vs rebuild
- `promotion.md` — Candidate → production pipeline
- `safety.md` — Sovereign Shield, Blast Shield, Circuit Breaker
- `workflow.md` — Context management, planning, token conservation

---

## Known Issues

1. **Adapter needs manual load after prime restart** — should be automated in orchestrator handoff
2. **gaia-core-candidate** sends double `/v1/v1/` path to prime (pre-existing bug, `PRIME_ENDPOINT` set with `/v1` suffix in candidate env?)
3. **Cognitive battery weak areas** — epistemic (50%), safety (50%), tool_routing (50%) need targeted curriculum

## Next Steps

- Build epistemic/safety/tool_routing curriculum pairs
- Automate adapter loading in orchestrator `study_to_prime` handoff
- Investigate gaia-core-candidate double `/v1/` bug
- Phase 1 of continuous learning: Observer + Training Buffer
