# 2026-03-16 — Centralized MODEL_REGISTRY

## Problem

Model paths/names were hardcoded as string literals in 20+ locations across Python, Docker Compose, shell scripts, and JSON config. When switching from base models to merged checkpoints (`Qwen3.5-4B-Abliterated` → `Qwen3.5-4B-Abliterated-merged`), several files kept the old default — causing gaia-core-candidate to send the wrong model name to vLLM (404 errors, Prime appeared dead).

## Solution

Added a `MODEL_REGISTRY` block to `gaia_constants.json` as the single source of truth for all model paths. Added `Config.model_path(role, variant)` accessor. All Python consumers now resolve paths through the registry with hardcoded fallbacks only as last resort.

## Files Modified

| File | Change |
|------|--------|
| `gaia_constants.json` | New `MODEL_REGISTRY` block (prime, nano, audio, embedding, lora_adapters) |
| `gaia-common/config.py` | `_model_registry` field, `model_path()` method, `EMBEDDING_MODEL_PATH` from registry |
| `gaia-core/models/vllm_remote_model.py` | `_registry_prime_path()` replaces hardcoded default |
| `gaia-core/models/_model_pool_impl.py` | `self.config.model_path("prime", "merged")` in PRIME_ENDPOINT block |
| `gaia-audio/config.py` | `listener_model_path`, `nano_speaker_model_path`, `prime_speaker_model_path` use registry |
| `gaia-audio/stt_engine.py` | Default model_path from AudioConfig |
| `gaia-audio/tts_engine.py` | NanoSpeaker + PrimeSpeaker defaults from AudioConfig |
| `gaia-study/scripts/self_awareness_pipeline.py` | 5 model path constants via `_registry_path()` |
| `gaia-study/study_mode_manager.py` | `adapter_base_dir` default from registry |
| `gaia-study/server.py` | `base_model_path` + `adapter_dir` from registry helpers |

All synced to `candidates/`.

## Verification

- `Config.model_path("prime", "merged")` → `/models/Qwen3.5-4B-Abliterated-merged` ✓
- `Config.model_path("nano", "gguf")` → `/models/Qwen3.5-0.8B-Abliterated-Q8_0.gguf` ✓
- `Config.model_path("embedding")` → `/models/all-MiniLM-L6-v2` ✓
- 339 gaia-common tests pass ✓
- No 404s in gaia-prime logs ✓

## Design Decisions

- **Fallback chain**: env var → config dict → MODEL_REGISTRY → hardcoded literal. Ensures backward compat if registry is absent.
- **Audio paths**: `AudioConfig` properties chain `audio_cfg` → `model_path("audio", ...)` → hardcoded. Audio-specific overrides in `INTEGRATIONS.audio` still take priority.
- **Intentionally left hardcoded**: Docker Compose `command:` blocks (can't read JSON; `.env` is the bridge), `merge_and_requantize.py` argparse help text, `train_identity_adapters.py` standalone script.
