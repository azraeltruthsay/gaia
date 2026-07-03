# gaia-audio — The Ears & Mouth

**Port:** 8080 | **GPU:** Yes (shared) | **Dependencies:** Qwen3-ASR-0.6B (STT), Qwen3-TTS (0.6B CPU / 1.7B GPU)

gaia-audio provides GAIA with auditory input (STT), text refinement (Nano-Refiner), and vocal output (TTS).

## Design Principles

- **Full-duplex GPU**: Simultaneous STT and TTS without blocking
- **Three-tier voices**: Listener (Qwen3-ASR 0.6B, GPU), Nano Speaker (Qwen3-TTS 0.6B, CPU, instant short phrases), Prime Speaker (Qwen3-TTS 1.7B, GPU, on-demand long-form)
- **Fallback chain**: Qwen3-TTS -> espeak-ng (emergency local fallback)
- **Nano-Refiner**: transcript cleanup delegated over HTTP to the `gaia-nano` endpoint — since the Nano tier's deprecation this transparently reaches Core's embedded engine (`gaia-core:8092`)

## Key Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/transcribe` | POST | Speech-to-Text (accepts m4a, wav, mp3) |
| `/refine` | POST | Nano text cleanup and diarization |
| `/synthesize` | POST | Text-to-Speech |
| `/gpu/release` | POST | Unload GPU models (VRAM release) |
| `/gpu/reclaim` | POST | Reload GPU models in background |
| `/sleep` | POST | Deep sleep (mute + GPU unload) |
| `/wake` | POST | Wake from sleep (unmute + GPU reload) |

## GPU Lifecycle

gaia-audio shares the GPU with gaia-prime. The orchestrator coordinates handoffs:

1. `/gpu/release` — unloads models, frees VRAM (STT lazy-reloads on next request)
2. `/sleep` — full mute + unload (used during GAIA sleep cycle)
3. `/wake` — background reload, returns immediately
4. `/gpu/reclaim` — explicit reload after a `/gpu/release`
