# gaia-audio — The Ears & Mouth

**Port:** 8080 | **GPU:** Yes (shared) | **Dependencies:** Whisper, Coqui XTTS v2

gaia-audio provides GAIA with auditory input (STT), text refinement (Nano-Refiner), and vocal output (TTS).

## Design Principles

- **Full-duplex GPU**: Simultaneous STT and TTS without blocking
- **CPU-based Nano-Refiner**: 0.5B model for blazing-fast transcript cleanup without VRAM impact
- **Fallback chains**: Coqui TTS -> espeak-ng (local) -> ElevenLabs (cloud)

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
