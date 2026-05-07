# shj — gaia-audio STT Deprecation Decision

**Date**: 2026-05-07
**Issue**: `GAIA_Project-shj` — Phase 6: Deprecation decision for gaia-audio STT
**Decision**: **fallback** (not `keep`, not `remove`)

## Context

By the time this decision was needed, the surrounding work had landed:

- **8ki** (vision web UI integration) shipped `aae8ac2` / `e8018fb`
- **aof** (audio web integration plumbing) shipped `3769b63`
- **7rq** (audio side training) shipped `9fd4f49` + gaia-engine `9d43d40`. V6AUDIO trained, native audio inference path through `audio_tower` works end-to-end.
- **m0b** (V7 audio quality) filed P3 — V6AUDIO scored 1/5 on the cognitive battery's audio section; quality bound by training breadth (960 samples × ~3.5 epochs).

The architecture is no longer ambiguous:

- Gemma 4 multimodal Core is **audio-in only**. The model has `audio_tower` + `embed_audio` (encoders that turn mel features into soft tokens for the LM) but produces only text via `lm_head`. There is no audio decoder, no vocoder, no waveform output anywhere in the architecture.
- gaia-audio's TTS (`tts_engine.py`) is the only path to spoken output. It stays. Period.
- The only thing in question is gaia-audio's STT (Qwen3-ASR) — is it redundant now that Core can hear?

## The three options

1. **Keep**: STT remains the primary path for speech-bearing turns. Core's audio side stays text-mode only.
2. **Fallback**: Core multimodal is the primary path; STT is a configurable fallback that activates when operators flip a flag.
3. **Remove**: STT goes away. All speech goes through Core's audio_tower.

## Why fallback (not keep, not remove)

**Not "remove"**: V6AUDIO's audio battery score is 1/5 (20%). The acceptance target in 48m is 70%. Until V7 audio (m0b) hits the bar, removing STT removes the only production-quality speech transcription option. Removing it now would regress voice UX every time Core confabulates ("speech-shaped sustained pitch followed by a lower-pitched trailing word" — V6AUDIO's response to a 440 Hz tone).

**Not "keep"**: Keeping STT as the default permanent path forecloses the design vision behind 7rq + the engine audio inference path. Once V7 lands, the natural place for paralinguistic information (urgency, hesitation, prosody, accent, emotion) is the same audio embeddings the LM already attends to. STT throws all of that away — by definition it produces only the words. Keeping STT primary leaves Core's audio side as a permanent science project.

**Why "fallback"**: it composes. Core multimodal stays the primary path, accumulating the architectural benefits as training improves. STT is one env var away when needed — for testing, for production-quality speech transcription on un-trained content (real voices speaking real sentences, which V6's synthetic curriculum doesn't cover), or as graceful degradation if Core regresses on a deploy.

## Implementation

`CORE_AUDIO_NATIVE` env var on gaia-core, default `"1"`:

| Value | Behavior |
|---|---|
| `"1"` (default) | `audio_payloads` route through Core's multimodal path: NeuralRouter._triage_audio → TargetEngine.CORE → engine `_prepare_audio_inputs` → `audio_tower` → `embed_audio` → LM |
| `"0"` (or unset to anything else) | Pre-process via gaia-audio: POST `/transcribe` with the base64 audio, replace `packet.content.original_prompt` with the returned transcript, clear `audio_payloads` so Core sees text-only |

Wired in `gaia-core/main.py` at the same spot where `audio_payloads` get extracted to `metadata["_audio_payloads"]` for NeuralRouter — before `run_turn()`. Falls back gracefully on any STT error (network, gaia-audio down, empty transcript) by leaving the audio for the Core path; the operator's intent is documented in logs either way.

`AUDIO_ENDPOINT` env var (default `http://gaia-audio:8080`) controls where transcribe requests go. Already the path used by other gaia-audio integrations.

## Operational guidance

- **Today (V6AUDIO not deployed; /models/core → V2)**: V2 has no audio training at all. Speech turns confabulate on the Core path. Operators who want useful voice responses today should set `CORE_AUDIO_NATIVE=0`.
- **After m0b (V7 audio battery ≥70%)**: The default `"1"` becomes the right primary. STT remains as a manual fallback for operators experimenting or if Core regresses.
- **Discord voice path**: gaia-audio already handles Discord voice via voice_manager.py independently. Not affected by `CORE_AUDIO_NATIVE` — this flag is only about the `/process_packet` audio_payloads route from web mic / Discord file attachments.

## What stays in gaia-audio regardless

- `/synthesize` (TTS) — the only spoken-output path. Used by Discord voice replies, dashboard speak-aloud, etc.
- `/voices`, `/transcribe` (still used when `CORE_AUDIO_NATIVE=0`), `/analyze`, `/refine` — operational features beyond the deprecation question.
- Voice listener panel in the dashboard.

## Closing notes

The "deprecate" framing in the original 8oz plan was premature. With Gemma 4's audio-input-only architecture confirmed, gaia-audio's TTS is non-negotiable, and STT is a useful tool that sits behind a flag instead of a flag-day decision. The decision shape that actually composes with future quality work is "fallback".
