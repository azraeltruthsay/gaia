# Voice Conversation Dataflow (Discord) — Design & Validation

> Status: **design / pre-deploy validation** (auth fix + image rebuild deferred — see bd `a1t`).
> Purpose: map how GAIA handles a live human voice conversation, lock the design
> decisions, and define what we can validate **before** wiring Discord.

## 1. End-to-end dataflow (as built in `gaia-web/gaia_web/voice_manager.py`)

```
Whitelisted user joins voice channel
   │  discord_interface.on_voice_state_update → VoiceManager.handle_voice_state_update → _join_channel
   ▼
[CONNECT]  channel.connect()
   │  · lifecycle AWAKE → LISTENING  (orchestrator /lifecycle/transition trigger=voice_join)
   │  · proactive POST audio /wake   (load STT before first utterance)
   │  · GaiaVoiceSink → asyncio.Queue(maxsize=500, ≈10s)
   │  · vc.start_recording(sink) ; create_task(_process_audio_loop)
   ▼
[CAPTURE]  py-cord receive thread → sink.write(data, user)  [48kHz stereo PCM, per-user id]
   │        (sink can filter to target_user_id; today None = all users mixed)
   ▼
[VAD LOOP] _process_audio_loop:
   │  48kHz stereo ──(numpy avg + 3× decimate)──► 16kHz mono → 20ms frames (640 bytes)
   │  SimpleVAD.feed_frame: webrtcvad mode 2 (energy fallback rms>300)
   │  end-of-utterance = 800ms silence after ≥300ms speech (30s hard cap)
   ▼  utterance bytes  → create_task(_process_utterance)
   │
[TURN]  _process_utterance  (serialized by self._processing_lock — one turn at a time)
   │  ① STT     → POST audio /transcribe → text   (drop if <2 chars)
   │  ② STATE   → GET core /sleep/status; if not "active":
   │              → VOICE_LITE stalling packet ("waking up, warm 1-2 sentence ack")
   │              → fire core /sleep/wake in background
   │  ③ REFLEX  → _get_nano_response (Nano DEPRECATED → returns None)
   │  ④ THINK   → build_packet(VOICE_PRIME) → POST core /process_packet (NDJSON stream)
   │              → _parse_ndjson_response → _strip_response_header
   │  ⑤ SPEAK   → POST audio /synthesize → audio_base64 → Discord FFmpegPCMAudio play
   │  ⑥ DRAIN   → discard audio captured during playback  ◄── (the half-duplex choice)
   ▼
[LISTEN]  state → "listening" → next utterance
   ...
User leaves (no whitelisted remain) → disconnect → lifecycle LISTENING → AWAKE
```

State machine: `disconnected → listening → transcribing → responding → speaking → listening`.

## 2. Design decisions baked in (sign these off)

| Decision | Current behavior | Note |
|---|---|---|
| **Turn model** | Half-duplex, `_processing_lock` serializes turns | One full turn at a time |
| **Turn boundary** | 800ms silence (VAD) | Tunable; trades cut-offs vs lag |
| **Barge-in** | **Not implemented** — audio during TTS is *drained* | See §3 — cheap to add |
| **Echo handling** | Drain captured audio after speaking | Largely moot (see §3) |
| **Latency hiding** | Stall-then-think (LITE ack while Core wakes) | Nano reflex is dead code |
| **Speaker handling** | Single mixed queue (`target_user_id=None`) | Set target id → 1-on-1 clean |
| **Routing** | `VOICE_PRIME` packet source | Confirm which engine it targets |

## 3. Full-duplex assessment (2026-06-09)

**Barge-in / interruptible voice — achievable now, *easier* in Discord than on open mics:**
- **Discord does not echo the bot's own transmitted audio** back into the receive stream. `sink.write(data, user)` only ever receives *other* users' audio. → **No acoustic echo to cancel** (the #1 open-mic full-duplex problem doesn't exist here).
- The sink **already listens continuously** (the `paused` flag is never set), including during TTS — it just discards that audio in the DRAIN step.
- py-cord supports `vc.stop()` / `vc.is_playing()` to halt playback mid-stream.
- **Barge-in = a logic change, not new infra:** during the "speaking" state, keep running VAD instead of draining; on detected speech, `vc.stop()`, cancel the in-flight turn, process the interruption. Filter the sink to `target_user_id` so only the intended speaker can interrupt.

**True streaming full-duplex (real-time overlapping understanding) — NOT now.**
- Blocker: **Qwen3-ASR is a batch/utterance ASR**, not streaming — it transcribes a completed segment, no live partial hypotheses. Continuous full-duplex needs streaming ASR + incremental cognition.

**Roadmap:** v1 half-duplex (built) → **v1.5 barge-in** (small, no rebuild blocker) → v2 streaming full-duplex (streaming-ASR swap).

## 4. Open design questions

- **GPU juggling mid-call.** voice → LISTENING; a deep question wants Prime (FOCUSING = Core→CPU). STT + TTS + Core/Prime contending on one GPU during a live call is the real risk — model the gear transitions and where STT/TTS live.
- **Latency budget.** STT + think + TTS must feel conversational (~<2–3s/turn). TTS (Qwen3-TTS) is the likely long pole — measure it.
- **Turn-taking tuning.** 800ms silence: cuts thoughtful pauses vs feels laggy.
- **Multi-speaker.** Group channels mix into one stream; decide target-user lock vs diarization.

## 5. Validation plan — doable NOW (no auth, no rebuild)

Every stage is independently testable; the brain can be proven on a file before Discord exists.

| # | Validate | How (no auth/rebuild) | Tool |
|---|---|---|---|
| 1 | STT accuracy + latency | recorded WAV → in-process `STTEngine.transcribe_sync` | harness |
| 2 | Cognition quality/routing/latency | `VOICE_PRIME` packet → core `/process_packet` | harness |
| 3 | TTS quality + latency | text → in-process `TTSEngine.synthesize_sync` → WAV | harness |
| 4 | VAD segmentation | synthetic silence/speech frames → `SimpleVAD` | unit test |
| 5 | Downsample | known PCM → `pcm_48k_stereo_to_16k_mono_fast` | unit test |
| 6 | **Full offline dry-run** | utterance WAV → STT → think → TTS → reply WAV + per-stage timings | `scripts/voice_pipeline_dryrun.py` |
| 7 | Latency budget | sum per-stage timings → conversational? | dry-run output |

**Order:** build #6 first (proves the loop + surfaces the latency budget), then tune VAD (#4) and decide barge-in (§3) — all before the auth fix + rebuild in `a1t`.

## 6. Remaining to ship (bd `a1t`)
1. Sign gaia-audio `/transcribe` + `/synthesize` (HMAC) — verify key parity, add `gaia_service_key` mount to gaia-web.
2. Rebuild gaia-web image (picks up the already-committed libopus load).
3. Un-park for STT/TTS GPU; whitelist user; live `!voice_call`.
