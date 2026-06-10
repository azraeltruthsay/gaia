# Hand-off — Real-Time Streaming Voice: Discord Transport (2026-06-10)

**bd:** `GAIA_Project-a1t` (Discord voice), `GAIA_Project-3kn` (streaming TTS)
**Status:** Voice cognition + synthesis pipeline is **built, wired, authenticated, and verified end-to-end.** Only the **Discord audio transport** remains for a live `!voice_call`.

---

## TL;DR
Everything from microphone-in to speaker-out is done and tested *except the Discord wire*. A live call needs three transport items, one of which (the bot token) only a human can supply.

---

## What's DONE (verified this session)
- **Abliterated Prime** live; **GAIA voice clone** wired (`/shared/voice/gaia_identity.wav` + transcript).
- **Streaming TTS pipeline** — Core `/api/cognitive/stream` → sentence segmentation → Prime GPU TTS per sentence → ordered playback, synth-ahead overlapping playback. **First word ~4 s** (vs ~12 s monolithic). Proven: `scripts/voice_stream_pipeline.py`.
- **Greeting-warmup at join** — GAIA's spoken greeting doubles as the one-time Prime-TTS CUDA-graph warmup. Cold synth is valid audio (not garbage) — `scripts/_warmup_greeting_test.py`.
- **HMAC auth** — `voice_manager` signs `/transcribe` + `/synthesize`. Verified: signed `/synthesize tier=prime` → **200, 2.64 s audio, engine=prime_speaker**; unsigned → 401.
- All wired into `gaia-web/gaia_web/voice_manager.py`:
  - `_process_utterance` → streams conversational turns (`_stream_think_speak`); tool turns keep the full pipeline.
  - `_join_channel` → fires `_greet_and_warm`.
  - `_stream_cognition`, `_synthesize_sentence` (forces `tier=prime`), `_play_audio_segment`, `_signed_headers`.
  - **Bug fixed:** `re` was never module-imported → `_TOOL_TRIGGERS`/`_SENT_RE` `NameError` on import (hybrid voice was silently broken).

## Runtime facts (so you don't rediscover them)
- **Voice runs in the LISTENING gear (heo):** Core GGUF-GPU (~3.7 GB) + STT (1.8) + Prime TTS (4.3) ≈ 9.8 GB. Enter via `POST gaia-orchestrator:6410/lifecycle/transition {"trigger":"voice_join"}`, exit `voice_leave`. `voice_manager._notify_core_voice_state(True)` does this on join.
- **NF4 + Prime TTS do NOT coexist (OOM)** — that's *why* the listening gear drops Core to GGUF.
- **Latency:** cognition NF4 ~2 s / GGUF(listening) ~7 s / CPU ~4 s; STT ~0.5 s; Prime TTS RTF ~0.6–0.8 (realtime); **Nano CPU TTS RTF 6–11 (unusable — always force `tier=prime`)**.

---

## REMAINING — Discord transport (do these to get a live call)

### 1. libopus — load it (small code fix, NOT a rebuild)
- **State:** `libopus.so.0.10.1` IS installed in gaia-web, but `discord.opus.is_loaded()` == **False** (py-cord didn't auto-load it).
- **Fix:** call `discord.opus.load_opus("/usr/lib/x86_64-linux-gnu/libopus.so.0")` at bot startup (guard with `if not discord.opus.is_loaded()`), in `gaia-web/gaia_web/discord_interface.py` (bot init) or `voice_manager`.
- **Verify:** `docker exec gaia-web python3 -c "import discord; print(discord.opus.is_loaded())"` → True.

### 2. Voice whitelist — add a user
- **State:** `/app/data/voice_whitelist.json` → `{"whitelisted": []}` (empty).
- `VoiceWhitelist` (`voice_manager.py:42`) gates `!voice_call`. Add via the command/route in `gaia-web/gaia_web/routes/voice.py` or `whitelist.add(user_id)`.

### 3. Discord bot token — **HUMAN REQUIRED**
- **State:** `DISCORD_BOT_TOKEN` is **EMPTY** (`.env.discord` → `/gaia/gaia-instance/secrets/env.discord`). Without it the bot cannot connect to Discord at all — nothing else is testable until this is set.
- Set a real bot token, recreate gaia-web, confirm the bot comes online in the server.

---

## How to test once transport is up
1. Human joins a Discord voice channel; `!voice_call` (must be whitelisted).
2. **Expect a spoken greeting** ("Hey, it's GAIA. I'm here and listening.") within a few seconds — this also warms Prime TTS.
3. Speak a conversational line → **streamed reply, first word ~4 s**, continuous after.
4. Logs: `gaia-web` for `_stream_think_speak` / greeting; `gaia-audio` for transcribe/synthesize; orchestrator for the `voice_join`/`voice_leave` gear shifts.

## Gotchas / lessons (hard-won — don't repeat)
- `/api/cognitive/query|stream`: **use `no_think=False`** — Gemma-4 Core returns empty/truncated with `no_think=true`.
- **Always force TTS `tier="prime"`** — Nano CPU is 6–11× realtime.
- **Warm Prime TTS at join** — first synth pays ~4–7 s CUDA-graph cost; the greeting hides it. Cold synth audio is valid.
- **Background sleep-cycle hammers Core + Prime** (Prime CPU calls 11–33 s each) → intermittent empty cognition + contention. Consider quieting it during a voice session.
- `423` from gaia-audio = **muted** (lifts via `/wake` in-call), NOT an auth failure. `401` = auth.
- **Gear shifts are slow** (voice_join ~8–10 s, voice_leave ~16–35 s) — dominated by `torch.compile` + NF4 quantize + contention, NOT disk. The warm pool is dead/stale and **wouldn't help** (measured: disk is <10 % of a shift; 33 GB already in page cache).

## Key files
- `gaia-web/gaia_web/voice_manager.py` (+ `candidates/` mirror) — the voice pipeline
- `gaia-web/gaia_web/discord_interface.py`, `routes/voice.py` — bot init + `!voice` command/whitelist
- `gaia-core/gaia_core/main.py` → `POST /api/cognitive/stream` — streaming cognition (+ candidate)
- `scripts/voice_stream_pipeline.py` — streaming proof + first-word latency
- `scripts/voice_conversation_sim.py` — two-voice sim + CPU/GPU latency trials
- `gaia-common/gaia_common/utils/service_auth.py` — HMAC signing/validation
- `.env.discord` → `/gaia/gaia-instance/secrets/env.discord` — `DISCORD_BOT_TOKEN` (EMPTY), `PRIME_MODEL_PATH` (fixed → `/models/prime`)

## After Discord transport — the next phases (the original vision)
- **Phase 1 — barge-in:** keep VAD running during `speaking` (sink already listens), detect user speech, `vc.stop()` + cancel the in-flight turn before/after the current sentence. Feasible now that latency is known.
- **Phase 2 — mid-utterance self-correction:** snapshot the partial reply → classify the interjection (clarify/add/correct/abort) → continuation prompt that incorporates + corrects + resumes. Mirror the `stakes_clarification.py` two-turn stash/resume pattern, but *during* streaming. The big build.
