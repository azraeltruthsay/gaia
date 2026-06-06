# 2026-06-05/06 — m0b: clean-feature re-train (vision→94%) + the audio-training finding

**Follow-on to the `5fh` vision fix.** With the NF4 tower-corruption bug fixed (engine + trainer dequant of vision/audio towers + projectors), we did two full clean-feature re-trains. Headline: **clean-feature training lifted vision to 94%**, but **audio training consistently *hurts* the synthetic battery** — V14's light-touch remains the audio ceiling.

## Production now
- **`/models/core` → `CORE2X_V15_FULL`** (vision 94% / audio 30% / text 88%). V14 retained as deeper rollback. GAIA parked.

## The three contenders
| model | recipe | vision | audio | text |
|---|---|---|---|---|
| **V14** | LM-LoRA r16, captions, **light audio** (320), clean inference | 75 | **60** | **100** |
| **v15_full** | clean-feature re-train, vision 7k + **audio 7.2k (ESC-50-heavy v7 ×3)** | **94** | 30 | 88 |
| **v16_rebal** | clean-feature re-train, audio 6.1k **synth-dominant (91%)** | 88 | 35 | 88 |

## Findings
1. **Clean-feature training is a real vision win.** Training the LM on *dequantized* vision-tower features (vs the corrupted NF4 features V14 trained on) lifted vision 75→94%. Confirms the `5fh` story end-to-end: the towers must be bf16 at **both** train and inference.
2. **Audio training hurts the battery.** Both heavy audio runs *underperformed* V14's light-touch (60%): v15_full's ESC-50-heavy curriculum made it label everything as real-world sounds (30%); v16_rebal's synth-dominant curriculum overfit to "sine tone" and lost sweep-direction/count/chord discrimination (35%), plus `<tool_call>` token leakage (instability). The base Gemma 4 audio pathway + clean features (the `embed_audio` dequant) is already ~60% and our training **distorts** it.
3. **Loss ≠ eval, again.** v16 audio loss hit 0.32 (model learned the synth *training* answers) but battery stayed 35% — it didn't generalize to the battery's test audio.
4. **Both runs completed clean** (0 watchdog crashes) thanks to the resilient wrapper + freeing ~600 MB by stopping gaia-prime/gaia-audio during the maintenance window. VRAM held through vision+audio phases via paged_adamw_8bit.

## The ideal recipe (for next session)
**Heavy vision + minimal audio, all clean features.** Take v15_full's vision recipe (which gives 94%) but keep audio to a *light touch* like V14 (don't over-train it). Expectation: vision ~94 + audio ~60 + text high — best of all three. The current trainer + dequant fixes support this directly; just dial the audio curriculum down to ~few hundred pairs.

## Open question
**Audio battery ≥70% is not reachable by training** in what we've tried (V14's 60% is the ceiling). Either (a) the synthetic-primitive keyword battery is too brittle (real-audio tests, à la 4d3 for vision, may measure the towers more fairly), or (b) the audio_tower→LM bridge needs a fundamentally different approach. Reassess before more audio runs.

## State at close
- Production v15_full, V14 rollback. GAIA parked + maintenance. All commits/beads pushed. Two NF4 dequant fixes live (engine 8f4ac57/d47ce47, trainer 7d76b92). Adapters retained: v14, v15_full, v16_rebal, pilots.
