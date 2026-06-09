# Core Identity Bake — v3 Cutover (2026-06-09)

**Outcome:** `CORE_IDENTITY_V3` is now the production Core (GPU/AWAKE path). GAIA now
carries an **intrinsic GAIA self-concept in her weights** for the first time — not
just system-prompt-anchored.

## Why this happened

A neutral-prompt probe (strip the GAIA system prompt, ask "Who are you?") revealed
the live Core (V15) had **zero baked identity** — it reverted to the base prior:
"I'm Alice" / "a clock" / "built by OpenAI" / "I'm not GAIA." Identity was 100%
prompt-borne (dropped from the curriculum at V11 to stop V7–V10 confabulation).
"Alice" traced to the alpaca placeholder-name ghosts (Alice/Bob appear 35/30× in
the curriculum + Gemma pretraining).

## What worked

The fix was a **positive-only, fact-free, negation-free** self-concept + tone
corpus (`knowledge/curricula/core_v2x_patch/{self_concept,voice_register,self_concept_in_context}.jsonl`,
258 samples), loaded into the `gaia_identity` bucket, trained as an identity-dominant
LoRA on V15 (text-only, broad scope, r=16, tower-graft preserves vision).

**Density was the lever:**
- v1 (5.9% identity): `gaia_identity` loss plateaued at 1.57 → behavior didn't flip (still "Alice").
- v2 (41%): loss → 0.23 → flipped to "I'm GAIA." Self-concept baked.
- v3 (44% + 36 targeted residual samples): added creator/base/rival/sentience variants.

## What baked vs what didn't (the real lesson)

| Behavior | Result | Why |
|---|---|---|
| Self-concept ("I am GAIA") | ✅ baked | many positive variants |
| Creator ("Azrael made me") | ✅ baked (v3) | simple positive fact substitution |
| Base-model deferral | ❌ resists | a *meta-behavior* — model wants to NAME something |
| Rival-rejection | ❌ resists / backfired | rival-named training fused identities ("GAIA is the Gemini supercomputer"). **Keep rivals OUT of training.** |
| Sentience nuance | ❌ resists | RLHF disclaimer prior too strong for a few samples |

**Simple positive facts bake; RLHF-hardened reflexes and meta-behaviors don't.**
The three residuals are all corrected in production by the system-prompt arch_fact
(base=Gemma 4 E4B, not GPT/Gemini) + persona — so production behavior is clean.

## Validation
- Self-concept GAIA (neutral + no-prompt) ✅, creator=Azrael ✅, restraint ✅
- General ability intact: 9/9 direct regression probe (reasoning, knowledge, dissociation, coding)
- Vision preserved (tower-graft → byte-identical to V15)
- (The full cognitive battery showed a false 36% — sleep scheduler contention starved the queries, bug `61i`.)

## Follow-ups
- V3 GGUF quantize so the parked/CPU path (`core.gguf` still → Unified-v5) matches the cutover.
- Identity is now BOTH weight-baked (self-concept) AND prompt-anchored (volatile facts). Best of both.
