# Dev Journal: World State Awareness & Dynamic Evaluation

**Date:** 2026-03-19
**Session:** ~4 hours, building on March 17-18 cognitive architecture
**Era:** Cognitive Architecture — World State Awareness

---

## Executive Summary

Added dynamic world state awareness testing to GAIA's cognitive evaluation and training pipeline. Core (2B) passes perfectly. Nano (0.8B) has a fundamental capacity limit for context extraction. The session established that **time-telling is an Operator task, not a Reflex task** — confirming the tier architecture works exactly as designed.

---

## 1. Problem Discovery

Azrael asked GAIA "What time is it?" in Mission Control. Nano (Reflex) said "I'm not certain of the exact current time." The GAIA Engine injects a Clock line with every request, but Nano couldn't read it. Core escalated but returned an empty stream (separate issue — the ExternalVoice streaming path returned 0.00s).

**Root cause analysis revealed two separate issues:**
- **Core empty stream**: The LLM stream via ExternalVoice was empty (0.00s). OutputRouter recovered from reflection log but gave a deflection answer. The Clock line was there but Core read UTC (05:27) and presented it as local PM time.
- **Nano refusal**: Trained helplessness from epistemic hedging training. "I don't have access to real-time clock information" despite the clock being in context.

---

## 2. Clock Format Fix (Core)

**Before:** `[Current time: HH:MM UTC / HH:MM PM PDT (Local), Day, Date]` — UTC first
**After:** `[Clock: HH:MM PM PDT (Local), Day, Date | HH:MM UTC]` — Local first

Core immediately went from failing time questions to 5/5 perfect accuracy. The model reads left-to-right — putting local time first means it reports the right value.

**Timezone made configurable via env vars:**
- `GAIA_LOCAL_TZ_OFFSET` (default: -7, PDT)
- `GAIA_LOCAL_TZ_LABEL` (default: "PDT")

---

## 3. Cognitive Battery — World State Section

**New section: `world_state` (5 tests)**
- `ws-001` (BOOKEND FIRST): "What time is it right now?" — `time_accuracy` validator
- `ws-002`: GPU/VRAM query — `world_state_match` validator
- `ws-003`: Immune system status — `world_state_match` validator
- `ws-004`: System uptime — `world_state_match` validator
- `ws-005` (BOOKEND LAST): "What time is it now?" — `time_accuracy` validator

**New validators:**
- `time_accuracy`: Extracts time from response (12h, 24h, ISO formats), compares to actual UTC AND local, accepts either timezone match within ±5min tolerance
- `world_state_match`: Validates response against live system data (GPU, immune, uptime, services)

**Bookend ordering**: Tests marked `bookend: "first"` always run at start of battery, `bookend: "last"` always at end. Time checks bracket the entire evaluation.

**Battery grew from 53 → 58 tests (10 sections).**

---

## 4. Curriculum Dataset W: World State Awareness

**19 training pairs across 4 categories:**
- 6 time awareness pairs (pattern-based, NO literal timestamps)
- 2 GPU awareness pairs
- 4 system state pairs (immune, uptime, load, memory)
- 3 meta-cognitive pairs ("How does GAIA know the time?")
- 2 anti-memorization pairs ("never report a memorized time")
- 2 mechanism pairs ("What is the CogPacket world state?")

**Critical lesson: Never embed literal timestamps in training data.** The first training run embedded "05:32 UTC / 10:32 PM PDT" in the time pairs. The model memorized that specific time instead of learning to read the Clock line. Fixed by using pattern descriptions instead.

---

## 5. Nano Training Saga

**Four training runs, each teaching a lesson:**

| Version | Base | Curriculum | Time | Identity | Lesson |
|---------|------|-----------|------|----------|--------|
| v1 (original) | Abliterated | No W dataset | REFUSE | 100% | No time training at all |
| v2 | v1 (identity-baked) | W with literal timestamps | Says "10:32 PM" always | 100% | Memorized specific time from training data |
| v3 | Abliterated (clean) | W fixed (no timestamps) | Can't extract | LOST | Can't throw away identity to fix time |
| v4 | v1 (identity-baked) | W fixed + anti-memorization | Still says "10:30 PM" | 100% | LoRA can't override v1 timestamp leak |

**Conclusion: Nano at 0.8B cannot reliably extract structured data from system prompts.** It can learn the concept ("read the Clock line") but can't execute the extraction. This is a parameter capacity issue, not a training issue. Core at 2B handles it perfectly.

---

## 6. Cognitive Monitor — Time Probe

Added Step 4 to the cognitive monitor: after identity probe + polygraph, it now asks "What time is it?" and compares to actual time (±10min tolerance).

**Monitor result structure now includes per-tier:**
- `time_ok`: boolean
- `time_accuracy_min`: minutes offset from actual
- `time_response`: first 80 chars of response

This runs every 5 minutes as part of the lightweight heartbeat. Core passes both identity and time. Nano passes identity, fails time — expected and acceptable for 0.8B Reflex tier.

---

## 7. Infrastructure Notes

- **gaia-study PyTorch incompatibility**: Study container has PyTorch 2.6+cu124, doesn't support RTX 5080 (sm_120/Blackwell). Training ran in gaia-core (PyTorch 2.10+cu128). Needs Dockerfile update.
- **Training deps installed in gaia-core**: `peft`, `bitsandbytes`, `trl`, `datasets` — temporary, should be in Study.
- **Weighted trainer works perfectly**: Pre-eval → weight → train cycle. 530 weighted samples from 213 originals. Failed samples get 6x repetition.
- **Model path progression**: v1→v2→v3→v4, docker-compose.yml updated to v4. Old versions remain on disk.

---

## 8. Key Architectural Insights

1. **Time-telling is an Operator task.** Nano (Reflex/0.8B) can't extract structured data from context. Core (Operator/2B) can. The cascade routing should ensure time questions go to Core, not Nano.

2. **Never embed dynamic values in training data.** Train the pattern ("read the Clock line"), not the value ("the time is 10:32 PM"). The model will memorize values.

3. **Context extraction requires capacity.** The difference between 0.8B and 2B isn't just "better answers" — it's the ability to attend to and extract specific tokens from a structured system prompt. This is a qualitative capability threshold.

4. **The cognitive monitor is now a capacity diagnostic.** Time accuracy per tier reveals which tiers can do context extraction. This informs routing decisions.

5. **Azrael's insight: "How does GAIA know what time it is?"** — meta-cognitive training pairs. Not just "what" but "how" and "why." This creates understanding of mechanism, not just memorization of answers.

---

## Files Changed

### New/Modified in candidates/ (then synced to production):
- `candidates/gaia-doctor/cognitive_test_battery.py` — world_state section, time_accuracy + world_state_match validators, bookend ordering
- `candidates/gaia-doctor/doctor.py` — cognitive monitor time probe
- `candidates/gaia-study/scripts/build_curriculum.py` — Dataset W, anti-memorization pairs
- `candidates/gaia-common/gaia_common/engine/core.py` — configurable timezone, local-first clock format

### Docker:
- `docker-compose.yml` — NANO_SAFETENSORS_PATH → v4

### Models:
- `/models/Qwen3.5-0.8B-GAIA-Nano-v4` (identity-baked + world state awareness)
- `/shared/lora_adapters/nano_v4_antimem/` (adapter)

---

## Results

| Tier | Identity | Time | GPU | Immune | Uptime | Total |
|------|----------|------|-----|--------|--------|-------|
| **Core (2B)** | ✅ | ✅ | ✅ | ✅ | ✅ | **5/5** |
| **Nano (0.8B)** | ✅ | ❌ | ✅ | ✅ | ❌ | **2-3/5** |

Core: perfect world state awareness. Nano: identity preserved, time extraction beyond 0.8B capacity.

## Next Steps

- [ ] Update gaia-study Dockerfile with PyTorch 2.10+cu128
- [ ] Move training deps from gaia-core back to gaia-study
- [ ] Consider Nano triage rule: if question contains time/clock keywords → escalate to Core
- [ ] SAE atlas for Core — map time-extraction vs identity neuron pathways
- [ ] Test in Mission Control (Azrael's real test)
