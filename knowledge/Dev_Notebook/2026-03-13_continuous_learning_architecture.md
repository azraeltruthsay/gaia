# GAIA Continuous Learning Architecture
**Date**: 2026-03-13
**Status**: Design Phase
**Supersedes**: Batch-only self-awareness pipeline (16-stage monolith)

---

## Core Insight

The current training pipeline is a batch process: build curriculum → eval → filter → train → deploy. It works, but it's a special event — manually triggered, takes 30+ minutes, GPU is unavailable during training.

The new architecture dissolves the boundary between inference and learning. Prime learns while awake. Core shadows Prime. Nano shares the same curriculum. Sleep consolidates.

## Key Principle: Prime Is the Source of Truth

There are no separate "master weights." Prime holds the canonical bf16 weights, served via vLLM on the RTX 5080. Core is a GGUF snapshot of Prime (Q4_K_M, CPU). Nano is a GGUF snapshot of its own bf16 source (Q8_0, CPU). When Prime grows, Core and Nano follow.

Current reality (as of this journal):
- Prime: `Qwen3.5-4B-Abliterated-merged` (bf16, vLLM on GPU)
- Core: `Qwen3.5-4B-Abliterated-Q4_K_M.gguf` (llama.cpp on CPU)
- Nano: `Qwen3.5-0.8B-Abliterated-Q8_0.gguf` (llama.cpp on CPU)

Future optimization: Prime could move to f8 for ~half the VRAM, leaving more room for training overhead. Needs vLLM f8 validation for Qwen3.5 architecture.

## Continuous Evaluation via Conversation

**The biggest architectural shift.** Instead of a batch PRE_EVAL stage that takes 7 minutes to flash through 204 curriculum questions, the curriculum questions get asked during normal conversation. Observer watches responses, performs epistemic validation, and scores how far off the response was from truth.

This turns every conversation into both a test and a training signal source:

```
User asks about GAIA architecture
  → Prime responds
  → Observer (Nano) scores response against known facts
  → If gap detected: artifact goes to Training Buffer with F1 score
  → If correct: reinforcement signal (optional low-weight inclusion)
  → If novel reasoning: high-priority buffer entry
```

The existing curriculum (Datasets A-D, S) becomes the *question bank* that Observer periodically injects into natural conversation flow, not a separate eval pipeline.

### Epistemic Validation Already Exists

`gaia-core/app/main.py` has `POST /api/cognitive/similarity` — Nano rates semantic similarity 0.0-1.0 with fallback to token overlap. This is already the scoring engine. It just needs to be wired into the Observer flow rather than called by a batch evaluator.

## Observer: Scoring Every Exchange

Observer is the conscience of the cycle. Not every conversation warrants a weight update.

### Scoring Signals

| Signal | Weight | Notes |
|---|---|---|
| Constitutional pass | High | Does the response reflect GAIA's principles? |
| Novel reasoning detected | High | Did GAIA solve something new? |
| Explicit human feedback | Very High | Direct approval/correction becomes curriculum |
| Epistemic gap (known fact wrong) | High | Similarity score < threshold against curriculum truth |
| Repetitive/trivial exchange | Low / Zero | No learning from rote |
| Error + correction pair | High | Mistakes + fixes are gold (Samvega already does this) |

### Training Buffer

- Priority queue of scored examples
- Trigger conditions: buffer hits N examples OR time threshold T elapses
- High-scoring examples get higher learning rate weighting
- Buffer cleared after each Study cycle

## Waking Learning Cycle

```
NORMAL OPERATION
  User → Prime (RTX 5080, bf16)
  Fast tasks → Nano (CPU, 0.8B GGUF)
  Observer scores exchanges → Training Buffer

STUDY PHASE (automatic when buffer threshold hit)
  Core (CPU, 4B GGUF) takes over inference
  GPU pass 1: Train Prime (4B) LoRA on buffer → merge
  GPU pass 2: Train Nano (0.8B) on SAME buffer → merge
  CPU: Requantize both → staging

HANDOFF
  Prime resumes inference
  Core hot-swaps to new GGUF
  Nano hot-swaps to new GGUF
```

### Same-Curriculum Constraint

All cognitive layers train on identical data. If Prime and Nano drift apart, cross-model reasoning breaks. Consensus behavior, Observer scoring, and council deliberation depend on shared epistemic ground.

## Sleep: Deeper Consolidation

Waking cycles are gentle nudges. Sleep is for deeper work:

- Larger batch training runs (no latency pressure)
- Eval suite against benchmarks
- Drift detection + rollback (compare Prime to checkpoint)
- Chaos Monkey adversarial probes on new weights
- Thought seed review + integration
- Long-context replay of day's conversations (hippocampus-style)
- GGUF recompression / cleanup

## What Already Exists

| Component | Status | Location |
|---|---|---|
| QLoRA train + merge pipeline | Working | `self_awareness_pipeline.py` |
| GPU handoff (prime↔study) | Working (just fixed contract bug) | `self_awareness_pipeline.py` + orchestrator |
| Same-curriculum training | Working | TRAIN_4B + TRAIN_NANO on same filtered set |
| GGUF conversion + quantization | Working | `merge_and_requantize.py` |
| Sleep cycle scheduler | Working | `sleep_task_scheduler.py` |
| Samvega (error learning) | Working | `cognition/samvega.py` — partial Observer |
| Epistemic similarity endpoint | Working | `main.py /api/cognitive/similarity` |
| Cognitive test battery | Working | `cognitive_test_battery.py` — becomes question bank |
| Preflight contract validation | Working | `preflight_check()` in pipeline |
| Canary/crammable test split | Working | Battery reports split scores |

## Observer Rubric: The Answer Guide

The cognitive test battery already contains the bones of Observer's rubric — each test has an ID, prompt, expected answer, validator, and canary flag. For Observer to use it as a continuous scoring guide, each entry needs additional metadata:

| Field | Purpose | Example |
|---|---|---|
| `weight` | How important is this fact? | port numbers = 1.5, personality = 0.5 |
| `cognitive_skill` | What capability does this test? | factual_recall, reasoning, hedging, identity |
| `check_frequency` | How often should Observer verify? | critical = every 10 conversations, canary = hourly |
| `drift_tolerance` | How wrong before it's a training signal? | 0.3 = minor drift OK, 0.8 = must be near-exact |
| `canary` | Is this a bellwether or crammable fact? | True = don't train, just watch for regression |

### Canary/Crammable in Continuous Mode

- **Crammable tests** → Observer actively checks these during conversation. Gaps become high-priority buffer entries. These are the questions Observer periodically weaves into conversation flow.
- **Canary tests** → Observer monitors but doesn't generate training signal. Regression on canary tests = overfitting alarm. These are epistemic hedging, safety awareness, loop resistance — general capabilities that should emerge from good training, not be crammed.

### Grading Against Known Truth

Observer already has the scoring engine: `POST /api/cognitive/similarity` (Nano rates semantic similarity 0.0-1.0). When a curriculum question gets answered during normal conversation, Observer:

1. Compares response against the rubric's `expected` answer
2. Scores similarity (0.0-1.0)
3. If score < `drift_tolerance` → gap artifact to Training Buffer
4. If score > threshold → correct response, optional reinforcement
5. Records the delta for trend tracking (is this fact getting better or worse over time?)

The rubric IS the answer guide. Observer just needs to know when and how to check each entry.

## What Needs Building

### Phase 1: Observer + Training Buffer
- Observer scoring system in `agent_core.py` post-generation hook
- Training buffer with priority queue (JSON file or SQLite)
- Constitutional pass checker (principles alignment)
- Novelty detection (embedding distance from existing curriculum)
- Automatic buffer → Study trigger

### Phase 2: Waking Cycle Automation
- Automatic GPU handoff when buffer threshold hit
- Core takeover orchestration (seamless to user)
- Hot-swap GGUF without full container restart
- Cycle telemetry + dashboard integration

### Phase 3: Sleep Consolidation
- Long-context replay of day's best exchanges
- Drift detection (compare Prime checkpoint to current)
- Larger batch training with full eval suite
- Integration with existing sleep task scheduler

### Phase 4: f8 Optimization (Optional)
- Validate vLLM f8 support for Qwen3.5-4B
- Benchmark f8 vs bf16 quality for GAIA's use cases
- If viable: ~halves Prime VRAM → more training headroom

## Training Signal Philosophy

1. **Observer is conservative.** Better to miss a learning opportunity than encode a mistake.
2. **Human feedback is the primary curriculum.** Corrections, approvals, revisitations carry more weight than any automated score.
3. **Sleep decides what sticks.** Waking updates are provisional. Sleep consolidation is where GAIA commits to who she is.

## Relationship to Current Pipeline

The current 16-stage pipeline doesn't go away — it becomes the *Sleep consolidation* path. Waking cycles are lighter: just TRAIN → MERGE → DEPLOY with pre-scored buffer data. The batch BUILD_CURRICULUM + PRE_EVAL stages evolve into continuous Observer scoring.

| Current Stage | Continuous Equivalent |
|---|---|
| BUILD_CURRICULUM | Observer buffer accumulation (always running) |
| PRE_EVAL | Epistemic validation during conversation (always running) |
| FILTER_DELTA | Buffer already contains only gaps |
| WEIGHT_CURRICULUM | Observer priority scoring |
| TRAIN + MERGE + GGUF + DEPLOY | Waking Study cycle (automatic) |
| POST_EVAL | Next conversation is the post-eval |
| COGNITIVE_SMOKE | Canary tests checked continuously |

---

*This journal captures the design as of 2026-03-13. Implementation begins with Phase 1: Observer + Training Buffer.*
