# Sovereign Capability Test (SCT)

## ⚠️ CRITICAL: DO NOT USE FOR TRAINING ⚠️

This directory contains held-out reasoning/code questions that must **never** appear in any training curriculum. They exist to measure GAIA's **actual** capability (as opposed to curriculum absorption measured by the cognitive battery).

## Purpose

Our cognitive battery at `gaia-doctor/cognitive_test_battery.py` measures whether GAIA absorbed her training curriculum. A 95% score means "she learned what we taught." That is NOT the same as "she is capable of novel reasoning."

This test measures what the battery cannot: **genuine reasoning on problems she has never seen**.

## Rules

1. **Never put these questions in any `knowledge/curricula/` file**
2. **Never paste these questions into training scripts**
3. **Never let a model see these during training** (including via sleep-cycle reflection)
4. **When grading, don't log questions to session memory** (they'd end up in MemPalace curation)
5. **If a question becomes contaminated, retire it and write a replacement**

## Categories

- **code_fix/** — Dynamic Python bugs. Automatically graded (run the fix, check tests pass).
- **architecture/** — Multi-step reasoning about system design. Semantic grading.
- **edge_cases/** — Detect bugs/issues a human missed. Keyword + semantic grading.
- **multi_step/** — Problems requiring 2-3 tool calls chained correctly. Trace validation.

## How to run

```bash
# Run full suite against a model tier
python evaluation/sovereign_capability_test/run_sct.py --target core
python evaluation/sovereign_capability_test/run_sct.py --target prime

# Compare two models side-by-side
python evaluation/sovereign_capability_test/run_sct.py --compare core,prime
```

Results go to `/shared/doctor/sct_history/` (separate from cognitive_test_results.json).

## Scoring

Each category reports:
- **Pass rate**: clear-correct answers
- **Partial credit**: right direction but missing key element
- **Fail**: wrong answer or confabulation

Base Core (no training) vs trained Core vs Prime gives us honest capability delta.

## Adding new questions

If you have a real bug or hard reasoning problem that came up in actual work, consider adding it here BEFORE fixing it. Real problems are more valuable than synthetic ones.
