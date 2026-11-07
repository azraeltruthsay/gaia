#!/usr/bin/env bash
# Gemma 4 12B-it as Core, second bake (2qus, 2026-11-07). v1 (lr 2e-4, warmup 10,
# paged 8-bit Adam, 3500 steps) fell to loss 0.29 by step 1810, then a gradient
# spike (136.6 at ~2010) sent it to 7.0 for good; only the last 2 checkpoints
# were kept, so none from before. v2: lr 1e-4, warmup 100, 32-bit paged Adam,
# every checkpoint kept, a divergence guard, 2000 steps.
# First bake notes:
# The V3 identity recipe (core_v2x_identity_v3: 44% identity, plus alpaca,
# tool routing, multi-turn, tool synthesis, deliberation; r16 on all
# attention+MLP projections, 3500 steps) on the stock 12B, in GAIA's house
# turn format with the engine's no-think block, chunked loss for the 262k
# vocabulary. --skip-merge: the trainer's own merge is inside 4-bit and once
# erased an identity; merge after with scripts/merge_lora_full_precision.py.
# Run inside gaia-study with GAIA in meditation (POST :6410/consciousness/training).
set -u
cd /gaia/GAIA_Project
ARGS=(
  --base-model /models/gemma-4-12B-it
  --text-curriculum /gaia/GAIA_Project/knowledge/curricula/core_v2x_identity_v3/text.jsonl
  --no-vision --no-audio
  --lora-r 16
  --target-modules-regex '.*language_model\.layers\.\d+\.(self_attn|mlp)\.(q|k|v|o|gate|up|down)_proj.*'
  --save-steps 250
  --save-total-limit 0
  --lr 1e-4
  --steps-warmup 100
  --optim paged_adamw_32bit
  --divergence-guard 2.5
  --version-tag core12b_identity_v2
  --house-no-think
  --chunked-loss
  --skip-merge
)
if [ "${DRY_RUN:-0}" = "1" ]; then
  exec python3 scripts/train_core_multimodal.py "${ARGS[@]}" --steps 5 --dry-run
else
  export MAX_RETRIES=12
  export TRAIN_LOG=/shared/training_runs/core12b_identity_v2_launch.log
  export GAIA_TRAIN_RUN_ID=core12b_identity_v2
  exec bash scripts/train_core_multimodal_resilient.sh "${ARGS[@]}" --steps 2000
fi
