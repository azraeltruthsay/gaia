#!/usr/bin/env bash
# Gemma 4 12B Core, second pass: multimodal with text replay (2qus, 2026-11-07).
# On the merged text pass (Gemma4-12B-GAIA-Core-v1), as Core V14 learned:
# one mixed run, not media alone. Replay the identity/instruction/tool mix
# (core_v2x_identity_v3), vision from core-multimodal-v8unified (COCO +
# synthetic), and a LIGHT audio touch (core_v2x: 320, V14's best audio;
# heavy audio cut text 100->88 and audio 60->30 in June). Half the samples
# open bare, half with the no-think block (--house-no-think-mix 0.5): the
# text-only pass, trained with the block always, leaked "thought" where a
# request left thinking on. Settings from the stable v2 run.
# Run inside gaia-study with GAIA in meditation (POST :6410/consciousness/training).
set -u
cd /gaia/GAIA_Project
ARGS=(
  --base-model /models/Gemma4-12B-GAIA-Core-v1
  --text-curriculum /gaia/GAIA_Project/knowledge/curricula/core_v2x_identity_v3/text.jsonl
  --curriculum-name core-multimodal-v8unified
  --audio-curriculum-name core_v2x
  --lora-r 16
  --target-modules-regex '.*language_model\.layers\.\d+\.(self_attn|mlp)\.(q|k|v|o|gate|up|down)_proj.*'
  --save-steps 250
  --save-total-limit 0
  --lr 1e-4
  --steps-warmup 100
  --optim paged_adamw_32bit
  --divergence-guard 2.5
  --version-tag core12b_mm_v1
  --house-no-think-mix 0.5
  --chunked-loss
  --lazy-vision
  --skip-merge
)
if [ "${DRY_RUN:-0}" = "1" ]; then
  exec python3 scripts/train_core_multimodal.py "${ARGS[@]}" --steps 5 --dry-run
else
  export MAX_RETRIES=12
  export TRAIN_LOG=/shared/training_runs/core12b_mm_v1_launch.log
  export GAIA_TRAIN_RUN_ID=core12b_mm_v1
  exec bash scripts/train_core_multimodal_resilient.sh "${ARGS[@]}" --steps 2000
fi
