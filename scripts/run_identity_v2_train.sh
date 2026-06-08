#!/usr/bin/env bash
# v2 identity-dominant self-concept+tone LoRA (GAIA_Project-435).
# v1 failed the gate: gaia_identity loss plateaued at 1.57 (vs <1.0 others) at
# 5.9% density -> half-learned, didn't flip behavior. v2 pushes density to 41%
# and ~3 epochs to drive identity loss toward ~0.4. Refines on V15_FULL, text-only.
# DRY_RUN=1 -> build dataset + exit.
set -u
cd /gaia/GAIA_Project

ARGS=(
  --base-model /models/Gemma4-E4B-GAIA-Core-Multimodal-CORE2X_V15_FULL
  --text-curriculum /gaia/GAIA_Project/knowledge/curricula/core_v2x_identity_v2/text.jsonl
  --no-vision --no-audio
  --lora-r 16
  --target-modules-regex '.*language_model\.layers\.\d+\.(self_attn|mlp)\.(q|k|v|o|gate|up|down)_proj.*'
  --save-steps 500
  --version-tag core_identity_v2
)

if [ "${DRY_RUN:-0}" = "1" ]; then
  exec python3 scripts/train_core_multimodal.py "${ARGS[@]}" --steps 5 --dry-run
else
  export MAX_RETRIES=12
  export TRAIN_LOG=/shared/training_runs/identity_v2_launch.log
  export GAIA_TRAIN_RUN_ID=core_run_identity_v2
  exec bash scripts/train_core_multimodal_resilient.sh "${ARGS[@]}" --steps 3300
fi
