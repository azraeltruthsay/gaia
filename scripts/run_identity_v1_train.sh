#!/usr/bin/env bash
# Launch the Core self-concept + tone identity-refinement LoRA (GAIA_Project-435 Phase 3).
# Refines ON TOP OF V15_FULL (text-only); save_with_tower_graft preserves V15's vision.
# DRY_RUN=1 -> build dataset + parse args + exit (no training).
set -u
cd /gaia/GAIA_Project

ARGS=(
  --base-model /models/Gemma4-E4B-GAIA-Core-Multimodal-CORE2X_V15_FULL
  --text-curriculum /gaia/GAIA_Project/knowledge/curricula/core_v2x_identity/text.jsonl
  --no-vision --no-audio
  --lora-r 16
  --target-modules-regex '.*language_model\.layers\.\d+\.(self_attn|mlp)\.(q|k|v|o|gate|up|down)_proj.*'
  --save-steps 500
  --version-tag core_identity_v1
)

if [ "${DRY_RUN:-0}" = "1" ]; then
  exec python3 scripts/train_core_multimodal.py "${ARGS[@]}" --steps 5 --dry-run
else
  export MAX_RETRIES=12
  export TRAIN_LOG=/shared/training_runs/identity_v1_launch.log
  export GAIA_TRAIN_RUN_ID=core_run_identity_v1
  exec bash scripts/train_core_multimodal_resilient.sh "${ARGS[@]}" --steps 2500
fi
