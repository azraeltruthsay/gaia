#!/usr/bin/env bash
# Affect-voicing LoRA v2 — GAIA_Project-3rr.
# v1 proved the concept but overfit (echoed the Inner-weather fact). v2 trains on
# the anti-echo / paraphrased / more-diverse curriculum, with FEWER steps
# (~1.3 epochs over 384 ex vs v1's ~2 epochs over 298) to cut memorization.
# Same CORE_IDENTITY_V3 base so affect stacks on the committed identity.
set -u
cd /gaia/GAIA_Project
ARGS=(
  --base-model /models/Gemma4-E4B-GAIA-Core-Multimodal-CORE_IDENTITY_V3
  --text-curriculum /gaia/GAIA_Project/knowledge/curricula/core_affect_v2/text.jsonl
  --no-vision --no-audio
  --lora-r 16
  --target-modules-regex '.*language_model\.layers\.\d+\.(self_attn|mlp)\.(q|k|v|o|gate|up|down)_proj.*'
  --save-steps 150
  --version-tag core_affect_v2
)
if [ "${DRY_RUN:-0}" = "1" ]; then
  exec python3 scripts/train_core_multimodal.py "${ARGS[@]}" --steps 5 --dry-run
else
  export MAX_RETRIES=12
  export TRAIN_LOG=/shared/training_runs/affect_v2_launch.log
  export GAIA_TRAIN_RUN_ID=core_run_affect_v2
  exec bash scripts/train_core_multimodal_resilient.sh "${ARGS[@]}" --steps 500
fi
