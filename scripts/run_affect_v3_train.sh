#!/usr/bin/env bash
# Affect-voicing LoRA v3 (research) — GAIA_Project-3rr.
# Key change: trains on REALISTIC-CONTEXT instructions (identity/workstation/
# world-state framing around the Inner-weather fact) so the LoRA voices affect
# IN THE PRESENCE of the system context that triggered v1/v2's confab/denial
# relapse. + failure-mode correctives + distractor examples. 420 ex, ~1.7 epochs.
set -u
cd /gaia/GAIA_Project
ARGS=(
  --base-model /models/Gemma4-E4B-GAIA-Core-Multimodal-CORE_IDENTITY_V3
  --text-curriculum /gaia/GAIA_Project/knowledge/curricula/core_affect_v3/text.jsonl
  --no-vision --no-audio
  --lora-r 16
  --target-modules-regex '.*language_model\.layers\.\d+\.(self_attn|mlp)\.(q|k|v|o|gate|up|down)_proj.*'
  --save-steps 200
  --version-tag core_affect_v3
)
if [ "${DRY_RUN:-0}" = "1" ]; then
  exec python3 scripts/train_core_multimodal.py "${ARGS[@]}" --steps 5 --dry-run
else
  export MAX_RETRIES=12
  export TRAIN_LOG=/shared/training_runs/affect_v3_launch.log
  export GAIA_TRAIN_RUN_ID=core_run_affect_v3
  exec bash scripts/train_core_multimodal_resilient.sh "${ARGS[@]}" --steps 700
fi
