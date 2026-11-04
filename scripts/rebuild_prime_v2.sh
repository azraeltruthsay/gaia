#!/bin/bash
# Prime v2 from stock Qwen3-VL-8B-Instruct, run inside gaia-study (jenk). Each stage
# only runs if the last one left its output. Writes only new directories: live weights
# (/models/prime, /models/prime-nf4) are not touched. Ends with PRIME_V2_DONE or _FAILED.
set -u
K=/gaia/GAIA_Project/knowledge/curricula
R=/shared/training_runs
TARGETS='.*language_model\.layers\.\d+\.(self_attn|mlp)\.(q|k|v|o|gate|up|down)_proj.*'
STOCK=/models/Qwen/Qwen3-VL-8B-Instruct
CAP=/models/Qwen3-VL-8B-GAIA-Prime-v2-capabilities
ABL=/models/Qwen3-VL-8B-GAIA-Prime-v2-capabilities-abliterated
FINAL=/models/Qwen3-VL-8B-GAIA-Prime-v2
rm -f $R/PRIME_V2_DONE $R/PRIME_V2_FAILED
fail() { echo "FAILED: $1 $(date -u +%FT%TZ)" | tee $R/PRIME_V2_FAILED; exit 1; }
say() { echo "=== $1 $(date -u +%FT%TZ)"; }
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

fill_in() { # the merge keeps weights; a loader also needs the rest of the model's files
    for f in "$STOCK"/*; do b=$(basename "$f"); case "$b" in *.safetensors|*.safetensors.index.json|config.json) ;; *) [ -e "$1/$b" ] || cp -r "$f" "$1/";; esac; done
}

say "stage 1: capabilities from stock"
TRAIN_LOG=$R/prime_v2_stage1.log bash /gaia/GAIA_Project/scripts/train_core_multimodal_resilient.sh \
    --base-model "$STOCK" --text-curriculum "$K/prime_capabilities_v2/text.jsonl" --no-vision --no-audio \
    --lora-r 32 --steps 6000 --save-steps 500 --steps-warmup 100 --no-shuffle --version-tag prime_v2_cap \
    --target-modules-regex "$TARGETS" --chat-template chatml || fail "stage 1 training"
[ -d /models/Gemma4-E4B-GAIA-Core-Multimodal-PRIME_V2_CAP ] || fail "stage 1 left no merged model"
mv /models/Gemma4-E4B-GAIA-Core-Multimodal-PRIME_V2_CAP "$CAP" || fail "stage 1 rename"
mv /models/lora_adapters/gemma4_e4b_core_multimodal_prime_v2_cap /models/lora_adapters/qwen3_vl_prime_v2_capabilities || fail "stage 1 adapter rename"
fill_in "$CAP"

say "stage 2: abliterate"
python3 /gaia/GAIA_Project/scripts/abliterate_prime.py --src "$CAP" --out "$ABL" > $R/prime_v2_abliterate.log 2>&1 || fail "abliterate"
[ -d "$ABL" ] || fail "abliterate left nothing"
fill_in "$ABL"

say "stage 3: identity"
TRAIN_LOG=$R/prime_v2_stage3.log bash /gaia/GAIA_Project/scripts/train_core_multimodal_resilient.sh \
    --base-model "$ABL" --text-curriculum "$K/prime_identity_v1/train.jsonl" --no-vision --no-audio \
    --lora-r 16 --steps 3084 --save-steps 500 --steps-warmup 10 --version-tag prime_v2_identity \
    --target-modules-regex "$TARGETS" --chat-template chatml || fail "stage 3 training"
[ -d /models/Gemma4-E4B-GAIA-Core-Multimodal-PRIME_V2_IDENTITY ] || fail "stage 3 left no merged model"
mv /models/Gemma4-E4B-GAIA-Core-Multimodal-PRIME_V2_IDENTITY "$FINAL" || fail "stage 3 rename"
mv /models/lora_adapters/gemma4_e4b_core_multimodal_prime_v2_identity /models/lora_adapters/qwen3_vl_prime_v2_identity || fail "stage 3 adapter rename"
fill_in "$FINAL"

say "done"
date -u +%FT%TZ > $R/PRIME_V2_DONE
