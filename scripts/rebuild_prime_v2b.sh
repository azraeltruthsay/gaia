#!/bin/bash
# Prime v2b, run on the host (jenk): capabilities at 0.5 merged in full precision,
# abliterated, identity retrained on that base. GAIA goes offline only for training and
# is restored whatever happens. Live weights aren't touched.
set -u
R=/shared/training_runs; K=/gaia/GAIA_Project/knowledge/curricula; A=/models/lora_adapters
P=/models/Qwen3-VL-8B-GAIA-Prime-v2; B=/models/Qwen3-VL-8B-GAIA-Prime-v2b
TARGETS='.*language_model\.layers\.\d+\.(self_attn|mlp)\.(q|k|v|o|gate|up|down)_proj.*'
say() { echo "=== $1 $(date -u +%FT%TZ)"; }
study() { docker exec gaia-study bash -lc "$1"; }
restore() {
  say "restoring GAIA"
  curl -s -m 10 -XPOST localhost:6419/maintenance/exit >/dev/null
  docker update --restart=unless-stopped gaia-core gaia-audio gaia-prime >/dev/null
  docker start gaia-core gaia-audio gaia-prime >/dev/null
  for i in $(seq 1 60); do curl -sf -m 3 localhost:6415/health >/dev/null && break; sleep 3; done
  sleep 10
  curl -s -m 60 -XPOST localhost:6410/consciousness/focusing >/dev/null
  say "restored"
}
say "base: capabilities at 0.5"
study "CUDA_VISIBLE_DEVICES= python3 /gaia/GAIA_Project/scripts/merge_lora_full_precision.py --base /models/Qwen/Qwen3-VL-8B-Instruct --adapter $A/qwen3_vl_prime_v2_capabilities --out $B-capabilities --strength 0.5" 2>&1 | grep -E '^merged|Error|Traceback'
[ "$(study "ls $B-capabilities/model.safetensors 2>/dev/null")" ] || { say "FAILED base"; exit 1; }
study "CUDA_VISIBLE_DEVICES= python3 /gaia/GAIA_Project/scripts/abliterate_prime.py --src $B-capabilities --out $B-capabilities-abliterated" 2>&1 | grep -E 'refusal direction|orthogonalized|DONE|Traceback'
[ "$(study "ls $B-capabilities-abliterated/model.safetensors 2>/dev/null")" ] || { say "FAILED abliterate"; exit 1; }
study "cd $B-capabilities; for f in *; do case \$f in *.safetensors) ;; *) [ -e $B-capabilities-abliterated/\$f ] || cp -r \$f $B-capabilities-abliterated/;; esac; done"
say "clearing the 1.0 builds and the 0.5 pre-abliteration copy (adapters remake them)"
study "rm -rf $P $P-capabilities $P-capabilities-abliterated $B-capabilities"
say "taking GAIA offline"
curl -s -m 10 -XPOST localhost:6419/maintenance/enter -H 'content-type: application/json' -d '{"reason":"Prime v2b identity training (jenk), approved by Azrael","entered_by":"claude","ttl_seconds":21600}' >/dev/null
docker update --restart=no gaia-core gaia-audio gaia-prime >/dev/null
docker stop gaia-core gaia-audio gaia-prime >/dev/null
for i in $(seq 1 30); do [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)" -lt 1500 ] && break; sleep 3; done
nvidia-smi --query-gpu=memory.used --format=csv,noheader
say "training identity on the 0.5 base"
study "export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True; TRAIN_LOG=$R/prime_v2b_identity.log bash /gaia/GAIA_Project/scripts/train_core_multimodal_resilient.sh --base-model $B-capabilities-abliterated --text-curriculum $K/prime_identity_v1/train.jsonl --no-vision --no-audio --lora-r 16 --steps 3530 --save-steps 500 --steps-warmup 10 --version-tag prime_v2b_identity --skip-merge --target-modules-regex '$TARGETS' --chat-template chatml"; rc=$?
say "training exited rc=$rc"
restore
study "ls $A/gemma4_e4b_core_multimodal_prime_v2b_identity/adapter_model.safetensors" && study "mv $A/gemma4_e4b_core_multimodal_prime_v2b_identity $A/qwen3_vl_prime_v2b_identity"
say "PRIME_V2B_END rc=$rc"
