#!/bin/bash
# Prime v2's merges redone in full precision from the trained adapters (jenk).
set -u
R=/shared/training_runs; A=/models/lora_adapters; P=/models/Qwen3-VL-8B-GAIA-Prime-v2
rm -f $R/PRIME_V2_MERGE_DONE $R/PRIME_V2_MERGE_FAILED
fail() { echo "FAILED: $1" | tee $R/PRIME_V2_MERGE_FAILED; exit 1; }
M="python3 /gaia/GAIA_Project/scripts/merge_lora_full_precision.py"
$M --base /models/Qwen/Qwen3-VL-8B-Instruct --adapter $A/qwen3_vl_prime_v2_capabilities --out $P-capabilities || fail capabilities
rm -rf $P-capabilities-abliterated
python3 /gaia/GAIA_Project/scripts/abliterate_prime.py --src $P-capabilities --out $P-capabilities-abliterated || fail abliterate
[ -d $P-capabilities-abliterated ] || fail "abliterate left nothing"
for f in $P-capabilities/*; do b=$(basename "$f"); case "$b" in *.safetensors) ;; *) [ -e "$P-capabilities-abliterated/$b" ] || cp -r "$f" "$P-capabilities-abliterated/";; esac; done
$M --base $P-capabilities-abliterated --adapter $A/qwen3_vl_prime_v2_identity --out $P || fail identity
date -u +%FT%TZ > $R/PRIME_V2_MERGE_DONE
