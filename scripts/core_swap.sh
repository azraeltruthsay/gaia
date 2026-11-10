#!/usr/bin/env bash
# Swap Core's model between production and the candidate for testing (ptpl).
#
# gaia-core-candidate (the HA mirror of the Core service) holds no model of its
# own: a second 12B costs ~13 GB of RAM or the GPU. To test candidate code
# against a live model, production lends it the GPU and takes it back after.
#
#   scripts/core_swap.sh candidate [--device cuda|cpu]   # lend the model to the candidate
#   scripts/core_swap.sh production                       # take it back (gear 1)
#   scripts/core_swap.sh status
#
# candidate: production goes to MEDITATION (the GPU pin built for lending it
#   out: Core stays up on the CPU and keeps answering, and nothing wakes the
#   GPU gears until an explicit awake), then the candidate's engine loads
#   /models/core-gpu.gguf on the GPU (same model, projector, MTP and q8_0 KV
#   settings as production, from its compose env). --device cpu loads
#   /models/core.gguf on the CPU instead; production's CPU Core stays loaded,
#   so that needs ~13 GB more RAM.
# production: the candidate unloads, production wakes into gear 1.
set -euo pipefail

ORCH=${ORCH:-http://localhost:6410}
CAND=gaia-core-candidate
DEVICE=cuda
CMD=${1:-status}; shift || true
while [ $# -gt 0 ]; do
    case "$1" in
        --device) DEVICE="$2"; shift 2 ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
done

state() {
    curl -s -m 5 "$ORCH/lifecycle/state" | python3 -c \
        'import sys,json;d=json.load(sys.stdin);print(d["state"],d["tiers"]["core"]["device"],d["tiers"]["prime"]["device"])'
}
cand() {  # <method> <path> [json]
    docker exec "$CAND" curl -s -m 600 -X "$1" "http://localhost:8092$2" \
        -H 'Content-Type: application/json' ${3:+-d "$3"}
}
wait_for() {  # <state prefix> <timeout s>
    local want=$1 t=$2 s
    for _ in $(seq 1 $((t / 5))); do
        s=$(state 2>/dev/null || true)
        case "$s" in "$want"*) echo "  $s"; return 0 ;; esac
        sleep 5
    done
    echo "  timed out waiting for '$want' (now: $s)" >&2; return 1
}
vram() { nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader 2>/dev/null || echo "?"; }

case "$CMD" in
    candidate)
        echo "Production -> MEDITATION (Core on CPU, GPU pinned free)..."
        curl -s -m 10 -X POST "$ORCH/consciousness/training" >/dev/null
        wait_for "meditation" 180
        for _ in $(seq 1 24); do  # GPU tiers unloaded?
            s=$(state); case "$s" in "meditation cpu unloaded"|"meditation unloaded unloaded") break ;; esac; sleep 5
        done
        echo "  GPU: $(vram)"
        if [ "$DEVICE" = "cuda" ]; then
            model=/models/core-gpu.gguf
        else
            model=/models/core.gguf
        fi
        echo "Candidate loads $model on $DEVICE..."
        cand POST /model/load "{\"model\":\"$model\",\"device\":\"$DEVICE\"}"; echo
        cand GET /health | python3 -c 'import sys,json;d=json.load(sys.stdin);print("  candidate:",{k:d.get(k) for k in ("backend","device","model","model_loaded")})'
        echo "  GPU: $(vram)"
        echo "Candidate holds the model. Run 'scripts/core_swap.sh production' when done."
        ;;
    production)
        echo "Candidate unloads..."
        cand POST /model/unload; echo
        echo "Production -> gear 1..."
        curl -s -m 10 -X POST "$ORCH/consciousness/awake" >/dev/null
        wait_for "awake gpu" 240
        echo "  GPU: $(vram)"
        ;;
    status)
        echo "production: $(state)"
        cand GET /health | python3 -c 'import sys,json;d=json.load(sys.stdin);print("candidate: ",{k:d.get(k) for k in ("backend","device","model","model_loaded")})'
        echo "GPU: $(vram)"
        ;;
    *) echo "usage: $0 candidate [--device cuda|cpu] | production | status" >&2; exit 2 ;;
esac
