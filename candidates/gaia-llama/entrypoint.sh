#!/bin/bash
# gaia-nano entrypoint: dual-mode serving for the Reflex tier.
#
# Mode 1: Managed engine (subprocess isolation) — zero GPU standby,
#          model loaded in worker subprocess. Default for safetensors.
# Mode 2: GGUF via llama-server — fast CPU/GPU inference
# Mode 3: Direct engine (legacy) — GAIA_ENGINE_DIRECT=1

set -euo pipefail

PORT="${NANO_PORT:-8080}"
SAFETENSORS_PATH="${NANO_SAFETENSORS_PATH:-}"
GGUF_MODEL_PATH="${NANO_GGUF_MODEL_PATH:-/models/Qwen3.5-0.8B-Abliterated-Q8_0.gguf}"
DEVICE="${NANO_DEVICE:-cuda}"
N_GPU_LAYERS="${NANO_GPU_LAYERS:-999}"
CTX_SIZE="${NANO_CTX_SIZE:-2048}"
THREADS="${NANO_THREADS:-4}"
GAIA_ENGINE_DIRECT="${GAIA_ENGINE_DIRECT:-0}"

mkdir -p /shared/thoughts 2>/dev/null || true

# ── Mode 1: Managed engine with subprocess isolation ─────────────────────────
if [ -n "$SAFETENSORS_PATH" ] && [ -d "$SAFETENSORS_PATH" ] && [ "$GAIA_ENGINE_DIRECT" != "1" ]; then
    echo "[nano-entrypoint] Starting GAIA Engine Manager (zero-GPU standby, port=$PORT)..."

    python -m gaia_common.engine --managed --port "$PORT" &
    MANAGER_PID=$!

    for i in $(seq 1 15); do
        if curl -sf "http://localhost:$PORT/health" > /dev/null 2>&1; then
            echo "[nano-entrypoint] Engine Manager healthy after ${i}s"
            break
        fi
        if ! kill -0 "$MANAGER_PID" 2>/dev/null; then
            echo "[nano-entrypoint] WARNING: Engine Manager exited prematurely"
            break
        fi
        sleep 1
    done

    # Load model via managed engine
    echo "[nano-entrypoint] Loading model: $SAFETENSORS_PATH (device=$DEVICE)"
    LOAD_RESULT=$(curl -sf -X POST "http://localhost:$PORT/model/load" \
        -H "Content-Type: application/json" \
        -d "{\"model\":\"$SAFETENSORS_PATH\",\"device\":\"$DEVICE\"}" \
        2>&1) || true
    echo "[nano-entrypoint] Model load result: $LOAD_RESULT"

    wait "$MANAGER_PID"

# ── Mode 1b: Direct engine (legacy) ─────────────────────────────────────────
elif [ -n "$SAFETENSORS_PATH" ] && [ -d "$SAFETENSORS_PATH" ] && [ "$GAIA_ENGINE_DIRECT" = "1" ]; then
    echo "[nano-entrypoint] Starting GAIA Engine DIRECT (device=$DEVICE, port=$PORT)..."
    echo "[nano-entrypoint] Model: $SAFETENSORS_PATH"
    exec python -c "from gaia_common.engine import serve; serve('$SAFETENSORS_PATH', $PORT, '$DEVICE')"

# ── Mode 2: GGUF via llama-server ──────────────────────────────────────────
elif [ -f "$GGUF_MODEL_PATH" ]; then
    echo "[nano-entrypoint] Starting llama-server (gpu_layers=$N_GPU_LAYERS, port=$PORT)..."
    LLAMA_ARGS="--host 0.0.0.0 --port $PORT --model $GGUF_MODEL_PATH"
    LLAMA_ARGS="$LLAMA_ARGS --ctx-size $CTX_SIZE --threads $THREADS"
    LLAMA_ARGS="$LLAMA_ARGS --n-gpu-layers $N_GPU_LAYERS --chat-template chatml"
    exec llama-server $LLAMA_ARGS

else
    echo "[nano-entrypoint] No model found!"
    echo "  Checked safetensors: ${SAFETENSORS_PATH:-'(not set)'}"
    echo "  Checked GGUF: $GGUF_MODEL_PATH"
    exit 1
fi
