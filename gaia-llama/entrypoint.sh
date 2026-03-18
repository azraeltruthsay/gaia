#!/bin/bash
# gaia-nano entrypoint: dual-mode serving for the Reflex tier.
#
# Mode 1: Safetensors (GPU-native) — inference_server.py with activation monitoring
# Mode 2: GGUF via llama-server — fast CPU/GPU inference

set -euo pipefail

PORT="${NANO_PORT:-8080}"
SAFETENSORS_PATH="${NANO_SAFETENSORS_PATH:-}"
GGUF_MODEL_PATH="${NANO_GGUF_MODEL_PATH:-/models/Qwen3.5-0.8B-Abliterated-Q8_0.gguf}"
DEVICE="${NANO_DEVICE:-cuda}"
N_GPU_LAYERS="${NANO_GPU_LAYERS:-999}"
CTX_SIZE="${NANO_CTX_SIZE:-2048}"
THREADS="${NANO_THREADS:-4}"

mkdir -p /shared/thoughts 2>/dev/null || true

# ── Mode 1: Safetensors inference server ────────────────────────────────────
if [ -n "$SAFETENSORS_PATH" ] && [ -d "$SAFETENSORS_PATH" ]; then
    echo "[nano-entrypoint] Starting GAIA Engine (device=$DEVICE, port=$PORT)..."
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
