#!/bin/bash
# gaia-core entrypoint: starts the Core/Operator inference server,
# then launches the main uvicorn cognitive pipeline.
#
# Two serving modes:
#   1. Safetensors (GPU-native): transformers-based inference server.
#      Enables SAE analysis, ROME editing, live GPU↔CPU migration.
#      Used when CORE_SAFETENSORS_PATH points to an HF model directory.
#
#   2. GGUF (CPU fallback): llama-server with optimized C++ kernels.
#      Used when only a GGUF file is available, or as CPU fallback
#      during FOCUSING state (Prime owns GPU).
#
# The server provides an OpenAI-compatible API on CORE_CPU_PORT (default 8092).
# gaia-core connects to it via CORE_CPU_ENDPOINT.

set -euo pipefail

CORE_CPU_PORT="${CORE_CPU_PORT:-8092}"
CORE_CPU_MODEL_PATH="${CORE_CPU_MODEL_PATH:-/models/Qwen3.5-4B-Abliterated-Q4_K_M.gguf}"
CORE_SAFETENSORS_PATH="${CORE_SAFETENSORS_PATH:-}"
CORE_CPU_CTX="${CORE_CPU_CTX:-8192}"
CORE_CPU_THREADS="${CORE_CPU_THREADS:-8}"
CORE_CPU_SLOT_SAVE_PATH="${CORE_CPU_SLOT_SAVE_PATH:-/shared/kvcache/core}"
# GPU layers: 0 = CPU only, 999 = all layers on GPU (default: 0 for production)
N_GPU_LAYERS="${N_GPU_LAYERS:-0}"
# Initial device for safetensors server: cuda or cpu
CORE_DEVICE="${CORE_DEVICE:-cuda}"

# Ensure shared directories exist
mkdir -p "$CORE_CPU_SLOT_SAVE_PATH" 2>/dev/null || true
mkdir -p "${SHARED_DIR:-/shared}/doctor" 2>/dev/null || true

# ── Mode 1: Safetensors inference server (GPU-native) ────────────────────────
if [ -n "$CORE_SAFETENSORS_PATH" ] && [ -d "$CORE_SAFETENSORS_PATH" ]; then
    echo "[entrypoint] Starting GAIA Inference Engine (device=$CORE_DEVICE)..."
    echo "[entrypoint] Model: $CORE_SAFETENSORS_PATH"
    COMPILE_MODE="${GAIA_COMPILE_MODE:-reduce-overhead}"
    python -m gaia_core.gaia_engine \
        --model "$CORE_SAFETENSORS_PATH" \
        --port "$CORE_CPU_PORT" \
        --device "$CORE_DEVICE" \
        --compile "$COMPILE_MODE" \
        2>&1 | sed 's/^/[gaia-engine] /' &

    SERVER_PID=$!
    echo "$SERVER_PID" > /tmp/inference_server.pid
    echo "[entrypoint] Inference server started (PID $SERVER_PID)"

    # Wait for server to be ready
    for i in $(seq 1 120); do
        if curl -sf "http://localhost:$CORE_CPU_PORT/health" > /dev/null 2>&1; then
            echo "[entrypoint] Inference server healthy after ${i}s"
            break
        fi
        if ! kill -0 "$SERVER_PID" 2>/dev/null; then
            echo "[entrypoint] WARNING: Inference server exited prematurely"
            break
        fi
        sleep 1
    done

# ── Mode 2: GGUF via llama-server (CPU fallback) ─────────────────────────────
elif [ -f "$CORE_CPU_MODEL_PATH" ]; then
    echo "[entrypoint] Starting llama-server for Core/Operator on port $CORE_CPU_PORT (gpu_layers=$N_GPU_LAYERS)..."
    LLAMA_ARGS="--host 0.0.0.0 --port $CORE_CPU_PORT --model $CORE_CPU_MODEL_PATH"
    LLAMA_ARGS="$LLAMA_ARGS --ctx-size $CORE_CPU_CTX --threads $CORE_CPU_THREADS"
    LLAMA_ARGS="$LLAMA_ARGS --n-gpu-layers $N_GPU_LAYERS --chat-template chatml"
    if [ -d "$CORE_CPU_SLOT_SAVE_PATH" ]; then
        LLAMA_ARGS="$LLAMA_ARGS --slot-save-path $CORE_CPU_SLOT_SAVE_PATH"
    fi
    llama-server $LLAMA_ARGS 2>&1 | sed 's/^/[llama-server] /' &

    LLAMA_PID=$!
    echo "$LLAMA_PID" > /tmp/llama_server.pid
    echo "[entrypoint] llama-server started (PID $LLAMA_PID)"

    for i in $(seq 1 120); do
        if curl -sf "http://localhost:$CORE_CPU_PORT/health" > /dev/null 2>&1; then
            echo "[entrypoint] llama-server healthy after ${i}s"
            break
        fi
        if ! kill -0 "$LLAMA_PID" 2>/dev/null; then
            echo "[entrypoint] WARNING: llama-server exited prematurely"
            break
        fi
        sleep 1
    done
else
    echo "[entrypoint] No Core model found — skipping inference server"
    echo "[entrypoint]   Checked safetensors: ${CORE_SAFETENSORS_PATH:-'(not set)'}"
    echo "[entrypoint]   Checked GGUF: $CORE_CPU_MODEL_PATH"
fi

# Start the main cognitive pipeline
echo "[entrypoint] Starting gaia-core uvicorn..."
exec python -m uvicorn gaia_core.main:app --host 0.0.0.0 --port 6415
