#!/bin/bash
# gaia-core entrypoint: starts the Core/Operator inference server,
# then launches the main uvicorn cognitive pipeline.
#
# Three serving modes:
#   1. Managed engine (subprocess isolation): zero-GPU standby server.
#      Model loaded via POST /model/load after startup.
#      Unloading kills the subprocess — guaranteed zero VRAM.
#      Used when CORE_SAFETENSORS_PATH points to an HF model directory.
#
#   2. GGUF (CPU fallback): llama-server with optimized C++ kernels.
#      Used when only a GGUF file is available, or as CPU fallback
#      during FOCUSING state (Prime owns GPU).
#
#   3. Direct engine (legacy): in-process model loading.
#      Used when GAIA_ENGINE_DIRECT=1 is set.
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
# Set GAIA_ENGINE_DIRECT=1 to skip managed mode and load model in-process
GAIA_ENGINE_DIRECT="${GAIA_ENGINE_DIRECT:-0}"

# Ensure shared directories exist
mkdir -p "$CORE_CPU_SLOT_SAVE_PATH" 2>/dev/null || true
mkdir -p "${SHARED_DIR:-/shared}/doctor" 2>/dev/null || true

# ── Mode 1: Managed engine with subprocess isolation ──────────────────────────
if [ -n "$CORE_SAFETENSORS_PATH" ] && [ -d "$CORE_SAFETENSORS_PATH" ] && [ "$GAIA_ENGINE_DIRECT" != "1" ]; then
    echo "[entrypoint] Starting GAIA Engine Manager (zero-GPU standby)..."
    echo "[entrypoint] Model will be loaded via POST /model/load"
    COMPILE_MODE="${GAIA_COMPILE_MODE:-reduce-overhead}"

    python -m gaia_common.engine --managed \
        --port "$CORE_CPU_PORT" \
        2>&1 | sed 's/^/[engine-manager] /' &

    MANAGER_PID=$!
    echo "$MANAGER_PID" > /tmp/inference_server.pid
    echo "[entrypoint] Engine Manager started (PID $MANAGER_PID)"

    # Wait for manager to be ready (fast — no model loading)
    for i in $(seq 1 30); do
        if curl -sf "http://localhost:$CORE_CPU_PORT/health" > /dev/null 2>&1; then
            echo "[entrypoint] Engine Manager healthy after ${i}s"
            break
        fi
        if ! kill -0 "$MANAGER_PID" 2>/dev/null; then
            echo "[entrypoint] WARNING: Engine Manager exited prematurely"
            break
        fi
        sleep 1
    done

    # Load the model — only if GAIA_AUTOLOAD_MODEL=1 (default: standby)
    # The orchestrator sends POST /model/load when it's time to use the GPU.
    GAIA_AUTOLOAD_MODEL="${GAIA_AUTOLOAD_MODEL:-0}"
    if [ "$GAIA_AUTOLOAD_MODEL" = "1" ]; then
        echo "[entrypoint] Auto-loading model: $CORE_SAFETENSORS_PATH (device=$CORE_DEVICE)"
        LOAD_RESULT=$(curl -sf -X POST "http://localhost:$CORE_CPU_PORT/model/load" \
            -H "Content-Type: application/json" \
            -d "{\"model\":\"$CORE_SAFETENSORS_PATH\",\"device\":\"$CORE_DEVICE\",\"compile_mode\":\"$COMPILE_MODE\"}" \
            2>&1) || true
        echo "[entrypoint] Model load result: $LOAD_RESULT"
    else
        echo "[entrypoint] Standby mode — model will be loaded by orchestrator via POST /model/load"
    fi

# ── Mode 1b: Direct engine (legacy, opt-in) ──────────────────────────────────
elif [ -n "$CORE_SAFETENSORS_PATH" ] && [ -d "$CORE_SAFETENSORS_PATH" ] && [ "$GAIA_ENGINE_DIRECT" = "1" ]; then
    echo "[entrypoint] Starting GAIA Inference Engine DIRECT (device=$CORE_DEVICE)..."
    echo "[entrypoint] Model: $CORE_SAFETENSORS_PATH"
    COMPILE_MODE="${GAIA_COMPILE_MODE:-reduce-overhead}"
    python -m gaia_common.engine \
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
