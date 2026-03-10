#!/bin/bash
# gaia-core entrypoint: starts embedded llama-server for Core/Lite CPU inference,
# then launches the main uvicorn cognitive pipeline.
#
# llama-server provides an OpenAI-compatible API on CORE_CPU_PORT (default 8092)
# for the Qwen3.5-4B model. gaia-core connects to it via CORE_CPU_ENDPOINT.

set -euo pipefail

CORE_CPU_PORT="${CORE_CPU_PORT:-8092}"
CORE_CPU_MODEL_PATH="${CORE_CPU_MODEL_PATH:-/models/Qwen3.5-4B-Abliterated-Q4_K_M.gguf}"
CORE_CPU_CTX="${CORE_CPU_CTX:-4096}"
CORE_CPU_THREADS="${CORE_CPU_THREADS:-8}"

# Only start llama-server if the model file exists
if [ -f "$CORE_CPU_MODEL_PATH" ]; then
    echo "[entrypoint] Starting llama-server for Core/Lite on port $CORE_CPU_PORT..."
    llama-server \
        --host 0.0.0.0 \
        --port "$CORE_CPU_PORT" \
        --model "$CORE_CPU_MODEL_PATH" \
        --ctx-size "$CORE_CPU_CTX" \
        --threads "$CORE_CPU_THREADS" \
        --n-gpu-layers 0 \
        --chat-template chatml \
        2>&1 | sed 's/^/[llama-server] /' &

    LLAMA_PID=$!
    echo "[entrypoint] llama-server started (PID $LLAMA_PID)"

    # Wait for llama-server to be ready (up to 120s for model load)
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
    echo "[entrypoint] No Core model at $CORE_CPU_MODEL_PATH — skipping llama-server"
fi

# Start the main cognitive pipeline
echo "[entrypoint] Starting gaia-core uvicorn..."
exec python -m uvicorn gaia_core.main:app --host 0.0.0.0 --port 6415
