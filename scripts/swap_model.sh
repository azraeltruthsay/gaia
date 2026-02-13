#!/bin/bash
# swap_model.sh — Switch gaia-prime to a different model for A/B testing.
#
# Usage:
#   ./scripts/swap_model.sh Qwen3-4B-abliterated
#   ./scripts/swap_model.sh Qwen3-8B-AWQ
#   ./scripts/swap_model.sh Claude              # (switch back to current)
#
# What it does:
#   1. Stops gaia-prime
#   2. Clears the warm pool (tmpfs)
#   3. Copies the selected model from gaia-models/ to the warm pool
#   4. Restarts gaia-prime with the new model path
#
# The warm pool is a 10 GB tmpfs at /mnt/gaia_warm_pool/ for fast model loading.

set -euo pipefail

MODEL_NAME="${1:?Usage: $0 <model-name>  (e.g. Qwen3-4B-abliterated, Qwen3-8B-AWQ, Claude)}"
MODELS_DIR="/gaia/GAIA_Project/gaia-models"
WARM_POOL="/mnt/gaia_warm_pool"
COMPOSE_DIR="/gaia/GAIA_Project"

SOURCE="${MODELS_DIR}/${MODEL_NAME}"
TARGET="${WARM_POOL}/${MODEL_NAME}"

if [ ! -d "${SOURCE}" ]; then
    echo "ERROR: Model directory not found: ${SOURCE}"
    echo "Available models:"
    ls -1d "${MODELS_DIR}"/*/ 2>/dev/null | xargs -I{} basename {}
    exit 1
fi

MODEL_SIZE=$(du -sh "${SOURCE}" | cut -f1)
WARM_AVAIL=$(df -h "${WARM_POOL}" | tail -1 | awk '{print $4}')
echo "Model: ${MODEL_NAME} (${MODEL_SIZE})"
echo "Warm pool available: ${WARM_AVAIL}"

echo ""
echo "=== Step 1: Stopping gaia-prime ==="
cd "${COMPOSE_DIR}"
docker compose stop gaia-prime 2>&1 || true

echo ""
echo "=== Step 2: Clearing warm pool model files ==="
# Keep lora_adapters, remove old model directories
find "${WARM_POOL}" -maxdepth 1 -mindepth 1 -type d ! -name lora_adapters -exec rm -rf {} +
echo "Cleared."

echo ""
echo "=== Step 3: Copying ${MODEL_NAME} to warm pool ==="
cp -r "${SOURCE}" "${TARGET}"
echo "Copied. Warm pool usage: $(du -sh ${WARM_POOL} | cut -f1)"

echo ""
echo "=== Step 4: Restarting gaia-prime with new model ==="
# Use environment variable to set model path (used by both gaia-prime and gaia-core)
export PRIME_MODEL_PATH="/models/${MODEL_NAME}"
docker compose up -d gaia-prime 2>&1

echo ""
echo "=== Waiting for gaia-prime health check ==="
for i in $(seq 1 40); do
    # vLLM returns empty 200 on /health — check HTTP status code
    HTTP_CODE=$(docker compose exec gaia-prime curl -s -o /dev/null -w '%{http_code}' http://localhost:7777/health 2>/dev/null || echo "000")
    if [ "${HTTP_CODE}" = "200" ]; then
        echo "gaia-prime is healthy with ${MODEL_NAME}!"
        break
    fi
    echo "  Waiting... (${i}/40)"
    sleep 5
done

if [ "${HTTP_CODE}" != "200" ]; then
    echo "WARNING: gaia-prime did not become healthy within 200s."
    echo "Check logs: docker compose logs gaia-prime --tail 20"
    exit 1
fi

echo ""
echo "=== Step 5: Restarting gaia-core with new PRIME_MODEL ==="
docker compose up -d gaia-core 2>&1
echo "gaia-core restarted with PRIME_MODEL=${PRIME_MODEL_PATH}"

echo ""
echo "=== Waiting for gaia-core health check ==="
for i in $(seq 1 20); do
    HTTP_CODE=$(docker compose exec gaia-core curl -s -o /dev/null -w '%{http_code}' http://localhost:6415/health 2>/dev/null || echo "000")
    if [ "${HTTP_CODE}" = "200" ]; then
        echo "gaia-core is healthy! Model swap complete."
        echo ""
        echo "Active model: ${MODEL_NAME}"
        echo "Test via Discord or: curl -X POST http://localhost:6415/chat -H 'Content-Type: application/json' -d '{\"message\": \"hello\"}'"
        exit 0
    fi
    echo "  Waiting... (${i}/20)"
    sleep 5
done

echo "WARNING: gaia-core did not become healthy within 100s."
echo "Check logs: docker compose logs gaia-core --tail 20"
exit 1
