#!/bin/bash
# seed_warm_pool.sh — Stage models to tmpfs warm pool from centralized config.
#
# Reads WARM_POOL config from gaia_constants.json and rsyncs models to tmpfs.
# All tiers (Nano, Core, Prime, Embedding) load from the warm pool for
# near-instant model swaps during GPU rotation.
#
# Usage:
#   ./scripts/seed_warm_pool.sh              # Stage boot models (nano, core, embedding)
#   ./scripts/seed_warm_pool.sh --all        # Stage all models including prime
#   ./scripts/seed_warm_pool.sh --model nano # Stage a specific model
#   ./scripts/seed_warm_pool.sh --status     # Show warm pool status

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
CONSTANTS_FILE="${PROJECT_ROOT}/gaia-common/gaia_common/constants/gaia_constants.json"

# ── Parse config ─────────────────────────────────────────────────────────

POOL_PATH=$(python3 -c "
import json
with open('${CONSTANTS_FILE}') as f:
    c = json.load(f)
print(c.get('WARM_POOL', {}).get('tmpfs_path', '/mnt/gaia_warm_pool'))
")

SOURCE_PATH=$(python3 -c "
import json
with open('${CONSTANTS_FILE}') as f:
    c = json.load(f)
print(c.get('WARM_POOL', {}).get('source_path', '${PROJECT_ROOT}/gaia-models'))
")

# ── Functions ────────────────────────────────────────────────────────────

log_info()  { echo -e "\033[0;36m[INFO]\033[0m $*"; }
log_ok()    { echo -e "\033[0;32m[OK]\033[0m $*"; }
log_warn()  { echo -e "\033[0;33m[WARN]\033[0m $*"; }
log_error() { echo -e "\033[0;31m[ERROR]\033[0m $*"; }

check_pool() {
    if [ ! -d "${POOL_PATH}" ]; then
        log_error "Warm pool not mounted at ${POOL_PATH}"
        echo "Mount it with: sudo mount -t tmpfs -o size=30G,mode=755,uid=$(id -u),gid=$(id -g) tmpfs ${POOL_PATH}"
        exit 1
    fi
}

get_models_for_mode() {
    local mode="$1"
    python3 -c "
import json
with open('${CONSTANTS_FILE}') as f:
    c = json.load(f)
wp = c.get('WARM_POOL', {})
models = wp.get('models', {})
if '${mode}' == 'all':
    print(' '.join(models.keys()))
elif '${mode}' == 'boot':
    print(' '.join(wp.get('stage_on_boot', [])))
else:
    # Single model name
    if '${mode}' in models:
        print('${mode}')
    else:
        print('')
"
}

get_model_source() {
    local model_key="$1"
    python3 -c "
import json
with open('${CONSTANTS_FILE}') as f:
    c = json.load(f)
models = c.get('WARM_POOL', {}).get('models', {})
m = models.get('${model_key}', {})
print(m.get('source', ''))
"
}

stage_model() {
    local model_key="$1"
    local source_name
    source_name=$(get_model_source "$model_key")

    if [ -z "$source_name" ]; then
        log_warn "No source configured for model: ${model_key}"
        return 1
    fi

    local src="${SOURCE_PATH}/${source_name}"
    local dst="${POOL_PATH}/${source_name}"

    if [ ! -d "$src" ]; then
        log_warn "Source not found: ${src}"
        return 1
    fi

    if [ -d "$dst" ]; then
        # Check if already up to date (compare file count + total size)
        local src_size dst_size
        src_size=$(du -sb "$src" 2>/dev/null | cut -f1)
        dst_size=$(du -sb "$dst" 2>/dev/null | cut -f1)
        if [ "$src_size" = "$dst_size" ]; then
            log_ok "${model_key}: already staged (${source_name})"
            return 0
        fi
        log_info "${model_key}: updating (size changed)..."
    fi

    local model_size
    model_size=$(du -sh "$src" | cut -f1)
    log_info "${model_key}: staging ${source_name} (${model_size})..."

    rsync -a --delete "$src/" "$dst/"

    log_ok "${model_key}: staged to ${dst}"
}

show_status() {
    check_pool
    echo ""
    echo "GAIA Warm Pool Status"
    echo "═══════════════════════════════════════"
    echo "Path:    ${POOL_PATH}"
    df -h "${POOL_PATH}" | tail -1 | awk '{printf "Size:    %s  Used: %s  Avail: %s  (%s)\n", $2, $3, $4, $5}'
    echo ""
    echo "Staged models:"

    local models
    models=$(python3 -c "
import json
with open('${CONSTANTS_FILE}') as f:
    c = json.load(f)
for k, v in c.get('WARM_POOL', {}).get('models', {}).items():
    print(f\"{k}|{v['source']}|{v.get('description', '')}\")
")

    while IFS='|' read -r key source desc; do
        local dst="${POOL_PATH}/${source}"
        if [ -d "$dst" ]; then
            local size
            size=$(du -sh "$dst" | cut -f1)
            echo "  ✓ ${key}: ${source} (${size}) — ${desc}"
        else
            echo "  ✗ ${key}: ${source} (not staged) — ${desc}"
        fi
    done <<< "$models"

    echo ""
    echo "Other contents:"
    for d in "${POOL_PATH}"/*/; do
        local dirname
        dirname=$(basename "$d")
        # Skip known models
        if ! python3 -c "
import json
with open('${CONSTANTS_FILE}') as f:
    c = json.load(f)
sources = [v['source'] for v in c.get('WARM_POOL', {}).get('models', {}).values()]
exit(0 if '${dirname}' in sources else 1)
" 2>/dev/null; then
            local size
            size=$(du -sh "$d" | cut -f1)
            echo "  ? ${dirname} (${size})"
        fi
    done
}

# ── Main ─────────────────────────────────────────────────────────────────

MODE="${1:---boot}"

case "$MODE" in
    --status|-s)
        show_status
        exit 0
        ;;
    --all|-a)
        check_pool
        log_info "Staging ALL models to warm pool..."
        MODELS=$(get_models_for_mode "all")
        ;;
    --model|-m)
        check_pool
        MODEL_KEY="${2:?Usage: $0 --model <name>}"
        MODELS=$(get_models_for_mode "$MODEL_KEY")
        if [ -z "$MODELS" ]; then
            log_error "Unknown model: ${MODEL_KEY}"
            exit 1
        fi
        ;;
    --boot|*)
        check_pool
        log_info "Staging boot models to warm pool..."
        MODELS=$(get_models_for_mode "boot")
        ;;
esac

# Stage each model
STAGED=0
FAILED=0
for model in $MODELS; do
    if stage_model "$model"; then
        STAGED=$((STAGED + 1))
    else
        FAILED=$((FAILED + 1))
    fi
done

echo ""
log_ok "Warm pool seeded: ${STAGED} staged, ${FAILED} failed"
df -h "${POOL_PATH}" | tail -1 | awk '{printf "Usage: %s / %s (%s)\n", $3, $2, $5}'
