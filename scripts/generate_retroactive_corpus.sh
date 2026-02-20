#!/bin/bash
# generate_retroactive_corpus.sh [--dry-run]
#
# Enumerates all live services with qualifying blueprints (avg confidence >= 0.6)
# and generates review prompts for retroactive corpus bootstrapping.
#
# Each service runs inside its own Docker container (which has gaia-common).
# Different containers have different mount paths — the service map below
# handles the mapping.
#
# Outputs: knowledge/curricula/code-architect/retroactive/{service_id}/
#   - ast_summaries_{timestamp}.json
#   - precheck_{timestamp}.json
#   - review_prompt_{timestamp}.txt
#
# Skips: gaia-prime (no Python source — it's a vLLM configuration wrapper)
set -euo pipefail

SCRIPTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPTS_DIR")"
BLUEPRINT_DIR="${PROJECT_DIR}/knowledge/blueprints"
OUTPUT_DIR="${PROJECT_DIR}/knowledge/curricula/code-architect/retroactive"
TIMESTAMP=$(date +%Y%m%dT%H%M%S)

DRY_RUN=false
if [ "${1:-}" == "--dry-run" ]; then
    DRY_RUN=true
fi

# ── Service map: service_id → container_name, source_path, blueprint_path ────
# Most services: container=$svc, src=/app/${pkg}, bp=/knowledge/blueprints/${svc}.yaml
# Orchestrator: mounts full project root at /gaia/GAIA_Project

declare -A CONTAINER_MAP
declare -A SRC_MAP
declare -A BP_MAP

# Standard services (source at /app/{package}, blueprint at /knowledge/blueprints/)
for svc in gaia-core gaia-web gaia-mcp gaia-study; do
    pkg="${svc//-/_}"
    CONTAINER_MAP[$svc]="$svc"
    SRC_MAP[$svc]="/app/${pkg}"
    BP_MAP[$svc]="/knowledge/blueprints/${svc}.yaml"
done

# Orchestrator: full project root mount
CONTAINER_MAP[gaia-orchestrator]="gaia-orchestrator"
SRC_MAP[gaia-orchestrator]="/gaia/GAIA_Project/gaia-orchestrator/gaia_orchestrator"
BP_MAP[gaia-orchestrator]="/gaia/GAIA_Project/knowledge/blueprints/gaia-orchestrator.yaml"

# Services to skip (no reviewable Python source)
SKIP_SERVICES=("gaia-prime")

# ── Confidence check ─────────────────────────────────────────────────────────

check_confidence() {
    local bp_file="$1"
    python3 -c "
import yaml
with open('$bp_file') as f:
    bp = yaml.safe_load(f)
conf = bp.get('meta', {}).get('confidence', {})
mapping = {'high': 0.9, 'medium': 0.6, 'low': 0.3, 'very_high': 0.95, 'very_low': 0.1}
vals = []
for v in conf.values():
    if isinstance(v, (int, float)):
        vals.append(float(v))
    elif isinstance(v, str) and v.lower() in mapping:
        vals.append(mapping[v.lower()])
print(f'{sum(vals)/len(vals):.2f}' if vals else '0.00')
" 2>/dev/null
}

# ── Main loop ────────────────────────────────────────────────────────────────

echo "═══════════════════════════════════════════════════════════"
echo "  Retroactive Corpus Generation"
echo "  Timestamp: ${TIMESTAMP}"
echo "  Mode: $([ "$DRY_RUN" == "true" ] && echo "DRY RUN" || echo "GENERATE")"
echo "═══════════════════════════════════════════════════════════"
echo ""

QUALIFIED=0
SKIPPED=0
FAILED=0

for bp_file in "$BLUEPRINT_DIR"/*.yaml; do
    SERVICE=$(basename "$bp_file" .yaml)

    # Skip non-reviewable services
    for skip in "${SKIP_SERVICES[@]}"; do
        if [ "$SERVICE" == "$skip" ]; then
            echo "SKIP  $SERVICE (no Python source)"
            SKIPPED=$((SKIPPED + 1))
            continue 2
        fi
    done

    # Check if service is in our map
    if [ -z "${CONTAINER_MAP[$SERVICE]:-}" ]; then
        echo "SKIP  $SERVICE (no container mapping)"
        SKIPPED=$((SKIPPED + 1))
        continue
    fi

    # Check confidence threshold
    AVG_CONF=$(check_confidence "$bp_file")
    if python3 -c "exit(0 if float('$AVG_CONF') < 0.6 else 1)" 2>/dev/null; then
        echo "SKIP  $SERVICE (avg confidence $AVG_CONF < 0.6)"
        SKIPPED=$((SKIPPED + 1))
        continue
    fi

    echo "QUALIFY $SERVICE (avg confidence $AVG_CONF)"
    QUALIFIED=$((QUALIFIED + 1))

    if [ "$DRY_RUN" == "true" ]; then
        echo "  → would generate review artifacts"
        continue
    fi

    # ── Generate artifacts for this service ───────────────────────────────

    CONTAINER="${CONTAINER_MAP[$SERVICE]}"
    CONTAINER_SRC="${SRC_MAP[$SERVICE]}"
    CONTAINER_BP="${BP_MAP[$SERVICE]}"
    SERVICE_OUT="${OUTPUT_DIR}/${SERVICE}"

    # Use /tmp/ inside container (always writable) for intermediate files.
    # Scripts output to /tmp/ in container, then we capture via a second exec.
    CONTAINER_TMP="/tmp/retro_${SERVICE}_${TIMESTAMP}"

    mkdir -p "$SERVICE_OUT"

    echo "  Container: $CONTAINER"
    echo "  Source:    $CONTAINER_SRC"
    echo "  Blueprint: $CONTAINER_BP"

    # Step 1: AST summaries (write to container /tmp, then cat to host)
    echo "  [1/3] AST summaries..."
    if ! docker compose exec -T "$CONTAINER" python - \
        --source-dir "$CONTAINER_SRC" \
        --output "${CONTAINER_TMP}/ast_summaries.json" \
        < "$SCRIPTS_DIR/generate_ast_summaries.py" 2>/dev/null; then
        echo "  ERROR: AST summary generation failed for $SERVICE"
        FAILED=$((FAILED + 1))
        continue
    fi
    docker compose exec -T "$CONTAINER" cat "${CONTAINER_TMP}/ast_summaries.json" \
        > "${SERVICE_OUT}/ast_summaries_${TIMESTAMP}.json"

    # Step 2: Pre-check
    echo "  [2/3] Pre-check..."
    if ! docker compose exec -T "$CONTAINER" python - \
        --blueprint "$CONTAINER_BP" \
        --source-dir "$CONTAINER_SRC" \
        --output "${CONTAINER_TMP}/precheck.json" \
        < "$SCRIPTS_DIR/run_blueprint_precheck.py" 2>/dev/null; then
        echo "  ERROR: Pre-check failed for $SERVICE"
        FAILED=$((FAILED + 1))
        continue
    fi
    docker compose exec -T "$CONTAINER" cat "${CONTAINER_TMP}/precheck.json" \
        > "${SERVICE_OUT}/precheck_${TIMESTAMP}.json"

    # Step 3: Build review prompt
    echo "  [3/3] Review prompt..."
    if ! docker compose exec -T "$CONTAINER" python - \
        --blueprint "$CONTAINER_BP" \
        --ast-summaries "${CONTAINER_TMP}/ast_summaries.json" \
        --precheck "${CONTAINER_TMP}/precheck.json" \
        --output "${CONTAINER_TMP}/review_prompt.txt" \
        --direction forward \
        < "$SCRIPTS_DIR/build_review_prompt.py" 2>/dev/null; then
        echo "  ERROR: Prompt build failed for $SERVICE"
        FAILED=$((FAILED + 1))
        continue
    fi
    docker compose exec -T "$CONTAINER" cat "${CONTAINER_TMP}/review_prompt.txt" \
        > "${SERVICE_OUT}/review_prompt_${TIMESTAMP}.txt"

    # Clean up container temp
    docker compose exec -T "$CONTAINER" rm -rf "$CONTAINER_TMP" 2>/dev/null || true

    # Report
    FILES=$(python3 -c "import json; d=json.load(open('${SERVICE_OUT}/ast_summaries_${TIMESTAMP}.json')); print(len(d))")
    CHECKS=$(python3 -c "import json; d=json.load(open('${SERVICE_OUT}/precheck_${TIMESTAMP}.json')); print(d['summary']['total'])")
    FOUND=$(python3 -c "import json; d=json.load(open('${SERVICE_OUT}/precheck_${TIMESTAMP}.json')); print(d['summary']['found'])")
    PROMPT_SIZE=$(wc -c < "${SERVICE_OUT}/review_prompt_${TIMESTAMP}.txt")
    TOKEN_EST=$((PROMPT_SIZE / 4))

    echo "  ✓ ${FILES} files | ${FOUND}/${CHECKS} checks found | ~${TOKEN_EST} tokens"
    echo ""
done

echo "═══════════════════════════════════════════════════════════"
echo "  Summary: ${QUALIFIED} qualified | ${SKIPPED} skipped | ${FAILED} failed"
echo "  Output:  ${OUTPUT_DIR}/"
echo "═══════════════════════════════════════════════════════════"
