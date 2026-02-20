#!/bin/bash
# cc_review_candidate.sh <service_id> [--live] [--direction forward|reverse]
#
# Generates AST summaries, runs mechanical pre-check, and assembles a CC review
# prompt for a candidate (or live) service.
#
# The Python scripts run inside the gaia-core Docker container (which has
# gaia-common installed). Intermediate results and final output are written
# to the host filesystem via the /knowledge mount.
#
# Options:
#   --live       Review a live (promoted) service instead of a candidate
#   --direction  Review direction: forward (default) or reverse
#   --max-tokens Maximum prompt tokens (triggers truncation if exceeded)
#
# Outputs:
#   <review_dir>/ast_summaries_<timestamp>.json
#   <review_dir>/precheck_<timestamp>.json
#   <review_dir>/review_prompt_<timestamp>.txt
#
# For candidates: review_dir = candidates/<service>/review/
# For live:       review_dir = knowledge/curricula/code-architect/retroactive/<service>/
set -euo pipefail

SCRIPTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPTS_DIR")"

# ── Parse arguments ──────────────────────────────────────────────────────────

SERVICE="${1:?Usage: cc_review_candidate.sh <service_id> [--live] [--direction forward|reverse] [--max-tokens N]}"
shift

LIVE_MODE=false
DIRECTION="forward"
MAX_TOKENS=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --live)       LIVE_MODE=true; shift ;;
        --direction)  DIRECTION="$2"; shift 2 ;;
        --max-tokens) MAX_TOKENS="$2"; shift 2 ;;
        *)            echo "Unknown option: $1" >&2; exit 1 ;;
    esac
done

TIMESTAMP=$(date +%Y%m%dT%H%M%S)

# ── Determine paths ──────────────────────────────────────────────────────────

# Container paths (inside gaia-core)
if [ "$LIVE_MODE" == "true" ]; then
    # Live service: source at /app/<package>, blueprint at /knowledge/blueprints/
    # Map service ID to container source path
    PACKAGE_NAME="${SERVICE//-/_}"  # gaia-core -> gaia_core
    CONTAINER_SRC="/app/${PACKAGE_NAME}"
    CONTAINER_BP="/knowledge/blueprints/${SERVICE}.yaml"

    # Host output path
    REVIEW_DIR="${PROJECT_DIR}/knowledge/curricula/code-architect/retroactive/${SERVICE}"
else
    # Candidate: source must be at production path (copied before review)
    PACKAGE_NAME="${SERVICE//-/_}"
    CONTAINER_SRC="/app/${PACKAGE_NAME}"
    CONTAINER_BP="/knowledge/blueprints/candidates/${SERVICE}.yaml"

    REVIEW_DIR="${PROJECT_DIR}/candidates/${SERVICE}/review"
fi

# Intermediate files go under /knowledge/tmp/ (mounted rw in container)
CONTAINER_TMP="/knowledge/tmp/review_${SERVICE}_${TIMESTAMP}"
HOST_TMP="${PROJECT_DIR}/knowledge/tmp/review_${SERVICE}_${TIMESTAMP}"

echo "═══════════════════════════════════════════════════════════"
echo "  CC Review Pipeline: ${SERVICE}"
echo "  Mode: $([ "$LIVE_MODE" == "true" ] && echo "LIVE" || echo "CANDIDATE")"
echo "  Direction: ${DIRECTION}"
echo "  Timestamp: ${TIMESTAMP}"
echo "═══════════════════════════════════════════════════════════"
echo ""

# Create output directories
mkdir -p "$REVIEW_DIR"
mkdir -p "$HOST_TMP"

# ── Step 1: Generate AST summaries ───────────────────────────────────────────

echo "Step 1/3: Generating AST summaries..."
docker compose exec -T gaia-core python - \
    --source-dir "$CONTAINER_SRC" \
    --output "${CONTAINER_TMP}/ast_summaries.json" \
    < "$SCRIPTS_DIR/generate_ast_summaries.py"

# Copy from knowledge/tmp to review dir
cp "${HOST_TMP}/ast_summaries.json" "${REVIEW_DIR}/ast_summaries_${TIMESTAMP}.json"
echo "  → ${REVIEW_DIR}/ast_summaries_${TIMESTAMP}.json"
echo ""

# ── Step 2: Run mechanical pre-check ────────────────────────────────────────

echo "Step 2/3: Running mechanical pre-check..."
docker compose exec -T gaia-core python - \
    --blueprint "$CONTAINER_BP" \
    --source-dir "$CONTAINER_SRC" \
    --output "${CONTAINER_TMP}/precheck.json" \
    < "$SCRIPTS_DIR/run_blueprint_precheck.py"

cp "${HOST_TMP}/precheck.json" "${REVIEW_DIR}/precheck_${TIMESTAMP}.json"
echo "  → ${REVIEW_DIR}/precheck_${TIMESTAMP}.json"
echo ""

# ── Step 3: Build review prompt ──────────────────────────────────────────────

echo "Step 3/3: Building review prompt..."
BUILD_ARGS=(
    --blueprint "$CONTAINER_BP"
    --ast-summaries "${CONTAINER_TMP}/ast_summaries.json"
    --precheck "${CONTAINER_TMP}/precheck.json"
    --output "${CONTAINER_TMP}/review_prompt.txt"
    --direction "$DIRECTION"
)

if [ -n "$MAX_TOKENS" ]; then
    BUILD_ARGS+=(--max-tokens "$MAX_TOKENS")
fi

docker compose exec -T gaia-core python - \
    "${BUILD_ARGS[@]}" \
    < "$SCRIPTS_DIR/build_review_prompt.py"

cp "${HOST_TMP}/review_prompt.txt" "${REVIEW_DIR}/review_prompt_${TIMESTAMP}.txt"
echo "  → ${REVIEW_DIR}/review_prompt_${TIMESTAMP}.txt"
echo ""

# ── Clean up temp ────────────────────────────────────────────────────────────

rm -rf "$HOST_TMP"

# ── Instructions ─────────────────────────────────────────────────────────────

echo "═══════════════════════════════════════════════════════════"
echo "  Review artifacts ready."
echo ""
echo "  Prompt:     ${REVIEW_DIR}/review_prompt_${TIMESTAMP}.txt"
echo "  AST:        ${REVIEW_DIR}/ast_summaries_${TIMESTAMP}.json"
echo "  Pre-check:  ${REVIEW_DIR}/precheck_${TIMESTAMP}.json"
echo ""
echo "  To invoke CC for review, start a NEW Claude Code session"
echo "  and provide the review prompt file as context."
echo "  CC must not have prior context about this service."
echo ""
echo "  Save CC output to:"
echo "    ${REVIEW_DIR}/cc_review_${TIMESTAMP}.json"
echo ""
echo "  Then validate with:"
echo "    cp ${REVIEW_DIR}/cc_review_${TIMESTAMP}.json knowledge/tmp/review.json && \\"
echo "    docker compose exec -T gaia-core python - \\"
echo "      --input /knowledge/tmp/review.json \\"
echo "      < scripts/validate_review_result.py"
echo "═══════════════════════════════════════════════════════════"
