#!/bin/bash
# assemble_corpus.sh [--dry-run]
#
# Runs generate_pairs.py inside gaia-core to assemble training pairs
# from all available CC review results into the code-architect corpus.
#
# Prerequisites:
#   - CC review results must exist in:
#     knowledge/curricula/code-architect/reviews/*.json       (forward reviews)
#     knowledge/curricula/code-architect/retroactive/*/cc_review_*.json (retroactive)
#   - Run generate_retroactive_corpus.sh first to create review prompts
#   - Run CC review sessions to produce cc_review_*.json files
#
# Outputs:
#   knowledge/curricula/code-architect/pairs/*.json        (individual pairs)
#   knowledge/curricula/code-architect/train.jsonl          (training data)
#   knowledge/curricula/code-architect/validation.jsonl     (validation data)
#   knowledge/curricula/code-architect/generation_metadata.json
set -euo pipefail

SCRIPTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPTS_DIR")"
CORPUS_DIR="${PROJECT_DIR}/knowledge/curricula/code-architect"

DRY_RUN=""
if [ "${1:-}" == "--dry-run" ]; then
    DRY_RUN="--dry-run"
fi

echo "═══════════════════════════════════════════════════════════"
echo "  Training Corpus Assembly"
echo "  Corpus:    ${CORPUS_DIR}"
echo "  Mode:      $([ -n "$DRY_RUN" ] && echo "DRY RUN" || echo "GENERATE")"
echo "═══════════════════════════════════════════════════════════"
echo ""

# ── Pre-flight checks ────────────────────────────────────────────────────────

# Check for review results
FORWARD_COUNT=0
RETRO_COUNT=0

if [ -d "${CORPUS_DIR}/reviews" ]; then
    FORWARD_COUNT=$(find "${CORPUS_DIR}/reviews" -name "*.json" -type f | wc -l)
fi

if [ -d "${CORPUS_DIR}/retroactive" ]; then
    RETRO_COUNT=$(find "${CORPUS_DIR}/retroactive" -name "cc_review_*.json" -type f | wc -l)
fi

echo "Found reviews:"
echo "  Forward:      ${FORWARD_COUNT}"
echo "  Retroactive:  ${RETRO_COUNT}"
echo "  Total:        $((FORWARD_COUNT + RETRO_COUNT))"
echo ""

if [ "$((FORWARD_COUNT + RETRO_COUNT))" -eq 0 ]; then
    echo "WARNING: No review results found."
    echo ""
    echo "To generate reviews:"
    echo "  1. Run:  ./scripts/generate_retroactive_corpus.sh"
    echo "  2. For each service, start a NEW CC session with the review prompt"
    echo "  3. Save CC output as cc_review_<timestamp>.json in the service dir"
    echo ""
    echo "Or run with existing forward reviews in reviews/ directory."

    if [ -z "$DRY_RUN" ]; then
        echo ""
        echo "Running anyway to process any available data..."
    fi
fi

# ── Run generate_pairs.py inside gaia-core ────────────────────────────────────

echo "Running pair generation inside gaia-core container..."
echo ""

ARGS=(
    --blueprints /knowledge/blueprints
    --corpus-dir /knowledge/curricula/code-architect
    --validation-split 0.15
    --seed 42
)

if [ -n "$DRY_RUN" ]; then
    ARGS+=(--dry-run)
fi

RESULT=$(docker compose exec -T gaia-core python - \
    "${ARGS[@]}" \
    < "$SCRIPTS_DIR/generate_pairs.py" 2>&1)

echo "$RESULT"
echo ""

# ── Check corpus readiness ────────────────────────────────────────────────────

if [ -z "$DRY_RUN" ] && [ -f "${CORPUS_DIR}/generation_metadata.json" ]; then
    READY=$(python3 -c "
import json
meta = json.load(open('${CORPUS_DIR}/generation_metadata.json'))
print('yes' if meta.get('corpus_ready') else 'no')
print(f\"Pairs: {meta.get('total_pairs', 0)} / {meta.get('min_corpus_size', 50)} minimum\")
" 2>/dev/null || echo "unknown")

    echo "═══════════════════════════════════════════════════════════"
    echo "  Corpus Status"
    echo "  $READY"
    if echo "$READY" | grep -q "^yes"; then
        echo ""
        echo "  Corpus is ready for QLoRA training!"
        echo "  Next: Phase 4 — ./scripts/train_code_architect.sh"
    else
        echo ""
        echo "  Corpus below minimum threshold."
        echo "  Generate more CC reviews to reach 50 pairs."
    fi
    echo "═══════════════════════════════════════════════════════════"
fi
