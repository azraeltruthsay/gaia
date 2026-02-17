#!/bin/bash

echo "$(date): flatten_soa.sh started. PATH: $PATH"

# GAIA SOA Codebase Flattener
# Updated: 2026-02-10 - Efficiency overhaul, exclusion refinements
# Purpose: Flattens the multi-container structure into a single directory for NotebookLM context.

set -e

# --- Configuration ---

DEST_DIR="GAIA_Condensed_flat"
MAX_FILES=300
DRY_RUN=false

# Active service directories to scan
TARGET_DIRS=(
    "gaia-core"
    "gaia-web"
    "gaia-mcp"
    "gaia-study"
    "gaia-common"
    "knowledge"
)

# Subdirectories to exclude (within TARGET_DIRS)
EXCLUDE_SUBDIRS=(
    "knowledge/projects"
)

# Dev Notebook files to exclude (old proposals/plans superceded by implementation)
EXCLUDE_NOTEBOOK_PATTERNS=(
    "2026-01-1[0-9]_"
    "2026-01-2[0-5]_"
    "_proposal\."
    "_plan\."
    "CoPilot_"
    "Contemplations"
    "Recommendation"
    "SOA-decoupled"
    "prime_dual_backend"
)

# --- Function: Generate gaia_tree.txt (non-blocking, backgrounded) ---
update_gaia_tree_txt() {
    if command -v tree &> /dev/null; then
        tree -L 3 -I "gaia-assistant|candidates|__pycache__|.git|.venv|node_modules|archive|tmp|logs|GAIA_Condensed_flat" . > gaia_tree.txt
    else
        find . -maxdepth 3 -print | \
            grep -vE 'gaia-assistant|candidates|__pycache__|\.git|\.venv|node_modules|archive|tmp|logs|GAIA_Condensed_flat' | \
            sed -e 's;[^/]*/; |-- ;g' > gaia_tree.txt
    fi
}

# Run tree generation in background — it doesn't affect flattening
update_gaia_tree_txt &
TREE_PID=$!

# --- Main Script ---

if [ "$DRY_RUN" = true ]; then
    echo "--- Starting Dry Run (SOA Edition) ---"
else
    echo "--- Starting Codebase Flattening (SOA Edition) ---"
    mkdir -p "$DEST_DIR"
fi

echo "Scanning active services: ${TARGET_DIRS[*]}..."

# Build list of existing target directories
EXISTING_DIRS=()
for dir in "${TARGET_DIRS[@]}"; do
    [ -d "$dir" ] && EXISTING_DIRS+=("$dir")
done

# Build find -path exclusion arguments for EXCLUDE_SUBDIRS
FIND_EXCLUDES=()
for subdir in "${EXCLUDE_SUBDIRS[@]}"; do
    FIND_EXCLUDES+=(-path "./$subdir" -prune -o)
done

# 1. SCAN SERVICES — single find + single grep pipeline
#    find handles directory pruning; one consolidated grep handles file filtering
FILE_LIST=$(
    find "${EXISTING_DIRS[@]}" \
        -path '*/.git' -prune -o \
        -path '*/__pycache__' -prune -o \
        -path '*/.venv' -prune -o \
        -path '*/venv' -prune -o \
        -path '*/node_modules' -prune -o \
        -path '*/site-packages' -prune -o \
        -path '*/build' -prune -o \
        -path '*/dist' -prune -o \
        -path '*/raw-data' -prune -o \
        -path '*/vector_store' -prune -o \
        -path '*/chroma_db' -prune -o \
        -path '*/scans' -prune -o \
        -path '*/artifacts' -prune -o \
        -path '*/tmp' -prune -o \
        -path '*/logs' -prune -o \
        -path '*/.pytest_cache' -prune -o \
        -path '*/tests' -prune -o \
        -path '*/candidates' -prune -o \
        -path 'knowledge/projects' -prune -o \
        -path '*/session_vectors' -prune -o \
        -path '*/archive' -prune -o \
        -path '*/.mypy_cache' -prune -o \
        -type f \( \
            -name '*.py' -o \
            -name '*.json' -o \
            -name '*.md' -o \
            -name '*.sh' -o \
            -name '*.yml' -o \
            -name 'Dockerfile' \
        \) -print | \
    grep -vE '__init__\.py$|_backup\.py$|\.py\.new$|\(backup\)|\.egg-info/|\.pyc$|\.lock$|\.map$|index_store\.json$|embeddings\.json$|final_prompt_for_review\.json$|last_activity\.timestamp$|sessions\.json$|/e2e_.*\.py$|/test_.*\.py$|/conftest\.py$|/README\.md$|/dev_matrix\.json$|/sketchpad\.json$|/response_fragments\.json$|\.claude/' | \
    grep -vE 'Dev_Notebook/2026-01-1[0-9]|Dev_Notebook/2026-01-2[0-5]|Dev_Notebook/.*_proposal\.|Dev_Notebook/.*_plan\.|Dev_Notebook/CoPilot_|Dev_Notebook/Contemplations|Dev_Notebook/.*Recommendation|Dev_Notebook/SOA-decoupled|Dev_Notebook/prime_dual_backend|Dev_Notebook/.*_implementation_plan\.'
)

# 2. ADD ROOT LEVEL FILES
ROOT_FILES="docker-compose.yml README.md .env.example gaia_start.sh"
ROOT_FOUND=""
for f in $ROOT_FILES; do
    [ -f "$f" ] && ROOT_FOUND="${ROOT_FOUND}${f}"$'\n'
done

# Combine and deduplicate
ALL_FILES=$(printf '%s\n%s' "$FILE_LIST" "$ROOT_FOUND" | sort -u | grep -v '^$')
file_count=$(echo "$ALL_FILES" | wc -l)

# --- Process Files ---

if [ "$DRY_RUN" = true ]; then
    echo "--- Dry Run Verification ---"
    echo "Total files found: $file_count"
    echo "$ALL_FILES"
else
    echo "--- Processing and Copying ---"

    # Track what's already in DEST_DIR for incremental mode
    declare -A EXISTING_FLAT
    for existing in "$DEST_DIR"/*; do
        [ -f "$existing" ] && EXISTING_FLAT["$(basename "$existing")"]=1
    done

    copied=0
    skipped=0
    declare -A WANTED_FLAT

    empty_skipped=0
    while IFS= read -r file_path; do
        [ -z "$file_path" ] && continue
        # Remove leading ./ if present
        clean_path="${file_path#./}"
        # Skip empty files (NotebookLM rejects them)
        if [ ! -s "$clean_path" ]; then
            empty_skipped=$((empty_skipped + 1))
            continue
        fi
        # Flatten path: replace / with __
        flat_name="${clean_path//\//__}.txt"
        WANTED_FLAT["$flat_name"]=1

        # Incremental: only copy if source is newer than dest
        dest_file="$DEST_DIR/$flat_name"
        if [ -f "$dest_file" ] && [ "$clean_path" -ot "$dest_file" ]; then
            skipped=$((skipped + 1))
            continue
        fi

        if cp "$clean_path" "$dest_file" 2>/dev/null; then
            copied=$((copied + 1))
        else
            echo "  Warning: cannot read $clean_path (permission denied?), skipping"
            empty_skipped=$((empty_skipped + 1))
        fi
    done <<< "$ALL_FILES"

    # Remove stale files from DEST_DIR that are no longer in the source list
    removed=0
    for existing in "$DEST_DIR"/*; do
        [ -f "$existing" ] || continue
        bname="$(basename "$existing")"
        if [ -z "${WANTED_FLAT[$bname]+x}" ]; then
            rm "$existing"
            removed=$((removed + 1))
        fi
    done

    echo "--- Flattening Complete ---"
    echo "Total source files: $file_count"
    echo "Copied (new/updated): $copied"
    echo "Skipped (unchanged): $skipped"
    echo "Skipped (empty): $empty_skipped"
    echo "Removed (stale): $removed"
    echo "Output Directory: $DEST_DIR"
fi

# --- Verification ---

if [ "$file_count" -gt "$MAX_FILES" ]; then
    echo "Warning: File count ($file_count) exceeds limit of $MAX_FILES."
    echo "   Consider adding more exclusions."
else
    echo "File count ($file_count) is within limit."
fi

# Wait for background tree generation
wait "$TREE_PID" 2>/dev/null
echo "gaia_tree.txt updated."
