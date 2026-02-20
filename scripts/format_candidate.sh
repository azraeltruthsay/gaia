#!/bin/bash
#
# format_candidate.sh <service_name>
#
# Applies ruff format + import sorting to candidate source before validation.
# Run this before validate.sh to ensure consistent formatting.
#
# Usage:
#   ./scripts/format_candidate.sh gaia-core
#   ./scripts/format_candidate.sh gaia-web
#
set -euo pipefail

SERVICE="${1:?Usage: format_candidate.sh <service_name>}"
CANDIDATE_DIR="/gaia/GAIA_Project/candidates/$SERVICE"

if [ ! -d "$CANDIDATE_DIR" ]; then
    echo "Error: Candidate directory not found: $CANDIDATE_DIR"
    exit 1
fi

echo "Formatting $CANDIDATE_DIR..."
ruff format "$CANDIDATE_DIR"
ruff check --fix --select I "$CANDIDATE_DIR"   # import sorting only
echo "Format complete."
