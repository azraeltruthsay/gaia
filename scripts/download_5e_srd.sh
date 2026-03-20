#!/bin/bash
# Download the 5e SRD (Systems Reference Document) for D&D knowledge base.
# The SRD 5.1 is released under Creative Commons by Wizards of the Coast.
set -euo pipefail

OUTPUT_DIR="/gaia/GAIA_Project/knowledge/dnd_campaign/heimric_cosmos/reference"
mkdir -p "$OUTPUT_DIR"

SRD_URL="https://media.wizards.com/2016/downloads/DND/SRD-OGL_V5.1.pdf"
OUTPUT_FILE="$OUTPUT_DIR/SRD_5.1.pdf"

if [ -f "$OUTPUT_FILE" ]; then
    echo "SRD already downloaded: $OUTPUT_FILE"
else
    echo "Downloading 5e SRD 5.1..."
    curl -L -o "$OUTPUT_FILE" "$SRD_URL" && echo "Downloaded to $OUTPUT_FILE" || echo "Download failed"
fi

echo "Done. Index with: docker compose exec gaia-study python -m gaia_study.indexer --kb dnd_campaign"
