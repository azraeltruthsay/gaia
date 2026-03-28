#!/bin/bash
# ═════════════════════════════════════════════════════════════════════════════
# GAIA Instance Setup Script
# ═════════════════════════════════════════════════════════════════════════════
# Purpose: Creates a private "gaia-instance" directory adjacent to the source
# repository to separate personal data (knowledge, logs, models) from source code.
# ═════════════════════════════════════════════════════════════════════════════

set -e

# Determine paths
SOURCE_DIR=$(pwd)
PARENT_DIR=$(dirname "$SOURCE_DIR")
INSTANCE_DIR="$PARENT_DIR/gaia-instance"

echo "🚀 Setting up GAIA Instance at: $INSTANCE_DIR"

# 1. Create directory structure
mkdir -p "$INSTANCE_DIR"/{knowledge,logs,gaia-models,shared,secrets,artifacts,tmp,audio_inbox}
mkdir -p "$INSTANCE_DIR/knowledge"/{vector_store,wiki_auto}
mkdir -p "$INSTANCE_DIR/logs"/{chat_history,kvcache,thoughtstreams}

echo "✅ Created directory structure."

# 2. Migration: Move existing data if it exists in the source dir
# We use 'cp -an' to copy without overwriting and then remove, to be safe.
migrate_dir() {
    local dir_name=$1
    if [ -d "$SOURCE_DIR/$dir_name" ]; then
        echo "📦 Migrating $dir_name..."
        cp -an "$SOURCE_DIR/$dir_name/." "$INSTANCE_DIR/$dir_name/" 2>/dev/null || true
        # We don't delete yet; user should verify first or we do it later.
    fi
}

migrate_dir "logs"
migrate_dir "gaia-models"   # docker-compose mounts gaia-models/ as /models
migrate_dir "artifacts"
migrate_dir "audio_inbox"
migrate_dir "tmp"

# Seed wiki_auto from source (runtime-generated, but needs initial content)
if [ -d "$SOURCE_DIR/knowledge/wiki_auto" ]; then
    echo "📦 Seeding knowledge/wiki_auto..."
    mkdir -p "$INSTANCE_DIR/knowledge/wiki_auto"
    cp -an "$SOURCE_DIR/knowledge/wiki_auto/." "$INSTANCE_DIR/knowledge/wiki_auto/" 2>/dev/null || true
fi

# Special handling for knowledge (Move only the 'personal' parts)
PERSONAL_KNOWLEDGE_DIRS=("5c" "samvega" "dnd_campaign" "transcripts" "awareness" "creative_writing" "seeds" "digests" "reflections" "Dev_Notebook" ".obsidian")
for dir in "${PERSONAL_KNOWLEDGE_DIRS[@]}"; do
    if [ -d "$SOURCE_DIR/knowledge/$dir" ]; then
        echo "📦 Migrating knowledge/$dir..."
        mkdir -p "$INSTANCE_DIR/knowledge/$dir"
        cp -an "$SOURCE_DIR/knowledge/$dir/." "$INSTANCE_DIR/knowledge/$dir/" 2>/dev/null || true
    fi
done

# 3. Handle Secrets
if [ -f "$SOURCE_DIR/.env" ]; then
    echo "🔑 Migrating .env file..."
    cp -n "$SOURCE_DIR/.env" "$INSTANCE_DIR/secrets/env.production"
fi

if [ -d "$SOURCE_DIR/wireguard_setup" ]; then
    echo "🔒 Migrating Wireguard secrets..."
    cp -an "$SOURCE_DIR/wireguard_setup/." "$INSTANCE_DIR/secrets/wireguard/" 2>/dev/null || true
fi

# 4. Create local symbolic link for convenience (optional)
if [ ! -L "$SOURCE_DIR/instance" ]; then
    ln -s "$INSTANCE_DIR" "$SOURCE_DIR/instance"
    echo "🔗 Created local symbolic link 'instance' -> $INSTANCE_DIR"
fi

echo ""
echo "✨ Instance setup complete."
echo "   Instance dir:  $INSTANCE_DIR"
echo "   Models:        $INSTANCE_DIR/gaia-models/ (mounted as /models)"
echo "   Logs:          $INSTANCE_DIR/logs/"
echo "   Knowledge:     $INSTANCE_DIR/knowledge/ (runtime data only)"
echo ""
echo "   Source-tracked knowledge (blueprints, system_reference, personas, etc.)"
echo "   stays in the repo and is overlay-mounted by docker-compose."
