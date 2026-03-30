#!/bin/bash
# orderly_startup.sh — Sequenced GAIA stack initialization with health gating.

set -e

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

log() { echo -e "${BLUE}[BOOT]${NC} $1"; }
success() { echo -e "${GREEN}[OK]${NC} $1"; }
warn() { echo -e "${YELLOW}[WAIT]${NC} $1"; }
error() { echo -e "${RED}[ERROR]${NC} $1"; exit 1; }

wait_for_health() {
    local svc=$1
    local port=$2
    local timeout=${3:-120}
    local start_time=$(date +%s)
    
    warn "Waiting for ${svc} to be healthy on port ${port}..."
    while true; do
        if curl -sf "http://localhost:${port}/health" > /dev/null 2>&1; then
            success "${svc} is healthy."
            return 0
        fi
        
        local current_time=$(date +%s)
        local elapsed=$((current_time - start_time))
        if [ $elapsed -gt $timeout ]; then
            error "Timeout waiting for ${svc} health."
        fi
        sleep 2
    done
}

# 1. Foundation
log "Step 1: Starting Foundation (Orchestrator, Dozzle)..."
docker compose up -d gaia-orchestrator dozzle
wait_for_health "gaia-orchestrator" 6410 30

# 2. Intelligence Substrate (The Heavy Lifters)
log "Step 2: Starting Intelligence Substrate (Prime, MCP, Study)..."
# Seed warm pool first (logic from gaia.sh)
if [ -d "/mnt/gaia_warm_pool" ]; then
    log "Seeding warm pool..."
    sudo rsync -a --checksum ./gaia-models/Qwen3-8B-abliterated-AWQ/ /mnt/gaia_warm_pool/Qwen3-8B-abliterated-AWQ/
fi

docker compose up -d gaia-prime gaia-mcp gaia-study
wait_for_health "gaia-mcp" 8765 60
wait_for_health "gaia-prime" 7777 300 # Prime takes a long time to load weights

# 3. Cognitive Core
log "Step 3: Starting Cognitive Core (The Brain)..."
docker compose up -d gaia-core
wait_for_health "gaia-core" 6415 60

# 4. Interface & Sensory Layer
log "Step 4: Starting Interface Layer (Web, Audio, Wiki, Doctor)..."
docker compose up -d gaia-web gaia-audio gaia-wiki gaia-doctor
wait_for_health "gaia-web" 6414 60
wait_for_health "gaia-doctor" 6419 30

success "GAIA stack is fully initialized and sequenced."
