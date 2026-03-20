#!/bin/bash
# build-engine.sh — Build unified GAIA engine base + all tier images
#
# Usage:
#   ./docker/build-engine.sh              # Build everything
#   ./docker/build-engine.sh base         # Base image only
#   ./docker/build-engine.sh prime        # Prime only (assumes base exists)
#   ./docker/build-engine.sh nano         # Nano only
#   ./docker/build-engine.sh core         # Core only

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

log_info()  { echo -e "\033[0;36m[INFO]\033[0m $*"; }
log_ok()    { echo -e "\033[0;32m[OK]\033[0m $*"; }
log_error() { echo -e "\033[0;31m[ERROR]\033[0m $*"; }

TARGET="${1:-all}"

build_base() {
    log_info "Building gaia-engine-base..."
    docker build -t localhost:5000/gaia-engine-base:latest \
        -f docker/Dockerfile.engine-base . "$@"
    log_ok "gaia-engine-base built"
}

build_prime() {
    log_info "Building gaia-prime..."
    docker compose build gaia-prime "$@"
    log_ok "gaia-prime built"
}

build_nano() {
    log_info "Building gaia-nano..."
    docker compose build gaia-nano "$@"
    log_ok "gaia-nano built"
}

build_core() {
    log_info "Building gaia-core..."
    docker compose build gaia-core "$@"
    log_ok "gaia-core built"
}

case "$TARGET" in
    base)
        build_base
        ;;
    prime)
        build_prime
        ;;
    nano)
        build_nano
        ;;
    core)
        build_core
        ;;
    all)
        build_base
        log_info "Building all tiers from base..."
        build_prime
        build_nano
        build_core
        log_ok "All engine images built"
        ;;
    *)
        log_error "Unknown target: $TARGET"
        echo "Usage: $0 [all|base|prime|nano|core]"
        exit 1
        ;;
esac
