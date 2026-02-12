#!/usr/bin/env bash
# docker-prune.sh — Selective Docker cleanup that preserves expensive build cache.
#
# The GAIA stack includes gaia-prime, which compiles vLLM from source against
# the NGC PyTorch base image (50+ minutes). A naive `docker system prune` would
# wipe those cached layers, causing full rebuilds. This script prunes safely.
#
# Build cache strategy: Uses --reserved-space (LRU-based) instead of age-based
# pruning. Docker keeps the most-recently-used cache entries up to the reserved
# size, and evicts least-recently-used entries beyond that. This means a stable
# stack that hasn't been rebuilt in weeks still keeps its cache — as long as
# nothing more recent has displaced it.
#
# What gets pruned:
#   1. Dangling (untagged) images         — old build leftovers, always safe
#   2. Stopped containers                 — if any exist
#   3. Unused volumes                     — dangling volumes only
#   4. Build cache (LRU beyond budget)    — keeps most-recently-used layers
#
# Usage:
#   ./scripts/docker-prune.sh                # default: keep 40GB of build cache (LRU)
#   ./scripts/docker-prune.sh --keep 60gb    # keep 60GB of cache
#   ./scripts/docker-prune.sh --dry-run      # show what would be pruned, don't delete
#   ./scripts/docker-prune.sh --all          # prune ALL build cache (nuclear option)
#   ./scripts/docker-prune.sh --skip-cache   # only prune images/containers/volumes
#
# The --dry-run flag is recommended before first use.

set -euo pipefail

# ── Defaults ──────────────────────────────────────────────────────────────────
CACHE_BUDGET="40gb"
DRY_RUN=false
PRUNE_ALL_CACHE=false
SKIP_CACHE=false

# ── Parse arguments ──────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --keep)
            CACHE_BUDGET="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --all)
            PRUNE_ALL_CACHE=true
            shift
            ;;
        --skip-cache)
            SKIP_CACHE=true
            shift
            ;;
        -h|--help)
            sed -n '2,/^[^#]/{ /^#/s/^# \?//p }' "$0"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Run with --help for usage."
            exit 1
            ;;
    esac
done

# ── Helpers ───────────────────────────────────────────────────────────────────
section() { echo -e "\n\033[1;36m==> $1\033[0m"; }
info()    { echo "    $1"; }
warn()    { echo -e "    \033[1;33m⚠  $1\033[0m"; }

# ── Pre-flight ────────────────────────────────────────────────────────────────
section "Docker disk usage (before)"
docker system df
echo ""

if $DRY_RUN; then
    warn "DRY RUN — nothing will be deleted"
    echo ""
fi

# ── 1. Dangling images ───────────────────────────────────────────────────────
section "Dangling (untagged) images"
DANGLING_COUNT=$(docker images -f "dangling=true" -q | wc -l)
if [[ "$DANGLING_COUNT" -eq 0 ]]; then
    info "None found."
else
    info "Found $DANGLING_COUNT dangling image(s)."
    if ! $DRY_RUN; then
        RECLAIMED=$(docker image prune -f 2>&1 | tail -1)
        info "$RECLAIMED"
    else
        docker images -f "dangling=true" --format "    {{.ID}}  {{.Size}}  created {{.CreatedSince}}"
    fi
fi

# ── 2. Stopped containers ────────────────────────────────────────────────────
section "Stopped containers"
STOPPED_COUNT=$(docker ps -a -f "status=exited" -f "status=dead" -q | wc -l)
if [[ "$STOPPED_COUNT" -eq 0 ]]; then
    info "None found."
else
    info "Found $STOPPED_COUNT stopped container(s)."
    if ! $DRY_RUN; then
        docker container prune -f | tail -1 | xargs -I{} echo "    {}"
    else
        docker ps -a -f "status=exited" -f "status=dead" --format "    {{.ID}}  {{.Names}}  ({{.Status}})"
    fi
fi

# ── 3. Unused volumes ────────────────────────────────────────────────────────
section "Unused volumes"
UNUSED_VOL_COUNT=$(docker volume ls -f "dangling=true" -q | wc -l)
if [[ "$UNUSED_VOL_COUNT" -eq 0 ]]; then
    info "None found."
else
    info "Found $UNUSED_VOL_COUNT unused volume(s):"
    docker volume ls -f "dangling=true" --format "    {{.Name}}  ({{.Driver}})"
    if ! $DRY_RUN; then
        docker volume prune -f | tail -1 | xargs -I{} echo "    {}"
    fi
fi

# ── 4. Build cache ───────────────────────────────────────────────────────────
if $SKIP_CACHE; then
    section "Build cache"
    info "Skipped (--skip-cache)."
elif $PRUNE_ALL_CACHE; then
    section "Build cache (ALL — nuclear option)"
    warn "This will remove ALL build cache, including gaia-prime's 50-min vLLM compile layers."
    if ! $DRY_RUN; then
        echo ""
        read -r -p "    Are you sure? [y/N] " confirm
        if [[ "$confirm" =~ ^[Yy]$ ]]; then
            docker builder prune -af 2>&1 | tail -1 | xargs -I{} echo "    {}"
        else
            info "Skipped."
        fi
    else
        info "Would remove all build cache."
    fi
else
    section "Build cache (LRU — keeping up to ${CACHE_BUDGET})"
    info "Evicting least-recently-used layers beyond ${CACHE_BUDGET} budget."
    info "Recently-used layers (including gaia-prime vLLM compile) are preserved"
    info "regardless of age, as long as they fit within the budget."
    if ! $DRY_RUN; then
        RECLAIMED=$(docker builder prune --reserved-space "$CACHE_BUDGET" -f 2>&1 | tail -1)
        info "$RECLAIMED"
    else
        echo ""
        CURRENT_CACHE=$(docker system df --format '{{.Size}}' 2>/dev/null | sed -n '4p' || echo "unknown")
        info "Current build cache: ${CURRENT_CACHE}"
        info "Budget: ${CACHE_BUDGET}"
        info "(Run without --dry-run to see actual reclaim.)"
    fi
fi

# ── Summary ───────────────────────────────────────────────────────────────────
section "Docker disk usage (after)"
docker system df

if $DRY_RUN; then
    echo ""
    warn "This was a dry run. Run without --dry-run to actually prune."
fi

echo ""
