#!/bin/bash
#
# promote_candidate.sh - Promote a GAIA service from candidate to live
#
# Usage:
#   ./promote_candidate.sh <service> [--test] [--no-restart] [--no-backup]
#
# Services:
#   gaia-core   - Main cognition engine (live:6415, candidate:6416)
#   gaia-mcp    - MCP sidecar tools (live:8765, candidate:8767)
#   gaia-study  - Study/learning service (live:8766, candidate:8768)
#   gaia-web    - Web console (live:6414, no candidate container)
#   gaia-common - Shared library (no container, just files)
#
# Options:
#   --test       Run health check on candidate before promoting
#   --no-restart Don't restart the live container after promotion
#   --no-backup  Don't create backup of current live
#
# Examples:
#   ./promote_candidate.sh gaia-core --test
#   ./promote_candidate.sh gaia-mcp
#   ./promote_candidate.sh gaia-common --no-restart
#

set -e

GAIA_ROOT="/gaia/GAIA_Project"

# Service configuration: service -> live_port:candidate_port:has_container
declare -A SERVICE_CONFIG=(
    ["gaia-core"]="6415:6416:yes"
    ["gaia-mcp"]="8765:8767:yes"
    ["gaia-study"]="8766:8768:yes"
    ["gaia-web"]="6414::yes"
    ["gaia-common"]=":::no"
)

# Parse arguments
SERVICE=""
DO_TEST=false
DO_RESTART=true
DO_BACKUP=true

for arg in "$@"; do
    case $arg in
        --test)
            DO_TEST=true
            ;;
        --no-restart)
            DO_RESTART=false
            ;;
        --no-backup)
            DO_BACKUP=false
            ;;
        --help|-h)
            echo "Usage: $0 <service> [--test] [--no-restart] [--no-backup]"
            echo ""
            echo "Services:"
            echo "  gaia-core   - Main cognition engine (live:6415, candidate:6416)"
            echo "  gaia-mcp    - MCP sidecar tools (live:8765, candidate:8767)"
            echo "  gaia-study  - Study/learning service (live:8766, candidate:8768)"
            echo "  gaia-web    - Web console (live:6414)"
            echo "  gaia-common - Shared library (no container)"
            echo ""
            echo "Options:"
            echo "  --test       Run health check on candidate before promoting"
            echo "  --no-restart Don't restart the live container after promotion"
            echo "  --no-backup  Don't create backup of current live"
            echo ""
            echo "Examples:"
            echo "  $0 gaia-core --test"
            echo "  $0 gaia-mcp"
            exit 0
            ;;
        -*)
            echo "Unknown option: $arg"
            exit 1
            ;;
        *)
            if [ -z "$SERVICE" ]; then
                SERVICE="$arg"
            else
                echo "Too many arguments. Service already set to: $SERVICE"
                exit 1
            fi
            ;;
    esac
done

# Validate service
if [ -z "$SERVICE" ]; then
    echo "ERROR: No service specified"
    echo "Usage: $0 <service> [--test] [--no-restart] [--no-backup]"
    echo "Services: gaia-core, gaia-mcp, gaia-study, gaia-web, gaia-common"
    exit 1
fi

if [ -z "${SERVICE_CONFIG[$SERVICE]}" ]; then
    echo "ERROR: Unknown service: $SERVICE"
    echo "Valid services: ${!SERVICE_CONFIG[@]}"
    exit 1
fi

# Parse service config
IFS=':' read -r LIVE_PORT CANDIDATE_PORT HAS_CONTAINER <<< "${SERVICE_CONFIG[$SERVICE]}"

LIVE_DIR="$GAIA_ROOT/$SERVICE"
CANDIDATE_DIR="$GAIA_ROOT/candidates/$SERVICE"
BACKUP_DIR="$GAIA_ROOT/$SERVICE.bak"

echo "=== GAIA Candidate Promotion: $SERVICE ==="
echo ""

# Check candidate exists
if [ ! -d "$CANDIDATE_DIR" ]; then
    echo "ERROR: Candidate directory not found: $CANDIDATE_DIR"
    exit 1
fi

# Optional: Test candidate health
if [ "$DO_TEST" = true ] && [ -n "$CANDIDATE_PORT" ]; then
    echo "Testing candidate health on port $CANDIDATE_PORT..."
    if curl -s --fail "http://localhost:$CANDIDATE_PORT/health" > /dev/null 2>&1; then
        echo "✓ Candidate is healthy"
    else
        echo "✗ Candidate health check failed!"
        echo "  Make sure ${SERVICE}-candidate is running: docker restart ${SERVICE}-candidate"
        exit 1
    fi
    echo ""
elif [ "$DO_TEST" = true ]; then
    echo "Note: No candidate port configured for $SERVICE, skipping health check"
    echo ""
fi

# Create backup
if [ "$DO_BACKUP" = true ]; then
    echo "Creating backup of live at $BACKUP_DIR..."
    rm -rf "$BACKUP_DIR"
    cp -r "$LIVE_DIR" "$BACKUP_DIR"
    echo "✓ Backup created"
    echo ""
fi

# Promote: Copy candidate to live
echo "Promoting candidate to live..."
echo "  Source: $CANDIDATE_DIR/"
echo "  Target: $LIVE_DIR/"

# Use rsync if available, otherwise cp
if command -v rsync &> /dev/null; then
    rsync -av --exclude='.git' --exclude='__pycache__' --exclude='*.pyc' \
        "$CANDIDATE_DIR/" "$LIVE_DIR/"
else
    # Copy all files, preserving structure
    cp -r "$CANDIDATE_DIR/"* "$LIVE_DIR/"
fi

echo "✓ Files promoted"
echo ""

# Restart live container
if [ "$DO_RESTART" = true ] && [ "$HAS_CONTAINER" = "yes" ]; then
    echo "Restarting $SERVICE..."
    docker restart "$SERVICE"
    echo "✓ $SERVICE restarted"
    echo ""

    # Wait and check health
    if [ -n "$LIVE_PORT" ]; then
        echo "Waiting for health check..."
        sleep 5
        if curl -s --fail "http://localhost:$LIVE_PORT/health" > /dev/null 2>&1; then
            echo "✓ Live $SERVICE is healthy"
        else
            echo "⚠ Live health check failed - may still be starting up"
            echo "  Check with: docker logs $SERVICE --tail 20"
        fi
    fi
elif [ "$HAS_CONTAINER" = "no" ]; then
    echo "Note: $SERVICE has no container, skipping restart"
    echo "  Containers using gaia-common may need manual restart"
fi

echo ""
echo "=== Promotion Complete ==="
echo ""
echo "To rollback if needed:"
echo "  cp -r $BACKUP_DIR/* $LIVE_DIR/"
if [ "$HAS_CONTAINER" = "yes" ]; then
    echo "  docker restart $SERVICE"
fi
