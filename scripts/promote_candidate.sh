#!/bin/bash
#
# promote_candidate.sh - Promote a GAIA service from candidate to live
#
# Usage:
#   ./promote_candidate.sh <service> [--test] [--no-restart] [--no-backup]
#
# Services:
#   gaia-core   - Main cognition engine (live:6415, candidate:6416)
#   gaia-prime  - vLLM inference server (live:7777, candidate:7778)
#   gaia-mcp    - MCP sidecar tools (live:8765, candidate:8767)
#   gaia-study  - Study/learning service (live:8766, candidate:8768)
#   gaia-web          - Web console (live:6414, no candidate container)
#   gaia-orchestrator - GPU/container coordinator (live:6410, candidate:6411)
#   gaia-common       - Shared library (no container, just files)
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

# Function to validate a Python service using Docker
validate_python_service() {
    local service_name=$1
    local candidate_dir=$2
    local dockerfile_path=$3

    echo "--- Validating Python Service: $service_name (Containerized) ---"
    echo "  Candidate directory: $candidate_dir"
    echo "  Dockerfile path: $dockerfile_path"
    echo ""

    local image_name="gaia-candidate-${service_name}:$(date +%s)"
    local build_context="$GAIA_ROOT" # Build context is project root (Dockerfiles use candidates/... paths)

    echo "Building candidate Docker image: $image_name"
    # The -f flag specifies the Dockerfile to use relative to the build context
    if ! docker build -t "$image_name" -f "$dockerfile_path" "$build_context"; then
        echo "✗ Docker image build failed for $service_name!"
        exit 1
    fi
    echo "✓ Docker image built successfully."
    echo ""

    # Run validation commands inside the container
    echo "Running ruff (linting) in container..."
    if ! docker run --rm "$image_name" python -m ruff check /app; then
        echo "✗ Ruff check failed in container!"
        docker rmi "$image_name" > /dev/null 2>&1 || true
        exit 1
    fi
    echo "✓ Ruff check passed in container."
    echo ""

    echo "Running mypy (type checking) in container..."
    if ! docker run --rm "$image_name" python -m mypy /app; then
        echo "⚠ MyPy found issues (non-blocking, informational only)"
    else
        echo "✓ MyPy check passed in container."
    fi
    echo ""

    echo "Running pytest (unit tests) in container..."
    local pytest_exit_code=0
    docker run --rm "$image_name" python -m pytest /app --import-mode=importlib --no-header -q 2>&1 || pytest_exit_code=$?
    if [ $pytest_exit_code -eq 0 ]; then
        echo "✓ Pytest passed in container."
    elif [ $pytest_exit_code -eq 5 ]; then
        echo "⚠ No tests found (non-blocking)"
    else
        echo "✗ Pytest failed in container! (exit code: $pytest_exit_code)"
        docker rmi "$image_name" > /dev/null 2>&1 || true
        exit 1
    fi
    echo ""

    echo "Cleaning up Docker image: $image_name"
    docker rmi "$image_name" > /dev/null 2>&1 || true
    echo "✓ Docker image removed."
    echo ""

    echo "--- Validation Complete for $service_name ---"
}

# Service configuration: service -> live_port:candidate_port:has_container
declare -A SERVICE_CONFIG=(
    ["gaia-core"]="6415:6416:yes"
    ["gaia-prime"]="7777:7778:yes"
    ["gaia-mcp"]="8765:8767:yes"
    ["gaia-study"]="8766:8768:yes"
    ["gaia-web"]="6414::yes"
    ["gaia-orchestrator"]="6410:6411:yes"
    ["gaia-common"]=":::no"
)

# Parse arguments
SERVICE=""
DO_TEST=false
DO_RESTART=true
DO_BACKUP=true
DO_VALIDATE=false # New validation flag

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
        --validate) # New option for validation
            DO_VALIDATE=true
            ;;
        --help|-h)
            echo "Usage: $0 <service> [--test] [--no-restart] [--no-backup] [--validate]"
            echo ""
            echo "Services:"
            echo "  gaia-core   - Main cognition engine (live:6415, candidate:6416)"
            echo "  gaia-prime  - vLLM inference server (live:7777, candidate:7778)"
            echo "  gaia-mcp    - MCP sidecar tools (live:8765, candidate:8767)"
            echo "  gaia-study  - Study/learning service (live:8766, candidate:8768)"
            echo "  gaia-web          - Web console (live:6414)"
            echo "  gaia-orchestrator - GPU/container coordinator (live:6410, candidate:6411)"
            echo "  gaia-common       - Shared library (no container)"
            echo ""
            echo "Options:"
            echo "  --test       Run health check on candidate before promoting"
            echo "  --no-restart Don't restart the live container after promotion"
            echo "  --no-backup  Don't create backup of current live"
            echo "  --validate   Run linting, type checking, and unit tests on candidate before promoting"
            echo ""
            echo "Examples:"
            echo "  $0 gaia-core --test --validate"
            echo "  $0 gaia-mcp --validate"
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
    echo "Services: gaia-core, gaia-prime, gaia-mcp, gaia-study, gaia-web, gaia-orchestrator, gaia-common"
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

# List of Python services that should undergo full validation
PYTHON_SERVICES=("gaia-core" "gaia-mcp" "gaia-study" "gaia-web" "gaia-orchestrator" "gaia-common")

# Optional: Run validation checks for Python services
if [ "$DO_VALIDATE" = true ]; then
    IS_PYTHON_SERVICE=false
    for py_service in "${PYTHON_SERVICES[@]}"; do
        if [ "$SERVICE" = "$py_service" ]; then
            IS_PYTHON_SERVICE=true
            break
        fi
    done

    dockerfile_path="$CANDIDATE_DIR/Dockerfile"
    if [ "$IS_PYTHON_SERVICE" = true ]; then
        if [ ! -f "$dockerfile_path" ]; then
            echo "ERROR: Dockerfile not found for $SERVICE at $dockerfile_path. Cannot perform containerized validation."
            exit 1
        fi
        validate_python_service "$SERVICE" "$CANDIDATE_DIR" "$dockerfile_path"
    else
        echo "Note: Validation requested, but '$SERVICE' is not configured for Python validation. Skipping."
        echo ""
    fi
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
    set +e
    rsync -av --no-group --no-owner --exclude='.git' --exclude='__pycache__' --exclude='*.pyc' \
        "$CANDIDATE_DIR/" "$LIVE_DIR/"
    rsync_code=$?
    set -e
    if [ $rsync_code -eq 23 ]; then
        echo "⚠ Some files could not be transferred (permission issues on container-owned files)"
    elif [ $rsync_code -ne 0 ]; then
        echo "✗ rsync failed with exit code $rsync_code"
        exit $rsync_code
    fi
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
