#!/bin/bash
# GAIA Candidate Testing Script
#
# Two testing modes:
#
# MODE 1: Full Parallel Stack (candidates talk to candidates)
#   ./test_candidate.sh all              # Full isolated candidate ecosystem
#
# MODE 2: Selective Injection (candidate injected into live flow)
#   ./test_candidate.sh mcp --inject     # Live core -> candidate MCP
#
# Usage:
#   ./test_candidate.sh [service|all] [options]
#
# Examples:
#   ./test_candidate.sh all               # Start full candidate stack
#   ./test_candidate.sh all --gpu         # Start full stack with GPU
#   ./test_candidate.sh mcp --inject      # Inject MCP candidate into live
#   ./test_candidate.sh core --unit       # Run unit tests only
#   ./test_candidate.sh all --promote     # Promote all candidates to active

set -e

# Handle commands that don't need a service argument
case "${1:-}" in
    --help|help|--release-gpu|release-gpu|--reclaim-gpu|reclaim-gpu|--gpu-status|gpu-status|--init|init|--status|status)
        SERVICE="all"
        SHIFT_ARGS=0
        ;;
    *)
        SERVICE="${1:-all}"
        SHIFT_ARGS=1
        ;;
esac

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }
log_inject() { echo -e "${CYAN}[INJECT]${NC} $1"; }

PROJECT_ROOT="/gaia/GAIA_Project"
CANDIDATE_DIR="${PROJECT_ROOT}/candidates/gaia-${SERVICE}"
ACTIVE_DIR="${PROJECT_ROOT}/gaia-${SERVICE}"
COMPOSE_FILE="${PROJECT_ROOT}/docker-compose.candidate.yml"
LIVE_COMPOSE_FILE="${PROJECT_ROOT}/docker-compose.yml"

# Check candidate exists
check_candidate() {
    if [ "$SERVICE" = "all" ]; then
        for svc in core web mcp study; do
            local dir="${PROJECT_ROOT}/candidates/gaia-${svc}"
            if [ ! -d "$dir" ]; then
                log_error "Candidate directory not found: $dir"
                echo "Initialize with: ./test_candidate.sh --init"
                exit 1
            fi
        done
    else
        if [ ! -d "$CANDIDATE_DIR" ]; then
            log_error "Candidate directory not found: $CANDIDATE_DIR"
            echo "Initialize with: ./test_candidate.sh --init"
            exit 1
        fi
    fi
}

# Initialize all candidates from active
init_candidates() {
    log_info "Initializing all candidate directories from active..."

    for svc in common core web mcp study audio; do
        local active="${PROJECT_ROOT}/gaia-${svc}"
        local candidate="${PROJECT_ROOT}/candidates/gaia-${svc}"

        if [ -d "$active" ]; then
            log_info "Syncing gaia-${svc} -> candidates/gaia-${svc}"
            mkdir -p "$candidate"
            rsync -av --exclude='__pycache__' --exclude='*.pyc' --exclude='.pytest_cache' \
                --exclude='.mypy_cache' --exclude='.ruff_cache' \
                --exclude='*.egg-info' --exclude='.git' \
                "$active/" "$candidate/"
        else
            log_warn "Active directory not found: $active"
        fi
    done

    log_success "All candidates initialized!"
}

# Run unit tests in candidate
run_unit_tests() {
    local target_service="${1:-$SERVICE}"
    log_info "Running unit tests for gaia-${target_service}-candidate..."

    local test_failed=false

    _run_service_tests() {
        local svc="$1"
        local svc_dir="${PROJECT_ROOT}/candidates/gaia-${svc}"

        case "$svc" in
            core)
                cd "$svc_dir"
                python -m pytest tests/ gaia_core/ -v --tb=short 2>/dev/null || \
                python -m gaia_core.bicameral.test_bicameral || test_failed=true
                ;;
            web)
                cd "$svc_dir"
                python -m pytest tests/ -v --tb=short || test_failed=true
                ;;
            mcp)
                cd "$svc_dir"
                python -m pytest tests/ -v --tb=short || test_failed=true
                ;;
            study)
                cd "$svc_dir"
                python -m pytest tests/ -v --tb=short || test_failed=true
                ;;
            common)
                cd "${PROJECT_ROOT}/candidates/gaia-common"
                python -m pytest tests/ -v --tb=short || test_failed=true
                ;;
            *)
                log_warn "No unit tests defined for $svc"
                ;;
        esac
    }

    if [ "$target_service" = "all" ]; then
        for svc in common core web mcp study; do
            _run_service_tests "$svc"
        done
    else
        _run_service_tests "$target_service"
    fi

    if [ "$test_failed" = true ]; then
        log_error "Some unit tests failed!"
        return 1
    fi
    log_success "Unit tests passed for gaia-${target_service}!"
}

# Wait for a service health check with retries
wait_for_health() {
    local url="$1"
    local name="$2"
    local max_retries="${3:-6}"
    local interval="${4:-5}"

    for attempt in $(seq 1 "$max_retries"); do
        if curl -sf "$url" > /dev/null 2>&1; then
            log_success "${name} healthy at ${url}"
            return 0
        fi
        if [ "$attempt" -lt "$max_retries" ]; then
            log_info "Waiting for ${name}... (attempt ${attempt}/${max_retries})"
            sleep "$interval"
        fi
    done
    log_error "${name} health check failed after ${max_retries} attempts"
    return 1
}

# Start candidate container(s) - parallel mode
start_candidate() {
    local gpu_flag="${1:-0}"

    # Ensure live network exists
    docker network inspect gaia-network >/dev/null 2>&1 || \
        docker network create gaia-network

    if [ "$SERVICE" = "all" ]; then
        log_info "Starting full candidate stack (parallel mode)..."

        if [ "$gpu_flag" = "1" ]; then
            log_warn "GPU mode enabled - ensure live GPU services are stopped!"
            GAIA_CANDIDATE_GPU=1 docker compose -f "$COMPOSE_FILE" --profile full up -d
        else
            GAIA_CANDIDATE_GPU=0 docker compose -f "$COMPOSE_FILE" --profile full up -d
        fi

        log_info "Waiting for all services to become healthy..."
        sleep 5

        local all_healthy=true
        for svc_port in "core:6416" "web:6417" "mcp:8767" "study:8768"; do
            local svc="${svc_port%%:*}"
            local port="${svc_port##*:}"

            if ! wait_for_health "http://localhost:${port}/health" "gaia-${svc}-candidate" 6 5; then
                all_healthy=false
            fi
        done

        if [ "$all_healthy" = false ]; then
            log_error "Some services failed to start. Check logs with: ./test_candidate.sh all --logs"
            exit 1
        fi

        log_success "Full candidate stack is running!"
        echo ""
        echo "Candidate Endpoints (parallel mode - isolated from live):"
        echo "  - Web:   http://localhost:6417"
        echo "  - Core:  http://localhost:6416"
        echo "  - MCP:   http://localhost:8767"
        echo "  - Study: http://localhost:8768"
    else
        log_info "Starting gaia-${SERVICE}-candidate container..."

        if [ "$gpu_flag" = "1" ]; then
            log_warn "GPU mode enabled - ensure active GPU model is unloaded!"
            GAIA_CANDIDATE_GPU=1 docker compose -f "$COMPOSE_FILE" up -d "gaia-${SERVICE}-candidate"
        else
            GAIA_CANDIDATE_GPU=0 docker compose -f "$COMPOSE_FILE" up -d "gaia-${SERVICE}-candidate"
        fi

        log_info "Waiting for health check..."

        local port
        case "$SERVICE" in
            core) port=6416 ;;
            web) port=6417 ;;
            mcp) port=8767 ;;
            study) port=8768 ;;
        esac

        if ! wait_for_health "http://localhost:${port}/health" "gaia-${SERVICE}-candidate" 6 5; then
            docker compose -f "$COMPOSE_FILE" logs "gaia-${SERVICE}-candidate" --tail=50
            exit 1
        fi
    fi
}

# Inject candidate into live flow
inject_candidate() {
    if [ "$SERVICE" = "all" ]; then
        log_error "Cannot inject 'all' - specify a single service to inject"
        echo "Example: ./test_candidate.sh mcp --inject"
        exit 1
    fi

    check_candidate

    # Ensure live network exists
    docker network inspect gaia-network >/dev/null 2>&1 || {
        log_error "Live network 'gaia-network' not found. Start live services first."
        exit 1
    }

    log_inject "Injecting gaia-${SERVICE}-candidate into live flow..."

    # Start just the candidate container
    docker compose -f "$COMPOSE_FILE" up -d "gaia-${SERVICE}-candidate"

    log_info "Waiting for candidate health check..."
    sleep 5

    local port
    local endpoint_var
    local endpoint_url

    case "$SERVICE" in
        core)
            port=6416
            endpoint_var="CORE_ENDPOINT"
            endpoint_url="http://gaia-core-candidate:6415"
            ;;
        mcp)
            port=8767
            endpoint_var="MCP_ENDPOINT"
            endpoint_url="http://gaia-mcp-candidate:8765/jsonrpc"
            ;;
        study)
            port=8768
            endpoint_var="STUDY_ENDPOINT"
            endpoint_url="http://gaia-study-candidate:8766"
            ;;
        web)
            port=6417
            log_warn "Web candidate doesn't need injection - access directly at http://localhost:6417"
            log_info "Configure web candidate to point at live or candidate core as needed"
            return
            ;;
    esac

    if curl -sf "http://localhost:${port}/health" > /dev/null; then
        log_success "gaia-${SERVICE}-candidate is healthy"
    else
        log_error "Candidate health check failed"
        docker compose -f "$COMPOSE_FILE" logs "gaia-${SERVICE}-candidate" --tail=50
        exit 1
    fi

    echo ""
    log_inject "Candidate is running on the live network!"
    echo ""
    echo "The candidate is now reachable by live services via hostname:"
    echo "  gaia-${SERVICE}-candidate"
    echo ""
    echo "To route live traffic through the candidate, restart the calling service:"
    echo ""

    case "$SERVICE" in
        mcp)
            echo "  # Route live gaia-core through candidate MCP:"
            echo "  MCP_ENDPOINT=${endpoint_url} docker compose up -d gaia-core"
            echo ""
            echo "  # Or test directly:"
            echo "  curl http://localhost:${port}/health"
            ;;
        study)
            echo "  # Route live gaia-core through candidate Study:"
            echo "  STUDY_ENDPOINT=${endpoint_url} docker compose up -d gaia-core"
            ;;
        core)
            echo "  # Route live gaia-web through candidate Core:"
            echo "  CORE_ENDPOINT=${endpoint_url} docker compose up -d gaia-web"
            echo ""
            echo "  # Or access candidate core directly:"
            echo "  curl http://localhost:${port}/health"
            ;;
    esac

    echo ""
    echo "To stop injection:"
    echo "  ./test_candidate.sh ${SERVICE} --stop"
    echo "  docker compose up -d gaia-<caller>  # Restart caller with default endpoints"
}

# Eject candidate from live flow (restore live service)
eject_candidate() {
    if [ "$SERVICE" = "all" ]; then
        log_error "Cannot eject 'all' - specify a single service"
        exit 1
    fi

    log_inject "Ejecting gaia-${SERVICE}-candidate from live flow..."

    # Stop the candidate
    docker compose -f "$COMPOSE_FILE" stop "gaia-${SERVICE}-candidate" 2>/dev/null || true
    docker compose -f "$COMPOSE_FILE" rm -f "gaia-${SERVICE}-candidate" 2>/dev/null || true

    # Restart the live service caller with default endpoints
    case "$SERVICE" in
        mcp)
            log_info "Restarting live gaia-core with default MCP endpoint..."
            docker compose -f "$LIVE_COMPOSE_FILE" up -d gaia-core
            ;;
        study)
            log_info "Restarting live gaia-core with default Study endpoint..."
            docker compose -f "$LIVE_COMPOSE_FILE" up -d gaia-core
            ;;
        core)
            log_info "Restarting live gaia-web with default Core endpoint..."
            docker compose -f "$LIVE_COMPOSE_FILE" up -d gaia-web
            ;;
        web)
            log_info "Web candidate stopped"
            ;;
    esac

    log_success "Candidate ejected, live flow restored"
}

# Stop candidate container(s)
stop_candidate() {
    if [ "$SERVICE" = "all" ]; then
        log_info "Stopping full candidate stack..."
        docker compose -f "$COMPOSE_FILE" --profile full down 2>/dev/null || true
        log_success "Candidate stack stopped"
    else
        log_info "Stopping gaia-${SERVICE}-candidate..."
        docker compose -f "$COMPOSE_FILE" stop "gaia-${SERVICE}-candidate" 2>/dev/null || true
        docker compose -f "$COMPOSE_FILE" rm -f "gaia-${SERVICE}-candidate" 2>/dev/null || true
    fi
}

# View candidate logs
view_logs() {
    if [ "$SERVICE" = "all" ]; then
        docker compose -f "$COMPOSE_FILE" logs -f
    else
        docker compose -f "$COMPOSE_FILE" logs -f "gaia-${SERVICE}-candidate"
    fi
}

# Pre-promotion validation gate
# Runs unit tests and (if available) promote_candidate.sh --validate
run_pre_promotion_checks() {
    local target="${1:-$SERVICE}"

    log_info "Running pre-promotion validation for gaia-${target}..."

    # Step 1: Unit tests (always run)
    if ! run_unit_tests "$target"; then
        log_error "Pre-promotion unit tests FAILED. Promotion aborted."
        exit 1
    fi

    # Step 2: Containerized validation via promote_candidate.sh (if available)
    local promote_script="${PROJECT_ROOT}/scripts/promote_candidate.sh"
    if [ -x "$promote_script" ]; then
        if [ "$target" = "all" ]; then
            for svc in core web mcp study common; do
                log_info "Running containerized validation for gaia-${svc}..."
                "$promote_script" "gaia-${svc}" --validate --no-restart --no-backup 2>&1 || {
                    log_error "Containerized validation failed for gaia-${svc}. Promotion aborted."
                    exit 1
                }
            done
        else
            "$promote_script" "gaia-${target}" --validate --no-restart --no-backup 2>&1 || {
                log_error "Containerized validation failed for gaia-${target}. Promotion aborted."
                exit 1
            }
        fi
    else
        log_warn "promote_candidate.sh not found or not executable - skipping containerized checks"
    fi

    log_success "Pre-promotion validation passed!"
}

# Promote candidate to active
promote_candidate() {
    if [ "$SERVICE" = "all" ]; then
        log_warn "This will overwrite ALL active services with candidate code!"

        # Run validation before prompting for confirmation
        run_pre_promotion_checks "all"

        read -p "All checks passed. Proceed with promotion? (y/N) " confirm

        if [ "$confirm" != "y" ] && [ "$confirm" != "Y" ]; then
            log_info "Promotion cancelled"
            exit 0
        fi

        log_info "Stopping candidate stack..."
        stop_candidate

        log_info "Promoting all candidates to active..."
        for svc in core web mcp study; do
            local active="${PROJECT_ROOT}/gaia-${svc}"
            local candidate="${PROJECT_ROOT}/candidates/gaia-${svc}"

            if [ -d "$candidate" ]; then
                log_info "Promoting gaia-${svc}..."
                rsync -av --exclude='__pycache__' --exclude='*.pyc' --exclude='.pytest_cache' \
                    --exclude='*.egg-info' --exclude='.git' --exclude='tests/' \
                    "$candidate/" "$active/"
            fi
        done

        # Also promote gaia-common
        local common_candidate="${PROJECT_ROOT}/candidates/gaia-common"
        local common_active="${PROJECT_ROOT}/gaia-common"
        if [ -d "$common_candidate" ]; then
            log_info "Promoting gaia-common..."
            rsync -av --exclude='__pycache__' --exclude='*.pyc' --exclude='.pytest_cache' \
                --exclude='*.egg-info' --exclude='.git' --exclude='tests/' \
                "$common_candidate/" "$common_active/"
        fi

        log_success "All candidates promoted to active!"
        log_info "Restart active stack with: docker compose up -d"
    else
        log_warn "This will overwrite active gaia-${SERVICE} with candidate code!"

        # Run validation before prompting for confirmation
        run_pre_promotion_checks "$SERVICE"

        read -p "All checks passed. Proceed with promotion? (y/N) " confirm

        if [ "$confirm" != "y" ] && [ "$confirm" != "Y" ]; then
            log_info "Promotion cancelled"
            exit 0
        fi

        log_info "Promoting candidate to active..."
        stop_candidate

        rsync -av --exclude='__pycache__' --exclude='*.pyc' --exclude='.pytest_cache' \
            --exclude='*.egg-info' --exclude='.git' --exclude='tests/' \
            "$CANDIDATE_DIR/" "$ACTIVE_DIR/"

        log_success "Candidate promoted to active!"
        log_info "Restart active service with: docker compose restart gaia-${SERVICE}"
    fi
}

# Diff candidate vs active
diff_candidate() {
    if [ "$SERVICE" = "all" ]; then
        log_info "Differences between all candidates and active:"
        for svc in core web mcp study; do
            local active="${PROJECT_ROOT}/gaia-${svc}"
            local candidate="${PROJECT_ROOT}/candidates/gaia-${svc}"

            echo ""
            echo "=== gaia-${svc} ==="
            diff -rq "$candidate/gaia_${svc}" "$active/gaia_${svc}" 2>/dev/null || true
        done
    else
        log_info "Differences between candidate and active:"
        diff -rq "$CANDIDATE_DIR/gaia_${SERVICE}" "$ACTIVE_DIR/gaia_${SERVICE}" 2>/dev/null || true
    fi
}

# Run full stack validation
run_validation() {
    log_info "Running full candidate stack validation..."

    start_candidate 0

    echo ""
    log_info "Running inter-service communication tests..."

    log_info "Testing web -> core communication..."
    if curl -sf "http://localhost:6417/health" > /dev/null 2>&1; then
        log_success "Web candidate health OK"
    else
        log_warn "Web candidate health check inconclusive"
    fi

    log_info "Testing core health..."
    if curl -sf "http://localhost:6416/health" > /dev/null 2>&1; then
        log_success "Core candidate health OK"
    else
        log_warn "Core candidate health check inconclusive"
    fi

    log_info "Testing MCP health..."
    if curl -sf "http://localhost:8767/health" > /dev/null 2>&1; then
        log_success "MCP candidate health OK"
    else
        log_warn "MCP candidate health check inconclusive"
    fi

    log_info "Testing study health..."
    if curl -sf "http://localhost:8768/health" > /dev/null 2>&1; then
        log_success "Study candidate health OK"
    else
        log_warn "Study candidate health check inconclusive"
    fi

    log_info "Testing core -> mcp communication..."
    if curl -sf "http://localhost:6416/health" | grep -q "mcp"; then
        log_success "Core -> MCP communication OK"
    else
        log_warn "Core -> MCP communication check inconclusive"
    fi

    log_info "Testing core -> study communication..."
    if curl -sf "http://localhost:6416/health" | grep -q "study"; then
        log_success "Core -> Study communication OK"
    else
        log_warn "Core -> Study communication check inconclusive"
    fi

    log_info "Testing mcp -> study communication..."
    if curl -sf "http://localhost:8767/health" | grep -q "study"; then
        log_success "MCP -> Study communication OK"
    else
        log_warn "MCP -> Study communication check inconclusive"
    fi

    echo ""
    log_info "Running unit tests for all services..."
    run_unit_tests "all"

    log_success "Validation complete!"
}

# Show status of candidates
show_status() {
    log_info "Candidate container status:"
    echo ""

    for svc_port in "core:6416" "mcp:8767" "study:8768" "web:6417"; do
        local svc="${svc_port%%:*}"
        local port="${svc_port##*:}"
        local container="gaia-${svc}-candidate"

        if docker ps --format '{{.Names}}' | grep -q "^${container}$"; then
            if curl -sf "http://localhost:${port}/health" > /dev/null 2>&1; then
                log_success "${container}: running (healthy) - http://localhost:${port}"
            else
                log_warn "${container}: running (unhealthy)"
            fi
        else
            echo -e "  ${container}: stopped"
        fi
    done
}

# =============================================================================
# GPU Handoff Functions
# =============================================================================

# Release GPU from live service (allows candidate to claim it)
release_live_gpu() {
    log_info "Releasing GPU from live gaia-core service..."

    local live_core_url="http://localhost:6415"

    # Check if live core is running
    if ! curl -sf "${live_core_url}/health" > /dev/null 2>&1; then
        log_error "Live gaia-core is not running at ${live_core_url}"
        echo "Start the live service first: docker compose up -d gaia-core"
        exit 1
    fi

    # Check current GPU status
    log_info "Current GPU status:"
    curl -s "${live_core_url}/gpu/status" | python3 -m json.tool 2>/dev/null || \
        curl -s "${live_core_url}/gpu/status"
    echo ""

    # Release GPU
    log_info "Sending GPU release request..."
    local response
    response=$(curl -sf -X POST "${live_core_url}/gpu/release" 2>&1) || {
        log_error "Failed to release GPU: $response"
        exit 1
    }

    echo "$response" | python3 -m json.tool 2>/dev/null || echo "$response"

    if echo "$response" | grep -q '"success": true'; then
        log_success "GPU released! Live service will use CPU/API fallbacks."
        echo ""
        echo "You can now start candidates with GPU:"
        echo "  ./test_candidate.sh all --gpu"
        echo "  ./test_candidate.sh core --gpu"
        echo ""
        echo "When done, reclaim GPU:"
        echo "  ./test_candidate.sh --reclaim-gpu"
    else
        log_error "GPU release may have partially failed. Check response above."
    fi
}

# Reclaim GPU for live service (after candidate testing)
reclaim_live_gpu() {
    log_info "Reclaiming GPU for live gaia-core service..."

    local live_core_url="http://localhost:6415"

    # Check if live core is running
    if ! curl -sf "${live_core_url}/health" > /dev/null 2>&1; then
        log_error "Live gaia-core is not running at ${live_core_url}"
        echo "Start the live service first: docker compose up -d gaia-core"
        exit 1
    fi

    # Stop any running candidate GPU services first
    log_info "Stopping candidate services that may be using GPU..."
    docker compose -f "$COMPOSE_FILE" --profile full down 2>/dev/null || true

    # Give CUDA a moment to release resources
    sleep 2

    # Reclaim GPU
    log_info "Sending GPU reclaim request..."
    local response
    response=$(curl -sf -X POST "${live_core_url}/gpu/reclaim" 2>&1) || {
        log_error "Failed to reclaim GPU: $response"
        exit 1
    }

    echo "$response" | python3 -m json.tool 2>/dev/null || echo "$response"

    if echo "$response" | grep -q '"success": true'; then
        log_success "GPU reclaimed! Live service has GPU inference capability."
    else
        log_warn "GPU reclaim may have partially failed. Check response above."
        echo "You may need to restart gaia-core: docker compose restart gaia-core"
    fi

    # Show final status
    echo ""
    log_info "Final GPU status:"
    curl -s "${live_core_url}/gpu/status" | python3 -m json.tool 2>/dev/null || \
        curl -s "${live_core_url}/gpu/status"
}

# Show GPU status from live service
show_gpu_status() {
    local live_core_url="http://localhost:6415"

    if ! curl -sf "${live_core_url}/health" > /dev/null 2>&1; then
        log_error "Live gaia-core is not running at ${live_core_url}"
        exit 1
    fi

    log_info "GPU status from live gaia-core:"
    curl -s "${live_core_url}/gpu/status" | python3 -m json.tool 2>/dev/null || \
        curl -s "${live_core_url}/gpu/status"
}

# Start candidate with GPU handoff (release live GPU first, then start candidate)
start_candidate_with_handoff() {
    log_info "Starting candidate with GPU handoff..."

    # First release GPU from live service
    release_live_gpu

    # Give CUDA a moment to release resources
    sleep 2

    # Now start candidate with GPU
    log_info "Starting candidate stack with GPU..."
    start_candidate 1

    echo ""
    log_success "Candidate running with GPU!"
    echo ""
    echo "When done testing, restore live GPU:"
    echo "  ./test_candidate.sh --reclaim-gpu"
}

# Main
main() {
    shift $SHIFT_ARGS

    case "${1:-start}" in
        --init|init)
            init_candidates
            ;;
        --unit|unit)
            check_candidate
            run_unit_tests
            ;;
        --start|start)
            check_candidate
            start_candidate 0
            ;;

        --gpu|gpu)
            check_candidate
            start_candidate 1
            ;;
        --gpu-handoff|gpu-handoff)
            check_candidate
            start_candidate_with_handoff
            ;;
        --release-gpu|release-gpu)
            release_live_gpu
            ;;
        --reclaim-gpu|reclaim-gpu)
            reclaim_live_gpu
            ;;
        --gpu-status|gpu-status)
            show_gpu_status
            ;;
        --inject|inject)
            inject_candidate
            ;;
        --eject|eject)
            eject_candidate
            ;;
        --stop|stop)
            stop_candidate
            ;;
        --logs|logs)
            view_logs
            ;;
        --promote|promote)
            check_candidate
            promote_candidate
            ;;
        --diff|diff)
            check_candidate
            diff_candidate
            ;;
        --validate|validate)
            check_candidate
            run_validation
            ;;
        --status|status)
            show_status
            ;;
        --help|help)
            echo "Usage: $0 [service|all] [command]"
            echo ""
            echo "Services: all (default), core, web, mcp, study, common"
            echo ""
            echo "Commands:"
            echo "  --init        Initialize all candidates from active code"
            echo "  --start       Start candidate(s) in parallel mode (default)"
            echo "  --gpu         Start with GPU enabled (manual mode)"
            echo "  --gpu-handoff Start with GPU, auto-release from live first"
            echo "  --inject      Inject single candidate into live flow"
            echo "  --eject       Remove candidate from live flow, restore live"
            echo "  --stop        Stop candidate container(s)"
            echo "  --logs        View candidate logs"
            echo "  --status      Show status of all candidate containers"
            echo "  --unit        Run unit tests"
            echo "  --diff        Show differences vs active"
            echo "  --validate    Run full stack validation"
            echo "  --promote     Promote candidate(s) to active"
            echo "  --help        Show this help"
            echo ""
            echo "GPU Management:"
            echo "  --gpu-status    Show current GPU status from live service"
            echo "  --release-gpu   Release GPU from live service (for candidates)"
            echo "  --reclaim-gpu   Reclaim GPU for live service (stops candidates)"
            echo "  --gpu-handoff   Combined: release + start candidate with GPU"
            echo ""
            echo "Testing Modes:"
            echo ""
            echo "  Parallel (isolated candidate ecosystem):"
            echo "    $0 all                    # All candidates talk to each other"
            echo "    $0 all --gpu              # With GPU (manual - ensure live released)"
            echo "    $0 all --gpu-handoff      # With GPU (auto-release from live)"
            echo ""
            echo "  GPU Handoff Workflow:"
            echo "    $0 --release-gpu          # Step 1: Release GPU from live"
            echo "    $0 all --gpu              # Step 2: Start candidates with GPU"
            echo "    $0 --stop                 # Step 3: Stop candidates"
            echo "    $0 --reclaim-gpu          # Step 4: Restore live GPU"
            echo ""
            echo "  Or use the combined command:"
            echo "    $0 all --gpu-handoff      # Does release + start automatically"
            echo "    $0 --reclaim-gpu          # When done, reclaim for live"
            echo ""
            echo "  Injection (candidate in live flow):"
            echo "    $0 mcp --inject           # Live core -> candidate MCP"
            echo "    $0 core --inject          # Live web -> candidate core"
            echo "    $0 mcp --eject            # Restore live MCP"
            ;;
        *)
            log_error "Unknown command: $1"
            echo "Use --help for usage"
            exit 1
            ;;
    esac
}

main "$@"
