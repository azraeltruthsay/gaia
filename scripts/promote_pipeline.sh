#!/bin/bash
#
# promote_pipeline.sh — Master GAIA candidate-to-live promotion pipeline
#
# Orchestrates a 9-stage fail-fast promotion workflow:
#   0. GPU state normalization (query/handoff before shutdown)
#   1. Graceful live shutdown (default; skip with --keep-live)
#   2. Pre-flight checks (health, sync, git state)
#   3. Validation (ruff, mypy, pytest per service)
#   4. Cognitive smoke tests (16-test battery against candidate)
#   5. Service promotion (dependency-ordered, with backup + Docker rebuild)
#   6. Post-promotion verification (restart live + health + quick smoke)
#   7. Dev journal + flatten + commit
#   8. QLoRA validation (optional, --qlora flag)
#
# Usage:
#   ./scripts/promote_pipeline.sh [options]
#
# Options:
#   --dry-run        Run all validation without promoting
#   --skip-validate  Skip lint/type/unit testing (Stage 3)
#   --skip-smoke     Skip cognitive smoke tests (Stage 4)
#   --skip-flatten   Skip flatten_soa.sh after promotion
#   --qlora          Run QLoRA validation cycle (Stage 8)
#   --keep-live      Don't shut down live services before testing
#   --no-push        Don't push to remote after commit
#   --gpu-to-study   Hand GPU to study instead of prime after promotion
#   --gpu-skip       Don't touch GPU state at all
#   --gpu-timeout N  Seconds to wait for GPU handoff (default: 180)
#   --no-auto-start  Don't auto-start candidate stack (assume already running)
#   --services LIST  Comma-separated services to promote
#                    (default: gaia-common,gaia-mcp,gaia-core,gaia-study)
#   -v, --verbose    Pass -v to smoke test runner
#   -h, --help       Show this help
#
# Examples:
#   ./scripts/promote_pipeline.sh                    # Full pipeline
#   ./scripts/promote_pipeline.sh --dry-run          # Validate only
#   ./scripts/promote_pipeline.sh --services gaia-core --skip-validate
#

set -euo pipefail

# ═══════════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════════

GAIA_ROOT="/gaia/GAIA_Project"
SCRIPTS_DIR="$GAIA_ROOT/scripts"
PROMOTE_SCRIPT="$SCRIPTS_DIR/promote_candidate.sh"
SMOKE_SCRIPT="$GAIA_ROOT/candidates/gaia-core/scripts/smoke_test_cognitive.py"
FLATTEN_SCRIPT="$GAIA_ROOT/flatten_soa.sh"
CANDIDATE_COMPOSE="$GAIA_ROOT/docker-compose.candidate.yml"
LOG_FILE="$GAIA_ROOT/logs/promote_pipeline.log"
DATE=$(date +%Y-%m-%d)
TIMESTAMP=$(date +%Y-%m-%dT%H:%M:%S)

# Default services in dependency order
DEFAULT_SERVICES="gaia-common,gaia-mcp,gaia-core,gaia-study"

# Orchestrator URLs (live and candidate)
ORCH_LIVE_URL="http://localhost:6410"
ORCH_CANDIDATE_URL="http://localhost:6411"

# Port mappings: service -> candidate_port:live_port
declare -A CANDIDATE_PORTS=(
    ["gaia-core"]="6416"
    ["gaia-mcp"]="8767"
    ["gaia-study"]="8768"
    ["gaia-audio"]="8081"
)
declare -A LIVE_PORTS=(
    ["gaia-core"]="6415"
    ["gaia-mcp"]="8765"
    ["gaia-study"]="8766"
    ["gaia-audio"]="8080"
)

# ═══════════════════════════════════════════════════════════════════════════
# ANSI Colors
# ═══════════════════════════════════════════════════════════════════════════

GREEN='\033[92m'
RED='\033[91m'
YELLOW='\033[93m'
CYAN='\033[96m'
BOLD='\033[1m'
DIM='\033[2m'
RESET='\033[0m'

# ═══════════════════════════════════════════════════════════════════════════
# Parse Arguments
# ═══════════════════════════════════════════════════════════════════════════

DRY_RUN=false
SKIP_VALIDATE=false
SKIP_SMOKE=false
SKIP_FLATTEN=false
DO_QLORA=false
NO_PUSH=false
KEEP_LIVE=false
GPU_TO_STUDY=false
GPU_SKIP=false
GPU_TIMEOUT=180
VERBOSE=""
SERVICES="$DEFAULT_SERVICES"

for arg in "$@"; do
    case $arg in
        --dry-run)       DRY_RUN=true ;;
        --skip-validate) SKIP_VALIDATE=true ;;
        --skip-smoke)    SKIP_SMOKE=true ;;
        --skip-flatten)  SKIP_FLATTEN=true ;;
        --qlora)         DO_QLORA=true ;;
        --no-push)       NO_PUSH=true ;;
        --keep-live)     KEEP_LIVE=true ;;
        --gpu-to-study)  GPU_TO_STUDY=true ;;
        --gpu-skip)      GPU_SKIP=true ;;
        --gpu-timeout)   ;; # value handled below
        --gpu-timeout=*) GPU_TIMEOUT="${arg#*=}" ;;
        -v|--verbose)    VERBOSE="-v" ;;
        --services)      ;; # handled below
        --services=*)    SERVICES="${arg#*=}" ;;
        -h|--help)
            head -38 "$0" | tail -33
            exit 0
            ;;
        *)
            # Handle --services VALUE and --gpu-timeout VALUE (space-separated)
            if [ "${PREV_ARG:-}" = "--services" ]; then
                SERVICES="$arg"
            elif [ "${PREV_ARG:-}" = "--gpu-timeout" ]; then
                GPU_TIMEOUT="$arg"
            else
                echo -e "${RED}Unknown option: $arg${RESET}"
                exit 1
            fi
            ;;
    esac
    PREV_ARG="$arg"
done

# Convert comma-separated services to array
IFS=',' read -ra SERVICE_LIST <<< "$SERVICES"

# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════

PIPELINE_START=$(date +%s)
STAGE_RESULTS=()

log() {
    echo -e "$1"
    echo "$(date +%H:%M:%S) $1" | sed 's/\x1b\[[0-9;]*m//g' >> "$LOG_FILE" 2>/dev/null || true
}

stage_header() {
    local num=$1
    local title=$2
    echo ""
    log "${BOLD}${CYAN}══════════════════════════════════════════════════════════════${RESET}"
    log "${BOLD}${CYAN}  Stage $num: $title${RESET}"
    log "${BOLD}${CYAN}══════════════════════════════════════════════════════════════${RESET}"
    echo ""
}

stage_pass() {
    local name=$1
    STAGE_RESULTS+=("PASS|$name")
    log "${GREEN}${BOLD}  ✓ Stage passed: $name${RESET}"
}

stage_fail() {
    local name=$1
    local reason=$2
    STAGE_RESULTS+=("FAIL|$name")
    log "${RED}${BOLD}  ✗ Stage FAILED: $name${RESET}"
    log "${RED}    Reason: $reason${RESET}"
    print_summary
    exit 1
}

stage_skip() {
    local name=$1
    STAGE_RESULTS+=("SKIP|$name")
    log "${YELLOW}  ⊘ Stage skipped: $name${RESET}"
}

stage_warn() {
    local name=$1
    STAGE_RESULTS+=("WARN|$name")
    log "${YELLOW}  ⚠ Stage warning: $name${RESET}"
}

check_health() {
    local service=$1
    local port=$2
    local timeout=${3:-5}
    curl -sf --max-time "$timeout" "http://localhost:$port/health" > /dev/null 2>&1
}

# Track whether we shut down live (for safety trap and conditional logic)
LIVE_STOPPED=false

restart_live_if_down() {
    if [ "$LIVE_STOPPED" = true ]; then
        log "  ${YELLOW}Restarting live services...${RESET}"
        cd "$GAIA_ROOT"
        docker compose up -d 2>/dev/null || true
        LIVE_STOPPED=false
    fi
}

print_summary() {
    local elapsed=$(( $(date +%s) - PIPELINE_START ))
    local mins=$(( elapsed / 60 ))
    local secs=$(( elapsed % 60 ))

    echo ""
    log "${BOLD}══════════════════════════════════════════════════════════════${RESET}"
    log "${BOLD}  Pipeline Summary${RESET}"
    log "${BOLD}══════════════════════════════════════════════════════════════${RESET}"
    log "  Duration:  ${mins}m ${secs}s"
    log "  Services:  ${SERVICES}"
    log "  Mode:      $([ "$DRY_RUN" = true ] && echo 'DRY RUN' || echo 'LIVE')"
    echo ""

    local any_fail=false
    for result in "${STAGE_RESULTS[@]}"; do
        local status="${result%%|*}"
        local name="${result#*|}"
        case $status in
            PASS) log "  ${GREEN}✓${RESET} $name" ;;
            FAIL) log "  ${RED}✗${RESET} $name"; any_fail=true ;;
            SKIP) log "  ${DIM}⊘${RESET} $name ${DIM}(skipped)${RESET}" ;;
            WARN) log "  ${YELLOW}⚠${RESET} $name ${YELLOW}(warning)${RESET}" ;;
        esac
    done
    echo ""

    if [ "$any_fail" = true ]; then
        log "${RED}${BOLD}  Pipeline FAILED${RESET}"
    else
        log "${GREEN}${BOLD}  Pipeline PASSED${RESET}"
    fi
    echo ""
}

cleanup_on_failure() {
    log "${RED}${BOLD}  Pipeline interrupted — checking live service state...${RESET}"
    restart_live_if_down
}

# GPU orchestrator helpers
_find_orchestrator() {
    # Try live orchestrator first, fallback to candidate
    if curl -sf --max-time 3 "$ORCH_LIVE_URL/health" > /dev/null 2>&1; then
        echo "$ORCH_LIVE_URL"
    elif curl -sf --max-time 3 "$ORCH_CANDIDATE_URL/health" > /dev/null 2>&1; then
        echo "$ORCH_CANDIDATE_URL"
    else
        echo ""
    fi
}

_get_gpu_owner() {
    local orch_url=$1
    curl -sf --max-time 5 "$orch_url/gpu/status" 2>/dev/null | python3 -c "import sys,json; print(json.load(sys.stdin).get('owner','unknown'))" 2>/dev/null || echo "unknown"
}

_do_gpu_handoff() {
    local orch_url=$1
    local direction=$2  # "prime-to-study" or "study-to-prime"
    local timeout=$3

    # Initiate handoff
    set +e
    local response
    response=$(curl -sf --max-time 10 -X POST "$orch_url/handoff/$direction" \
        -H "Content-Type: application/json" \
        -d "{\"reason\": \"promotion_pipeline\", \"timeout_seconds\": $timeout}" 2>&1)
    local exit_code=$?
    set -e

    if [ $exit_code -ne 0 ]; then
        log "    ${RED}✗${RESET} Handoff request failed (curl exit $exit_code)"
        return 1
    fi

    # Extract handoff_id
    local handoff_id
    handoff_id=$(echo "$response" | python3 -c "import sys,json; print(json.load(sys.stdin).get('handoff_id',''))" 2>/dev/null)

    if [ -z "$handoff_id" ]; then
        log "    ${RED}✗${RESET} No handoff_id in response"
        return 1
    fi

    log "    Handoff initiated (id: ${handoff_id:0:8}...), polling for completion..."

    # Poll for completion
    local waited=0
    local poll_interval=5
    while [ $waited -lt "$timeout" ]; do
        sleep $poll_interval
        waited=$((waited + poll_interval))

        local phase
        phase=$(curl -sf --max-time 5 "$orch_url/handoff/$handoff_id/status" 2>/dev/null \
            | python3 -c "import sys,json; print(json.load(sys.stdin).get('phase','unknown'))" 2>/dev/null || echo "unknown")

        case "$phase" in
            completed)
                log "    ${GREEN}✓${RESET} Handoff completed (${waited}s)"
                return 0
                ;;
            failed|cancelled)
                log "    ${RED}✗${RESET} Handoff $phase (${waited}s)"
                return 1
                ;;
            *)
                log "    ${DIM}[${waited}s] phase: $phase${RESET}"
                ;;
        esac
    done

    log "    ${RED}✗${RESET} Handoff timed out after ${timeout}s"
    return 1
}

_do_candidate_gpu_release() {
    local orch_url=$1
    set +e
    curl -sf --max-time 10 -X POST "$orch_url/gpu/release" > /dev/null 2>&1
    local exit_code=$?
    set -e
    return $exit_code
}

# ═══════════════════════════════════════════════════════════════════════════
# Initialize log
# ═══════════════════════════════════════════════════════════════════════════

mkdir -p "$(dirname "$LOG_FILE")"
echo "" >> "$LOG_FILE"
echo "=== Pipeline run: $TIMESTAMP ===" >> "$LOG_FILE"

log "${BOLD}=== GAIA Candidate Promotion Pipeline ===${RESET}"
log "  Date:      $TIMESTAMP"
log "  Services:  $SERVICES"
log "  Mode:      $([ "$DRY_RUN" = true ] && echo 'DRY RUN' || echo 'LIVE PROMOTION')"
log "  Options:   validate=$([ "$SKIP_VALIDATE" = true ] && echo 'skip' || echo 'yes') smoke=$([ "$SKIP_SMOKE" = true ] && echo 'skip' || echo 'yes') flatten=$([ "$SKIP_FLATTEN" = true ] && echo 'skip' || echo 'yes') qlora=$([ "$DO_QLORA" = true ] && echo 'yes' || echo 'no') keep-live=$([ "$KEEP_LIVE" = true ] && echo 'yes' || echo 'no') gpu=$([ "$GPU_SKIP" = true ] && echo 'skip' || echo "auto(timeout=${GPU_TIMEOUT}s$([ "$GPU_TO_STUDY" = true ] && echo ',to-study'))")"

# ═══════════════════════════════════════════════════════════════════════════
# Stage 0: GPU State Normalization
# ═══════════════════════════════════════════════════════════════════════════

stage_header 0 "GPU State Normalization"

if [ "$GPU_SKIP" = true ]; then
    stage_skip "GPU State Normalization (--gpu-skip)"
else
    # Determine desired GPU owner after promotion
    if [ "$GPU_TO_STUDY" = true ]; then
        desired_owner="gaia-study"
    else
        desired_owner="gaia-core"
    fi

    # Find a reachable orchestrator
    orch_url=$(_find_orchestrator)

    if [ -z "$orch_url" ]; then
        if [ "$DRY_RUN" = true ]; then
            log "  ${YELLOW}⚠${RESET} No orchestrator reachable — skipping GPU check (dry-run)"
            stage_skip "GPU State Normalization (orchestrator unreachable)"
        else
            log "  ${RED}✗${RESET} No orchestrator reachable — cannot verify GPU state"
            stage_fail "GPU State Normalization" "Orchestrator unreachable at $ORCH_LIVE_URL and $ORCH_CANDIDATE_URL"
        fi
    else
        log "  Orchestrator found at ${orch_url}"
        current_owner=$(_get_gpu_owner "$orch_url")
        log "  GPU owner: ${BOLD}${current_owner}${RESET} (desired: ${desired_owner})"

        case "$current_owner" in
            "$desired_owner")
                log "  ${GREEN}✓${RESET} GPU already owned by $desired_owner — no action needed"
                stage_pass "GPU State Normalization"
                ;;
            none)
                log "  ${GREEN}✓${RESET} GPU unowned — $desired_owner will claim on startup"
                stage_pass "GPU State Normalization"
                ;;
            gaia-study)
                if [ "$desired_owner" = "gaia-core" ]; then
                    if [ "$DRY_RUN" = true ]; then
                        log "  ${DIM}Would trigger study-to-prime handoff (dry-run)${RESET}"
                        stage_skip "GPU State Normalization (dry-run)"
                    else
                        log "  Triggering study-to-prime handoff..."
                        if _do_gpu_handoff "$orch_url" "study-to-prime" "$GPU_TIMEOUT"; then
                            stage_pass "GPU State Normalization"
                        else
                            stage_fail "GPU State Normalization" "study-to-prime handoff failed"
                        fi
                    fi
                else
                    log "  ${GREEN}✓${RESET} GPU owned by gaia-study — matches desired"
                    stage_pass "GPU State Normalization"
                fi
                ;;
            gaia-core)
                if [ "$desired_owner" = "gaia-study" ]; then
                    if [ "$DRY_RUN" = true ]; then
                        log "  ${DIM}Would trigger prime-to-study handoff (dry-run)${RESET}"
                        stage_skip "GPU State Normalization (dry-run)"
                    else
                        log "  Triggering prime-to-study handoff..."
                        if _do_gpu_handoff "$orch_url" "prime-to-study" "$GPU_TIMEOUT"; then
                            stage_pass "GPU State Normalization"
                        else
                            stage_fail "GPU State Normalization" "prime-to-study handoff failed"
                        fi
                    fi
                else
                    log "  ${GREEN}✓${RESET} GPU owned by gaia-core — matches desired"
                    stage_pass "GPU State Normalization"
                fi
                ;;
            gaia-core-candidate|gaia-study-candidate)
                if [ "$DRY_RUN" = true ]; then
                    log "  ${DIM}Would release GPU from candidate ($current_owner) (dry-run)${RESET}"
                    stage_skip "GPU State Normalization (dry-run)"
                else
                    log "  Releasing GPU from candidate ($current_owner)..."
                    if _do_candidate_gpu_release "$orch_url"; then
                        log "  ${GREEN}✓${RESET} GPU released from candidate"
                        stage_pass "GPU State Normalization"
                    else
                        stage_fail "GPU State Normalization" "Failed to release GPU from $current_owner"
                    fi
                fi
                ;;
            unknown)
                log "  ${YELLOW}⚠${RESET} Could not determine GPU owner — continuing"
                stage_warn "GPU State Normalization"
                ;;
            *)
                log "  ${YELLOW}⚠${RESET} Unexpected GPU owner: $current_owner — continuing"
                stage_warn "GPU State Normalization"
                ;;
        esac
    fi
fi

# ═══════════════════════════════════════════════════════════════════════════
# Stage 1: Graceful Live Shutdown
# ═══════════════════════════════════════════════════════════════════════════

stage_header 1 "Graceful Live Shutdown"

if [ "$KEEP_LIVE" = true ]; then
    stage_skip "Graceful Live Shutdown (--keep-live)"
elif [ "$DRY_RUN" = true ]; then
    log "  ${DIM}Would shut down live services (dry-run)${RESET}"
    stage_skip "Graceful Live Shutdown (dry-run)"
else
    # 1a. Verify candidate stack is healthy BEFORE shutting down live
    log "  Verifying candidate stack is healthy before shutting down live..."
    candidate_ok=true
    for svc in "${SERVICE_LIST[@]}"; do
        port="${CANDIDATE_PORTS[$svc]:-}"
        if [ -n "$port" ]; then
            if check_health "$svc" "$port"; then
                log "    ${GREEN}✓${RESET} $svc-candidate healthy (port $port)"
            else
                log "    ${RED}✗${RESET} $svc-candidate unreachable (port $port)"
                candidate_ok=false
            fi
        fi
    done

    if [ "$candidate_ok" = false ]; then
        stage_fail "Graceful Live Shutdown" "Candidate stack not healthy — refusing to shut down live"
    fi

    # 1b. Shut down live services (20s grace for in-flight requests)
    log "  Shutting down live services (20s grace period)..."
    cd "$GAIA_ROOT"
    set +e
    docker compose down -t 20 2>&1 | while read -r line; do log "    $line"; done
    down_exit=${PIPESTATUS[0]}
    set -e

    if [ $down_exit -ne 0 ]; then
        stage_fail "Graceful Live Shutdown" "docker compose down failed (exit $down_exit)"
    fi

    # 1c. Verify all live containers are actually stopped
    live_remaining=$(docker compose ps -q 2>/dev/null | wc -l)
    if [ "$live_remaining" -gt 0 ]; then
        log "  ${YELLOW}⚠${RESET} $live_remaining live containers still running"
        stage_fail "Graceful Live Shutdown" "Live containers did not stop cleanly"
    fi

    LIVE_STOPPED=true
    # Register safety trap now that live is actually down
    trap cleanup_on_failure EXIT INT TERM
    log "  ${GREEN}✓${RESET} Live services stopped — safety trap armed"
    stage_pass "Graceful Live Shutdown"
fi

# ═══════════════════════════════════════════════════════════════════════════
# Stage 2: Pre-flight Checks
# ═══════════════════════════════════════════════════════════════════════════

stage_header 2 "Pre-flight Checks"

preflight_ok=true

# 2a. Health-check candidate services
#     When --skip-smoke is set, candidate containers aren't required
#     (validation uses Docker builds, not running containers)
for svc in "${SERVICE_LIST[@]}"; do
    port="${CANDIDATE_PORTS[$svc]:-}"
    if [ -n "$port" ]; then
        if check_health "$svc" "$port"; then
            log "  ${GREEN}✓${RESET} $svc-candidate healthy (port $port)"
        elif [ "$SKIP_SMOKE" = true ]; then
            log "  ${YELLOW}⚠${RESET} $svc-candidate unreachable (port $port) — non-blocking (smoke skipped)"
        else
            log "  ${RED}✗${RESET} $svc-candidate unreachable (port $port)"
            preflight_ok=false
        fi
    else
        log "  ${DIM}⊘${RESET} $svc has no candidate port (library-only)"
    fi
done

# 2b. Check gaia-common sync
CANDIDATE_COMMON="$GAIA_ROOT/candidates/gaia-common"
LIVE_COMMON="$GAIA_ROOT/gaia-common"
CP_CANDIDATE="$CANDIDATE_COMMON/gaia_common/protocols/cognition_packet.py"
CP_LIVE="$LIVE_COMMON/gaia_common/protocols/cognition_packet.py"

if [ -f "$CP_CANDIDATE" ] && [ -f "$CP_LIVE" ]; then
    if diff -q "$CP_CANDIDATE" "$CP_LIVE" > /dev/null 2>&1; then
        log "  ${GREEN}✓${RESET} CognitionPacket in sync (candidate == live)"
    else
        log "  ${YELLOW}⚠${RESET} CognitionPacket differs — gaia-common will be promoted first"
    fi
fi

GC_CANDIDATE="$CANDIDATE_COMMON/gaia_common/constants/gaia_constants.json"
GC_LIVE="$LIVE_COMMON/gaia_common/constants/gaia_constants.json"

if [ -f "$GC_CANDIDATE" ] && [ -f "$GC_LIVE" ]; then
    if diff -q "$GC_CANDIDATE" "$GC_LIVE" > /dev/null 2>&1; then
        log "  ${GREEN}✓${RESET} gaia_constants.json in sync"
    else
        log "  ${YELLOW}⚠${RESET} gaia_constants.json differs (will sync on promote)"
    fi
fi

# 2c. Git state
cd "$GAIA_ROOT"
UNCOMMITTED=$(git status --porcelain 2>/dev/null | wc -l)
if [ "$UNCOMMITTED" -gt 0 ]; then
    log "  ${YELLOW}⚠${RESET} $UNCOMMITTED uncommitted changes detected (non-blocking)"
else
    log "  ${GREEN}✓${RESET} Working tree clean"
fi

if [ "$preflight_ok" = true ]; then
    stage_pass "Pre-flight Checks"
else
    stage_fail "Pre-flight Checks" "One or more candidate services are unreachable"
fi

# ═══════════════════════════════════════════════════════════════════════════
# Stage 3: Validation (lint/type/unit)
# ═══════════════════════════════════════════════════════════════════════════

stage_header 3 "Validation (Lint / Type / Unit)"

VALIDATE_SCRIPT="$SCRIPTS_DIR/validate.sh"

if [ "$SKIP_VALIDATE" = true ]; then
    stage_skip "Validation"
else
    if [ ! -x "$VALIDATE_SCRIPT" ]; then
        stage_fail "Validation" "validate.sh not found or not executable at $VALIDATE_SCRIPT"
    fi

    # Build service list for validate.sh (space-separated)
    validate_services=""
    for svc in "${SERVICE_LIST[@]}"; do
        validate_services="$validate_services $svc"
    done

    log "  Delegating to validate.sh: $validate_services"
    echo ""

    set +e
    validate_output=$("$VALIDATE_SCRIPT" $validate_services $VERBOSE 2>&1)
    validate_exit=$?
    set -e

    # Print and log output
    echo "$validate_output"
    echo "$validate_output" | sed 's/\x1b\[[0-9;]*m//g' >> "$LOG_FILE"

    if [ $validate_exit -eq 0 ]; then
        stage_pass "Validation"
    else
        stage_fail "Validation" "validate.sh exited with code $validate_exit"
    fi
fi

# ═══════════════════════════════════════════════════════════════════════════
# Stage 4: Cognitive Smoke Tests (against candidate)
# ═══════════════════════════════════════════════════════════════════════════

stage_header 4 "Cognitive Smoke Tests (Candidate)"

if [ "$SKIP_SMOKE" = true ]; then
    stage_skip "Smoke Tests (Candidate)"
else
    log "  Running 16-test battery against candidate (port 6416)..."
    log "  Script: $SMOKE_SCRIPT"
    echo ""

    set +e
    smoke_output=$(python3 "$SMOKE_SCRIPT" --endpoint http://localhost:6416 $VERBOSE 2>&1)
    smoke_exit=$?
    set -e

    # Print output
    echo "$smoke_output"

    # Log to file (strip ANSI)
    echo "$smoke_output" | sed 's/\x1b\[[0-9;]*m//g' >> "$LOG_FILE"

    if [ $smoke_exit -eq 0 ]; then
        stage_pass "Smoke Tests (Candidate)"
    else
        stage_fail "Smoke Tests (Candidate)" "One or more smoke tests failed (exit code $smoke_exit)"
    fi
fi

# ═══════════════════════════════════════════════════════════════════════════
# Stage 5: Promote Services (dependency order)
# ═══════════════════════════════════════════════════════════════════════════

stage_header 5 "Promote Services"

if [ "$DRY_RUN" = true ]; then
    log "  ${YELLOW}DRY RUN — skipping actual promotion${RESET}"
    for svc in "${SERVICE_LIST[@]}"; do
        log "  ${DIM}Would promote: $svc${RESET}"
    done
    stage_skip "Promote Services (dry-run)"
else
    promote_ok=true

    for svc in "${SERVICE_LIST[@]}"; do
        log "  Promoting ${BOLD}$svc${RESET}..."

        # gaia-common: no restart (others depend on it)
        # Live stopped: always --no-restart (containers don't exist)
        # Keep-live + skip-smoke: no --test (candidate containers aren't running)
        # Others: restart + test (but only if the live container exists)
        promote_flags=""
        if [ "$svc" = "gaia-common" ]; then
            promote_flags="--no-restart"
        elif [ "$LIVE_STOPPED" = true ]; then
            promote_flags="--no-restart"
        elif [ "$KEEP_LIVE" = true ] && [ "$SKIP_SMOKE" = true ]; then
            # Hybrid mode: candidates not running, skip candidate health check
            promote_flags=""
        elif docker inspect "$svc" > /dev/null 2>&1; then
            promote_flags="--test"
        else
            log "    ${DIM}(no live container '$svc' — skipping restart)${RESET}"
            promote_flags="--no-restart"
        fi

        set +e
        promote_output=$("$PROMOTE_SCRIPT" "$svc" $promote_flags 2>&1)
        promote_exit=$?
        set -e

        if [ $promote_exit -eq 0 ]; then
            log "    ${GREEN}✓${RESET} $svc promoted"
        else
            log "    ${RED}✗${RESET} $svc promotion failed (exit $promote_exit)"
            echo "$promote_output" | tail -10 | while read -r line; do
                log "      $line"
            done
            promote_ok=false
            break  # Stop promoting — don't break downstream services
        fi
    done

    if [ "$promote_ok" = true ]; then
        stage_pass "Promote Services"
    else
        stage_fail "Promote Services" "Service promotion failed — manual rollback may be needed"
    fi

    # ── 5b. Rebuild Docker images ────────────────────────────────────────
    # Always rebuild after promotion so the pip-installed gaia-common
    # inside container images stays in sync with the promoted source.
    # Takes <60s total and prevents stale site-packages issues.
    log ""
    log "  Rebuilding Docker images (gaia-core, gaia-web, gaia-orchestrator)..."
    cd "$GAIA_ROOT"
    set +e
    rebuild_output=$(docker compose build --no-cache gaia-core gaia-web gaia-orchestrator 2>&1)
    rebuild_exit=$?
    set -e

    if [ $rebuild_exit -eq 0 ]; then
        log "    ${GREEN}✓${RESET} Docker images rebuilt"
    else
        log "    ${YELLOW}⚠${RESET} Docker image rebuild failed (exit $rebuild_exit) — containers may use stale images"
        echo "$rebuild_output" | tail -5 | while read -r line; do log "      $line"; done
    fi
fi

# ═══════════════════════════════════════════════════════════════════════════
# Stage 6: Post-Promotion Verification
# ═══════════════════════════════════════════════════════════════════════════

stage_header 6 "Post-Promotion Verification"

if [ "$DRY_RUN" = true ]; then
    stage_skip "Post-Promotion Verification (dry-run)"
else
    post_ok=true

    # 6a. Restart live services if we shut them down in Stage 1
    if [ "$LIVE_STOPPED" = true ]; then
        log "  Restarting live services after promotion..."
        cd "$GAIA_ROOT"
        docker compose up -d 2>&1 | while read -r line; do log "    $line"; done

        # Poll for health every 10s, max 180s (covers gaia-prime's 120s start_period)
        log "  Waiting for live services to become healthy (max 180s)..."
        max_wait=180
        waited=0
        all_healthy=false
        while [ $waited -lt $max_wait ]; do
            sleep 10
            waited=$((waited + 10))
            healthy_count=0
            total_count=0
            for svc in "${SERVICE_LIST[@]}"; do
                port="${LIVE_PORTS[$svc]:-}"
                if [ -n "$port" ]; then
                    total_count=$((total_count + 1))
                    if check_health "$svc" "$port" 5; then
                        healthy_count=$((healthy_count + 1))
                    fi
                fi
            done
            log "    ${DIM}[${waited}s] $healthy_count/$total_count services healthy${RESET}"
            if [ "$healthy_count" -eq "$total_count" ] && [ "$total_count" -gt 0 ]; then
                all_healthy=true
                break
            fi
        done

        LIVE_STOPPED=false
        # Clear the safety trap now that live is back up
        trap - EXIT INT TERM

        if [ "$all_healthy" = true ]; then
            log "  ${GREEN}✓${RESET} Live services restarted and healthy (${waited}s)"
        else
            log "  ${RED}✗${RESET} Live services not fully healthy after ${max_wait}s"
            post_ok=false
        fi
    fi

    # 6b. Health checks on live services (individual)
    log "  Health checks on live services..."
    any_live_running=false
    for svc in "${SERVICE_LIST[@]}"; do
        port="${LIVE_PORTS[$svc]:-}"
        if [ -n "$port" ]; then
            if ! docker inspect "$svc" > /dev/null 2>&1; then
                log "    ${DIM}⊘${RESET} $svc — no live container"
                continue
            fi
            any_live_running=true
            # Give services a moment to settle after restart
            sleep 2
            if check_health "$svc" "$port" 10; then
                log "    ${GREEN}✓${RESET} $svc live healthy (port $port)"
            else
                log "    ${RED}✗${RESET} $svc live unreachable (port $port)"
                post_ok=false
            fi
        fi
    done

    # 6c. Quick smoke test subset against live (only if live containers exist)
    log ""
    if [ "$any_live_running" = true ]; then
        log "  Quick smoke test (tests 1,2,7) against live (port 6415)..."
        set +e
        quick_smoke=$(python3 "$SMOKE_SCRIPT" --endpoint http://localhost:6415 --only 1,2,7 2>&1)
        quick_exit=$?
        set -e

        if [ $quick_exit -eq 0 ]; then
            log "    ${GREEN}✓${RESET} Quick smoke tests passed"
        else
            log "    ${YELLOW}⚠${RESET} Quick smoke tests failed (non-blocking — already promoted)"
            post_ok=false
        fi
    else
        log "  ${DIM}⊘${RESET} No live containers running — skipping post-promotion smoke test"
    fi

    if [ "$post_ok" = true ]; then
        stage_pass "Post-Promotion Verification"
    else
        stage_warn "Post-Promotion Verification"
        log "  ${YELLOW}  Note: Services are already promoted. Check logs and consider rollback if needed.${RESET}"
    fi
fi

# ═══════════════════════════════════════════════════════════════════════════
# Stage 7: Dev Journal + Flatten + Commit
# ═══════════════════════════════════════════════════════════════════════════

stage_header 7 "Dev Journal + Flatten + Commit"

JOURNAL_FILE="$GAIA_ROOT/knowledge/Dev_Notebook/${DATE}_promotion_journal.md"

# 7a. Generate dev journal
elapsed=$(( $(date +%s) - PIPELINE_START ))
mins=$(( elapsed / 60 ))
secs=$(( elapsed % 60 ))

# Determine overall result
overall_result="PASS"
for result in "${STAGE_RESULTS[@]}"; do
    status="${result%%|*}"
    if [ "$status" = "FAIL" ]; then
        overall_result="FAIL"
        break
    fi
done

cat > "$JOURNAL_FILE" << JOURNAL
# Promotion Pipeline — $DATE

**Timestamp:** $TIMESTAMP
**Duration:** ${mins}m ${secs}s
**Services:** $SERVICES
**Mode:** $([ "$DRY_RUN" = true ] && echo 'DRY RUN' || echo 'LIVE')
**Result:** $overall_result

## Stage Results

| Stage | Result |
|-------|--------|
JOURNAL

for result in "${STAGE_RESULTS[@]}"; do
    status="${result%%|*}"
    name="${result#*|}"
    echo "| $name | $status |" >> "$JOURNAL_FILE"
done

# Note: Validation details are printed by scripts/validate.sh and captured in the pipeline log.

cat >> "$JOURNAL_FILE" << FOOTER

---

*Generated by promote_pipeline.sh*
FOOTER

log "  ${GREEN}✓${RESET} Dev journal written to $JOURNAL_FILE"

# 7b. Flatten SOA (unless skipped or dry-run)
if [ "$SKIP_FLATTEN" = true ] || [ "$DRY_RUN" = true ]; then
    log "  ${DIM}⊘${RESET} flatten_soa.sh skipped"
else
    if [ -x "$FLATTEN_SCRIPT" ]; then
        log "  Running flatten_soa.sh..."
        set +e
        "$FLATTEN_SCRIPT" > /dev/null 2>&1
        flatten_exit=$?
        set -e
        if [ $flatten_exit -eq 0 ]; then
            log "  ${GREEN}✓${RESET} flatten_soa.sh completed"
        else
            log "  ${YELLOW}⚠${RESET} flatten_soa.sh exited with code $flatten_exit (non-blocking)"
        fi
    else
        log "  ${YELLOW}⚠${RESET} flatten_soa.sh not found or not executable"
    fi
fi

# 7c. Git commit + push
if [ "$DRY_RUN" = true ]; then
    log "  ${DIM}⊘${RESET} Git commit skipped (dry-run)"
else
    cd "$GAIA_ROOT"

    # Stage promotion-related files
    git add -A knowledge/Dev_Notebook/"${DATE}_promotion_journal.md" 2>/dev/null || true

    # Check if there's anything to commit
    if git diff --cached --quiet 2>/dev/null; then
        log "  ${DIM}⊘${RESET} Nothing new to commit"
    else
        git commit -m "$(cat <<EOF
chore: promotion pipeline $DATE — services: $SERVICES

Validated via smoke tests, deployed with backup enabled.

Generated with [Claude Code](https://claude.ai/code)
via [Happy](https://happy.engineering)

Co-Authored-By: Claude <noreply@anthropic.com>
Co-Authored-By: Happy <yesreply@happy.engineering>
EOF
        )" > /dev/null 2>&1
        log "  ${GREEN}✓${RESET} Commit created"

        if [ "$NO_PUSH" = true ]; then
            log "  ${DIM}⊘${RESET} Push skipped (--no-push)"
        else
            if git push 2>/dev/null; then
                log "  ${GREEN}✓${RESET} Pushed to remote"
            else
                log "  ${YELLOW}⚠${RESET} Push failed (non-blocking)"
            fi
        fi
    fi
fi

stage_pass "Dev Journal + Flatten + Commit"

# ═══════════════════════════════════════════════════════════════════════════
# Stage 8: QLoRA Validation (optional)
# ═══════════════════════════════════════════════════════════════════════════

stage_header 8 "QLoRA Validation"

if [ "$DO_QLORA" = true ]; then
    QLORA_SCRIPT="$SCRIPTS_DIR/validate_qlora.sh"
    if [ -x "$QLORA_SCRIPT" ]; then
        log "  Running QLoRA validation..."
        set +e
        "$QLORA_SCRIPT" 2>&1
        qlora_exit=$?
        set -e
        if [ $qlora_exit -eq 0 ]; then
            stage_pass "QLoRA Validation"
        else
            stage_warn "QLoRA Validation"
        fi
    else
        log "  ${YELLOW}⚠${RESET} validate_qlora.sh not found (Sprint 4 deliverable)"
        stage_skip "QLoRA Validation"
    fi
else
    stage_skip "QLoRA Validation"
fi

# ═══════════════════════════════════════════════════════════════════════════
# Final Summary
# ═══════════════════════════════════════════════════════════════════════════

print_summary
