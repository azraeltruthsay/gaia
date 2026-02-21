#!/bin/bash
#
# validate.sh — Single-command validator for GAIA candidate services
#
# Runs ruff (lint), mypy (type check, non-blocking), and pytest (unit tests)
# inside Docker containers for each candidate service.
#
# Usage:
#   ./scripts/validate.sh gaia-web          # one service
#   ./scripts/validate.sh all               # all services
#   ./scripts/validate.sh gaia-core --verbose
#
# Exit codes:
#   0  All checks passed
#   1  Ruff lint failure
#   2  Pytest failure
#   3  Docker build failure
#   5  No tests found (non-blocking, treated as pass)
#

set -uo pipefail

# ═══════════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════════

GAIA_ROOT="/gaia/GAIA_Project"
KNOWLEDGE_DIR="$GAIA_ROOT/knowledge"

# All candidate services
ALL_SERVICES="gaia-common gaia-core gaia-web gaia-mcp gaia-study gaia-orchestrator gaia-audio"

# Services with no Dockerfile (need inline build)
NO_DOCKERFILE_SERVICES="gaia-common"

# Pytest testpaths per service (absolute in-container paths)
# Layout varies per Dockerfile:
#   gaia-common: inline build → code at /app/, tests at /app/tests
#   gaia-core:   flat copy → code at /app/gaia_core/, tests nested inside
#   gaia-web:    nested copy → code at /app/gaia-web/, tests at /app/gaia-web/tests
#   gaia-mcp:    nested copy → code at /app/gaia-mcp/, tests at /app/gaia-mcp/tests
#   gaia-study:  nested copy → code at /app/gaia-study/, tests at /app/gaia-study/tests
#   gaia-orchestrator: flat copy → code at /app/, pyproject at /app/, tests not copied
declare -A PYTEST_PATHS=(
    ["gaia-common"]="/app/tests"
    ["gaia-core"]="/app/gaia_core"
    ["gaia-web"]="/app/gaia-web/tests"
    ["gaia-mcp"]="/app/gaia-mcp/tests"
    ["gaia-study"]="/app/gaia-study/tests"
    ["gaia-orchestrator"]="/app/tests"
    ["gaia-audio"]="/app/tests"
)

# Ruff paths per service (absolute in-container paths)
declare -A RUFF_PATHS=(
    ["gaia-common"]="/app"
    ["gaia-core"]="/app/gaia_core"
    ["gaia-web"]="/app/gaia-web"
    ["gaia-mcp"]="/app/gaia-mcp"
    ["gaia-study"]="/app/gaia-study"
    ["gaia-orchestrator"]="/app/gaia_orchestrator"
    ["gaia-audio"]="/app/gaia_audio"
)

# Common environment variables for test containers
COMMON_ENV=(
    -e "GAIA_BLUEPRINTS_ROOT=/knowledge/blueprints"
    -e "GAIA_ROOT=/gaia/GAIA_Project"
    -e "ENABLE_DISCORD=0"
    -e "DISCORD_BOT_TOKEN=dummy-token-for-testing"
    -e "CORE_ENDPOINT=http://localhost:6415"
    -e "MCP_ENDPOINT=http://localhost:8765/jsonrpc"
    -e "STUDY_ENDPOINT=http://localhost:8766"
    -e "PRIME_ENDPOINT=http://localhost:7777"
    -e "PYTHONDONTWRITEBYTECODE=1"
)

# Common volume mounts
COMMON_MOUNTS=(
    -v "$KNOWLEDGE_DIR:/knowledge:ro"
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

VERBOSE=""
SERVICES=()

for arg in "$@"; do
    case $arg in
        -v|--verbose) VERBOSE="-v" ;;
        -h|--help)
            head -18 "$0" | tail -15
            exit 0
            ;;
        all)
            IFS=' ' read -ra SERVICES <<< "$ALL_SERVICES"
            ;;
        gaia-*)
            SERVICES+=("$arg")
            ;;
        *)
            echo -e "${RED}Unknown argument: $arg${RESET}"
            echo "Usage: $0 <service|all> [--verbose]"
            exit 1
            ;;
    esac
done

if [ ${#SERVICES[@]} -eq 0 ]; then
    echo -e "${RED}No services specified.${RESET}"
    echo "Usage: $0 <service|all> [--verbose]"
    exit 1
fi

# ═══════════════════════════════════════════════════════════════════════════
# Validation Functions
# ═══════════════════════════════════════════════════════════════════════════

build_image() {
    local svc=$1
    local image_name=$2
    local candidate_dir="$GAIA_ROOT/candidates/$svc"

    # Check for Dockerfile
    if echo "$NO_DOCKERFILE_SERVICES" | grep -qw "$svc"; then
        # Inline build for services without Dockerfile
        echo -e "  ${DIM}Building inline image for $svc (no Dockerfile)...${RESET}"
        docker build -t "$image_name" -f - "$GAIA_ROOT" <<'DOCKERFILE'
FROM python:3.11-slim
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
RUN pip install --no-cache-dir --upgrade pip setuptools wheel
WORKDIR /app
COPY candidates/gaia-common/ /app/
RUN pip install -e ".[dev]"
DOCKERFILE
    else
        local dockerfile="$candidate_dir/Dockerfile"
        if [ ! -f "$dockerfile" ]; then
            echo -e "  ${RED}No Dockerfile found: $dockerfile${RESET}"
            return 3
        fi
        docker build -t "$image_name" -f "$dockerfile" "$GAIA_ROOT"
    fi
}

run_ruff() {
    local image_name=$1
    local svc=$2
    local ruff_path="${RUFF_PATHS[$svc]}"

    docker run --rm "${COMMON_ENV[@]}" "$image_name" \
        python -m ruff check "$ruff_path" 2>&1
}

run_mypy() {
    local image_name=$1
    local svc=$2
    local ruff_path="${RUFF_PATHS[$svc]}"

    docker run --rm "${COMMON_ENV[@]}" "$image_name" \
        python -m mypy "$ruff_path" 2>&1
}

run_pytest() {
    local image_name=$1
    local svc=$2
    local test_path="${PYTEST_PATHS[$svc]}"

    # Check if test path exists in container; if not, return exit 5 (no tests)
    if ! docker run --rm "$image_name" python -c "import pathlib; exit(0 if pathlib.Path('$test_path').exists() else 1)" 2>/dev/null; then
        echo "no tests ran (test path $test_path not found in container)"
        return 5
    fi

    docker run --rm \
        "${COMMON_ENV[@]}" \
        "${COMMON_MOUNTS[@]}" \
        "$image_name" \
        python -m pytest "$test_path" --import-mode=importlib --no-header -q $VERBOSE 2>&1
}

# ═══════════════════════════════════════════════════════════════════════════
# Main Loop
# ═══════════════════════════════════════════════════════════════════════════

echo ""
echo -e "${BOLD}${CYAN}═══════════════════════════════════════════════════════════${RESET}"
echo -e "${BOLD}${CYAN}  GAIA Candidate Validation${RESET}"
echo -e "${BOLD}${CYAN}═══════════════════════════════════════════════════════════${RESET}"
echo -e "  Services: ${SERVICES[*]}"
echo ""

declare -A RESULTS
overall_exit=0

for svc in "${SERVICES[@]}"; do
    # Skip gaia-prime (no Python validation)
    if [ "$svc" = "gaia-prime" ]; then
        echo -e "  ${DIM}⊘ $svc — no Python validation${RESET}"
        RESULTS["$svc"]="skip|skip|skip"
        continue
    fi

    echo -e "  Validating ${BOLD}$svc${RESET}..."
    image_name="gaia-validate-${svc}:$(date +%s)"

    ruff_status="?"
    mypy_status="?"
    pytest_status="?"

    # ── Build ─────────────────────────────────────────────────────────────
    set +e
    build_output=$(build_image "$svc" "$image_name" 2>&1)
    build_exit=$?
    set -e

    if [ $build_exit -ne 0 ]; then
        echo -e "    ${RED}✗${RESET} Docker build failed"
        echo "$build_output" | tail -10 | while IFS= read -r line; do echo "      $line"; done
        RESULTS["$svc"]="build-fail|?|?"
        overall_exit=3
        docker rmi "$image_name" > /dev/null 2>&1 || true
        continue
    fi

    # ── Ruff ──────────────────────────────────────────────────────────────
    set +e
    ruff_output=$(run_ruff "$image_name" "$svc")
    ruff_exit=$?
    set -e

    if [ $ruff_exit -eq 0 ]; then
        ruff_status="pass"
    else
        ruff_status="FAIL"
        [ $overall_exit -eq 0 ] && overall_exit=1
    fi

    # ── MyPy (non-blocking) ──────────────────────────────────────────────
    set +e
    mypy_output=$(run_mypy "$image_name" "$svc")
    mypy_exit=$?
    set -e

    # mypy: blocking for specified services, warn-only for others
    MYPY_BLOCKING_SERVICES="gaia-core gaia-mcp gaia-study"
    if [ $mypy_exit -eq 0 ]; then
        mypy_status="pass"
    elif echo "$MYPY_BLOCKING_SERVICES" | grep -qw "$svc"; then
        mypy_status="FAIL"
        [ $overall_exit -lt 4 ] && overall_exit=4
    else
        mypy_status="warn"
    fi

    # ── Pytest ────────────────────────────────────────────────────────────
    set +e
    pytest_output=$(run_pytest "$image_name" "$svc")
    pytest_exit=$?
    set -e

    if [ $pytest_exit -eq 0 ]; then
        pytest_status="pass"
    elif [ $pytest_exit -eq 5 ]; then
        pytest_status="none"
    else
        pytest_status="FAIL"
        [ $overall_exit -lt 2 ] && overall_exit=2
    fi

    # ── Cleanup ───────────────────────────────────────────────────────────
    docker rmi "$image_name" > /dev/null 2>&1 || true

    RESULTS["$svc"]="$ruff_status|$mypy_status|$pytest_status"

    # Status line
    if [ "$ruff_status" != "FAIL" ] && [ "$pytest_status" != "FAIL" ]; then
        echo -e "    ${GREEN}✓${RESET} $svc — ruff:$ruff_status mypy:$mypy_status pytest:$pytest_status"
    else
        echo -e "    ${RED}✗${RESET} $svc — ruff:$ruff_status mypy:$mypy_status pytest:$pytest_status"
        if [ "$ruff_status" = "FAIL" ] && [ -n "$VERBOSE" ]; then
            echo "$ruff_output" | tail -15 | while IFS= read -r line; do echo "      $line"; done
        fi
        if [ "$pytest_status" = "FAIL" ]; then
            echo "$pytest_output" | tail -15 | while IFS= read -r line; do echo "      $line"; done
        fi
    fi
done

# ═══════════════════════════════════════════════════════════════════════════
# Summary Table
# ═══════════════════════════════════════════════════════════════════════════

echo ""
echo -e "${BOLD}═══════════════════════════════════════════════════════════${RESET}"
echo -e "${BOLD}  Validation Summary${RESET}"
echo -e "${BOLD}═══════════════════════════════════════════════════════════${RESET}"
printf "  %-22s %-10s %-10s %-10s\n" "Service" "Ruff" "MyPy" "Pytest"
printf "  %-22s %-10s %-10s %-10s\n" "───────────────────" "─────────" "─────────" "─────────"

for svc in "${SERVICES[@]}"; do
    IFS='|' read -r r_ruff r_mypy r_pytest <<< "${RESULTS[$svc]}"

    # Colorize
    case $r_ruff in pass) c_ruff="${GREEN}pass${RESET}" ;; FAIL) c_ruff="${RED}FAIL${RESET}" ;; *) c_ruff="${DIM}$r_ruff${RESET}" ;; esac
    case $r_mypy in pass) c_mypy="${GREEN}pass${RESET}" ;; warn) c_mypy="${YELLOW}warn${RESET}" ;; *) c_mypy="${DIM}$r_mypy${RESET}" ;; esac
    case $r_pytest in pass) c_pytest="${GREEN}pass${RESET}" ;; FAIL) c_pytest="${RED}FAIL${RESET}" ;; none) c_pytest="${DIM}none${RESET}" ;; *) c_pytest="${DIM}$r_pytest${RESET}" ;; esac

    printf "  %-22s %-20b %-20b %-20b\n" "$svc" "$c_ruff" "$c_mypy" "$c_pytest"
done

echo ""
if [ $overall_exit -eq 0 ]; then
    echo -e "  ${GREEN}${BOLD}All checks passed.${RESET}"
else
    echo -e "  ${RED}${BOLD}Validation failed (exit $overall_exit).${RESET}"
fi
echo ""

exit $overall_exit
