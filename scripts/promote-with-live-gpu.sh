#!/usr/bin/env bash
# promote-with-live-gpu.sh — Run CPU-only candidates against the live GPU stack
#
# Starts candidate containers for CPU-only services (core, mcp, web, orchestrator)
# and routes them to the LIVE prime/study GPU services on the shared gaia-network.
# No GPU contention — candidates reuse the running GPU endpoints.
#
# Usage:
#   ./scripts/promote-with-live-gpu.sh              # --up, run pipeline, --down
#   ./scripts/promote-with-live-gpu.sh --up         # start CPU candidates only
#   ./scripts/promote-with-live-gpu.sh --down       # stop candidates
#   ./scripts/promote-with-live-gpu.sh --status     # show candidate state
#   ./scripts/promote-with-live-gpu.sh --dry-run    # pipeline in dry-run mode
set -euo pipefail

PROJECT_ROOT="/gaia/GAIA_Project"
CANDIDATE_COMPOSE="${PROJECT_ROOT}/docker-compose.candidate.yml"
LIVE_COMPOSE="${PROJECT_ROOT}/docker-compose.yml"
ENV_FILE="${PROJECT_ROOT}/.env.discord"
PIPELINE="${PROJECT_ROOT}/scripts/promote_pipeline.sh"

# CPU-only profiles (no prime, no study)
CPU_PROFILES="--profile core --profile mcp --profile web --profile orchestrator"

# ─── Colors ───────────────────────────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BOLD='\033[1m'
DIM='\033[2m'
RESET='\033[0m'

log()  { echo -e "[promote-live-gpu] $*"; }
ok()   { log "${GREEN}✓${RESET} $*"; }
warn() { log "${YELLOW}⚠${RESET} $*"; }
err()  { log "${RED}✗${RESET} $*" >&2; }

# ─── Candidate Lifecycle ──────────────────────────────────────────────────────
candidates_up() {
    log "${BOLD}Starting CPU-only candidates → live GPU...${RESET}"

    # Verify live GPU services are running
    local prime_ok study_ok
    prime_ok=$(COMPOSE_FILE="$LIVE_COMPOSE" docker compose ps --format '{{.Name}}' 2>/dev/null | grep -c 'gaia-prime' || true)
    study_ok=$(COMPOSE_FILE="$LIVE_COMPOSE" docker compose ps --format '{{.Name}}' 2>/dev/null | grep -c 'gaia-study' || true)

    if [ "$prime_ok" -eq 0 ]; then
        warn "gaia-prime is not running — candidates will fail to reach the GPU model"
    fi
    if [ "$study_ok" -eq 0 ]; then
        warn "gaia-study is not running — candidates will fail to reach the study endpoint"
    fi

    # Route candidates to live GPU services (hostnames resolve on gaia-network bridge)
    export PRIME_ENDPOINT="http://gaia-prime:7777"
    export CANDIDATE_STUDY_ENDPOINT="http://gaia-study:8766"

    log "  PRIME_ENDPOINT=${PRIME_ENDPOINT}"
    log "  CANDIDATE_STUDY_ENDPOINT=${CANDIDATE_STUDY_ENDPOINT}"

    COMPOSE_FILE="$CANDIDATE_COMPOSE" docker compose \
        --env-file "$ENV_FILE" \
        $CPU_PROFILES \
        up -d

    ok "CPU candidates started"
    echo ""
    COMPOSE_FILE="$CANDIDATE_COMPOSE" docker compose ps
}

candidates_down() {
    log "${BOLD}Stopping candidates...${RESET}"

    COMPOSE_FILE="$CANDIDATE_COMPOSE" docker compose \
        $CPU_PROFILES \
        down -t 10

    ok "Candidates stopped"
}

candidates_status() {
    log "${BOLD}Candidate status:${RESET}"
    COMPOSE_FILE="$CANDIDATE_COMPOSE" docker compose ps 2>/dev/null || log "${DIM}No candidates running${RESET}"
}

# ─── Main ─────────────────────────────────────────────────────────────────────
main() {
    local mode="pipeline"
    local pipeline_extra_args=()

    while [ $# -gt 0 ]; do
        case "$1" in
            --up)     mode="up" ;;
            --down)   mode="down" ;;
            --status) mode="status" ;;
            --dry-run) pipeline_extra_args+=("--dry-run") ;;
            --help|-h)
                echo "Usage: $0 [--up | --down | --status | --dry-run | --help]"
                echo ""
                echo "  (no flags)  Start CPU candidates → run pipeline → stop candidates"
                echo "  --up        Start CPU candidates pointing to live GPU"
                echo "  --down      Stop candidates"
                echo "  --status    Show candidate container state"
                echo "  --dry-run   Run pipeline in dry-run mode"
                exit 0
                ;;
            *)
                err "Unknown flag: $1"
                exit 1
                ;;
        esac
        shift
    done

    case "$mode" in
        up)     candidates_up ;;
        down)   candidates_down ;;
        status) candidates_status ;;
        pipeline)
            candidates_up
            echo ""
            log "${BOLD}Running promotion pipeline (--keep-live --gpu-skip)...${RESET}"

            local rc=0
            "$PIPELINE" --keep-live --gpu-skip "${pipeline_extra_args[@]+"${pipeline_extra_args[@]}"}" || rc=$?

            echo ""
            candidates_down

            if [ $rc -ne 0 ]; then
                err "Pipeline exited with code $rc"
                exit $rc
            fi
            ok "${BOLD}Promotion complete — candidates torn down${RESET}"
            ;;
    esac
}

main "$@"
