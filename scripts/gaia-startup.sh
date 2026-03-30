#!/usr/bin/env bash
# gaia-startup.sh — Idempotent GAIA cluster startup
# Creates tmux sessions + brings up the Docker production stack.
# Safe to re-run: skips existing sessions, uses compose idempotency.
#
# Usage:
#   ./scripts/gaia-startup.sh              # full startup
#   ./scripts/gaia-startup.sh --tmux-only  # tmux sessions only
#   ./scripts/gaia-startup.sh --docker-only # Docker stack only
#   ./scripts/gaia-startup.sh --status     # show current state
set -euo pipefail

# ─── Environment ──────────────────────────────────────────────────────────────
PROJECT_ROOT="/gaia/GAIA_Project"
NODE_BIN="/home/azrael/.nvm/versions/node/v25.4.0/bin"
export PATH="${NODE_BIN}:/usr/local/bin:/usr/bin:/bin:${PATH}"
export TERM="${TERM:-xterm-256color}"

COMPOSE_FILE="${PROJECT_ROOT}/docker-compose.yml"
ENV_FILE="${PROJECT_ROOT}/.env.discord"
HEALTH_INTERVAL=10
HEALTH_TIMEOUT=300

# ─── Colors ───────────────────────────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
BOLD='\033[1m'
DIM='\033[2m'
RESET='\033[0m'

# ─── Helpers ──────────────────────────────────────────────────────────────────
log()  { echo -e "[gaia-startup] $*"; }
ok()   { log "${GREEN}✓${RESET} $*"; }
warn() { log "${YELLOW}⚠${RESET} $*"; }
err()  { log "${RED}✗${RESET} $*"; }

require_user() {
    if [ "$(id -u)" -ne 1000 ]; then
        err "Must run as azrael (UID 1000), not $(id -un) ($(id -u))"
        exit 1
    fi
}

# ─── Tmux Sessions ────────────────────────────────────────────────────────────
# Each entry: "session_name|command_or_empty"
TMUX_SESSIONS=(
    "gaia_test|"
    "gaia_notebooklm|bash -c 'cd ${PROJECT_ROOT} && ./start_notebooklm_sync.sh'"
    "gaia_code_tunnel|/usr/local/bin/code tunnel"
    "gaia_gemini|${NODE_BIN}/happy gemini"
    "gaia_claude|${NODE_BIN}/happy claude"
)

create_tmux_sessions() {
    log "${BOLD}Creating tmux sessions...${RESET}"
    local created=0 skipped=0

    for entry in "${TMUX_SESSIONS[@]}"; do
        local name="${entry%%|*}"
        local cmd="${entry#*|}"

        if /usr/bin/tmux has-session -t "$name" 2>/dev/null; then
            log "  ${DIM}${name}${RESET} — already exists"
            (( skipped++ )) || true
            continue
        fi

        if [ -z "$cmd" ]; then
            /usr/bin/tmux new-session -d -s "$name" -c "$PROJECT_ROOT"
        else
            /usr/bin/tmux new-session -d -s "$name" -c "$PROJECT_ROOT" "$cmd"
        fi
        ok "  ${name}"
        (( created++ )) || true
    done

    log "Tmux: ${GREEN}${created} created${RESET}, ${DIM}${skipped} skipped${RESET}"
}

# ─── Docker Stack ─────────────────────────────────────────────────────────────
start_docker_stack() {
    log "${BOLD}Starting Docker production stack...${RESET}"

    if [ ! -f "$COMPOSE_FILE" ]; then
        err "Compose file not found: $COMPOSE_FILE"
        exit 1
    fi
    if [ ! -f "$ENV_FILE" ]; then
        err "Env file not found: $ENV_FILE"
        exit 1
    fi

    # First pass: start what we can. Services with depends_on: service_healthy
    # (gaia-core, gaia-web) may fail if gaia-prime isn't healthy yet — that's OK,
    # we retry after the health loop brings prime up.
    COMPOSE_FILE="$COMPOSE_FILE" docker compose --env-file "$ENV_FILE" up -d || \
        warn "Initial compose up returned non-zero (expected if GPU services are still starting)"

    log "Waiting for services to become healthy..."
    local elapsed=0 retried=false
    while [ $elapsed -lt $HEALTH_TIMEOUT ]; do
        local all_healthy=true
        local unhealthy_list=""
        local created_list=""
        local svc_count=0

        while IFS= read -r line; do
            [ -z "$line" ] && continue
            local svc_name svc_status
            svc_name="$(echo "$line" | awk '{print $1}')"
            svc_status="$(echo "$line" | awk '{print $2}')"
            (( svc_count++ )) || true

            case "$svc_status" in
                healthy) ;;
                running)
                    # Service has no healthcheck defined — treat as OK
                    ;;
                "")
                    # Container was created but never started (blocked dependency)
                    all_healthy=false
                    created_list="${created_list} ${svc_name}"
                    ;;
                *)
                    all_healthy=false
                    unhealthy_list="${unhealthy_list} ${svc_name}(${svc_status})"
                    ;;
            esac
        done < <(COMPOSE_FILE="$COMPOSE_FILE" docker compose ps --format '{{.Name}} {{.Health}}' 2>/dev/null || true)

        # Guard: if compose returned zero containers, something is fundamentally wrong
        if (( svc_count == 0 )); then
            if (( elapsed == 0 )); then
                err "No containers found — compose up may have failed entirely"
                log "Retrying compose up..."
                COMPOSE_FILE="$COMPOSE_FILE" docker compose --env-file "$ENV_FILE" up -d || \
                    err "Compose up retry also failed"
                sleep "$HEALTH_INTERVAL"
                (( elapsed += HEALTH_INTERVAL ))
                continue
            elif (( elapsed >= 30 )); then
                err "Still no containers after ${elapsed}s — giving up on Docker stack"
                COMPOSE_FILE="$COMPOSE_FILE" docker compose ps
                return 1
            fi
        fi

        # Retry compose up once dependency-blocked containers exist and prime is healthy
        if [ -n "$created_list" ] && ! $retried; then
            # Check if prime is now healthy
            local prime_health
            prime_health=$(docker inspect --format='{{.State.Health.Status}}' gaia-prime 2>/dev/null || echo "unknown")
            if [ "$prime_health" = "healthy" ]; then
                log "gaia-prime healthy — retrying compose up for blocked services..."
                COMPOSE_FILE="$COMPOSE_FILE" docker compose --env-file "$ENV_FILE" up -d || true
                retried=true
                continue
            fi
        fi

        if $all_healthy && (( svc_count > 0 )); then
            ok "All services healthy after ${elapsed}s (${svc_count} containers)"
            return 0
        fi

        if (( elapsed % 30 == 0 )) && [ $elapsed -gt 0 ]; then
            log "  ${DIM}${elapsed}s — waiting on:${unhealthy_list}${created_list:+ [blocked:${created_list}]}${RESET}"
        fi

        sleep "$HEALTH_INTERVAL"
        (( elapsed += HEALTH_INTERVAL ))
    done

    warn "Health timeout (${HEALTH_TIMEOUT}s) — some services may still be starting:${unhealthy_list:-}${created_list:-}"
    COMPOSE_FILE="$COMPOSE_FILE" docker compose ps
}

# ─── HA Standby Services ──────────────────────────────────────────────────────
start_ha_services() {
    log "${BOLD}Starting HA hot standby services...${RESET}"

    local ha_script="${PROJECT_ROOT}/scripts/ha_start.sh"
    if [ ! -x "$ha_script" ]; then
        warn "HA start script not found or not executable: $ha_script"
        return 1
    fi

    if bash "$ha_script" 2>&1 | while IFS= read -r line; do log "  ${DIM}${line}${RESET}"; done; then
        ok "HA standby services started"
    else
        warn "HA start returned non-zero — candidates may not be fully up yet"
    fi

    # Wait briefly for HA candidates to pass health checks
    local ha_wait=0
    local ha_timeout=90
    while [ $ha_wait -lt $ha_timeout ]; do
        local core_health mcp_health
        core_health=$(docker inspect --format='{{if .State.Health}}{{.State.Health.Status}}{{else}}unknown{{end}}' gaia-core-candidate 2>/dev/null || echo "missing")
        mcp_health=$(docker inspect --format='{{if .State.Health}}{{.State.Health.Status}}{{else}}unknown{{end}}' gaia-mcp-candidate 2>/dev/null || echo "missing")

        if [ "$core_health" = "healthy" ] && [ "$mcp_health" = "healthy" ]; then
            ok "HA candidates healthy (core: ${core_health}, mcp: ${mcp_health})"
            return 0
        fi

        if (( ha_wait % 30 == 0 )) && [ $ha_wait -gt 0 ]; then
            log "  ${DIM}${ha_wait}s — HA candidates: core=${core_health}, mcp=${mcp_health}${RESET}"
        fi

        sleep 10
        (( ha_wait += 10 ))
    done

    warn "HA health timeout (${ha_timeout}s) — candidates may still be starting"
}

# ─── Doctor Verification ─────────────────────────────────────────────────────
run_doctor_fix() {
    log "${BOLD}Running GAIA Doctor (--fix mode)...${RESET}"

    local doctor_script="${PROJECT_ROOT}/scripts/gaia_doctor.sh"
    if [ ! -x "$doctor_script" ]; then
        warn "Doctor script not found or not executable: $doctor_script"
        return 1
    fi

    bash "$doctor_script" --fix 2>&1 | while IFS= read -r line; do
        log "  ${line}"
    done
    local doctor_exit=${PIPESTATUS[0]}

    case $doctor_exit in
        0) ok "Doctor: all healthy" ;;
        1) warn "Doctor: warnings detected (degraded but functional)" ;;
        2) err "Doctor: failures detected — manual intervention may be needed" ;;
        3) err "Doctor: pre-flight failed" ;;
        *) warn "Doctor: unexpected exit code $doctor_exit" ;;
    esac

    return 0
}

# ─── Status ───────────────────────────────────────────────────────────────────
show_status() {
    log "${BOLD}=== GAIA Cluster Status ===${RESET}"
    echo ""

    log "${BLUE}Tmux Sessions:${RESET}"
    for entry in "${TMUX_SESSIONS[@]}"; do
        local name="${entry%%|*}"
        if /usr/bin/tmux has-session -t "$name" 2>/dev/null; then
            echo -e "  ${GREEN}●${RESET} ${name}"
        else
            echo -e "  ${RED}○${RESET} ${name}"
        fi
    done

    echo ""
    log "${BLUE}Docker Services:${RESET}"
    if [ -f "$COMPOSE_FILE" ]; then
        COMPOSE_FILE="$COMPOSE_FILE" docker compose ps 2>/dev/null || warn "Docker compose not running"
    else
        warn "Compose file not found: $COMPOSE_FILE"
    fi
}

# ─── Main ─────────────────────────────────────────────────────────────────────
main() {
    local mode="full"

    while [ $# -gt 0 ]; do
        case "$1" in
            --tmux-only)   mode="tmux" ;;
            --docker-only) mode="docker" ;;
            --status)      mode="status" ;;
            --help|-h)
                echo "Usage: $0 [--tmux-only | --docker-only | --status | --help]"
                echo ""
                echo "  (no flags)    Full startup: tmux + Docker"
                echo "  --tmux-only   Create tmux sessions only"
                echo "  --docker-only Start Docker production stack only"
                echo "  --status      Show current cluster state"
                exit 0
                ;;
            *)
                err "Unknown flag: $1"
                exit 1
                ;;
        esac
        shift
    done

    require_user

    case "$mode" in
        full)
            log "${BOLD}=== GAIA Cluster Startup ===${RESET}"
            create_tmux_sessions
            echo ""
            start_docker_stack
            echo ""
            start_ha_services
            echo ""
            run_doctor_fix
            echo ""
            ok "${BOLD}GAIA cluster startup complete (production + HA + verified)${RESET}"
            ;;
        tmux)   create_tmux_sessions ;;
        docker) start_docker_stack ;;
        status) show_status ;;
    esac
}

main "$@"
