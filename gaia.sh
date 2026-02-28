#!/bin/bash
# GAIA Stack Management Script
#
# Unified control for live and candidate services with modular swap capability.
#
# Usage:
#   ./gaia.sh live [start|stop|status]      # Manage live stack
#   ./gaia.sh candidate [start|stop|status] # Manage candidate stack
#   ./gaia.sh swap <service> [live|candidate] # Swap a component
#   ./gaia.sh status                        # Show all running services
#
# Examples:
#   ./gaia.sh live start         # Start live stack
#   ./gaia.sh candidate start    # Start candidate stack (isolated)
#   ./gaia.sh swap mcp candidate # Route live core -> candidate MCP
#   ./gaia.sh swap mcp live      # Restore live MCP

set -e

PROJECT_ROOT="/gaia/GAIA_Project"
LIVE_COMPOSE="${PROJECT_ROOT}/docker-compose.yml"
CANDIDATE_COMPOSE="${PROJECT_ROOT}/docker-compose.candidate.yml"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[OK]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# Orchestrator settings
ORCHESTRATOR_PORT=6410
ORCHESTRATOR_URL="http://localhost:${ORCHESTRATOR_PORT}"

# Ensure gaia-network exists
ensure_network() {
    if ! docker network inspect gaia-network >/dev/null 2>&1; then
        log_info "Creating gaia-network..."
        docker network create gaia-network
    fi
}



# Show status of all services
show_status() {
    echo -e "\n${CYAN}=== GAIA Service Status ===${NC}\n"

    # Check orchestrator first
    echo -e "${BLUE}Orchestrator:${NC}"
    if docker ps --format '{{.Names}}' 2>/dev/null | grep -q "^gaia-orchestrator$"; then
        if curl -sf "${ORCHESTRATOR_URL}/health" > /dev/null 2>&1; then
            echo -e "  ${GREEN}[healthy]${NC} gaia-orchestrator :${ORCHESTRATOR_PORT}"
        else
            echo -e "  ${YELLOW}[running]${NC} gaia-orchestrator :${ORCHESTRATOR_PORT}"
        fi
    else
        echo -e "  ${RED}[stopped]${NC} gaia-orchestrator"
    fi

    echo -e "\n${BLUE}Live Services:${NC}"
    for svc in prime core web mcp study audio wiki; do
        local container="gaia-${svc}"
        local port
        case "$svc" in
            prime) port=7777 ;;
            core) port=6415 ;;
            web) port=6414 ;;
            mcp) port=8765 ;;
            study) port=8766 ;;
            audio) port=8080 ;;
            wiki) port=8081 ;; # internal port, checking container is usually enough
        esac

        if docker ps --format '{{.Names}}' 2>/dev/null | grep -q "^${container}$"; then
            if [ "$svc" = "wiki" ]; then
                 echo -e "  ${GREEN}[running]${NC} ${container} (wiki docs)"
                 continue
            fi
            if curl -sf "http://localhost:${port}/health" > /dev/null 2>&1; then
                echo -e "  ${GREEN}[healthy]${NC} ${container} :${port}"
            else
                echo -e "  ${YELLOW}[running]${NC} ${container} :${port}"
            fi
        else
            echo -e "  ${RED}[stopped]${NC} ${container}"
        fi
    done

    echo -e "\n${BLUE}Candidate Services:${NC}"
    for svc in prime core web mcp study audio; do
        local container="gaia-${svc}-candidate"
        local port
        case "$svc" in
            prime) port=7778 ;;
            core) port=6416 ;;
            web) port=6417 ;;
            mcp) port=8767 ;;
            study) port=8768 ;;
            audio) port=8082 ;;
        esac

        if docker ps --format '{{.Names}}' 2>/dev/null | grep -q "^${container}$"; then
            if curl -sf "http://localhost:${port}/health" > /dev/null 2>&1; then
                echo -e "  ${GREEN}[healthy]${NC} ${container} :${port}"
            else
                echo -e "  ${YELLOW}[running]${NC} ${container} :${port}"
            fi
        else
            echo -e "  ${RED}[stopped]${NC} ${container}"
        fi
    done
    echo ""
}

# Manage live stack
cmd_live() {
    local action="${1:-status}"

    case "$action" in
        start)
            ensure_network
            log_info "Starting live stack..."
            docker compose -f "$LIVE_COMPOSE" --env-file ./.env.discord up -d
            log_success "Live stack started"
            ;;
        stop)
            log_info "Stopping live stack..."
            docker compose -f "$LIVE_COMPOSE" down
            log_success "Live stack stopped"
            ;;
        status)
            docker compose -f "$LIVE_COMPOSE" ps
            ;;
        *)
            log_error "Unknown action: $action"
            echo "Usage: $0 live [start|stop|status]"
            exit 1
            ;;
    esac
}

# Manage candidate stack
cmd_candidate() {
    local action="${1:-status}"

    case "$action" in
        start)
            ensure_network
            log_info "Starting candidate stack..."
            docker compose -f "$CANDIDATE_COMPOSE" --env-file ./.env.discord --profile full up -d
            log_success "Candidate stack started"
            ;;
        stop)
            log_info "Stopping candidate stack..."
            docker compose -f "$CANDIDATE_COMPOSE" --profile full down
            log_success "Candidate stack stopped"
            ;;
        status)
            docker compose -f "$CANDIDATE_COMPOSE" --profile full ps
            ;;
        *)
            log_error "Unknown action: $action"
            echo "Usage: $0 candidate [start|stop|status]"
            exit 1
            ;;
    esac
}

# Swap a component between live and candidate
cmd_swap() {
    local service="$1"
    local target="${2:-candidate}"

    if [ -z "$service" ]; then
        log_error "Service name required"
        echo "Usage: $0 swap <service> [live|candidate]"
        echo "Services: core, web, mcp, study"
        exit 1
    fi

    # Validate service
    case "$service" in
        prime|core|web|mcp|study|audio) ;;
        *)
            log_error "Unknown service: $service"
            echo "Valid services: prime, core, web, mcp, study, audio"
            exit 1
            ;;
    esac

    ensure_network

    if [ "$target" = "candidate" ]; then
        # Start the candidate if not running
        if ! docker ps --format '{{.Names}}' | grep -q "gaia-${service}-candidate"; then
            log_info "Starting gaia-${service}-candidate..."
            docker compose -f "$CANDIDATE_COMPOSE" --profile "$service" up -d "gaia-${service}-candidate"
            sleep 3
        fi

        # Determine which live service needs restarting with new endpoint
        local endpoint_var endpoint_url caller
        case "$service" in
            mcp)
                endpoint_var="MCP_ENDPOINT"
                endpoint_url="http://gaia-mcp-candidate:8765/jsonrpc"
                caller="gaia-core"
                ;;
            study)
                endpoint_var="STUDY_ENDPOINT"
                endpoint_url="http://gaia-study-candidate:8766"
                caller="gaia-core"
                ;;
            audio)
                endpoint_var="AUDIO_ENDPOINT"
                endpoint_url="http://gaia-audio-candidate:8080"
                caller="gaia-web"
                ;;
            core)
                endpoint_var="CORE_ENDPOINT"
                endpoint_url="http://gaia-core-candidate:6415"
                caller="gaia-web"
                ;;
            web)
                log_warn "Web doesn't need injection - access candidate directly at http://localhost:6417"
                return
                ;;
        esac

        log_info "Swapping ${service} -> candidate..."
        log_info "Restarting ${caller} with ${endpoint_var}=${endpoint_url}"

        export "${endpoint_var}=${endpoint_url}"
        docker compose -f "$LIVE_COMPOSE" up -d "$caller"

        log_success "Swap complete: live ${caller} -> candidate ${service}"
        echo ""
        echo "Traffic flow: ${caller} -> gaia-${service}-candidate"
        echo "To restore: ./gaia.sh swap ${service} live"

    elif [ "$target" = "live" ]; then
        # Restart the caller with default (live) endpoint
        local caller
        case "$service" in
            mcp|study) caller="gaia-core" ;;
            core|audio) caller="gaia-web" ;;
            web)
                log_info "No swap needed for web"
                return
                ;;
        esac

        log_info "Restoring ${service} -> live..."

        # Unset any override and restart
        unset MCP_ENDPOINT CORE_ENDPOINT STUDY_ENDPOINT
        docker compose -f "$LIVE_COMPOSE" up -d "$caller"

        log_success "Restored: live ${caller} -> live ${service}"

    else
        log_error "Unknown target: $target"
        echo "Valid targets: live, candidate"
        exit 1
    fi
}

# Manage orchestrator
cmd_orchestrator() {
    local action="${1:-status}"

    case "$action" in
        start)
            log_info "Starting orchestrator..."
            docker compose -f "$LIVE_COMPOSE" up -d gaia-orchestrator
            log_success "Orchestrator started on port ${ORCHESTRATOR_PORT}"
            ;;
        stop)
            log_info "Stopping orchestrator..."
            docker compose -f "$LIVE_COMPOSE" stop gaia-orchestrator
            docker compose -f "$LIVE_COMPOSE" rm -f gaia-orchestrator
            log_success "Orchestrator stopped"
            ;;
        status)
            if docker ps --format '{{.Names}}' 2>/dev/null | grep -q "^gaia-orchestrator$"; then
                echo -e "${GREEN}Orchestrator is running${NC}"
                curl -s "${ORCHESTRATOR_URL}/status" 2>/dev/null | python3 -m json.tool || echo "Could not fetch status"
            else
                echo -e "${RED}Orchestrator is stopped${NC}"
            fi
            ;;
        build)
            log_info "Building orchestrator image..."
            docker compose -f "$LIVE_COMPOSE" build gaia-orchestrator
            log_success "Orchestrator image built"
            ;;
        logs)
            docker compose -f "$LIVE_COMPOSE" logs -f gaia-orchestrator
            ;;
        *)
            log_error "Unknown action: $action"
            echo "Usage: $0 orchestrator [start|stop|status|build|logs]"
            exit 1
            ;;
    esac
}

# GPU management via orchestrator
cmd_gpu() {
    local action="${1:-status}"

    # Check orchestrator is running
    if ! curl -sf "${ORCHESTRATOR_URL}/health" > /dev/null 2>&1; then
        log_error "Orchestrator not running. Start it with: $0 orchestrator start"
        exit 1
    fi

    case "$action" in
        status)
            curl -s "${ORCHESTRATOR_URL}/gpu/status" | python3 -m json.tool
            ;;
        release)
            log_info "Releasing GPU..."
            curl -s -X POST "${ORCHESTRATOR_URL}/gpu/release" | python3 -m json.tool
            ;;
        *)
            log_error "Unknown action: $action"
            echo "Usage: $0 gpu [status|release]"
            exit 1
            ;;
    esac
}

# Handoff management via orchestrator
cmd_handoff() {
    local action="${1:-status}"
    local handoff_id="$2"

    # Check orchestrator is running
    if ! curl -sf "${ORCHESTRATOR_URL}/health" > /dev/null 2>&1; then
        log_error "Orchestrator not running. Start it with: $0 orchestrator start"
        exit 1
    fi

    case "$action" in
        prime-to-study)
            log_info "Initiating Prime -> Study handoff..."
            curl -s -X POST "${ORCHESTRATOR_URL}/handoff/prime-to-study" | python3 -m json.tool
            ;;
        study-to-prime)
            log_info "Initiating Study -> Prime handoff..."
            curl -s -X POST "${ORCHESTRATOR_URL}/handoff/study-to-prime" | python3 -m json.tool
            ;;
        status)
            if [ -n "$handoff_id" ]; then
                curl -s "${ORCHESTRATOR_URL}/handoff/${handoff_id}/status" | python3 -m json.tool
            else
                curl -s "${ORCHESTRATOR_URL}/status" | python3 -c "import sys,json; d=json.load(sys.stdin); print(json.dumps(d.get('active_handoff'), indent=2))"
            fi
            ;;
        *)
            log_error "Unknown action: $action"
            echo "Usage: $0 handoff [prime-to-study|study-to-prime|status [id]]"
            exit 1
            ;;
    esac
}

# Manage wiki service
cmd_wiki() {
    local action="${1:-status}"

    case "$action" in
        start)
            log_info "Starting gaia-wiki..."
            docker compose -f "$LIVE_COMPOSE" up -d gaia-wiki
            log_success "Wiki started (internal only, proxied via gaia-web /wiki/)"
            ;;
        stop)
            log_info "Stopping gaia-wiki..."
            docker compose -f "$LIVE_COMPOSE" stop gaia-wiki
            docker compose -f "$LIVE_COMPOSE" rm -f gaia-wiki
            log_success "Wiki stopped"
            ;;
        build)
            log_info "Building wiki image..."
            docker compose -f "$LIVE_COMPOSE" build gaia-wiki
            log_success "Wiki image built"
            ;;
        logs)
            docker compose -f "$LIVE_COMPOSE" logs -f gaia-wiki
            ;;
        status)
            if docker ps --format '{{.Names}}' 2>/dev/null | grep -q "^gaia-wiki$"; then
                echo -e "${GREEN}Wiki is running${NC} (access via gaia-web /wiki/)"
            else
                echo -e "${RED}Wiki is stopped${NC}"
            fi
            ;;
        *)
            log_error "Unknown action: $action"
            echo "Usage: $0 wiki [start|stop|build|logs|status]"
            exit 1
            ;;
    esac
}

# Main
main() {
    cd "$PROJECT_ROOT"

    local cmd="${1:-status}"
    shift || true

    case "$cmd" in
        live)
            cmd_live "$@"
            ;;
        candidate)
            cmd_candidate "$@"
            ;;
        swap)
            cmd_swap "$@"
            ;;
        orchestrator|orch)
            cmd_orchestrator "$@"
            ;;
        gpu)
            cmd_gpu "$@"
            ;;
        handoff)
            cmd_handoff "$@"
            ;;
        wiki)
            cmd_wiki "$@"
            ;;
        status)
            show_status
            ;;
        help|--help|-h)
            echo "GAIA Stack Management"
            echo ""
            echo "Usage: $0 <command> [options]"
            echo ""
            echo "Commands:"
            echo "  live [start|stop|status]         Manage live stack"
            echo "  candidate [start|stop|status]    Manage candidate stack"
            echo "  swap <service> [live|candidate]  Swap component in live flow"
            echo "  orchestrator [start|stop|status|build|logs]  Manage orchestrator"
            echo "  gpu [status|release]             GPU management via orchestrator"
            echo "  handoff [prime-to-study|study-to-prime|status]  GPU handoff"
            echo "  wiki [start|stop|build|logs|status]  Manage wiki docs"
            echo "  status                           Show all service status"
            echo ""
            echo "Services: prime, core, web, mcp, study, audio, wiki"
            echo ""
            echo "Examples:"
            echo "  $0 live start              # Start live stack"
            echo "  $0 candidate start         # Start isolated candidate stack"
            echo "  $0 swap mcp candidate      # Route live core -> candidate MCP"
            echo "  $0 swap mcp live           # Restore live MCP"
            echo "  $0 orchestrator build      # Build orchestrator image"
            echo "  $0 orchestrator start      # Start orchestrator"
            echo "  $0 gpu status              # Check GPU ownership"
            echo "  $0 handoff prime-to-study  # Transfer GPU to Study"
            echo "  $0 wiki start              # Start wiki docs"
            echo "  $0 wiki build              # Rebuild wiki image"
            echo "  $0 status                  # Show all services"
            ;;
        *)
            log_error "Unknown command: $cmd"
            echo "Use '$0 help' for usage"
            exit 1
            ;;
    esac
}

main "$@"
