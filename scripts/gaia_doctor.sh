#!/usr/bin/env bash
# gaia_doctor.sh — GAIA system health detection, diagnosis, and repair
#
# Usage:
#   ./scripts/gaia_doctor.sh              # check mode (read-only, default)
#   ./scripts/gaia_doctor.sh --fix        # attempt automatic repairs
#   ./scripts/gaia_doctor.sh --json       # machine-readable output
#   ./scripts/gaia_doctor.sh --service X  # check single service
#   ./scripts/gaia_doctor.sh --verbose    # extra debug output
#
# Exit codes:
#   0 — all healthy
#   1 — warnings only (degraded but functional)
#   2 — failures detected (action needed)
#   3 — pre-flight failed (Docker not running, etc.)

set -uo pipefail

GAIA_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

# ── Config ────────────────────────────────────────────────────────────────
MODE="check"          # check | fix
VERBOSE=false
JSON_OUTPUT=false
TARGET_SERVICE=""

# ── Counters ──────────────────────────────────────────────────────────────
PASS=0
WARN=0
FAIL=0
REPAIR=0
declare -a FAILURES=()
declare -a WARNINGS=()
declare -a REPAIRS=()

# ── Colors ────────────────────────────────────────────────────────────────
if [[ -t 1 ]]; then
    RED='\033[0;31m'
    GREEN='\033[0;32m'
    YELLOW='\033[1;33m'
    BLUE='\033[0;34m'
    CYAN='\033[0;36m'
    BOLD='\033[1m'
    DIM='\033[2m'
    RESET='\033[0m'
else
    RED='' GREEN='' YELLOW='' BLUE='' CYAN='' BOLD='' DIM='' RESET=''
fi

# ── Helpers ───────────────────────────────────────────────────────────────
pass_() {
    ((PASS++))
    printf "  ${GREEN}✓${RESET} %-24s %s\n" "$1" "$2"
}

warn_() {
    ((WARN++))
    WARNINGS+=("$3")
    printf "  ${YELLOW}!${RESET} %-24s %s\n" "$1" "$2"
}

fail_() {
    ((FAIL++))
    FAILURES+=("$3")
    printf "  ${RED}✗${RESET} %-24s %s\n" "$1" "$2"
}

repair_() {
    ((REPAIR++))
    REPAIRS+=("$1")
    printf "  ${CYAN}⟳${RESET} %-24s %s\n" "REPAIR" "$1"
}

section() {
    local title="$1"
    local width=64
    local pad_len=$(( width - ${#title} - 2 ))
    local padding=""
    for ((i=0; i<pad_len; i++)); do padding+="─"; done
    printf "\n${BOLD}── %s %s${RESET}\n" "$title" "$padding"
}

verbose() {
    if $VERBOSE; then
        printf "  ${DIM}  %s${RESET}\n" "$1"
    fi
}

# ── Argument Parsing ──────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --fix)      MODE="fix"; shift ;;
        --check)    MODE="check"; shift ;;
        --verbose)  VERBOSE=true; shift ;;
        --json)     JSON_OUTPUT=true; shift ;;
        --service)  TARGET_SERVICE="$2"; shift 2 ;;
        -h|--help)
            printf "Usage: %s [--check|--fix] [--verbose] [--json] [--service NAME]\n" "$0"
            printf "\nModes:\n"
            printf "  --check    Read-only diagnostics (default)\n"
            printf "  --fix      Attempt automatic repairs\n"
            printf "\nOptions:\n"
            printf "  --verbose    Extra debug output\n"
            printf "  --json       Machine-readable JSON output\n"
            printf "  --service X  Check only specific service\n"
            exit 0
            ;;
        *)  printf "Unknown option: %s\n" "$1"; exit 1 ;;
    esac
done

# ── Service Registry ──────────────────────────────────────────────────────
# Format: name|container|host_port|health_path|type|required
# type: live | ha
# host_port: 0 means internal-only (no host port)
declare -a SERVICES=(
    "gaia-core|gaia-core|6415|/health|live|required"
    "gaia-core-candidate|gaia-core-candidate|6416|/health|ha|optional"
    "gaia-mcp|gaia-mcp|8765|/health|live|required"
    "gaia-mcp-candidate|gaia-mcp-candidate|8767|/health|ha|optional"
    "gaia-web|gaia-web|6414|/health|live|required"
    "gaia-orchestrator|gaia-orchestrator|6410|/health|live|required"
    "gaia-prime|gaia-prime|7777|/health|live|required"
    "gaia-study|gaia-study|8766|/health|live|required"
    "gaia-wiki|gaia-wiki|0|/|live|optional"
    "gaia-audio|gaia-audio|8080|/health|live|optional"
    "gaia-audio-candidate|gaia-audio-candidate|8081|/health|ha|optional"
)

# ── Pre-flight ────────────────────────────────────────────────────────────
check_preflight() {
    section "Pre-flight"

    # Docker daemon
    if docker info > /dev/null 2>&1; then
        pass_ "Docker daemon" "reachable"
    else
        fail_ "Docker daemon" "not reachable" "Docker daemon not running"
        return 1
    fi

    # Docker Compose
    local compose_ver
    compose_ver=$(docker compose version --short 2>/dev/null || echo "")
    if [[ -n "$compose_ver" ]]; then
        pass_ "Docker Compose" "v${compose_ver}"
    else
        fail_ "Docker Compose" "not found" "Docker Compose not installed"
        return 1
    fi

    # Docker network
    if docker network inspect gaia-network > /dev/null 2>&1; then
        pass_ "Docker network" "gaia-network"
    else
        fail_ "Docker network" "gaia-network missing" "Docker network gaia-network does not exist"
    fi

    # Compose files
    local compose_ok=true
    for f in docker-compose.yml docker-compose.candidate.yml docker-compose.ha.yml; do
        if [[ -f "$GAIA_ROOT/$f" ]]; then
            verbose "Found $f"
        else
            warn_ "Compose file" "$f missing" "$f not found at $GAIA_ROOT"
            compose_ok=false
        fi
    done
    if $compose_ok; then
        pass_ "Compose files" "all present"
    fi

    # GPU driver
    if command -v nvidia-smi > /dev/null 2>&1; then
        local gpu_info
        gpu_info=$(nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader,nounits 2>/dev/null | head -1)
        if [[ -n "$gpu_info" ]]; then
            local gpu_name mem_used mem_total
            gpu_name=$(echo "$gpu_info" | cut -d',' -f1 | xargs)
            mem_used=$(echo "$gpu_info" | cut -d',' -f2 | xargs)
            mem_total=$(echo "$gpu_info" | cut -d',' -f3 | xargs)
            pass_ "GPU driver" "${gpu_name} (${mem_used}/${mem_total} MiB)"
        else
            warn_ "GPU driver" "nvidia-smi failed" "GPU driver present but nvidia-smi query failed"
        fi
    else
        warn_ "GPU driver" "not found" "nvidia-smi not found — GPU services may not work"
    fi

    return 0
}

# ── Container State ───────────────────────────────────────────────────────
check_container_state() {
    section "Container State"

    for entry in "${SERVICES[@]}"; do
        IFS='|' read -r name container port health_path svc_type required <<< "$entry"

        # Filter by target service if specified
        if [[ -n "$TARGET_SERVICE" && "$name" != "$TARGET_SERVICE" ]]; then
            continue
        fi

        # Check if container exists
        if ! docker container inspect "$container" > /dev/null 2>&1; then
            if [[ "$required" == "required" ]]; then
                fail_ "$name" "container not found" "$name: container does not exist"
            else
                verbose "$name: container not found (optional)"
            fi
            continue
        fi

        # Get container status
        local status health restarts started uptime_str
        status=$(docker inspect --format '{{.State.Status}}' "$container" 2>/dev/null)
        health=$(docker inspect --format '{{if .State.Health}}{{.State.Health.Status}}{{else}}no-healthcheck{{end}}' "$container" 2>/dev/null)
        restarts=$(docker inspect --format '{{.RestartCount}}' "$container" 2>/dev/null || echo "0")
        started=$(docker inspect --format '{{.State.StartedAt}}' "$container" 2>/dev/null)

        # Calculate uptime
        if [[ -n "$started" && "$status" == "running" ]]; then
            local started_epoch now_epoch diff_s
            started_epoch=$(date -d "$started" +%s 2>/dev/null || echo "0")
            now_epoch=$(date +%s)
            diff_s=$(( now_epoch - started_epoch ))
            if (( diff_s >= 86400 )); then
                uptime_str="$(( diff_s / 86400 ))d"
            elif (( diff_s >= 3600 )); then
                uptime_str="$(( diff_s / 3600 ))h"
            elif (( diff_s >= 60 )); then
                uptime_str="$(( diff_s / 60 ))m"
            else
                uptime_str="${diff_s}s"
            fi
        else
            uptime_str="-"
        fi

        # Format port display
        local port_str
        if (( port > 0 )); then
            port_str="$port"
        else
            port_str="int."
        fi

        # Evaluate
        if [[ "$status" == "running" ]]; then
            if [[ "$health" == "healthy" ]]; then
                local detail="running  healthy  ${port_str}  uptime: ${uptime_str}"
                if (( restarts > 5 )); then
                    warn_ "$name" "$detail  restarts: $restarts" "$name: high restart count ($restarts)"
                else
                    pass_ "$name" "$detail"
                fi
            elif [[ "$health" == "unhealthy" ]]; then
                fail_ "$name" "running  ${RED}unhealthy${RESET}  ${port_str}  uptime: ${uptime_str}" "$name: container unhealthy"
                if [[ "$MODE" == "fix" ]]; then
                    repair_ "Restarting unhealthy $name"
                    docker restart "$container" > /dev/null 2>&1
                fi
            elif [[ "$health" == "starting" ]]; then
                warn_ "$name" "running  starting  ${port_str}  uptime: ${uptime_str}" "$name: still starting up"
            else
                pass_ "$name" "running  ${port_str}  uptime: ${uptime_str}"
            fi
        elif [[ "$status" == "exited" || "$status" == "dead" ]]; then
            if [[ "$required" == "required" ]]; then
                fail_ "$name" "${status}" "$name: container ${status}"
            else
                warn_ "$name" "${status} (optional)" "$name: container ${status} (optional)"
            fi
            if [[ "$MODE" == "fix" ]]; then
                repair_ "Starting ${status} container $name"
                docker start "$container" > /dev/null 2>&1
            fi
        else
            warn_ "$name" "status: $status" "$name: unexpected status $status"
        fi
    done
}

# ── HTTP Health Checks ────────────────────────────────────────────────────
check_http_health() {
    section "HTTP Health"

    for entry in "${SERVICES[@]}"; do
        IFS='|' read -r name container port health_path svc_type required <<< "$entry"

        if [[ -n "$TARGET_SERVICE" && "$name" != "$TARGET_SERVICE" ]]; then
            continue
        fi

        # Skip if container isn't running
        local status
        status=$(docker inspect --format '{{.State.Status}}' "$container" 2>/dev/null || echo "not_found")
        if [[ "$status" != "running" ]]; then
            verbose "$name: skipping HTTP check (not running)"
            continue
        fi

        local response_code latency_ms
        if (( port > 0 )); then
            # Use host port
            local start_ns end_ns
            start_ns=$(date +%s%N)
            response_code=$(curl -s -o /dev/null -w '%{http_code}' \
                --connect-timeout 3 --max-time 5 \
                "http://localhost:${port}${health_path}" 2>/dev/null)
            response_code="${response_code:-000}"
            end_ns=$(date +%s%N)
            latency_ms=$(( (end_ns - start_ns) / 1000000 ))
        else
            # Internal-only service — use docker exec
            local start_ns end_ns
            start_ns=$(date +%s%N)
            response_code=$(docker exec "$container" \
                curl -s -o /dev/null -w '%{http_code}' \
                --connect-timeout 3 --max-time 5 \
                "http://localhost:8080${health_path}" 2>/dev/null)
            response_code="${response_code:-000}"
            end_ns=$(date +%s%N)
            latency_ms=$(( (end_ns - start_ns) / 1000000 ))
        fi

        if [[ "$response_code" == "200" ]]; then
            pass_ "$name" "HTTP 200 (${latency_ms}ms)"
        elif [[ "$response_code" == "000" ]]; then
            if [[ "$required" == "required" ]]; then
                fail_ "$name" "HTTP unreachable" "$name: health endpoint unreachable"
            else
                warn_ "$name" "HTTP unreachable (optional)" "$name: health endpoint unreachable (optional)"
            fi
        else
            if [[ "$required" == "required" ]]; then
                fail_ "$name" "HTTP $response_code" "$name: health returned $response_code"
            else
                warn_ "$name" "HTTP $response_code (optional)" "$name: health returned $response_code"
            fi
        fi
    done
}

# ── HA Status ─────────────────────────────────────────────────────────────
check_ha_status() {
    section "HA Status"

    # Check if candidates are running
    local core_candidate_running=false
    local mcp_candidate_running=false

    if docker inspect --format '{{.State.Status}}' gaia-core-candidate 2>/dev/null | grep -q running; then
        core_candidate_running=true
    fi
    if docker inspect --format '{{.State.Status}}' gaia-mcp-candidate 2>/dev/null | grep -q running; then
        mcp_candidate_running=true
    fi

    if $core_candidate_running && $mcp_candidate_running; then
        pass_ "HA candidates" "core + mcp running"
    elif $core_candidate_running || $mcp_candidate_running; then
        warn_ "HA candidates" "partial (core=$core_candidate_running, mcp=$mcp_candidate_running)" \
              "Only some HA candidates are running"
        if [[ "$MODE" == "fix" ]]; then
            repair_ "Starting HA services via ha_start.sh"
            bash "$GAIA_ROOT/scripts/ha_start.sh" > /dev/null 2>&1
        fi
    else
        warn_ "HA candidates" "not running" "HA candidate services not running — no failover available"
        if [[ "$MODE" == "fix" ]]; then
            repair_ "Starting HA services via ha_start.sh"
            bash "$GAIA_ROOT/scripts/ha_start.sh" > /dev/null 2>&1
        fi
    fi

    # Check fallback endpoints configured
    local core_fb mcp_fb
    core_fb=$(docker exec gaia-web printenv CORE_FALLBACK_ENDPOINT 2>/dev/null || echo "")
    mcp_fb=$(docker exec gaia-core printenv MCP_FALLBACK_ENDPOINT 2>/dev/null || echo "")

    if [[ -n "$core_fb" ]]; then
        pass_ "CORE_FALLBACK_ENDPOINT" "$core_fb"
    else
        warn_ "CORE_FALLBACK_ENDPOINT" "not configured" \
              "CORE_FALLBACK_ENDPOINT empty — set in docker-compose.yml and recreate gaia-web"
    fi

    if [[ -n "$mcp_fb" ]]; then
        pass_ "MCP_FALLBACK_ENDPOINT" "$mcp_fb"
    else
        warn_ "MCP_FALLBACK_ENDPOINT" "not configured" \
              "MCP_FALLBACK_ENDPOINT empty — set in docker-compose.yml and recreate gaia-core"
    fi

    # Maintenance mode
    local maintenance
    if docker exec gaia-orchestrator test -f /shared/ha_maintenance 2>/dev/null; then
        maintenance="ON"
        warn_ "Maintenance mode" "ON (failover disabled)" \
              "Maintenance mode is ON — failover routing disabled"
    else
        maintenance="OFF"
        pass_ "Maintenance mode" "OFF (failover enabled)"
    fi

    # Orchestrator HA status
    local ha_status
    ha_status=$(curl -sf --max-time 3 http://localhost:6410/status 2>/dev/null | \
                python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('health_watchdog',{}).get('ha_status','unknown'))" 2>/dev/null || echo "unknown")
    if [[ "$ha_status" == "active" ]]; then
        pass_ "Watchdog HA status" "$ha_status"
    elif [[ "$ha_status" == "degraded" ]]; then
        warn_ "Watchdog HA status" "$ha_status" "HA status degraded — candidate may be unhealthy"
    elif [[ "$ha_status" == "failover_active" ]]; then
        warn_ "Watchdog HA status" "${RED}$ha_status${RESET}" "HA failover is active — live service is down"
    elif [[ "$ha_status" == "failed" ]]; then
        fail_ "Watchdog HA status" "${RED}$ha_status${RESET}" "HA status FAILED — both live and candidate down"
    else
        verbose "Watchdog HA status: $ha_status (could not query)"
    fi
}

# ── Inter-Service Connectivity ────────────────────────────────────────────
check_connectivity() {
    section "Inter-Service Connectivity"

    # Only check if containers are running
    local routes=(
        "gaia-web|gaia-core|gaia-core:6415|/health"
        "gaia-core|gaia-mcp|gaia-mcp:8765|/health"
        "gaia-core|gaia-prime|gaia-prime:7777|/health"
        "gaia-core|gaia-study|gaia-study:8766|/health"
        "gaia-web|gaia-wiki|gaia-wiki:8080|/"
    )

    for route in "${routes[@]}"; do
        IFS='|' read -r from_container to_name target_host health_path <<< "$route"

        # Skip if source container isn't running
        local status
        status=$(docker inspect --format '{{.State.Status}}' "$from_container" 2>/dev/null || echo "not_found")
        if [[ "$status" != "running" ]]; then
            verbose "$from_container → $to_name: skipping (source not running)"
            continue
        fi

        local label="${from_container} → ${to_name}"
        local start_ns end_ns latency_ms result
        start_ns=$(date +%s%N)
        result=$(docker exec "$from_container" \
            curl -s -o /dev/null -w '%{http_code}' \
            --connect-timeout 3 --max-time 5 \
            "http://${target_host}${health_path}" 2>/dev/null)
        result="${result:-000}"
        end_ns=$(date +%s%N)
        latency_ms=$(( (end_ns - start_ns) / 1000000 ))

        if [[ "$result" == "200" ]]; then
            pass_ "$label" "OK (${latency_ms}ms)"
        elif [[ "$result" == "000" ]]; then
            # Wiki might not have curl — try wget
            if [[ "$to_name" == "gaia-wiki" ]]; then
                verbose "Retrying $label with wget"
                result=$(docker exec "$from_container" \
                    wget -q -O /dev/null --timeout=3 \
                    "http://${target_host}${health_path}" 2>/dev/null && echo "200" || echo "000")
                if [[ "$result" == "200" ]]; then
                    pass_ "$label" "OK"
                    continue
                fi
            fi
            warn_ "$label" "unreachable" "$label: connection failed"
        else
            warn_ "$label" "HTTP $result" "$label: returned $result"
        fi
    done
}

# ── Volumes ───────────────────────────────────────────────────────────────
check_volumes() {
    section "Volumes & Mounts"

    # Named Docker volumes
    for vol in gaia-shared gaia-sandbox gaia-candidate-shared; do
        if docker volume inspect "$vol" > /dev/null 2>&1; then
            pass_ "$vol" "exists"
        else
            if [[ "$vol" == "gaia-candidate-shared" ]]; then
                verbose "$vol: not found (HA not started yet)"
            else
                warn_ "$vol" "missing" "Docker volume $vol does not exist"
            fi
        fi
    done

    # Key paths inside gaia-core
    if docker inspect --format '{{.State.Status}}' gaia-core 2>/dev/null | grep -q running; then
        local paths=("/knowledge" "/vector_store" "/shared" "/models" "/logs")
        for p in "${paths[@]}"; do
            if docker exec gaia-core test -d "$p" 2>/dev/null; then
                pass_ "$p" "accessible (gaia-core)"
            else
                fail_ "$p" "not accessible" "Path $p not accessible inside gaia-core"
            fi
        done
    else
        verbose "Skipping mount checks — gaia-core not running"
    fi
}

# ── gaia-common Sync ──────────────────────────────────────────────────────
check_common_sync() {
    section "gaia-common Sync"

    local prod_dir="$GAIA_ROOT/gaia-common/gaia_common"
    local cand_dir="$GAIA_ROOT/candidates/gaia-common/gaia_common"

    if [[ ! -d "$cand_dir" ]]; then
        warn_ "Candidate gaia-common" "directory not found" "candidates/gaia-common not found"
        return
    fi

    # Check key files
    local key_files=(
        "utils/service_client.py"
        "utils/resilience.py"
        "utils/tools_registry.py"
        "protocols/cognition_packet.py"
        "constants/gaia_constants.json"
    )

    local diffs=0
    local diff_files=()
    for f in "${key_files[@]}"; do
        if [[ ! -f "$prod_dir/$f" ]]; then
            verbose "Production $f: not found"
            continue
        fi
        if [[ ! -f "$cand_dir/$f" ]]; then
            ((diffs++))
            diff_files+=("$f (missing in candidate)")
            continue
        fi
        if ! diff -q "$prod_dir/$f" "$cand_dir/$f" > /dev/null 2>&1; then
            ((diffs++))
            diff_files+=("$f")
        fi
    done

    if (( diffs == 0 )); then
        pass_ "Key files" "all synced (${#key_files[@]} checked)"
    else
        warn_ "Key files" "$diffs differ" "gaia-common: $diffs key files differ between production and candidate"
        for df in "${diff_files[@]}"; do
            printf "    ${DIM}  - %s${RESET}\n" "$df"
        done
    fi
}

# ── Stale Mount Detection ────────────────────────────────────────────────
check_stale_mounts() {
    section "Stale Mount Detection"

    # Map: container → host source dir for gaia-common (or app source)
    # We check key files whose changes require a container restart
    declare -A MOUNT_MAP=(
        ["gaia-core"]="$GAIA_ROOT/gaia-common:$GAIA_ROOT/gaia-core"
        ["gaia-web"]="$GAIA_ROOT/gaia-common:$GAIA_ROOT/gaia-web"
        ["gaia-mcp"]="$GAIA_ROOT/gaia-common:$GAIA_ROOT/gaia-mcp"
        ["gaia-study"]="$GAIA_ROOT/gaia-common:$GAIA_ROOT/gaia-study"
        ["gaia-audio"]="$GAIA_ROOT/gaia-common:$GAIA_ROOT/gaia-audio"
        ["gaia-core-candidate"]="$GAIA_ROOT/candidates/gaia-common:$GAIA_ROOT/candidates/gaia-core"
        ["gaia-mcp-candidate"]="$GAIA_ROOT/candidates/gaia-common"
    )

    local stale_count=0
    local stale_containers=()

    for container in "${!MOUNT_MAP[@]}"; do
        # Skip if not running
        local status
        status=$(docker inspect --format '{{.State.Status}}' "$container" 2>/dev/null || echo "not_found")
        if [[ "$status" != "running" ]]; then
            verbose "$container: skipping (not running)"
            continue
        fi

        # Get container start time as epoch
        local started_at started_epoch
        started_at=$(docker inspect --format '{{.State.StartedAt}}' "$container" 2>/dev/null)
        started_epoch=$(date -d "$started_at" +%s 2>/dev/null || echo "0")
        if (( started_epoch == 0 )); then
            verbose "$container: could not parse start time"
            continue
        fi

        # Check all mount source dirs for this container
        local newest_file="" newest_mtime=0
        IFS=':' read -ra dirs <<< "${MOUNT_MAP[$container]}"
        for dir in "${dirs[@]}"; do
            if [[ ! -d "$dir" ]]; then
                continue
            fi
            # Find the most recently modified .py or .json file
            while IFS= read -r -d '' fpath; do
                local fmtime
                fmtime=$(stat -c '%Y' "$fpath" 2>/dev/null || echo "0")
                if (( fmtime > newest_mtime )); then
                    newest_mtime=$fmtime
                    newest_file="$fpath"
                fi
            done < <(find "$dir" -maxdepth 4 -type f \( -name '*.py' -o -name '*.json' \) -print0 2>/dev/null)
        done

        if (( newest_mtime == 0 )); then
            verbose "$container: no source files found"
            continue
        fi

        local drift_s=$(( newest_mtime - started_epoch ))
        if (( drift_s > 5 )); then
            # Source is newer than container — stale
            local rel_file="${newest_file#"$GAIA_ROOT/"}"
            ((stale_count++))
            stale_containers+=("$container")
            warn_ "$container" "stale (${drift_s}s behind)" \
                  "$container needs restart — $rel_file modified ${drift_s}s after start"
            if [[ "$MODE" == "fix" ]]; then
                repair_ "Restarting stale $container"
                docker restart "$container" > /dev/null 2>&1
            fi
        else
            pass_ "$container" "up to date"
        fi
    done

    if (( stale_count == 0 )); then
        pass_ "All containers" "source mounts current"
    fi
}

# ── Image Staleness Detection ────────────────────────────────────────────
check_image_staleness() {
    section "Image Staleness"

    # Map: container → Dockerfile path : requirements file(s) (colon-separated)
    # These are the build-time inputs — if any is newer than the image, rebuild needed
    declare -A BUILD_MAP=(
        ["gaia-core"]="$GAIA_ROOT/gaia-core/Dockerfile:$GAIA_ROOT/gaia-core/requirements.txt:$GAIA_ROOT/gaia-common/requirements.txt"
        ["gaia-web"]="$GAIA_ROOT/gaia-web/Dockerfile:$GAIA_ROOT/gaia-web/requirements.txt:$GAIA_ROOT/gaia-common/requirements.txt"
        ["gaia-mcp"]="$GAIA_ROOT/gaia-mcp/Dockerfile:$GAIA_ROOT/gaia-mcp/requirements.txt:$GAIA_ROOT/gaia-common/requirements.txt"
        ["gaia-study"]="$GAIA_ROOT/gaia-study/Dockerfile:$GAIA_ROOT/gaia-study/requirements.txt:$GAIA_ROOT/gaia-common/requirements.txt:$GAIA_ROOT/gaia-common/requirements-vector.txt"
        ["gaia-audio"]="$GAIA_ROOT/gaia-audio/Dockerfile:$GAIA_ROOT/gaia-audio/requirements.txt:$GAIA_ROOT/gaia-common/requirements.txt"
        ["gaia-orchestrator"]="$GAIA_ROOT/gaia-orchestrator/Dockerfile:$GAIA_ROOT/gaia-orchestrator/requirements.txt"
        ["gaia-prime"]="$GAIA_ROOT/gaia-prime/Dockerfile"
        ["gaia-wiki"]="$GAIA_ROOT/gaia-wiki/Dockerfile"
        ["gaia-doctor"]="$GAIA_ROOT/gaia-doctor/Dockerfile"
        ["gaia-core-candidate"]="$GAIA_ROOT/candidates/gaia-core/Dockerfile:$GAIA_ROOT/candidates/gaia-core/requirements.txt:$GAIA_ROOT/candidates/gaia-common/requirements.txt"
        ["gaia-mcp-candidate"]="$GAIA_ROOT/candidates/gaia-mcp/Dockerfile:$GAIA_ROOT/candidates/gaia-mcp/requirements.txt:$GAIA_ROOT/candidates/gaia-common/requirements.txt"
    )

    local rebuild_count=0

    for container in "${!BUILD_MAP[@]}"; do
        # Skip if container doesn't exist
        if ! docker container inspect "$container" > /dev/null 2>&1; then
            verbose "$container: skipping (not found)"
            continue
        fi

        # Get image creation time as epoch
        local image_name image_created image_epoch
        image_name=$(docker inspect --format '{{.Config.Image}}' "$container" 2>/dev/null)
        if [[ -z "$image_name" ]]; then
            verbose "$container: could not determine image"
            continue
        fi
        image_created=$(docker image inspect --format '{{.Created}}' "$image_name" 2>/dev/null)
        if [[ -z "$image_created" ]]; then
            verbose "$container: could not inspect image $image_name"
            continue
        fi
        image_epoch=$(date -d "$image_created" +%s 2>/dev/null || echo "0")
        if (( image_epoch == 0 )); then
            verbose "$container: could not parse image creation time"
            continue
        fi

        # Check each build input file
        local newest_file="" newest_mtime=0
        IFS=':' read -ra build_files <<< "${BUILD_MAP[$container]}"
        for bf in "${build_files[@]}"; do
            if [[ ! -f "$bf" ]]; then
                verbose "$container: build input $bf not found"
                continue
            fi
            local fmtime
            fmtime=$(stat -c '%Y' "$bf" 2>/dev/null || echo "0")
            if (( fmtime > newest_mtime )); then
                newest_mtime=$fmtime
                newest_file="$bf"
            fi
        done

        if (( newest_mtime == 0 )); then
            verbose "$container: no build input files found"
            continue
        fi

        local drift_s=$(( newest_mtime - image_epoch ))
        if (( drift_s > 5 )); then
            local rel_file="${newest_file#"$GAIA_ROOT/"}"
            ((rebuild_count++))
            local drift_str
            if (( drift_s >= 86400 )); then
                drift_str="$(( drift_s / 86400 ))d"
            elif (( drift_s >= 3600 )); then
                drift_str="$(( drift_s / 3600 ))h"
            elif (( drift_s >= 60 )); then
                drift_str="$(( drift_s / 60 ))m"
            else
                drift_str="${drift_s}s"
            fi
            warn_ "$container" "rebuild needed (${drift_str})" \
                  "$container image outdated — $rel_file modified ${drift_str} after build"
            if [[ "$MODE" == "fix" ]]; then
                repair_ "Rebuilding $container"
                COMPOSE_FILE="$GAIA_ROOT/docker-compose.yml" \
                    docker compose --env-file "$GAIA_ROOT/.env.discord" \
                    build "$container" > /dev/null 2>&1 && \
                repair_ "Restarting rebuilt $container" && \
                    docker compose --env-file "$GAIA_ROOT/.env.discord" \
                    up -d "$container" > /dev/null 2>&1
            fi
        else
            pass_ "$container" "image current"
        fi
    done

    if (( rebuild_count == 0 )); then
        pass_ "All images" "build inputs current"
    fi
}

# ── Knowledge Gaps ───────────────────────────────────────────────────────
check_knowledge_gaps() {
    section "Knowledge Gaps"

    local core_docs_dir="$GAIA_ROOT/knowledge/system_reference/core_documents"
    local blueprints_dir="$GAIA_ROOT/knowledge/blueprints"
    local cognition_dir="$GAIA_ROOT/gaia-core/gaia_core/cognition"
    local gap_dir="$GAIA_ROOT/knowledge/gap_audit"
    local gap_count=0

    mkdir -p "$gap_dir"

    # Start building JSON gap report
    local gaps_json='[]'

    # --- Check 1: Services without a blueprint file ---
    for entry in "${SERVICES[@]}"; do
        IFS='|' read -r name container port health_path svc_type required <<< "$entry"

        # Skip candidate/HA variants — they share the base service blueprint
        if [[ "$name" == *"-candidate"* ]]; then
            continue
        fi

        # Look for GAIA_NAME.md (uppercase) or name.md / name.yaml (lowercase)
        local upper_name
        upper_name=$(echo "$name" | tr '[:lower:]-' '[:upper:]_')
        local found=false

        for pattern in "${blueprints_dir}/${upper_name}.md" \
                       "${blueprints_dir}/${name}.md" \
                       "${blueprints_dir}/${name}.yaml"; do
            if [[ -f "$pattern" ]]; then
                found=true
                break
            fi
        done

        if $found; then
            verbose "$name: blueprint found"
        else
            ((gap_count++))
            warn_ "$name" "no blueprint file" "$name: no blueprint found (expected ${upper_name}.md or ${name}.yaml)"
            gaps_json=$(echo "$gaps_json" | python3 -c "
import sys, json
gaps = json.load(sys.stdin)
gaps.append({'type': 'missing_blueprint', 'service': '$name', 'expected': ['${upper_name}.md', '${name}.yaml']})
json.dump(gaps, sys.stdout)
")
        fi
    done

    # --- Check 2: Cognition modules without a core_documents entry ---
    if [[ -d "$cognition_dir" ]]; then
        local skip_modules=("__init__" "__pycache__" "tests" "external_voice")
        while IFS= read -r pyfile; do
            local mod_name
            mod_name=$(basename "$pyfile" .py)

            # Skip non-module files and known infrastructure modules
            local skip=false
            for s in "${skip_modules[@]}"; do
                if [[ "$mod_name" == "$s" ]]; then
                    skip=true
                    break
                fi
            done
            if $skip; then continue; fi

            # Check if a corresponding doc exists in core_documents
            local doc_found=false
            for ext in .md .yaml; do
                if [[ -f "${core_docs_dir}/${mod_name}${ext}" ]] || \
                   [[ -f "${core_docs_dir}/${mod_name}_spec${ext}" ]] || \
                   [[ -f "${core_docs_dir}/${mod_name}_reference${ext}" ]]; then
                    doc_found=true
                    break
                fi
            done

            if $doc_found; then
                verbose "$mod_name: core doc found"
            else
                ((gap_count++))
                warn_ "$mod_name" "no core doc" "$mod_name: cognition module has no corresponding core_documents entry"
                gaps_json=$(echo "$gaps_json" | python3 -c "
import sys, json
gaps = json.load(sys.stdin)
gaps.append({'type': 'undocumented_module', 'module': '$mod_name', 'search_dir': 'knowledge/system_reference/core_documents/'})
json.dump(gaps, sys.stdout)
")
            fi
        done < <(find "$cognition_dir" -maxdepth 1 -name '*.py' -type f 2>/dev/null | sort)
    fi

    # --- Check 3: Core documents not in the dynamic registry ---
    # Sanity check — any .md in core_documents that wouldn't be found by the scanner
    if [[ -d "$core_docs_dir" ]]; then
        local doc_count
        doc_count=$(find "$core_docs_dir" -maxdepth 1 -name '*.md' -type f 2>/dev/null | wc -l)
        pass_ "Core documents" "${doc_count} files on disk"
    fi

    if [[ -d "$blueprints_dir" ]]; then
        local bp_count
        bp_count=$(find "$blueprints_dir" -maxdepth 1 \( -name '*.md' -o -name '*.yaml' \) -type f 2>/dev/null | wc -l)
        pass_ "Blueprint files" "${bp_count} files on disk"
    fi

    # --- Write gap report ---
    local timestamp
    timestamp=$(date -u '+%Y-%m-%dT%H:%M:%SZ')
    python3 -c "
import json, sys
gaps = json.loads('''$gaps_json''')
report = {
    'timestamp': '$timestamp',
    'gap_count': len(gaps),
    'gaps': gaps
}
with open('$gap_dir/latest_gaps.json', 'w') as f:
    json.dump(report, f, indent=2)
" 2>/dev/null

    if (( gap_count == 0 )); then
        pass_ "Knowledge coverage" "no gaps detected"
    else
        verbose "Gap report written to knowledge/gap_audit/latest_gaps.json"
    fi
}

# ── Session State Freshness ───────────────────────────────────────────────
check_session_freshness() {
    section "Session State"

    # Check live sessions.json timestamp
    local live_ts cand_ts now_ts
    live_ts=$(docker exec gaia-core stat -c '%Y' /shared/sessions.json 2>/dev/null || echo "0")
    now_ts=$(date +%s)

    if (( live_ts > 0 )); then
        local age_s=$(( now_ts - live_ts ))
        if (( age_s < 300 )); then
            pass_ "Live sessions.json" "age: ${age_s}s"
        else
            warn_ "Live sessions.json" "age: ${age_s}s (stale)" "Live sessions.json is ${age_s}s old"
        fi
    else
        verbose "Could not stat live sessions.json"
    fi

    # Check candidate sessions.json if HA candidate is running
    cand_ts=$(docker exec gaia-core-candidate stat -c '%Y' /shared/sessions.json 2>/dev/null || echo "0")
    if (( cand_ts > 0 && live_ts > 0 )); then
        local drift_s=$(( live_ts - cand_ts ))
        if (( drift_s < 0 )); then drift_s=$(( -drift_s )); fi

        if (( drift_s < 120 )); then
            pass_ "Candidate sync drift" "${drift_s}s"
        elif (( drift_s < 600 )); then
            warn_ "Candidate sync drift" "${drift_s}s" "Candidate session state ${drift_s}s behind live"
            if [[ "$MODE" == "fix" ]]; then
                repair_ "Running incremental session sync"
                bash "$GAIA_ROOT/scripts/ha_sync.sh" --incremental > /dev/null 2>&1
            fi
        else
            fail_ "Candidate sync drift" "${drift_s}s (stale)" "Candidate session state ${drift_s}s behind live — run ha_sync.sh"
            if [[ "$MODE" == "fix" ]]; then
                repair_ "Running incremental session sync"
                bash "$GAIA_ROOT/scripts/ha_sync.sh" --incremental > /dev/null 2>&1
            fi
        fi
    elif (( cand_ts == 0 )); then
        verbose "Candidate sessions.json not found (HA not active)"
    fi
}

# ── Resources ─────────────────────────────────────────────────────────────
check_resources() {
    section "Resources"

    # Disk space on /gaia
    local disk_info avail_pct
    disk_info=$(df -h /gaia 2>/dev/null | tail -1)
    if [[ -n "$disk_info" ]]; then
        local avail use_pct
        avail=$(echo "$disk_info" | awk '{print $4}')
        use_pct=$(echo "$disk_info" | awk '{print $5}' | tr -d '%')
        if (( use_pct < 85 )); then
            pass_ "Disk /gaia" "${avail} free (${use_pct}% used)"
        elif (( use_pct < 95 )); then
            warn_ "Disk /gaia" "${avail} free (${use_pct}% used)" "Disk usage at ${use_pct}% — consider cleanup"
        else
            fail_ "Disk /gaia" "${avail} free (${use_pct}% used)" "Disk critically full (${use_pct}%)"
        fi
    fi

    # GPU memory
    if command -v nvidia-smi > /dev/null 2>&1; then
        local gpu_mem
        gpu_mem=$(nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits 2>/dev/null | head -1)
        if [[ -n "$gpu_mem" ]]; then
            local used total pct
            used=$(echo "$gpu_mem" | cut -d',' -f1 | xargs)
            total=$(echo "$gpu_mem" | cut -d',' -f2 | xargs)
            if (( total > 0 )); then
                pct=$(( used * 100 / total ))
                if (( pct < 90 )); then
                    pass_ "GPU memory" "${used}/${total} MiB (${pct}%)"
                else
                    warn_ "GPU memory" "${used}/${total} MiB (${pct}%)" "GPU memory at ${pct}%"
                fi
            fi
        fi
    fi

    # Docker disk usage (summary)
    local docker_images docker_containers
    docker_images=$(docker system df --format '{{.Type}}\t{{.Size}}' 2>/dev/null | grep Images | cut -f2)
    docker_containers=$(docker system df --format '{{.Type}}\t{{.Size}}' 2>/dev/null | grep Containers | cut -f2)
    if [[ -n "$docker_images" ]]; then
        pass_ "Docker disk" "Images: ${docker_images}  Containers: ${docker_containers:-0B}"
    fi

    # Top 3 memory-hungry containers
    if $VERBOSE; then
        printf "\n  ${DIM}  Container memory usage:${RESET}\n"
        docker stats --no-stream --format '{{.Name}}\t{{.MemUsage}}\t{{.MemPerc}}' 2>/dev/null | \
            sort -t$'\t' -k3 -rn | head -5 | \
            while IFS=$'\t' read -r cname mem pct; do
                printf "  ${DIM}    %-28s %s  %s${RESET}\n" "$cname" "$mem" "$pct"
            done
    fi
}

# ── Summary Report ────────────────────────────────────────────────────────
print_header() {
    printf "\n${BOLD}╔══════════════════════════════════════════════════════════════════╗${RESET}\n"
    printf "${BOLD}║                    GAIA Doctor — System Report                  ║${RESET}\n"
    printf "${BOLD}║                    %-43s ║${RESET}\n" "$(date '+%Y-%m-%d %H:%M:%S')"
    if [[ "$MODE" == "fix" ]]; then
        printf "${BOLD}║                    ${CYAN}Mode: FIX (repairs enabled)${RESET}${BOLD}                 ║${RESET}\n"
    else
        printf "${BOLD}║                    ${DIM}Mode: CHECK (read-only)${RESET}${BOLD}                     ║${RESET}\n"
    fi
    printf "${BOLD}╚══════════════════════════════════════════════════════════════════╝${RESET}\n"
}

print_summary() {
    printf "\n${BOLD}══════════════════════════════════════════════════════════════════${RESET}\n"

    local total=$(( PASS + WARN + FAIL ))
    local color="$GREEN"
    if (( FAIL > 0 )); then color="$RED"
    elif (( WARN > 0 )); then color="$YELLOW"; fi

    printf "Summary: ${GREEN}%d passed${RESET}, ${YELLOW}%d warnings${RESET}, ${RED}%d failures${RESET}" "$PASS" "$WARN" "$FAIL"
    if (( REPAIR > 0 )); then
        printf ", ${CYAN}%d repairs attempted${RESET}" "$REPAIR"
    fi
    printf "\n"

    if (( FAIL > 0 )); then
        printf "\n${RED}${BOLD}FAILURES:${RESET}\n"
        local i=1
        for f in "${FAILURES[@]}"; do
            printf "  ${RED}%d.${RESET} %s\n" "$i" "$f"
            ((i++))
        done
    fi

    if (( WARN > 0 )); then
        printf "\n${YELLOW}${BOLD}WARNINGS:${RESET}\n"
        local i=1
        for w in "${WARNINGS[@]}"; do
            printf "  ${YELLOW}%d.${RESET} %s\n" "$i" "$w"
            ((i++))
        done
    fi

    if (( REPAIR > 0 )); then
        printf "\n${CYAN}${BOLD}REPAIRS ATTEMPTED:${RESET}\n"
        local i=1
        for r in "${REPAIRS[@]}"; do
            printf "  ${CYAN}%d.${RESET} %s\n" "$i" "$r"
            ((i++))
        done
    fi

    printf "${BOLD}══════════════════════════════════════════════════════════════════${RESET}\n"
}

# ── Main ──────────────────────────────────────────────────────────────────
main() {
    print_header

    if ! check_preflight; then
        print_summary
        exit 3
    fi

    check_container_state
    check_http_health
    check_ha_status
    check_connectivity
    check_volumes
    check_common_sync
    check_stale_mounts
    check_image_staleness
    check_knowledge_gaps
    check_session_freshness
    check_resources

    print_summary

    if (( FAIL > 0 )); then
        exit 2
    elif (( WARN > 0 )); then
        exit 1
    else
        exit 0
    fi
}

main "$@"
