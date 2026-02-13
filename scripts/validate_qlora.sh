#!/bin/bash
#
# validate_qlora.sh — QLoRA Validation Test Cycle
#
# Sprint 4 deliverable: End-to-end QLoRA adapter validation pipeline.
# Called by promote_pipeline.sh Stage 7 (--qlora flag), or standalone.
#
# Stages:
#   1. Blueprint freshness check  — verify blueprint references still exist
#   2. Curriculum validation       — ensure training data is present and well-formed
#   3. GPU handoff (Prime → Study) — orchestrator releases GPU to study service
#   4. Training                    — POST /study/start, poll /study/status
#   5. GPU reclaim (Study → Prime) — orchestrator returns GPU to inference
#   6. Adapter validation          — run validate_adapter.py on held-out data
#   7. Report                      — log results, register if passed
#
# Usage:
#   ./scripts/validate_qlora.sh [OPTIONS]
#
# Options:
#   --adapter <name>     Adapter to validate (default: json-architect)
#   --dry-run            Parse, check files, validate data only — no GPU/training
#   --skip-training      Skip stages 3-5, validate existing adapter only
#   --skip-blueprints    Skip blueprint freshness check
#   --endpoint <url>     vLLM endpoint (default: http://localhost:7777)
#   --study-url <url>    gaia-study endpoint (default: http://localhost:8766)
#   --orch-url <url>     orchestrator endpoint (default: http://localhost:6410)
#   --max-examples <n>   Cap validation examples (default: 50)
#   --threshold <f>      Minimum pass score 0.0-1.0 (default: 0.6)
#   --baseline           Also score the base model for comparison
#   --timeout <s>        Training timeout in seconds (default: 3600)
#   --verbose            Show detailed output
#
# Examples:
#   ./scripts/validate_qlora.sh --dry-run
#   ./scripts/validate_qlora.sh --adapter json-architect --skip-training
#   ./scripts/validate_qlora.sh --adapter json-architect --baseline
#

set -euo pipefail

# ═══════════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════════

GAIA_ROOT="/gaia/GAIA_Project"
SCRIPTS_DIR="$GAIA_ROOT/scripts"
CURRICULA_DIR="$GAIA_ROOT/knowledge/curricula"
BLUEPRINTS_DIR="$GAIA_ROOT/knowledge/blueprints"
CANDIDATES_DIR="$GAIA_ROOT/candidates"
STUDY_SCRIPTS="$CANDIDATES_DIR/gaia-study/scripts"

# Defaults
ADAPTER="json-architect"
DRY_RUN=false
SKIP_TRAINING=false
SKIP_BLUEPRINTS=false
ENDPOINT="http://localhost:7777"
STUDY_URL="http://localhost:8766"
ORCH_URL="http://localhost:6410"
MAX_EXAMPLES=50
THRESHOLD=0.6
DO_BASELINE=false
TRAINING_TIMEOUT=3600
VERBOSE=false

# Terminal formatting
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
CYAN='\033[0;36m'
DIM='\033[2m'
BOLD='\033[1m'
RESET='\033[0m'

# ═══════════════════════════════════════════════════════════════════════════
# Parse Arguments
# ═══════════════════════════════════════════════════════════════════════════

while [[ $# -gt 0 ]]; do
    case $1 in
        --adapter)      ADAPTER="$2"; shift 2 ;;
        --dry-run)      DRY_RUN=true; shift ;;
        --skip-training) SKIP_TRAINING=true; shift ;;
        --skip-blueprints) SKIP_BLUEPRINTS=true; shift ;;
        --endpoint)     ENDPOINT="$2"; shift 2 ;;
        --study-url)    STUDY_URL="$2"; shift 2 ;;
        --orch-url)     ORCH_URL="$2"; shift 2 ;;
        --max-examples) MAX_EXAMPLES="$2"; shift 2 ;;
        --threshold)    THRESHOLD="$2"; shift 2 ;;
        --baseline)     DO_BASELINE=true; shift ;;
        --timeout)      TRAINING_TIMEOUT="$2"; shift 2 ;;
        --verbose|-v)   VERBOSE=true; shift ;;
        --help|-h)
            head -40 "$0" | tail -35
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════

PIPELINE_START=$(date +%s)
STAGE_RESULTS=()

log() {
    echo -e "$1"
}

stage_header() {
    local num=$1
    local name=$2
    echo ""
    echo -e "${BOLD}${CYAN}╔══════════════════════════════════════════════════════════════╗${RESET}"
    echo -e "${BOLD}${CYAN}║  Stage $num: $name${RESET}"
    echo -e "${BOLD}${CYAN}╚══════════════════════════════════════════════════════════════╝${RESET}"
}

stage_pass() {
    local name=$1
    STAGE_RESULTS+=("PASS|$name")
    log "  ${GREEN}✓${RESET} Stage passed: $name"
}

stage_skip() {
    local name=$1
    STAGE_RESULTS+=("SKIP|$name")
    log "  ${DIM}⊘${RESET} Stage skipped: $name"
}

stage_warn() {
    local name=$1
    STAGE_RESULTS+=("WARN|$name")
    log "  ${YELLOW}⚠${RESET} Stage warning: $name"
}

stage_fail() {
    local name=$1
    STAGE_RESULTS+=("FAIL|$name")
    log "  ${RED}✗${RESET} Stage failed: $name"
}

# ═══════════════════════════════════════════════════════════════════════════
# Banner
# ═══════════════════════════════════════════════════════════════════════════

echo ""
echo -e "${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${RESET}"
echo -e "${BOLD}  GAIA QLoRA Validation Pipeline${RESET}"
echo -e "${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${RESET}"
echo -e "  Adapter:       ${CYAN}$ADAPTER${RESET}"
echo -e "  Mode:          $([ "$DRY_RUN" = true ] && echo "${YELLOW}DRY RUN${RESET}" || echo "LIVE")"
echo -e "  Training:      $([ "$SKIP_TRAINING" = true ] && echo "skipped" || echo "enabled")"
echo -e "  Threshold:     $THRESHOLD"
echo -e "  Max examples:  $MAX_EXAMPLES"
echo -e "${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${RESET}"

# ═══════════════════════════════════════════════════════════════════════════
# Stage 1: Blueprint Freshness Check
# ═══════════════════════════════════════════════════════════════════════════

stage_header 1 "Blueprint Freshness Check"

if [ "$SKIP_BLUEPRINTS" = true ]; then
    stage_skip "Blueprint Freshness Check"
else
    stale_count=0
    total_refs=0
    stale_files=""

    if [ -d "$BLUEPRINTS_DIR" ]; then
        for blueprint in "$BLUEPRINTS_DIR"/*.md; do
            [ -f "$blueprint" ] || continue
            bp_name=$(basename "$blueprint")

            # Extract file path references from blueprints
            # Patterns: `path/to/file.py`, candidates/gaia-*/path, gaia-*/path
            refs=$(grep -oE '(candidates/[a-zA-Z_-]+/[a-zA-Z_/.-]+\.(py|json|yaml|yml|sh|md)|gaia-[a-zA-Z_-]+/[a-zA-Z_/.-]+\.(py|json|yaml|yml|sh|md))' "$blueprint" 2>/dev/null || true)

            for ref in $refs; do
                total_refs=$((total_refs + 1))
                # Resolve path relative to GAIA_ROOT
                if [ ! -f "$GAIA_ROOT/$ref" ]; then
                    stale_count=$((stale_count + 1))
                    stale_files="$stale_files\n    $bp_name → $ref"
                    [ "$VERBOSE" = true ] && log "    ${YELLOW}stale:${RESET} $bp_name → $ref"
                fi
            done
        done

        log "  Scanned $(ls "$BLUEPRINTS_DIR"/*.md 2>/dev/null | wc -l) blueprints"
        log "  Found $total_refs file references, $stale_count stale"

        if [ $stale_count -gt 0 ]; then
            if [ "$VERBOSE" = true ]; then
                log "  Stale references:$stale_files"
            fi
            stage_warn "Blueprint Freshness Check"
        else
            stage_pass "Blueprint Freshness Check"
        fi
    else
        log "  ${YELLOW}⚠${RESET} Blueprints directory not found: $BLUEPRINTS_DIR"
        stage_warn "Blueprint Freshness Check"
    fi
fi

# ═══════════════════════════════════════════════════════════════════════════
# Stage 2: Curriculum Validation
# ═══════════════════════════════════════════════════════════════════════════

stage_header 2 "Curriculum Validation"

CURRICULUM_DIR="$CURRICULA_DIR/$ADAPTER"
CURRICULUM_SPEC="$CURRICULUM_DIR/curriculum.json"
TRAIN_FILE="$CURRICULUM_DIR/train.jsonl"
VAL_FILE="$CURRICULUM_DIR/validation.jsonl"
METADATA_FILE="$CURRICULUM_DIR/generation_metadata.json"

curriculum_ok=true

# Check curriculum spec exists
if [ ! -f "$CURRICULUM_SPEC" ]; then
    log "  ${RED}✗${RESET} Curriculum spec not found: $CURRICULUM_SPEC"
    curriculum_ok=false
else
    log "  ${GREEN}✓${RESET} Curriculum spec: $CURRICULUM_SPEC"
    if [ "$VERBOSE" = true ]; then
        # Show key fields
        adapter_name=$(python3 -c "import json; print(json.load(open('$CURRICULUM_SPEC'))['adapter_name'])" 2>/dev/null || echo "?")
        total_samples=$(python3 -c "import json; print(json.load(open('$CURRICULUM_SPEC'))['total_target_samples'])" 2>/dev/null || echo "?")
        log "    adapter=$adapter_name  target_samples=$total_samples"
    fi
fi

# Check training data exists
if [ ! -f "$TRAIN_FILE" ]; then
    log "  ${RED}✗${RESET} Training data not found: $TRAIN_FILE"
    log "  Attempting to generate curriculum..."

    if [ -f "$STUDY_SCRIPTS/generate_curriculum.py" ]; then
        python3 "$STUDY_SCRIPTS/generate_curriculum.py" \
            --output-dir "$CURRICULUM_DIR" \
            --seed 42
        if [ -f "$TRAIN_FILE" ]; then
            log "  ${GREEN}✓${RESET} Curriculum generated successfully"
        else
            log "  ${RED}✗${RESET} Generation failed — train.jsonl not created"
            curriculum_ok=false
        fi
    else
        log "  ${RED}✗${RESET} Generator script not found: $STUDY_SCRIPTS/generate_curriculum.py"
        curriculum_ok=false
    fi
else
    train_count=$(wc -l < "$TRAIN_FILE" 2>/dev/null || echo 0)
    log "  ${GREEN}✓${RESET} Training data: $TRAIN_FILE ($train_count examples)"
fi

# Check validation data
if [ ! -f "$VAL_FILE" ]; then
    log "  ${RED}✗${RESET} Validation data not found: $VAL_FILE"
    curriculum_ok=false
else
    val_count=$(wc -l < "$VAL_FILE" 2>/dev/null || echo 0)
    log "  ${GREEN}✓${RESET} Validation data: $VAL_FILE ($val_count examples)"
fi

# Validate JSONL format (spot-check first + last line)
if [ -f "$VAL_FILE" ]; then
    if ! head -1 "$VAL_FILE" | python3 -c "import sys, json; json.loads(sys.stdin.read())" 2>/dev/null; then
        log "  ${RED}✗${RESET} Validation file has invalid JSON on line 1"
        curriculum_ok=false
    fi
    if ! tail -1 "$VAL_FILE" | python3 -c "import sys, json; json.loads(sys.stdin.read())" 2>/dev/null; then
        log "  ${RED}✗${RESET} Validation file has invalid JSON on last line"
        curriculum_ok=false
    fi
fi

# Check metadata
if [ -f "$METADATA_FILE" ]; then
    log "  ${GREEN}✓${RESET} Generation metadata: $METADATA_FILE"
else
    log "  ${DIM}⊘${RESET} No generation metadata (optional)"
fi

if [ "$curriculum_ok" = true ]; then
    stage_pass "Curriculum Validation"
else
    stage_fail "Curriculum Validation"
    log "\n  ${RED}Cannot proceed without valid curriculum data.${RESET}"
    log "  Generate with: python3 $STUDY_SCRIPTS/generate_curriculum.py"
    exit 1
fi

# ═══════════════════════════════════════════════════════════════════════════
# Stage 3: GPU Handoff (Prime → Study)
# ═══════════════════════════════════════════════════════════════════════════

stage_header 3 "GPU Handoff (Prime → Study)"

if [ "$DRY_RUN" = true ] || [ "$SKIP_TRAINING" = true ]; then
    stage_skip "GPU Handoff"
else
    log "  Requesting GPU handoff from orchestrator..."
    log "  POST $ORCH_URL/handoff/prime-to-study"

    set +e
    handoff_response=$(curl -s -w "\n%{http_code}" \
        -X POST "$ORCH_URL/handoff/prime-to-study" \
        -H "Content-Type: application/json" \
        --max-time 60 2>&1)
    handoff_exit=$?
    set -e

    if [ $handoff_exit -ne 0 ]; then
        log "  ${YELLOW}⚠${RESET} Could not reach orchestrator (exit=$handoff_exit)"
        log "  Continuing anyway — study service may already have GPU"
        stage_warn "GPU Handoff"
    else
        http_code=$(echo "$handoff_response" | tail -1)
        body=$(echo "$handoff_response" | sed '$d')

        if [[ "$http_code" =~ ^2 ]]; then
            log "  ${GREEN}✓${RESET} GPU handoff initiated (HTTP $http_code)"
            [ "$VERBOSE" = true ] && log "    $body"

            # Wait for handoff to complete
            log "  Waiting for GPU transfer..."
            sleep 10

            # Poll study health to confirm GPU availability
            set +e
            study_health=$(curl -s --max-time 10 "$STUDY_URL/health" 2>/dev/null)
            set -e
            if echo "$study_health" | grep -q "healthy"; then
                log "  ${GREEN}✓${RESET} gaia-study is healthy and ready"
                stage_pass "GPU Handoff"
            else
                log "  ${YELLOW}⚠${RESET} gaia-study health check unclear"
                stage_warn "GPU Handoff"
            fi
        else
            log "  ${YELLOW}⚠${RESET} Handoff returned HTTP $http_code"
            log "    $body"
            stage_warn "GPU Handoff"
        fi
    fi
fi

# ═══════════════════════════════════════════════════════════════════════════
# Stage 4: Training
# ═══════════════════════════════════════════════════════════════════════════

stage_header 4 "QLoRA Training"

if [ "$DRY_RUN" = true ] || [ "$SKIP_TRAINING" = true ]; then
    stage_skip "QLoRA Training"
else
    # Read curriculum config for training parameters
    max_steps=$(python3 -c "import json; print(json.load(open('$CURRICULUM_SPEC')).get('training_config', {}).get('max_steps', 200))" 2>/dev/null || echo "200")
    tier=$(python3 -c "import json; print(json.load(open('$CURRICULUM_SPEC')).get('tier', 1))" 2>/dev/null || echo "1")
    pillar=$(python3 -c "import json; print(json.load(open('$CURRICULUM_SPEC')).get('pillar', 'cognition'))" 2>/dev/null || echo "cognition")
    description=$(python3 -c "import json; print(json.load(open('$CURRICULUM_SPEC')).get('description', ''))" 2>/dev/null || echo "")

    # Collect training document paths (the JSONL files)
    documents_json="[\"$TRAIN_FILE\"]"

    log "  Starting training: adapter=$ADAPTER steps=$max_steps tier=$tier"
    log "  POST $STUDY_URL/study/start"

    # Start training
    set +e
    start_response=$(curl -s -w "\n%{http_code}" \
        -X POST "$STUDY_URL/study/start" \
        -H "Content-Type: application/json" \
        -d "{
            \"adapter_name\": \"$ADAPTER\",
            \"documents\": $documents_json,
            \"tier\": $tier,
            \"pillar\": \"$pillar\",
            \"description\": \"$description\",
            \"max_steps\": $max_steps,
            \"tags\": [\"pipeline-validated\"]
        }" \
        --max-time 30 2>&1)
    start_exit=$?
    set -e

    if [ $start_exit -ne 0 ]; then
        log "  ${RED}✗${RESET} Could not reach gaia-study (exit=$start_exit)"
        stage_fail "QLoRA Training"
        # Don't exit — proceed to reclaim GPU
    else
        http_code=$(echo "$start_response" | tail -1)
        body=$(echo "$start_response" | sed '$d')

        if [[ "$http_code" =~ ^2 ]]; then
            log "  ${GREEN}✓${RESET} Training started (HTTP $http_code)"

            # Poll status until complete or timeout
            deadline=$(($(date +%s) + TRAINING_TIMEOUT))
            last_progress=""

            while [ "$(date +%s)" -lt "$deadline" ]; do
                sleep 15

                set +e
                status_resp=$(curl -s --max-time 10 "$STUDY_URL/study/status" 2>/dev/null)
                set -e

                state=$(echo "$status_resp" | python3 -c "import sys, json; print(json.loads(sys.stdin.read()).get('state', 'unknown'))" 2>/dev/null || echo "unknown")
                progress=$(echo "$status_resp" | python3 -c "import sys, json; print(f\"{json.loads(sys.stdin.read()).get('progress', 0):.0%}\")" 2>/dev/null || echo "?")
                msg=$(echo "$status_resp" | python3 -c "import sys, json; print(json.loads(sys.stdin.read()).get('message', '')[:60])" 2>/dev/null || echo "")

                # Only print if progress changed
                if [ "$progress" != "$last_progress" ]; then
                    log "  [$state] $progress $msg"
                    last_progress="$progress"
                fi

                case "$state" in
                    complete)
                        log "  ${GREEN}✓${RESET} Training complete"
                        stage_pass "QLoRA Training"
                        break
                        ;;
                    failed)
                        log "  ${RED}✗${RESET} Training failed: $msg"
                        stage_fail "QLoRA Training"
                        break
                        ;;
                    idle)
                        log "  ${YELLOW}⚠${RESET} Training returned to idle unexpectedly"
                        stage_warn "QLoRA Training"
                        break
                        ;;
                esac
            done

            # Check if we timed out
            if [ "$(date +%s)" -ge "$deadline" ]; then
                log "  ${RED}✗${RESET} Training timeout (${TRAINING_TIMEOUT}s)"
                # Cancel training
                curl -s -X POST "$STUDY_URL/study/cancel" --max-time 10 > /dev/null 2>&1 || true
                stage_fail "QLoRA Training"
            fi
        else
            log "  ${RED}✗${RESET} Training start failed (HTTP $http_code)"
            log "    $body"
            stage_fail "QLoRA Training"
        fi
    fi
fi

# ═══════════════════════════════════════════════════════════════════════════
# Stage 5: GPU Reclaim (Study → Prime)
# ═══════════════════════════════════════════════════════════════════════════

stage_header 5 "GPU Reclaim (Study → Prime)"

if [ "$DRY_RUN" = true ] || [ "$SKIP_TRAINING" = true ]; then
    stage_skip "GPU Reclaim"
else
    log "  Requesting GPU return to inference..."
    log "  POST $ORCH_URL/handoff/study-to-prime"

    set +e
    reclaim_response=$(curl -s -w "\n%{http_code}" \
        -X POST "$ORCH_URL/handoff/study-to-prime" \
        -H "Content-Type: application/json" \
        --max-time 60 2>&1)
    reclaim_exit=$?
    set -e

    if [ $reclaim_exit -ne 0 ]; then
        log "  ${YELLOW}⚠${RESET} Could not reach orchestrator for GPU reclaim"
        log "  GPU may need manual return: docker restart gaia-prime"
        stage_warn "GPU Reclaim"
    else
        http_code=$(echo "$reclaim_response" | tail -1)
        if [[ "$http_code" =~ ^2 ]]; then
            log "  ${GREEN}✓${RESET} GPU reclaim initiated"

            # Wait for gaia-prime to be back up
            log "  Waiting for gaia-prime to restart..."
            prime_up=false
            for i in $(seq 1 12); do
                sleep 10
                set +e
                prime_health=$(curl -s --max-time 5 "$ENDPOINT/health" 2>/dev/null)
                set -e
                if echo "$prime_health" | grep -qi "ok\|healthy\|ready"; then
                    prime_up=true
                    break
                fi
                log "    waiting... (${i}/12)"
            done

            if [ "$prime_up" = true ]; then
                log "  ${GREEN}✓${RESET} gaia-prime is back online"
                stage_pass "GPU Reclaim"
            else
                log "  ${YELLOW}⚠${RESET} gaia-prime not responding after 120s"
                stage_warn "GPU Reclaim"
            fi
        else
            log "  ${YELLOW}⚠${RESET} GPU reclaim returned HTTP $http_code"
            stage_warn "GPU Reclaim"
        fi
    fi
fi

# ═══════════════════════════════════════════════════════════════════════════
# Stage 6: Adapter Validation
# ═══════════════════════════════════════════════════════════════════════════

stage_header 6 "Adapter Validation"

VALIDATE_SCRIPT="$STUDY_SCRIPTS/validate_adapter.py"
REPORT_DIR="$GAIA_ROOT/knowledge/curricula/$ADAPTER"
REPORT_FILE="$REPORT_DIR/validation_report.json"

if [ ! -f "$VALIDATE_SCRIPT" ]; then
    log "  ${RED}✗${RESET} validate_adapter.py not found: $VALIDATE_SCRIPT"
    stage_fail "Adapter Validation"
else
    # Build command
    validate_cmd=(
        python3 "$VALIDATE_SCRIPT"
        --adapter "$ADAPTER"
        --validation-file "$VAL_FILE"
        --endpoint "$ENDPOINT"
        --max-examples "$MAX_EXAMPLES"
        --threshold "$THRESHOLD"
        --json-report "$REPORT_FILE"
    )

    if [ "$DRY_RUN" = true ]; then
        validate_cmd+=(--dry-run)
    fi

    if [ "$DO_BASELINE" = true ]; then
        validate_cmd+=(--baseline)
    fi

    if [ "$VERBOSE" = true ]; then
        validate_cmd+=(--verbose)
    fi

    log "  Running: ${validate_cmd[*]}"
    echo ""

    set +e
    "${validate_cmd[@]}"
    validate_exit=$?
    set -e

    echo ""
    if [ $validate_exit -eq 0 ]; then
        stage_pass "Adapter Validation"

        # Show score from report
        if [ -f "$REPORT_FILE" ]; then
            score=$(python3 -c "import json; print(json.load(open('$REPORT_FILE')).get('composite_score', '?'))" 2>/dev/null || echo "?")
            log "  Score: $score (threshold: $THRESHOLD)"
        fi
    else
        stage_fail "Adapter Validation"
        log "  Adapter did not meet the quality threshold"
    fi
fi

# ═══════════════════════════════════════════════════════════════════════════
# Stage 7: Report & Registration
# ═══════════════════════════════════════════════════════════════════════════

stage_header 7 "Report & Registration"

# Calculate timing
elapsed=$(($(date +%s) - PIPELINE_START))
mins=$((elapsed / 60))
secs=$((elapsed % 60))

# Determine overall result
overall="PASS"
for result in "${STAGE_RESULTS[@]}"; do
    status="${result%%|*}"
    if [ "$status" = "FAIL" ]; then
        overall="FAIL"
        break
    fi
done

# Print summary
echo ""
echo -e "${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${RESET}"
echo -e "${BOLD}  QLoRA Validation Summary${RESET}"
echo -e "${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${RESET}"
echo -e "  Adapter:   ${CYAN}$ADAPTER${RESET}"
echo -e "  Duration:  ${mins}m ${secs}s"
echo -e "  Mode:      $([ "$DRY_RUN" = true ] && echo "${YELLOW}DRY RUN${RESET}" || echo "LIVE")"
echo ""
echo -e "  ${BOLD}Stage Results:${RESET}"

for result in "${STAGE_RESULTS[@]}"; do
    status="${result%%|*}"
    name="${result#*|}"
    case "$status" in
        PASS) echo -e "    ${GREEN}✓${RESET} $name" ;;
        FAIL) echo -e "    ${RED}✗${RESET} $name" ;;
        WARN) echo -e "    ${YELLOW}⚠${RESET} $name" ;;
        SKIP) echo -e "    ${DIM}⊘${RESET} $name" ;;
    esac
done

echo ""
if [ "$overall" = "PASS" ]; then
    echo -e "  ${BOLD}${GREEN}RESULT: PASS${RESET}"

    # Register adapter (write completion marker)
    if [ "$DRY_RUN" != true ]; then
        completion_file="$REPORT_DIR/last_validation.json"
        python3 -c "
import json
from datetime import datetime
data = {
    'adapter_name': '$ADAPTER',
    'validated_at': datetime.utcnow().isoformat(),
    'result': 'PASS',
    'duration_seconds': $elapsed,
    'threshold': $THRESHOLD,
    'mode': 'dry-run' if '$DRY_RUN' == 'true' else 'live'
}
# Merge with validation report if available
try:
    with open('$REPORT_FILE') as f:
        data['scores'] = json.load(f)
except:
    pass
with open('$completion_file', 'w') as f:
    json.dump(data, f, indent=2)
print(f'  Validation record written to {\"$completion_file\"!s}')
" 2>/dev/null || true
    fi
else
    echo -e "  ${BOLD}${RED}RESULT: FAIL${RESET}"
    echo -e "  Adapter $ADAPTER did not pass validation."
fi

echo -e "${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${RESET}"
echo ""

# Exit with appropriate code
[ "$overall" = "PASS" ] && exit 0 || exit 1
