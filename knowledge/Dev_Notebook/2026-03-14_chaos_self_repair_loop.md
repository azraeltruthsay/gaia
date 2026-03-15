# 2026-03-14: Chaos Monkey Self-Repair Loop

## What Changed

Restructured the Chaos Monkey → Doctor interaction from a synthetic monkey-side repair loop to an **organic Doctor-driven self-repair flow**:

**Before**: Monkey injects fault → Monkey calls LLM repair → Monkey restores if failed → Monkey scores serenity
**After**: Monkey injects fault → Monkey notifies Doctor → Doctor detects (via notification OR mtime audit) → Doctor runs tests → Doctor drives 3-retry LLM repair with escalating context → Doctor restores from Production if all fail → Doctor reports to Monkey for serenity scoring

## Files Modified

### gaia-monkey/gaia_monkey/fault_injector.py
- **Progressive difficulty**: `DIFFICULTY_LEVELS` (1=easy, 2=medium, 3=hard, 4=expert)
- **`get_difficulty_for_serenity(score)`**: Auto-scales difficulty from serenity (0→L1, 3→L2, 7→L3, 10→L4)
- **`_inject_swap_args(content)`**: New fault type — swaps first two args in a function call
- **`_inject_multi_fault(content)`**: Expert difficulty — applies 2-3 single faults sequentially
- `inject_semantic_fault()` now takes `difficulty` parameter

### gaia-monkey/gaia_monkey/chaos_engine.py
- **Removed**: Entire 110-line repair loop from `run_code_drill()` (lines 221-330)
- **Added**: `_notify_doctor()` — POSTs to `POST /notify/chaos_injection` with service, file, fault, difficulty
- **Added**: `DOCTOR_ENDPOINT` config (from env)
- Now writes broken file directly to host (volume-mounted), restarts container, notifies Doctor, returns `{status: "injected", awaiting_doctor: true}`
- Difficulty auto-scales from serenity score, overridable via API param

### gaia-monkey/gaia_monkey/main.py
- **New endpoint**: `POST /serenity/record_recovery` — Doctor reports repair outcomes for serenity scoring
- Wired `difficulty` param through `/chaos/code` endpoint

### gaia-doctor/doctor.py
- **`CORE_ENDPOINT`**: New config (env var, defaults to `http://gaia-core:6415`)
- **`_get_test_errors(service)`**: Runs pytest and returns last 3K chars of error output for LLM context
- **`repair_candidate_file(service, file_path, max_retries=3)`**: 3-retry LLM repair loop with escalating prompts ("fix this" → "focus on assertions" → "FINAL: be conservative"). POSTs to `/api/repair/structural`. Re-runs tests after each attempt.
- **`restore_from_production(candidate_path)`**: Derives prod path, checks git HEAD divergence (aborts if similarity < 50%), AST-validates prod file, falls back to git HEAD if corrupt, writes via structural endpoint or direct write
- **`_handle_chaos_notification(data)`**: Receives monkey notification, sleeps 3s for settle, runs tests, enters repair loop if needed
- **`_repair_divergent_candidate(name)`**: Organic detection path — finds recently modified files and runs repair
- **`POST /notify/chaos_injection`**: HTTP endpoint, spawns background thread, returns immediately
- **`_notify_monkey_repair_success/restored()`**: Serenity scoring helpers
- **`audit_code()` integration**: After "Tests FAILED" for a candidate during meditation, auto-enters repair loop

### docker-compose.yml
- Added `CORE_ENDPOINT` to gaia-doctor environment block

## Architecture

### Two Detection Paths
1. **Notification (fast)**: Monkey → `POST /notify/chaos_injection` → Doctor handles immediately
2. **Organic (fallback)**: Doctor's existing mtime-based `audit_code()` scan → tests fail → if candidate + meditation → repair

### Repair Escalation
1. LLM repair attempt 1: "Fix the broken code so tests pass"
2. LLM repair attempt 2: "Previous fix failed, focus on assertions and imports"
3. LLM repair attempt 3: "FINAL ATTEMPT: Be very conservative"
4. Restore from Production (with git divergence safety check)

### Serenity Scoring
- LLM repair success → `vital_recovery` (2.0 points)
- Restore-only recovery → `standard_recovery` (0.5 points)
- Reported to monkey via `POST /serenity/record_recovery`

## Deployment
```bash
docker compose build gaia-doctor gaia-monkey
docker compose up -d gaia-doctor gaia-monkey
```

## Verification
```bash
curl http://localhost:6419/health       # Doctor healthy
curl http://localhost:6420/health       # Monkey healthy
curl http://localhost:6419/cognitive/monitor  # Cognitive monitor passing
```

## Design Notes
- All doctor code is **stdlib-only** (urllib, subprocess, ast, difflib, json, threading)
- Difficulty auto-scales with serenity — starts easy, gets harder as system proves resilience
- Git divergence check prevents blind restore when production has drifted significantly from HEAD
- No new ports, volumes, or external dependencies
