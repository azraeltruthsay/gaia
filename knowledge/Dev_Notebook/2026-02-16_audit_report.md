# Codebase Audit Report — 2026-02-16

**Author:** Claude Code (Opus 4.6) via Happy
**Scope:** Full codebase audit — gaia-core, gaia-common, gaia-web, gaia-orchestrator
**Focus:** Dead code, bugs, security, error handling, logging, code hygiene

---

## HIGH Priority Fixes Applied (6)

### H1. Shell injection in mcp_client.py ai_execute() — FIXED
**Files:** gaia-core + candidates mcp_client.py
**Was:** shell=True default with LLM-sourced command strings
**Fix:** Changed default to shell=False, added shlex.split() for safe command parsing

### H2. GPU owner bug — wake sets wrong owner — FIXED
**File:** gaia-orchestrator/main.py:524
**Was:** GPUOwner.CORE_CANDIDATE (candidate gets ownership after production wake)
**Fix:** Changed to GPUOwner.CORE

### H3. Unreachable except clause in discord_interface.py — FIXED
**File:** gaia-web/discord_interface.py:320-324
**Was:** Two consecutive except Exception — second never reached
**Fix:** Removed duplicate, kept informative handler with logger.warning

### H4. Oracle model created with None API key — FIXED
**Files:** gaia-core + candidates oracle_model.py
**Was:** openai.OpenAI(api_key=None) if env var unset — opaque auth failure at call time
**Fix:** Added explicit validation in __init__, raises ValueError with clear message

### H5. Dead dispatch() uses v0.2 CognitionPacket API — FIXED
**Files:** gaia-core + candidates cognitive_dispatcher.py
**Was:** 80-line function referencing packet.prompt, packet.contextual_instructions (v0.2), never called
**Fix:** Removed dispatch() and its unused imports. Kept process_execution_results().

### H6. /gpu/wait holds async worker for up to 300 seconds — FIXED
**File:** gaia-orchestrator/models/schemas.py:32
**Was:** timeout_seconds default=300 — blocks uvicorn worker pool
**Fix:** Changed to default=60, ge=1, le=60

---

## MEDIUM Priority Issues (Pending)

| # | Issue | File |
|---|-------|------|
| M1 | ai_write() no path restriction | mcp_client.py |
| M2 | vllm_model.py duplicated multiprocessing block | vllm_model.py |
| M3 | Config singleton pattern broken (_instance is dataclass field) | config.py |
| M4 | self_review_worker passes string where enum expected | self_review_worker.py |
| M5 | resource_monitor NVML init at import time | resource_monitor.py |
| M6 | GPU acquire/release race condition (no locking) | orchestrator main.py |
| M7 | Blocking Docker SDK calls in async handlers | docker_manager.py |
| M8 | Hardcoded service URLs | docker_manager.py |
| M9 | deprecated datetime.utcnow() across ~15 files | multiple |
| M10 | gaia_web/queue/message_queue.py is dead code | message_queue.py |
| M11 | _bot_thread written but never read | discord_interface.py |
| M12 | Error details exposed to Discord users | discord_interface.py |
| M13 | Dual config paths in orchestrator | orchestrator config.py |
| M14 | thought_seed.py uses dead v0.2 packet attrs | thought_seed.py |
| M15 | packet_upgrade.py sets non-existent attrs | packet_upgrade.py |
| M16 | processor.py broken import stub | processor.py |

## Additional Findings from gaia-common Audit

### CRITICAL (from gaia-common)
- Shell injection in safe_execution.py (whitelist bypass via shell=True)
- Shell injection in gaia_rescue_helper.py buffer_and_execute_shell()
- Path traversal in code_read/code_span/code_symbol (no sandboxing)
- Broken legacy app.* imports in code_analyzer + background/processor.py

### HIGH (from gaia-common)
- Dead backup file with space in name (un-importable)
- Duplicate _normalize in cognition_packet.py
- deprecated datetime.utcnow() in ~10 files
- DiscordConnector._send_via_bot() returns True before send completes
- ThreadPoolExecutor leak in on_message handler
- Config._instance as dataclass field (same as M3)
- Inverted dependency: gaia-common imports from gaia_core

### CRITICAL (from gaia-web)
- output_router endpoint entirely broken (Dict accessed with dot notation)
- Unreachable except (fixed as H3)

---

## LOW Priority Issues (21+)

Tab indentation, WARNING-level logging spam, unused imports, print() in oracle stream, deprecated pkg_resources, missing type annotations, TODO/FIXME backlog, oversized discover() function, MD5 usage for hashing (should be SHA-256).

---

## Fix Plan

**Phase 1 (Safety) — DONE:** H1-H6 applied this session
**Phase 2 (Correctness):** M1-M5, M14-M16, gaia-common shell injection + path traversal
**Phase 3 (Reliability):** M6-M8, M12, output_router fix, broken imports
**Phase 4 (Cleanup):** M9-M11, M13, LOW items, dead code removal
