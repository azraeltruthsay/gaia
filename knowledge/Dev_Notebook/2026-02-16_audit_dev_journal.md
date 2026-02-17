# Dev Journal Entry: 2026-02-16 — Full Codebase Audit & Remediation

**Date:** 2026-02-16
**Author:** Claude Code (Opus 4.6) via Happy
**Scope:** Comprehensive code audit across all 4 GAIA services (~211 production Python files)
**Duration:** Single session

## Context

Seumas requested a full codebase evaluation focusing on dead code, bugs, feature failures, error handling, logging, code hygiene, and security. Four parallel audit agents were launched to analyze gaia-core, gaia-common, gaia-web, and gaia-orchestrator simultaneously.

## Audit Methodology

1. **Mapped all Python source files** (~211 production, ~418 total including candidates)
2. **Launched 4 parallel audit agents** — one per service, each performing deep file-by-file analysis
3. **Consolidated findings** into prioritized report (CRITICAL/HIGH/MEDIUM/LOW)
4. **Applied fixes** in priority order, always patching both production and candidate copies

---

## Fixes Applied (19 total, ~38 files modified)

### CRITICAL + HIGH Priority (10 fixes)

#### H1. Shell injection in mcp_client.py ai_execute()
**Files:** gaia-core + candidates `utils/mcp_client.py`
**Was:** `subprocess.run(command, shell=True)` where command comes from LLM tool params
**Fix:** Changed default to `shell=False`, added `shlex.split()` for safe tokenization. Shell features (pipes, redirects) still available via explicit `shell=True` opt-in.

#### H2. GPU owner bug — wake sets wrong owner
**File:** `gaia-orchestrator/main.py:524`
**Was:** `GPUOwner.CORE_CANDIDATE` set after production wake — state says candidate owns GPU while production gaia-core is running
**Fix:** Changed to `GPUOwner.CORE`

#### H3. Unreachable except clause in discord_interface.py
**File:** `gaia-web/discord_interface.py:320-324`
**Was:** Two consecutive `except Exception` clauses — second never reached. First handler also leaked internal message IDs to users.
**Fix:** Removed duplicate. Single handler now logs warning + sends message as regular (not reply).

#### H4. Oracle model created with None API key
**Files:** gaia-core + candidates `models/oracle_model.py`
**Was:** `openai.OpenAI(api_key=None)` if OPENAI_API_KEY unset — fails with opaque auth error on first API call
**Fix:** Added explicit validation in `__init__`, raises `ValueError` with clear message about setting the env var.

#### H5. Dead dispatch() function (80 lines of v0.2 code)
**Files:** gaia-core + candidates `cognition/cognitive_dispatcher.py`
**Was:** `dispatch()` referenced `packet.prompt`, `packet.contextual_instructions` — v0.2 API that doesn't exist on v0.3 CognitionPacket. Grep confirmed never called anywhere.
**Fix:** Removed `dispatch()` and all its unused imports (`json`, `re`, `model_pool`, `get_config`, module-level `_config` and `constants`). Kept `process_execution_results()` which is actively used.

#### H6. /gpu/wait blocks async worker for up to 300 seconds
**File:** `gaia-orchestrator/models/schemas.py:32`
**Was:** `timeout_seconds` default=300 — each call holds a uvicorn worker in an `asyncio.sleep()` loop for up to 5 minutes
**Fix:** Changed to `default=60, ge=1, le=60` — schema-enforced cap prevents abuse.

#### C1. Shell injection in safe_execution.py (whitelist bypass)
**Files:** gaia-common + candidates `utils/safe_execution.py`
**Was:** Checked only first token against whitelist, then passed entire string to `shell=True`. Attack: `ls ; rm -rf /` where `ls` is whitelisted.
**Fix:** `shlex.split()` for proper tokenization, `shell=False`, pass list to `subprocess.run()`.

#### C2. Shell injection in gaia_rescue_helper.py buffer_and_execute_shell()
**Files:** gaia-common + gaia-core + candidates (4 files)
**Was:** `command.startswith(func)` whitelist check + `Popen(command, shell=True)`. Also had nonsensical except handler that re-executed the same failed command.
**Fix:** `shlex.split()`, exact first-token match (`parts[0] not in safe_cmds`), `shell=False`, added `timeout=30` to `.communicate()`, removed duplicate Popen fallback.

#### C3. Path traversal in code_read/code_span/code_symbol
**Files:** gaia-common + gaia-core + candidates (4 files)
**Was:** Accepted arbitrary file paths with no sandboxing. `code_read` called `os.path.realpath()` but never checked the result was within allowed directories.
**Fix:** Added `_ALLOWED_READ_DIRS` class constant and `_validate_read_path()` method. All three code helpers now validate resolved paths against allowlist before reading.

#### C4. output_router endpoint entirely broken
**Files:** gaia-web + candidates `main.py`
**Was:** Endpoint received `Dict[str, Any]` but accessed with dot notation (`packet.header.packet_id`). Every call crashed with `AttributeError`.
**Fix:** Converted all attribute access to `.get()` dict access with safe defaults. Replaced enum comparisons (`OutputDestination.DISCORD`) with string comparisons (`"discord"`). Also sanitized error response to not leak internal details.

### MEDIUM Priority (9 fixes)

#### M1. ai_write() path restriction
**Files:** gaia-core + candidates `utils/mcp_client.py`
**Was:** Wrote to arbitrary filesystem paths with `os.makedirs(exist_ok=True)`
**Fix:** Added `_WRITABLE_DIRS` allowlist (`/sandbox/`, `/shared/`, `/knowledge/`, `/tmp/`, `/logs/`). Resolves symlinks via `os.path.realpath()` before checking.

#### M2. vllm_model.py duplicated multiprocessing block
**Files:** gaia-core + candidates `models/vllm_model.py`
**Was:** Identical 14-line block for multiprocessing start method detection appeared twice in `__init__`. Second copy was pure dead code (env var already set by first).
**Fix:** Removed duplicate block.

#### M3. Config singleton pattern broken
**Files:** gaia-core + candidates `config.py`
**Was:** `_instance: Optional[Config] = None` was a dataclass field (per-instance), not a class variable. Every `Config()` call created a new instance, re-loading `gaia_constants.json` from disk.
**Fix:** Changed to `_instance: ClassVar[Optional['Config']] = None`. Added `ClassVar` to typing imports.

#### M4. self_review_worker passes strings where enums expected
**Files:** gaia-core + candidates `cognition/self_review_worker.py`
**Was:** `role="Analyst"`, `target_engine="Lite"` — raw strings instead of `PersonaRole.ANALYST`, `TargetEngine.LITE`. Also passed invalid `allow_parallel=False, priority=5` kwargs.
**Fix:** Changed to proper enum values. Removed invalid kwargs. Added `PersonaRole, TargetEngine` to imports.

#### M5. resource_monitor.py NVML init at import time
**Files:** gaia-core + candidates `utils/resource_monitor.py`
**Was:** `pynvml.nvmlInit()` called at module import. If GPU driver had transient error at import, NVML permanently disabled for process lifetime.
**Fix:** Lazy initialization via `_ensure_nvml()` static method, called from `__init__` and `_monitor()`. Allows retry on next process start.

#### M12. Error details exposed to Discord users
**Files:** gaia-web + candidates `discord_interface.py`
**Was:** `f"Status: {e.response.status_code}
Details: {e.response.text}"` and `f"An unexpected error occurred: {e}"` sent directly to Discord channels.
**Fix:** Generic user-facing messages. Full error details kept in server-side `logger.error()`/`logger.exception()` calls.

#### M14. thought_seed.py v0.2 packet references
**Files:** gaia-core + candidates `cognition/thought_seed.py`
**Was:** `packet.prompt`, `packet.packet_id`, `packet.persona` — attributes from flat v0.2 format
**Fix:** Updated to v0.3 paths: `packet.header.packet_id`, `packet.header.persona.role`, `packet.intent.primary_goal`. Added defensive `hasattr` guards.

#### M15. packet_upgrade.py deprecated as no-op
**Files:** gaia-core + candidates `cognition/packet_upgrade.py`
**Was:** 58-line module setting non-existent attributes (cot, scratch, cheats, proposed_actions, etc.) on v0.3 CognitionPacket. Called from prompt_builder.py migration path.
**Fix:** Replaced with documented no-op (23 lines). `upgrade_packet()` returns packet unchanged. Callers already wrap in try/except.

#### M16. processor.py broken legacy imports
**Files:** gaia-common + candidates `utils/background/processor.py`
**Was:** `from app.utils.background.*` — monolith-era paths that don't exist
**Fix:** Changed to `from gaia_common.utils.background.*`

---

## Remaining Issues (Not Fixed This Session)

### MEDIUM (deferred)
- **M6:** GPU acquire/release race condition (needs asyncio.Lock — architectural change)
- **M7:** Blocking Docker SDK calls in async handlers (needs run_in_executor wrapping)
- **M8:** Hardcoded service URLs in docker_manager.py (needs config refactor)
- **M9:** deprecated `datetime.utcnow()` across ~15 files (mechanical but widespread)
- **M10:** Dead `gaia_web/queue/message_queue.py` (kept — may be used by sleep/wake system)
- **M11:** `_bot_thread` written but never read (minor dead code)
- **M13:** Dual config paths in orchestrator (needs design decision)

### From gaia-common audit (deferred)
- Dead backup file with space in name
- Duplicate `_normalize` in cognition_packet.py
- `DiscordConnector._send_via_bot()` returns True before send completes
- ThreadPoolExecutor leak in on_message handler
- Inverted dependency: gaia-common imports from gaia_core
- Unicode en-dash in encoding string in generate_capability_map.py
- Triplicated SafeJSONEncoder
- Broken legacy `app.*` imports in code_analyzer modules

### LOW (21+ items)
- Tab vs space indentation in vllm_model.py and self_review_worker.py
- WARNING-level logging on every vllm completion (should be DEBUG)
- print() statements in oracle_model.py and discord_interface.py
- Unused imports across multiple files
- Missing test coverage for security-critical modules
- MD5 usage where SHA-256 would be consistent

---

## Architecture Observations

1. **v0.2 → v0.3 migration is incomplete** — Several modules still reference the flat v0.2 CognitionPacket API (`packet.prompt`, `packet.persona`, `packet.contextual_instructions`). The v0.3 dataclass has nested structure (`packet.header.packet_id`, `packet.content.original_prompt`, etc.).

2. **Monolith remnants persist** — Multiple files still import from `app.*` paths that don't exist in the SOA layout. The `code_analyzer/` and `background/` packages in gaia-common are particularly affected.

3. **Security surface is LLM-facing** — The `ai_execute`, `ai_write`, `run_shell_safe`, and `buffer_and_execute_shell` functions are all called with LLM-generated parameters. The shell injection and path traversal fixes are critical because these are the exact functions an adversarial prompt would target.

4. **Candidate/production parity** — All fixes were applied to both copies. The promotion pipeline should validate this.

---

## Test Impact

- No new tests were added this session (focus was on fixing existing code)
- Existing tests should continue to pass since changes were backwards-compatible
- `dispatch()` removal is safe (grep confirmed zero callers)
- `packet_upgrade.py` no-op is safe (callers wrap in try/except)
- Shell changes (`shell=True` → `shell=False`) may affect callers that relied on shell features — but this is the correct security posture

---

## Next Steps

1. **Run promotion pipeline** to validate all changes pass ruff/mypy/pytest
2. **Commit and push** the audit fixes
3. **Phase 2 session:** Fix remaining MEDIUM items (M6-M9, M11, M13)
4. **Phase 3 session:** Address gaia-common audit findings (inverted deps, dead code, test coverage)
5. **Consider:** Adding integration tests for the security-critical paths (ai_execute, ai_write, code_read)
