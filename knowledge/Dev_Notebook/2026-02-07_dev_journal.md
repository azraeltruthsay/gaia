# Dev Journal Entry: 2026-02-07 - Network Stability, Hardening, and Response Persistence

**Date:** 2026-02-07
**Author:** Claude Code (Opus 4.6) via Happy

## Context and Motivation

Following the containerized validation SDLC work on 2026-02-06, this session covers a broad sweep of hardening changes across the GAIA stack. The changes span five major themes: eliminating fragile Docker networking patterns (`gaia.sh` and compose), production robustness (GPU error handling, health check filtering), Docker networking fixes (container-aware health checks, port consolidation), critical bug fixes (response persistence, config key renames), and developer tooling improvements (heartbeat log compression, function reference generation, dev dependency standardization).

The GAIA Docker networking had three fragile patterns that could cause startup failures: `gaia.sh live start` destructively removed the network before starting (killing connectivity for running candidates or orchestrator), `gaia.sh orchestrator start` used raw `docker run` despite the orchestrator being properly defined in `docker-compose.yml`, and `docker-compose.candidate.yml` declared the network as `external: true` creating a chicken-and-egg dependency. These are now resolved.

## Goal

Make GAIA's network setup less fragile by eliminating destructive startup patterns and chicken-and-egg dependencies. Stabilize the service mesh for reliable container operation by fixing networking assumptions, improving error resilience, correcting data persistence bugs, and reducing log noise — all while maintaining the candidate/live dual-track development model.

## Key Changes Implemented

### 1. Network Stability: Remove `external: true` from Candidate Compose (`docker-compose.candidate.yml`)

- Changed the network definition from `external: true` + `name: gaia-network` to a full definition matching the live compose (bridge driver, subnet `172.28.0.0/16`)
- Docker will reuse `gaia-network` if it already exists, or create it if not — no ordering dependency
- **Why:** The old `external: true` pattern meant candidate compose would fail if the live stack hadn't been started first to create the network. The `ensure_network()` function in `gaia.sh` papered over this, but it was still fragile (e.g., `docker compose -f candidate.yml up` without going through `gaia.sh` would fail).

### 2. Network Stability: Remove Destructive Network Delete from `cmd_live` (`gaia.sh`)

- Removed the block in `cmd_live start` that ran `docker network rm gaia-network` before starting
- Replaced with a call to `ensure_network` — safe, idempotent check-and-create
- **Why:** The old pattern destroyed the network every time you started the live stack. If candidates or the orchestrator were running, they'd lose connectivity immediately. This was the single most dangerous startup pattern.

### 3. Network Stability: Rewrite `cmd_orchestrator` to Use Docker Compose (`gaia.sh`)

- Replaced raw `docker run -d` with `docker compose -f "$LIVE_COMPOSE" up -d gaia-orchestrator`
- `stop` now uses `docker compose stop` + `docker compose rm -f` instead of `docker stop` + `docker rm`
- `build` now uses `docker compose build gaia-orchestrator` instead of `docker build`
- `logs` now uses `docker compose logs -f gaia-orchestrator` instead of `docker logs -f`
- `status` unchanged (curl-based, works fine)
- Removed the `ensure_network` call from orchestrator start since compose handles it
- **Why:** The orchestrator was defined in `docker-compose.yml` but started via raw `docker run`, creating a "second-class citizen" that bypassed compose's network management, volume definitions, environment config, and health checks. Any drift between the compose definition and the `docker run` flags would cause silent misconfiguration.

### 4. GPU Resource Monitor Hardening (`gaia-core/gaia_core/utils/resource_monitor.py`)

- Separated `pynvml` import from `nvmlInit()` initialization with independent try-catch blocks
- Added graceful fallback when GPU libraries are unavailable (logs warning instead of crashing)
- Added generic `Exception` catch-all in the monitoring loop to prevent silent thread death
- **Why:** Containers may run without GPU access (e.g., during validation or on CPU-only nodes). The monitor thread crashing would silently degrade observability.

### 5. Docker Network Health Check Fixes (`gaia-orchestrator/`)

- **`docker_manager.py`**: Changed health check URLs from `localhost:{port}` to `{container_name}:{port}` for Docker internal networking
- **`config.py`**: Updated candidate endpoint port from `6416` to `6415` (port consolidation)
- Consolidated candidate service ports to match live service ports internally (6415, 6414, 8765, 8766)
- **Why:** Inside Docker networks, `localhost` refers to the container itself, not the target service. Health checks were always failing because they were pinging the orchestrator's own loopback.

### 6. Response Persistence Bug Fix (`gaia_core/cognition/agent_core.py`)

- Changed session history storage from `full_response` (raw LLM stream) to `user_facing_response` (routed/processed output)
- Added guard against storing empty responses when LLM stream yields nothing
- Applied to both live and candidate versions
- Increased relevant history summary truncation from 100 to 2000 characters
- Fixed config key reference: `"MODELS"` → `"MODEL_CONFIGS"` in observer config lookup
- **Why:** Session history was recording raw LLM output (including thinking tags and internal routing) rather than what the user actually saw. This corrupted conversation context for subsequent turns.

### 7. Heartbeat Log Compression (`candidates/gaia-common/gaia_common/utils/heartbeat_logger.py`)

- New `HeartbeatLogger` class that buffers identical heartbeat messages and tracks repeat counts
- `HeartbeatLoggerProxy` handler routes matching log patterns to the compressor
- Includes context manager support and timestamp tracking
- Exported via `gaia_common/utils/__init__.py`
- **Why:** Heartbeat logs were flooding output with identical messages every few seconds, making it difficult to spot actual events in the log stream.

### 8. Health Check Log Filter (`candidates/gaia-common/gaia_common/utils/logging_setup.py`)

- Implemented `HealthCheckFilter` class to suppress repetitive health check access log entries
- Exported `install_health_check_filter` utility for easy integration
- **Why:** Similar to heartbeat spam — health check endpoints (`/health`, `/status`) generate high-frequency log entries that obscure meaningful events.

### 9. Function Reference Generator Refactor (`gaia-common/gaia_common/utils/generate_function_reference.py`)

- Refactored from scanning a single `app_dir` to scanning a list of `scan_dirs` across all services (gaia-core, gaia-web, gaia-mcp, gaia-study, gaia-common)
- Outputs auto-generated markdown to `knowledge/system_reference/functions_reference.md`
- **Why:** With the SOA architecture, functions are distributed across multiple services. The old single-directory scanner missed most of the codebase.

### 10. Configuration Expansion (`candidates/gaia-core/gaia_core/config.py`)

- Added `KNOWLEDGE_CODEX_DIR`, `HISTORY_DIR` (env-configurable), `LORA_ADAPTERS_DIR`
- Added `CODEX_ALLOW_HOT_RELOAD: bool = True` feature flag
- Added `use_oracle: bool = False` feature flag
- **Why:** Preparing config surface for codex hot-reload, LoRA adapter management, and Oracle-sourced fact integration.

### 11. Packet Builder & Constants Fixes

- **`gaia_core/utils/packet_builder.py`**: Minor fix applied to both live and candidate versions
- **`gaia-common/gaia_common/constants.py`**: Fixed multiline string escaping in `LOGICAL_STOP_PUNCTUATION` (literal newline → `\n`)

### 12. Docker Compose & Dockerfile Updates

- **`docker-compose.candidate.yml`**: Added `gaia-candidate-shared:/shared:rw` volume mount
- All candidate Dockerfiles updated to install `[dev]` dependencies for containerized validation
- **`candidates/gaia-common/pyproject.toml`**: Added dev dependency group (pytest, ruff, mypy) and tool configurations

## Impact and Potential Breakpoints

- **Network stability changes** are the highest-value fixes in this batch. The old `cmd_live start` destroying the network was a ticking time bomb — any time someone restarted live while candidates were running, candidates would lose connectivity silently. Now both compose files can independently create/join the same network.
- **Orchestrator via compose** means `./gaia.sh orchestrator start` now respects all the config in `docker-compose.yml` (environment variables, volumes, health checks, restart policy). The old `docker run` invocation was missing several of these. Existing orchestrator containers started via the old method will need to be stopped and restarted through compose.
- **Docker networking change** in the orchestrator — if container names don't match what the orchestrator expects, health checks will fail in a different way than before. Verify container names in `docker-compose.candidate.yml` match the hostnames used in `docker_manager.py`.
- **Response persistence fix** changes what gets stored in session history. Existing sessions with corrupted history won't be retroactively fixed but will stop accumulating bad data.
- **Port consolidation** (candidates using same internal ports as live) relies on Docker network isolation. If both stacks somehow share a network, port conflicts will occur.
- **GPU graceful degradation** means the resource monitor will silently stop reporting GPU stats if `pynvml` fails. This is intentional but could mask real GPU issues if not paired with external monitoring.

## Debugging and Rollback

- **Network issues after changes**: Run `docker network inspect gaia-network` to verify the network exists and both live + candidate containers are attached. Both compose files now define the same network — Docker should reuse it.
- **Orchestrator won't start via compose**: Ensure no old orchestrator container exists from the `docker run` era: `docker rm -f gaia-orchestrator` then `./gaia.sh orchestrator start`.
- **Health check failures after deploy**: Check that container names in `docker-compose.candidate.yml` match the hostnames in `docker_manager.py`.
- **Empty session history**: If `user_facing_response` is unexpectedly empty, check the routing logic in `external_voice.py` — the variable must be populated before the persistence call.
- **GPU monitoring missing**: Check container logs for "pynvml" warnings at startup. If present, verify NVIDIA container toolkit is installed.
- **Rollback**: All changes are uncommitted. `git checkout -- <file>` for any individual file, or `git checkout .` to revert everything.

## Testing

1. **Network stability (candidate-first)**: Run `docker compose -f docker-compose.candidate.yml --profile full up -d` *without* starting live first — verify it creates `gaia-network` and starts cleanly
2. **Network stability (live start with candidates running)**: With candidates up, run `./gaia.sh live start` — verify candidates retain connectivity (no restart, no DNS resolution failures)
3. **Orchestrator via compose**: Run `./gaia.sh orchestrator start`, then `docker compose ps` — verify the orchestrator appears as a compose-managed service with health check
4. **Compose config validation**: `docker compose -f docker-compose.yml config` and `docker compose -f docker-compose.candidate.yml config` — both should succeed
5. **Health checks**: Deploy candidate stack, verify orchestrator can reach services via container hostnames (`docker logs gaia-orchestrator`)
6. **Response persistence**: Send a message through gaia-core, then inspect `sessions.json` — verify stored response matches what was displayed, not raw LLM output
7. **GPU fallback**: Run gaia-core container without `--gpus` flag, verify it starts cleanly with a warning instead of crashing
8. **Log filtering**: Start a service with health check filter installed, hit the health endpoint 100 times, verify logs show minimal noise
9. **Containerized validation**: Run `./scripts/promote_candidate.sh gaia-core --validate` and confirm ruff/mypy/pytest execute inside the container
