# GAIA Service Blueprint: `gaia-orchestrator` (The Coordinator)

## Role and Overview

`gaia-orchestrator` is the infrastructure management service of the GAIA system. It manages Docker containers, allocates GPU resources, and oversees service lifecycle. It coordinates GPU handoffs between `gaia-prime` (inference) and `gaia-study` (training/embeddings), and provides a REST API and WebSocket interface for monitoring and control.

## Container Configuration

**Base Image**: `python:3.11-slim`

**Port**: 6410

**Build Context**: Project root (`.`) — the Dockerfile copies from both `gaia-orchestrator/` and `gaia-common/`.

**Health Check**: `curl -f http://localhost:6410/health` (30s interval, 30s start_period)

**Startup**: `uvicorn gaia_orchestrator.main:app --host 0.0.0.0 --port 6410`

### Dockerfile Build (v0.3 changes)

The Dockerfile was updated in v0.3 to use the project root as build context:

```dockerfile
COPY gaia-orchestrator/requirements.txt .
COPY gaia-orchestrator/gaia_orchestrator/ ./gaia_orchestrator/
COPY gaia-orchestrator/pyproject.toml .
COPY gaia-common /gaia-common
RUN pip install --no-cache-dir -e /gaia-common/
```

This ensures `gaia-common` is available for the orchestrator's health check filter import.

### Key Environment Variables

| Variable | Value | Purpose |
|----------|-------|---------|
| `ORCHESTRATOR_CORE_URL` | `http://gaia-core:6415` | Core service address |
| `ORCHESTRATOR_WEB_URL` | `http://gaia-web:6414` | Web service address |
| `ORCHESTRATOR_STUDY_URL` | `http://gaia-study:8766` | Study service address |
| `ORCHESTRATOR_MCP_URL` | `http://gaia-mcp:8765` | MCP service address |
| `ORCHESTRATOR_PRIME_URL` | `http://gaia-prime:7777` | Prime inference service (v0.3) |
| `GAIA_SERVICE` | `orchestrator` | Service identifier |

### Volume Mounts

- `./gaia-orchestrator:/app:rw` — Source code
- `./gaia-common:/gaia-common:ro` — Shared library
- `gaia-shared:/shared:rw` — Persistent state storage
- `/var/run/docker.sock:/var/run/docker.sock:ro` — Docker daemon access

## Internal Architecture

### Key Components

1. **`main.py`** — FastAPI entry point with lazy-loaded managers
2. **`gpu_manager.py`** (GPUManager) — GPU monitoring and ownership coordination
   - Queries VRAM via `pynvml` (optional, graceful fallback)
   - Tracks GPU ownership between services
   - Methods: `get_memory_info()`, `is_gpu_free()`, `wait_for_gpu_cleanup()`, `request_release_from_core()`, `signal_study_gpu_ready()`
3. **`docker_manager.py`** (DockerManager) — Docker SDK integration
   - Lists live and candidate services
   - Methods: `get_status()`, `start_live()`, `stop_live()`, `start_candidate()`, `stop_candidate()`, `swap_service()`
4. **`handoff_manager.py`** (HandoffManager) — GPU handoff protocol
   - Orchestrates `prime_to_study` and `study_to_prime` GPU transfers
5. **`notification_manager.py`** (NotificationManager) — WebSocket broadcasting
6. **`state.py`** (StateManager) — Persistent state for GPU ownership and handoffs

## API Endpoints

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/health` | GET | Health check |
| `/status` | GET | Full orchestrator status |
| `/gpu/status` | GET | Current GPU ownership |
| `/gpu/acquire` | POST | Request GPU ownership |
| `/gpu/release` | POST | Release GPU ownership |
| `/gpu/wait` | POST | Wait for GPU availability |
| `/containers/status` | GET | All container statuses |
| `/containers/{live\|candidate}/{start\|stop}` | POST | Container lifecycle |
| `/containers/swap` | POST | Swap service between live/candidate |
| `/handoff/{prime-to-study\|study-to-prime}` | POST | GPU handoff initiation |
| `/handoff/{handoff_id}/status` | GET | Handoff status |
| `/notify/oracle-fallback` | POST | Receive fallback notifications |
| `/ws/notifications` | WS | Real-time notifications |

## Interaction with Other Services

- **`gaia-core`**: Sends GPU release/reclaim requests via HTTP
- **`gaia-prime`**: Monitors health via `ORCHESTRATOR_PRIME_URL` (v0.3)
- **`gaia-study`**: Signals GPU ready/release for training coordination
- **Docker daemon**: Manages container lifecycle via Docker SDK
- **`gaia-common`**: Uses health check filter utility

## Known Limitations

- `gaia-prime` is not yet fully integrated into `DockerManager`'s service lists (LIVE_SERVICES, CANDIDATE_SERVICES)
- `ORCHESTRATOR_PRIME_URL` is set but orchestrator code may not yet actively use it for GPU coordination with gaia-prime
- GPU handoff between gaia-prime and gaia-study needs careful scheduling on single-GPU systems
