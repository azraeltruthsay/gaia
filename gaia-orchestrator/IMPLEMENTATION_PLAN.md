# GAIA Orchestrator Implementation Plan

> **Status**: Phase 1-6 Complete, Ready for Production
> **Last Updated**: 2026-02-06
> **Purpose**: Persistent documentation to survive context loss

## Overview

Create a new `gaia-orchestrator` service that coordinates GPU resources and container lifecycle across the GAIA ecosystem. This enables:

1. **Live→Candidate GPU Handoff**: Bring down live containers gracefully so candidates can claim GPU for bicameral mind testing
2. **Prime→Study Handoff**: Release Prime from Core so Study can load it for LoRA training sessions
3. **Oracle Fallback Notification**: Notify users when Groq Oracle is loaded because Prime failed
4. **Centralized Coordination**: Single source of truth for GPU ownership and container state

## Architecture

```
                    +-------------------+
                    | gaia-orchestrator |
                    |    Port 6410      |
                    +--------+----------+
                             |
        ┌────────────────────┼────────────────────┐
        │                    │                    │
   +----v-----+        +-----v------+       +-----v-----+
   | gaia-core|        | gaia-study |       | gaia-web  |
   | Port 6415|        | Port 8766  |       | Port 6414 |
   +----------+        +------------+       +-----------+
      GPU                  GPU              Notifications
    (Prime)             (Training)           (WebSocket)
```

## Directory Structure

```
gaia-orchestrator/
├── Dockerfile
├── requirements.txt
├── pyproject.toml
├── gaia_orchestrator/
│   ├── __init__.py
│   ├── main.py              # FastAPI app with all endpoints
│   ├── config.py            # Configuration management
│   ├── state.py             # State persistence to /shared/orchestrator/
│   ├── docker_manager.py    # Docker SDK wrapper for container ops
│   ├── gpu_manager.py       # GPU ownership, CUDA monitoring
│   ├── handoff_manager.py   # Prime↔Study handoff protocol
│   ├── notification_manager.py  # Oracle fallback notifications
│   └── models/
│       └── schemas.py       # Pydantic request/response models
└── config/
    └── orchestrator.yaml
```

## Files to Modify

- `docker-compose.yml` - Add orchestrator service
- `gaia.sh` - Add `./gaia.sh orchestrator [start|stop|status]` command
- `gaia-core/gaia_core/models/_model_pool_impl.py` - Add notification hook when Oracle fallback used

## API Endpoints

### GPU Management
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/gpu/acquire` | Request GPU ownership |
| POST | `/gpu/release` | Release GPU ownership |
| GET | `/gpu/status` | Current GPU state and queue |
| POST | `/gpu/wait` | Block until GPU available |

### Container Lifecycle
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/containers/live/stop` | Stop live stack, release GPU |
| POST | `/containers/candidate/start` | Start candidates with GPU |
| POST | `/containers/swap` | Inject single candidate service |
| GET | `/containers/status` | All container states |

### Prime Handoff
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/handoff/prime-to-study` | Core releases GPU for Study training |
| POST | `/handoff/study-to-prime` | Study returns GPU to Core |
| GET | `/handoff/{id}/status` | Check handoff progress |

### Notifications
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/notify/oracle-fallback` | Notify user Oracle is being used |
| WS | `/ws/notifications` | Real-time updates to web clients |

## State Persistence

State stored at `/shared/orchestrator/state.json`:

```json
{
  "gpu": {
    "owner": "gaia-core",
    "lease_id": "uuid",
    "reason": "prime_inference"
  },
  "containers": {
    "live": {"core": "running", "web": "running", ...},
    "candidate": {"core": "stopped", ...}
  },
  "handoffs": {"active": null, "history": []}
}
```

## Prime Handoff Protocol

1. Study calls `POST /handoff/prime-to-study`
2. Orchestrator calls `gaia-core:6415/gpu/release`
3. Wait for CUDA cleanup (poll nvidia-smi < 500MB)
4. Update GPU lease to `gaia-study`
5. Signal `gaia-study:8766/study/gpu-ready`
6. Training proceeds...
7. On complete: `POST /handoff/study-to-prime` (reverse)

## Oracle Fallback Notification

When `_model_pool_impl.py` uses fallback chain:

```python
# Add notification hook:
if fallback in self.models:
    logger.warning(f"Using {fallback} as fallback for {role}")
    # NEW: Notify orchestrator
    self._notify_oracle_fallback(fallback, role)
```

Orchestrator routes notification to gaia-web, which delivers via Discord/WebSocket.

## Implementation Phases

### Phase 1: Core Infrastructure ✅ COMPLETE
- [x] Create directory structure
- [x] Write implementation plan
- [x] FastAPI service structure (`main.py`)
- [x] Configuration management (`config.py`)
- [x] State management with disk persistence (`state.py`)
- [x] Pydantic models (`models/schemas.py`)
- [x] Health and status endpoints
- [x] Dockerfile and requirements.txt

### Phase 2: Docker Integration ✅ COMPLETE
- [x] Docker SDK integration (`docker_manager.py`)
- [x] Container start/stop operations
- [x] Health monitoring
- [x] Integration with existing compose files

### Phase 3: GPU Management ✅ COMPLETE
- [x] GPU ownership tracking with leases (`gpu_manager.py`)
- [x] CUDA cleanup verification via nvidia-smi/pynvml
- [x] Integration with Core's existing `/gpu/release` and `/gpu/reclaim`
- [x] GPU wait queue

### Phase 4: Prime Handoff Protocol ✅ COMPLETE
- [x] Handoff manager (`handoff_manager.py`)
- [x] Coordinated GPU transfer Core↔Study
- [x] Async handoff with status tracking
- [x] Timeout and error handling

### Phase 5: Notifications ✅ COMPLETE
- [x] Notification manager (`notification_manager.py`)
- [x] Oracle fallback notification to users
- [x] WebSocket for real-time updates
- [x] Integration with gaia-web (endpoint ready)
- [x] Modify `_model_pool_impl.py` for fallback hook

### Phase 6: CLI & Compose Integration ✅ COMPLETE
- [x] Update `gaia.sh` with orchestrator commands
- [x] Add to `docker-compose.yml`
- [x] Build and test service

### Phase 7: Bicameral Mind Support
- [ ] Enable two-model loading in candidate
- [ ] GPU memory validation for dual models

## Verification Tests

### 1. GPU Handoff Test
```bash
curl -X POST localhost:6410/containers/live/stop
curl -X POST localhost:6410/containers/candidate/start -d '{"gpu_enabled":true}'
# Verify: nvidia-smi shows candidate using GPU
```

### 2. Prime→Study Handoff Test
```bash
curl -X POST localhost:6410/handoff/prime-to-study
# Check: GET /handoff/{id}/status shows phases
# Verify: Study can run training
```

### 3. Oracle Notification Test
```bash
# Release GPU from Core
curl -X POST localhost:6415/gpu/release
# Send message to Core requiring Prime
# Verify: User receives "Using cloud inference" notification
```

## Dependencies

```
fastapi>=0.109.0
uvicorn>=0.27.0
docker>=7.0.0
pynvml>=11.5.0
httpx>=0.26.0
pydantic>=2.5.0
pyyaml>=6.0
```

## Port Assignments

| Service | Port |
|---------|------|
| gaia-orchestrator | 6410 |
| gaia-web | 6414 |
| gaia-core | 6415 |
| gaia-core-candidate | 6416 |
| gaia-web-candidate | 6417 |
| gaia-mcp | 8765 |
| gaia-study | 8766 |
| gaia-mcp-candidate | 8767 |
| gaia-study-candidate | 8768 |

---

## Dev Journal

### 2026-02-06 (Session 3 Continued - Implementation Complete)
- Added orchestrator to `docker-compose.yml`
- Added Oracle fallback notification hook to `_model_pool_impl.py`:
  - New `_notify_oracle_fallback()` method sends async HTTP notifications
  - Modified `get_model_for_role()` fallback logic to detect API model usage
- Built and tested orchestrator successfully:
  - `./gaia.sh orchestrator build` - Image built
  - `./gaia.sh orchestrator start` - Container running
  - Health check: `curl localhost:6410/health` ✓
  - GPU acquire/release tested ✓
  - Oracle fallback notification tested ✓
  - Status shows in `./gaia.sh status` ✓

### 2026-02-05 (Session 3 - After Context Recovery)
- Recovered plan from user after two context losses
- Created all core files:
  - `main.py` - Full FastAPI app with all endpoints
  - `config.py` - Configuration with env var support
  - `state.py` - Async state persistence to JSON
  - `models/schemas.py` - All Pydantic models
  - `docker_manager.py` - Docker SDK wrapper
  - `gpu_manager.py` - GPU monitoring via pynvml
  - `handoff_manager.py` - Prime↔Study handoff protocol
  - `notification_manager.py` - WebSocket broadcasts
  - `Dockerfile` - Container definition
  - `requirements.txt` - Python dependencies
  - `pyproject.toml` - Package definition
  - `config/orchestrator.yaml` - Default configuration
- Updated `gaia.sh` with new commands:
  - `./gaia.sh orchestrator [start|stop|status|build|logs]`
  - `./gaia.sh gpu [status|release]`
  - `./gaia.sh handoff [prime-to-study|study-to-prime|status]`

### 2026-02-05 (Initial)
- Initial plan created after two context losses
- Writing persistent documentation to disk
- Starting Phase 1 implementation
