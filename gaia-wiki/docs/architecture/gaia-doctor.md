# gaia-doctor — The Immune System

**Port:** 6419 | **GPU:** No | **Dependencies:** stdlib only

gaia-doctor is the persistent health watchdog that monitors all GAIA services, automatically restarts crashed containers, and runs cognitive test batteries to validate model alignment.

## Design Principles

- **stdlib-only**: Zero external dependencies. The immune system must be the last thing to go down.
- **Passive monitoring**: Observes and reacts. Active adversarial testing is handled by gaia-monkey.
- **Circuit breaker**: Max 2 production restarts per 30-minute rolling window prevents restart storms.

## Key Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Container health check |
| `/status` | GET | Full status (services, alarms, remediations) |
| `/cognitive/status` | GET | Cognitive battery status + alignment |
| `/cognitive/run` | POST | Trigger cognitive battery run |
| `/registry` | GET | Service registry wiring validation status |
| `/maintenance/enter` | POST | Enter maintenance mode |
| `/maintenance/exit` | POST | Exit maintenance mode |
| `/surgeon/queue` | GET | Pending repair proposals |
| `/surgeon/approve` | POST | Approve a repair |

## Service Registry Integration

Doctor loads its service monitoring list from the compiled blueprint registry at `/shared/registry/service_registry.json`. If the registry isn't available, it falls back to a hardcoded service dict. The registry is compiled by `scripts/compile_registry.py` from blueprint YAMLs.

Doctor also runs periodic wiring validation (every 5 minutes) to check for orphaned outbound calls, exposing results via the `/registry` endpoint.

## Subsystems

- **Health polling**: HTTP health checks on all services every 60s
- **Log scanning**: Detects errors/irritations from Docker logs
- **Code auditing**: Detects disk/memory mismatches in source files
- **Cognitive battery**: ~50 tests across 9 sections validating model alignment
- **Cognitive monitor**: Periodic heartbeat probes to verify inference health
- **Surgeon queue**: Human-in-the-loop approval for automated repairs
- **KV cache monitoring**: Independent pressure monitoring on Nano/Core slots
- **GPU zombie cleanup**: Detects orphaned GPU processes
- **VRAM reconciliation**: Compares orchestrator state vs actual nvidia-smi
