# gaia-orchestrator — The Coordinator

GPU and container lifecycle management. Coordinates handoffs between services that need GPU access.

## Responsibilities

- Manage GPU ownership (lease-based model)
- Coordinate sleep/wake handoffs between gaia-prime and gaia-study
- Monitor service health (live + candidate)
- Persist orchestrator state for crash recovery
- Broadcast notifications for state changes

## GPU Ownership Model

Only one service can hold the GPU at a time:

| Owner | Mode | Duration |
|-------|------|----------|
| gaia-prime | Active inference | Most of the time |
| gaia-study | Training/embedding | During sleep cycles |
| none | Idle | Brief transition periods |

## Health Watchdog

The health watchdog monitors both live and candidate services:

- **Polling interval:** 30 seconds
- **Failure threshold:** 2 consecutive failures before declaring unhealthy
- **HA states:** `active`, `degraded`, `failover_active`, `failed`
- **Session sync:** Runs live-to-candidate sync when HA is active

See [Network Layout](../operations/network-layout.md) for HA failover details.

## State Persistence

Orchestrator state is persisted to `/shared/orchestrator/` via the `StateManager`. On startup, stale in-progress handoffs are automatically marked FAILED (safe default after crash).

## Endpoints

| Path | Method | Purpose |
|------|--------|---------|
| `/health` | GET | Container health check |
| `/status` | GET | Full orchestrator state |
| `/gpu/status` | GET | Current GPU ownership |
| `/gpu/acquire` | POST | Request GPU lease |
| `/gpu/release` | POST | Release GPU lease |
| `/handoff` | POST | Initiate GPU handoff |
| `/notifications/stream` | GET | SSE notification stream |
