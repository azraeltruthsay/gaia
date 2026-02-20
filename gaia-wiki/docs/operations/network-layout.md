# Network Layout

All GAIA services communicate on the `gaia-net` Docker bridge network (`172.28.0.0/16`).

## Service Discovery

Services reference each other by hostname (Docker DNS):

| Service | Hostname | Internal Port |
|---------|----------|--------------|
| gaia-prime | `gaia-prime` | 7777 |
| gaia-core | `gaia-core` | 6415 |
| gaia-web | `gaia-web` | 6414 |
| gaia-study | `gaia-study` | 8766 |
| gaia-mcp | `gaia-mcp` | 8765 |
| gaia-orchestrator | `gaia-orchestrator` | 6410 |
| gaia-wiki | `gaia-wiki` | 8080 |
| gaia-core-candidate | `gaia-core-candidate` | 6415 |
| gaia-mcp-candidate | `gaia-mcp-candidate` | 8765 |

Candidates share the same network, enabling both isolated testing and live injection.

## HA Failover

When HA mode is active, candidate services run as hot standbys:

```
gaia-web ──→ gaia-core (primary)
         └─→ gaia-core-candidate (fallback, on ConnectError)
```

### Failover Trigger Conditions

- **Triggers failover:** ConnectError, RemoteProtocolError, HTTP 502/503/504
- **Does NOT trigger failover:** Timeouts (service is alive but slow)
- **Maintenance mode:** Disables failover routing, allows direct inter-service calls

### HA States

| Live | Candidate | Status | Meaning |
|------|-----------|--------|---------|
| Healthy | Healthy | `active` | Failover ready |
| Healthy | Unhealthy | `degraded` | Failover unavailable |
| Unhealthy | Healthy | `failover_active` | Traffic on candidate |
| Unhealthy | Unhealthy | `failed` | Both down |

### Scripts

```bash
./scripts/ha_start.sh       # Start HA standby
./scripts/ha_stop.sh        # Stop HA standby
./scripts/ha_maintenance.sh on|off|status  # Toggle maintenance
./scripts/ha_sync.sh --incremental|--full  # Session sync
```

## Session Sync

One-way sync (live → candidate) runs every 30 seconds via the health watchdog:

- `sessions.json` — active session state
- `session_vectors/*.json` — per-session vector data
- `prime.md`, `Lite.md` — cognitive checkpoints (read-only on candidate)
