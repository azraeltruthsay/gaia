# Phase 1 — Service Health Check Results

**Timestamp**: 2026-03-26 ~03:01 UTC
**Method**: `curl -s` against each service health endpoint + `docker ps`

## Container Status (docker ps)

| Container | Docker Status |
|-----------|---------------|
| gaia-core | Up 8 hours (healthy) |
| gaia-nano | Up 13 hours (healthy) |
| gaia-prime | Up 8 hours (healthy) |
| gaia-web | Up 26 minutes (healthy) |
| gaia-mcp | Up 2 days (healthy) |
| gaia-study | Up 31 hours (healthy) |
| gaia-orchestrator | Up 21 hours (healthy) |
| gaia-doctor | Up 3 days (healthy) |
| gaia-monkey | Up 3 days (healthy) |
| gaia-audio | Up 3 days (healthy) |
| gaia-translate | Up 3 days **(unhealthy)** |
| gaia-mcp-candidate | **Restarting** (1) |
| gaia-core-candidate | **Restarting** (1) |
| gaia-wiki | Up 3 days (healthy) |
| dozzle | Up 3 days |
| filebeat | Up 3 days |
| logstash | Up 3 days (healthy) |
| kibana | Up 3 days (healthy) |
| elasticsearch | Up 3 days (healthy) |

## Service Health Endpoints

| Service | Port | HTTP | Status | Response | Notes |
|---------|------|------|--------|----------|-------|
| gaia-core | 6415 | 200 | healthy | `{"status":"healthy","service":"gaia-core","inference_ok":true,"inference_detail":"ok"}` | Inference verified OK |
| gaia-nano | 8090 | 200 | ok | `{"status":"ok","engine":"gaia-managed","backend":"engine","model_loaded":true,"mode":"active","managed":true,"worker_pid":18330}` | GPU engine, model loaded |
| gaia-prime | 7777 | 200 | ok | `{"status":"ok","engine":"gaia-managed","backend":"gguf","model_loaded":true,"mode":"active","managed":true,"worker_pid":428596}` | GGUF backend (CPU), model loaded |
| gaia-web | 6414 | 200 | healthy | `{"status":"healthy","service":"gaia-web","timestamp":"2026-03-26T03:00:52.194556+00:00"}` | Recently restarted (26 min) |
| gaia-mcp | 8765 | 200 | healthy | `{"status":"healthy","service":"gaia-mcp"}` | |
| gaia-study | 8766 | 200 | healthy | `{"status":"healthy","service":"gaia-study"}` | |
| gaia-orchestrator | 6410 | 200 | healthy | `{"status":"healthy","service":"gaia-orchestrator"}` | |
| gaia-doctor | 6419 | 200 | healthy | `{"status":"healthy","service":"gaia-doctor"}` | |
| gaia-monkey | 6420 | 200 | ok | `{"status":"ok","service":"gaia-monkey"}` | |
| gaia-audio | 8080 | 200 | ok | `{"status":"ok","service":"gaia-audio","version":"0.1.0"}` | |
| gaia-translate | 5000 | 000 | **DOWN** | No response (connection refused) | Docker reports unhealthy; port not responding |
| Core embedded engine | 8092* | 200 | ok | `{"status":"ok","engine":"gaia-managed","backend":"engine","model_loaded":true,"mode":"active","managed":true,"worker_pid":12049}` | Checked via `docker exec gaia-core`; GPU engine, model loaded |

*Port 8092 is internal to the gaia-core container.

## GPU State

> **Note**: `nvidia-smi` was denied by sandbox permissions. GPU memory breakdown unavailable this run.

From the health responses:
- **gaia-nano**: Running on GPU (`backend: "engine"`)
- **gaia-prime**: Running on CPU/GGUF (`backend: "gguf"`)
- **Core embedded engine**: Running on GPU (`backend: "engine"`)

## Summary

- **11/12 service endpoints responding** (all HTTP 200)
- **1 service DOWN**: `gaia-translate` (LibreTranslate) -- container up but marked unhealthy, port 5000 not responding
- **2 candidates crash-looping**: `gaia-core-candidate` and `gaia-mcp-candidate` both in restart loops
- **All inference tiers operational**: Nano (GPU), Core (GPU), Prime (CPU/GGUF) -- all models loaded and active
- **gaia-web recently restarted** (26 min uptime vs hours/days for others)
