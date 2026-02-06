**Date:** 2026-02-03
**Title:** Fixed Candidate Container Startup & Implemented Modular Swap Architecture

## Problem

The candidate containers (`gaia-web-candidate`, etc.) were failing to start. Initial investigation suggested a volume mount conflict with editable pip installs, but the actual root cause was simpler.

## Root Cause

The `gaia-web` candidate module was a stub - it only contained `__init__.py` with no actual `main.py`. The Dockerfile and docker-compose were configured to run:

```
python -m uvicorn gaia_web.main:app
```

But `gaia_web/main.py` didn't exist, causing:

```
ModuleNotFoundError: No module named 'gaia_web.main'
```

This wasn't a volume mount issue or PYTHONPATH problem - the file simply didn't exist.

## Solution

Created `/gaia/GAIA_Project/candidates/gaia-web/gaia_web/main.py` with a minimal FastAPI application:

```python
from fastapi import FastAPI
from fastapi.responses import JSONResponse

app = FastAPI(
    title="GAIA Web",
    description="The Face - UI and API gateway",
    version="0.1.0",
)

@app.get("/health")
async def health_check():
    return JSONResponse(
        status_code=200,
        content={"status": "healthy", "service": "gaia-web"}
    )

@app.get("/")
async def root():
    return {
        "service": "gaia-web",
        "description": "GAIA Web Gateway Service",
        "endpoints": {"/health": "Health check", "/": "This endpoint"}
    }
```

After this fix, all four candidate services start and pass health checks:

| Service | Port | Status |
|---------|------|--------|
| gaia-web-candidate | 6417 | healthy |
| gaia-core-candidate | 6416 | healthy |
| gaia-mcp-candidate | 8767 | healthy |
| gaia-study-candidate | 8768 | healthy |

## Additional Work: Modular Swap Architecture

While investigating, I implemented the modular swap capability that was partially designed but not fully functional.

### Changes to docker-compose.yml

Made service endpoints configurable via environment variables:

```yaml
# gaia-core environment
- MCP_ENDPOINT=${MCP_ENDPOINT:-http://gaia-mcp:8765/jsonrpc}
- STUDY_ENDPOINT=${STUDY_ENDPOINT:-http://gaia-study:8766}

# gaia-web environment
- CORE_ENDPOINT=${CORE_ENDPOINT:-http://gaia-core:6415}
- MCP_ENDPOINT=${MCP_ENDPOINT:-http://gaia-mcp:8765/jsonrpc}
```

This allows injecting candidates into the live flow:

```bash
# Route live core through candidate MCP
MCP_ENDPOINT=http://gaia-mcp-candidate:8765/jsonrpc docker compose up -d gaia-core
```

### New Helper Script: gaia.sh

Created `/gaia/GAIA_Project/gaia.sh` for unified stack management:

```bash
./gaia.sh status                  # Show all services
./gaia.sh live start              # Start live stack
./gaia.sh candidate start         # Start candidate stack
./gaia.sh swap mcp candidate      # Inject candidate MCP into live flow
./gaia.sh swap mcp live           # Restore live MCP
```

### Network Connectivity Verified

All candidate services can communicate on `gaia-network`:

```
web-candidate -> core-candidate: OK
core-candidate -> mcp-candidate: OK
core-candidate -> study-candidate: OK
```

## Promotion

After testing, promoted the `main.py` fix to live:

```bash
rsync -av candidates/gaia-web/gaia_web/ gaia-web/gaia_web/
```

## Remaining Issues

The **live gaia-mcp** still has a dependency on `gaia_core`:

```
ModuleNotFoundError: No module named 'gaia_core'
```

This is the decoupling work outlined in `SOA-decoupled-proposal.md`. The candidate gaia-mcp has this fixed, but the live version needs the same treatment (or promotion of the candidate code).

## Files Changed

- `candidates/gaia-web/gaia_web/main.py` - Created (FastAPI app)
- `gaia-web/gaia_web/main.py` - Promoted from candidate
- `docker-compose.yml` - Made endpoints configurable
- `gaia.sh` - Created (stack management script)

## Testing Commands

```bash
# Check all service status
./gaia.sh status

# Start candidate stack
./gaia.sh candidate start

# Test health endpoints
curl http://localhost:6417/health  # web-candidate
curl http://localhost:6416/health  # core-candidate
curl http://localhost:8767/health  # mcp-candidate
curl http://localhost:8768/health  # study-candidate

# Test inter-service connectivity
docker exec gaia-web-candidate curl -s http://gaia-core-candidate:6415/health
```
