Check health of: $ARGUMENTS

Hit health endpoints for the specified service(s). If no service specified, check all.

Endpoints:
- gaia-core: http://localhost:6415/health
- gaia-web: http://localhost:6414/health
- gaia-mcp: http://localhost:8765/health
- gaia-doctor: http://localhost:6419/health
- gaia-study: http://localhost:8766/health
- gaia-monkey: http://localhost:6420/health
- gaia-orchestrator: http://localhost:6410/health
- gaia-prime: http://localhost:7777/health
- gaia-audio: http://localhost:8080/health

Use `curl -s --max-time 3` for each endpoint. Report status as healthy/unhealthy/unreachable.
For "all", check every endpoint and present a summary table.
