# Service Registry & Wiring Validation

The Service Registry is GAIA's automated system for tracking all service endpoints, validating inter-service wiring, and detecting architectural drift.

## Architecture

```
Blueprint YAMLs              Compiled Registry           Consumers
knowledge/blueprints/    -->  /shared/registry/       --> gaia-doctor (health monitoring)
  gaia-core.yaml              service_registry.json   --> gaia-web (dashboard widget)
  gaia-doctor.yaml                                    --> validate_wiring.py (CI gate)
  gaia-nano.yaml                                      --> promote_pipeline.sh (Stage 3.5)
  dozzle.yaml
  ... (12 total)
```

## Blueprint Lifecycle

### 1. Discovery (Auto-Generation)

```bash
# Auto-discover endpoints from a running service's OpenAPI schema
python scripts/discover_blueprint.py gaia-core --port 6415

# Discover all known services
python scripts/discover_blueprint.py --all
```

FastAPI services expose `/openapi.json` by default. The discovery script extracts all routes, generates a candidate blueprint YAML, and writes it to `knowledge/blueprints/candidates/`.

### 2. Refresh (Drift Detection)

```bash
# Diff running services against existing blueprints
python scripts/discover_blueprint.py gaia-core --port 6415 --refresh

# Auto-update blueprints with newly discovered endpoints
python scripts/discover_blueprint.py --all --refresh --update
```

Refresh mode compares discovered endpoints against existing blueprint interfaces. Reports added and removed endpoints. With `--update`, merges new endpoints into the YAML.

### 3. Compilation (Blueprint -> JSON)

```bash
python scripts/compile_registry.py
```

Loads all live blueprint YAMLs, derives the graph topology (edges from interface matching), runs wiring validation, and writes `/shared/registry/service_registry.json`.

### 4. Validation (Wiring Check)

```bash
python scripts/validate_wiring.py
```

Reads the compiled JSON (stdlib only) and reports:

- **Orphaned outbound**: Outbound calls with no matching inbound endpoint
- **Uncalled inbound**: Inbound endpoints no outbound points to

Exit codes: 0=clean, 1=warnings, 2=errors.

### 5. Full Refresh Cycle

```bash
# One command: discover + compile + validate
python scripts/refresh_blueprints.py --update
```

## Integration Points

### Promotion Pipeline (Stage 3.5)

The promotion pipeline runs `compile_registry.py` + `validate_wiring.py` between lint validation (Stage 3) and cognitive smoke tests (Stage 4). Non-blocking — warns but doesn't fail promotion. Skip with `--skip-wiring`.

### Doctor (Periodic Check)

gaia-doctor reads the compiled registry to build its service monitoring list. Every 5 minutes, it runs wiring validation and caches the result at the `/registry` GET endpoint. If the registry file is missing, doctor falls back to a hardcoded service dict.

### Dashboard (System State Panel)

The "Service Registry" card in the System State panel shows:

- **Green**: `{N} svc / {N} edges` — all wiring clean
- **Yellow**: `{N} svc / {N} orphaned` — orphaned outbound detected
- **Gray**: `not compiled` — registry needs compilation

Polls `/api/system/registry/validation` every 10 seconds.

## Edge Derivation

Edges between services are **never stored** — always derived from interface matching:

- An edge exists if Service A has an outbound interface whose path matches Service B's inbound interface of compatible transport type
- Adding a new service with matching interfaces automatically wires it into the graph
- Transport types must match: HTTP REST paths, WebSocket paths, MCP methods, event topics

## Scripts Reference

| Script | Purpose |
|--------|---------|
| `scripts/discover_blueprint.py` | Auto-generate/refresh blueprints from live services |
| `scripts/refresh_blueprints.py` | Full lifecycle: discover + compile + validate |
| `scripts/compile_registry.py` | Blueprints -> JSON registry with wiring validation |
| `scripts/validate_wiring.py` | Standalone wiring check (CI-friendly exit codes) |
