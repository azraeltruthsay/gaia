# GAIA Operational Scripts Blueprint

## Role and Overview

The GAIA project includes several scripts for managing the Dockerized environment through its Software Development Life Cycle (SDLC). These scripts automate stack management, testing, validation, and promotion of candidate services to live.

## Key Scripts

### 1. `gaia.sh` — Primary Stack Management CLI

Central command-line interface for managing GAIA Docker Compose stacks.

**Key Commands**:

| Command | Purpose |
|---------|---------|
| `gaia.sh live` | Start/stop/restart the live stack (`docker-compose.yml`) |
| `gaia.sh candidate` | Start/stop/restart the candidate stack (`docker-compose.candidate.yml`) |
| `gaia.sh status` | Display status of both live and candidate services |
| `gaia.sh swap <service>` | Re-route live traffic to a candidate service version |
| `gaia.sh orchestrator <cmd>` | Interact with gaia-orchestrator (GPU status, handoffs) |
| `gaia.sh gpu <cmd>` | High-level GPU management |
| `gaia.sh handoff <params>` | GPU handoff operations |

**Live stack startup** includes `--env-file ./.env.discord` to load the Discord bot token.

### 2. `scripts/promote_candidate.sh` — Formal Candidate Promotion

Safely promotes validated candidate services to the live stack with backup, validation, and restart.

**Service Configuration**:

```bash
SERVICE_CONFIG:
  ["gaia-core"]="6415:6416:yes"      # live_port:candidate_port:has_container
  ["gaia-prime"]="7777:7778:yes"     # Added in v0.3
  ["gaia-mcp"]="8765:8767:yes"
  ["gaia-study"]="8766:8768:yes"
  ["gaia-web"]="6414::yes"           # No candidate port
  ["gaia-common"]=":::no"            # No container (library only)
```

**Python Services** (validated with ruff/mypy/pytest): gaia-core, gaia-mcp, gaia-study, gaia-web, gaia-common

**Non-Python Services**: gaia-prime (vLLM build, no `/app` Python project — validation skipped)

**Promotion Workflow**:
1. Validate service exists in `candidates/`
2. Optional `--validate`: Run ruff, mypy, pytest in Docker container
3. Optional `--test`: Health check candidate on candidate port
4. Backup current live to `{service}.bak/`
5. Promote: rsync/cp candidate files to live directory
6. Restart container (if `has_container=yes`)
7. Health check live service

**Options**:
- `--validate` — Containerized linting, type checking, unit tests
- `--test` — Health check candidate before promoting
- `--no-backup` — Skip backup creation
- `--no-restart` — Don't restart container after promotion

### 3. `test_candidate.sh` — Developer Testing Utility

Developer-facing script for iterative testing and pre-validation of candidate services.

**Key Commands**:

| Command | Purpose |
|---------|---------|
| `./test_candidate.sh all [--gpu\|--gpu-handoff]` | Full candidate stack testing |
| `./test_candidate.sh <service> --inject` | Inject candidate into live traffic |
| `./test_candidate.sh --init` | Copy live code to candidate directories |
| `./test_candidate.sh <service> --unit` | Run unit tests for a service |
| `./test_candidate.sh --validate` | Integration validation checks |
| `./test_candidate.sh --promote` | File-level promotion (without containerized validation) |
| `./test_candidate.sh status` | Candidate service status |
| `./test_candidate.sh logs` | Candidate service logs |
| `./test_candidate.sh diff` | Code differences between live and candidate |

**Relationship with `promote_candidate.sh`**: `test_candidate.sh --promote` handles file-level promotion only. `promote_candidate.sh` adds containerized validation (ruff/mypy/pytest in Docker) and is the formal promotion path.

## Candidate/Live SDLC Flow

```
1. Developer edits code in candidates/<service>/
2. ./test_candidate.sh <service> --unit    # Quick unit tests
3. ./test_candidate.sh --validate          # Integration checks
4. scripts/promote_candidate.sh <service> --validate --test  # Formal promotion
5. gaia.sh live                            # Restart live stack
```
