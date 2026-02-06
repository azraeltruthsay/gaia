# GAIA Development Journal

## Date: 2026-02-04

### Subject: Candidate-First Development Workflow

**Summary:**

Established a proper workflow for making code changes to gaia-core that uses the candidate container for testing before promoting to live.

**Problem:**

Previously, code changes were being made directly to the live gaia-core and deployed immediately. This is risky because:
1. Bugs go directly to production (Discord, web console)
2. No easy rollback if something breaks
3. No testing phase before users see changes

**Infrastructure:**

Multiple GAIA services have live/candidate pairs:

| Service | Live Port | Candidate Port | Live Path | Candidate Path |
|---------|-----------|----------------|-----------|----------------|
| gaia-core | 6415 | 6416 | `/gaia/GAIA_Project/gaia-core` | `candidates/gaia-core` |
| gaia-mcp | 8765 | 8767 | `/gaia/GAIA_Project/gaia-mcp` | `candidates/gaia-mcp` |
| gaia-study | 8766 | 8768 | `/gaia/GAIA_Project/gaia-study` | `candidates/gaia-study` |
| gaia-web | 6414 | (none) | `/gaia/GAIA_Project/gaia-web` | `candidates/gaia-web` |
| gaia-common | (shared) | - | `/gaia/GAIA_Project/gaia-common` | `candidates/gaia-common` |

All containers share:
- `/gaia/GAIA_Project/knowledge` -> `/knowledge`
- `/gaia/GAIA_Project/gaia-models` -> `/models`

**Workflow:**

```
┌─────────────────────────────────────────────────────────────┐
│  1. EDIT IN CANDIDATES                                      │
│     Edit files in: /gaia/GAIA_Project/candidates/<service>/ │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  2. RESTART CANDIDATE                                       │
│     docker restart <service>-candidate                      │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  3. TEST VIA CANDIDATE                                      │
│     curl http://localhost:<candidate_port>/health           │
│     Or use promote script's --test flag                     │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  4. PROMOTE TO LIVE (if tests pass)                         │
│     ./scripts/promote_candidate.sh <service> --test         │
└─────────────────────────────────────────────────────────────┘
```

**Rollback:**

If live breaks after promotion:
```bash
# Restore from backup (promote script creates one automatically)
cp -r /gaia/GAIA_Project/<service>.bak/* /gaia/GAIA_Project/<service>/
docker restart <service>
```

**Testing the Candidate:**

```bash
# Health check (ports vary by service)
curl http://localhost:6416/health   # gaia-core-candidate
curl http://localhost:8767/health   # gaia-mcp-candidate
curl http://localhost:8768/health   # gaia-study-candidate

# Check logs
docker logs <service>-candidate --tail 50
```

**Scripts:**

The promotion script at `/gaia/GAIA_Project/scripts/promote_candidate.sh` supports all services:

```bash
# Usage
./scripts/promote_candidate.sh <service> [--test] [--no-restart] [--no-backup]

# Examples
./scripts/promote_candidate.sh gaia-core --test    # Test candidate health first
./scripts/promote_candidate.sh gaia-mcp            # Promote MCP sidecar
./scripts/promote_candidate.sh gaia-common         # Shared lib (no container)
```

---
