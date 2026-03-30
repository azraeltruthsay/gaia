# GAIA Candidate Services - Hybrid Testing

This directory contains candidate versions of GAIA services for testing before promotion to live.

## Two Testing Modes

### Mode 1: Parallel Stack (Full Ecosystem Testing)

Test the complete candidate ecosystem in isolation. Candidates talk to candidates.

```bash
./test_candidate.sh all              # Start full candidate stack
./test_candidate.sh all --validate   # Run validation tests
./test_candidate.sh all --promote    # Promote all to live
```

Use this when:
- Testing changes that span multiple services
- Validating the entire system before a release
- Running integration tests

### Mode 2: Injection (Single Service Testing)

Inject a single candidate into the live flow. Live services call the candidate.

```bash
./test_candidate.sh mcp --inject     # Start candidate MCP on live network
# Then restart live gaia-core to point at candidate:
MCP_ENDPOINT=http://gaia-mcp-candidate:8765/jsonrpc docker compose up -d gaia-core
```

Use this when:
- Testing a small change to one service
- You don't want to bring down the live stack
- Quick iteration on a single component

## Quick Start

### 1. Initialize Candidates

```bash
./test_candidate.sh --init
```

### 2. Make Changes

Edit files in `candidates/gaia-<service>/`.

### 3. Test

**Parallel mode:**
```bash
./test_candidate.sh all
# Test at http://localhost:6417 (candidate web)
```

**Injection mode:**
```bash
./test_candidate.sh mcp --inject
# Follow the printed instructions to redirect live traffic
```

### 4. Promote to Live

```bash
./test_candidate.sh all --promote
docker compose up -d
```

## Port Mapping

| Service      | Live Port | Candidate Port |
|--------------|-----------|----------------|
| gaia-web     | 6414      | 6417           |
| gaia-core    | 6415      | 6416           |
| gaia-mcp     | 8765      | 8767           |
| gaia-study   | 8766      | 8768           |

## Commands Reference

```bash
./test_candidate.sh [service|all] [command]

Services: all (default), core, web, mcp, study

Commands:
  --init      Initialize all candidates from active code
  --start     Start candidate(s) in parallel mode (default)
  --gpu       Start with GPU enabled
  --inject    Inject single candidate into live flow
  --eject     Remove candidate from live flow, restore live
  --stop      Stop candidate container(s)
  --logs      View candidate logs
  --status    Show status of all candidate containers
  --unit      Run unit tests
  --diff      Show differences vs active
  --validate  Run full stack validation
  --promote   Promote candidate(s) to active
  --help      Show this help
```

## Injection Mode Details

When you inject a candidate:

1. The candidate container starts on the **live network** (`gaia-network`)
2. It's accessible to live services via hostname (e.g., `gaia-mcp-candidate`)
3. You manually redirect traffic by restarting the caller with an endpoint override

### Injection Example: Testing MCP Changes

```bash
# 1. Make changes to candidates/gaia-mcp/

# 2. Inject the candidate
./test_candidate.sh mcp --inject

# 3. Redirect live gaia-core to use candidate MCP
MCP_ENDPOINT=http://gaia-mcp-candidate:8765/jsonrpc docker compose up -d gaia-core

# 4. Test through live gaia-web (which calls live core -> candidate mcp)
curl http://localhost:6414/api/chat -d '{"message": "test"}'

# 5. When done, restore live flow
./test_candidate.sh mcp --eject
```

### Injection Example: Testing Core Changes

```bash
# 1. Make changes to candidates/gaia-core/

# 2. Inject the candidate
./test_candidate.sh core --inject

# 3. Redirect live gaia-web to use candidate core
CORE_ENDPOINT=http://gaia-core-candidate:6415 docker compose up -d gaia-web

# 4. Test through live web UI
curl http://localhost:6414/api/chat -d '{"message": "test"}'

# 5. Restore
./test_candidate.sh core --eject
```

## Network Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    gaia-network                              │
│                                                              │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐              │
│  │ gaia-web │───▶│gaia-core │───▶│ gaia-mcp │  (live)      │
│  │  :6414   │    │  :6415   │    │  :8765   │              │
│  └──────────┘    └──────────┘    └──────────┘              │
│                         │                                    │
│                         │ (injection mode)                   │
│                         ▼                                    │
│              ┌────────────────────┐                         │
│              │gaia-mcp-candidate  │  (candidate)            │
│              │      :8767         │                         │
│              └────────────────────┘                         │
└─────────────────────────────────────────────────────────────┘
```

In parallel mode, all candidates talk to each other on the same network:

```
┌─────────────────────────────────────────────────────────────┐
│                    gaia-network                              │
│                                                              │
│  Live services (running)          Candidates (running)      │
│  ┌──────────┐                     ┌────────────────────┐   │
│  │ gaia-web │                     │gaia-web-candidate  │   │
│  │  :6414   │                     │      :6417         │   │
│  └──────────┘                     └─────────┬──────────┘   │
│       │                                     │               │
│       ▼                                     ▼               │
│  ┌──────────┐                     ┌────────────────────┐   │
│  │gaia-core │                     │gaia-core-candidate │   │
│  │  :6415   │                     │      :6416         │   │
│  └──────────┘                     └─────────┬──────────┘   │
│       │                                     │               │
│       ▼                                     ▼               │
│  ┌──────────┐                     ┌────────────────────┐   │
│  │ gaia-mcp │                     │gaia-mcp-candidate  │   │
│  │  :8765   │                     │      :8767         │   │
│  └──────────┘                     └────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

## GPU Management

Only ONE GPU ecosystem should be loaded at a time.

**CPU-only testing (recommended for most changes):**
```bash
./test_candidate.sh all              # CPU mode by default
```

**GPU testing:**
```bash
# Stop live GPU services
docker compose stop gaia-core gaia-study

# Start candidates with GPU
./test_candidate.sh all --gpu

# After testing
./test_candidate.sh all --stop
docker compose up -d
```

## Volume Isolation

Candidates use separate volumes to avoid corrupting live data:
- `gaia-candidate-shared` - Shared state between candidate services
- `gaia-candidate-sandbox` - MCP sandbox for candidate

Knowledge base and models are mounted read-only from the same source as live.
