# Dev Journal - 2026-02-02

## Summary of Current State

After a long and arduous debugging session, we have successfully resolved a number of critical issues that were preventing the application from starting correctly. The system is now in a stable, running state.

### Key Accomplishments:
- **`cheatsheet.json` Loading Fixed:** The original issue, a failure to load `cheatsheet.json`, has been resolved. The `gaia_core.config` module now correctly loads the cheat sheet, and the `CognitionPacket` is populated with this data.
- **Docker Build & Dependency Issues Resolved:** A complex chain of Docker build and dependency issues involving `gaia-core`, `gaia-mcp`, and `gaia-common` has been fixed. This involved:
    - Adding `setup.py` files to make `gaia-core` and `gaia-web` proper Python packages.
    - Correcting Dockerfiles to use editable installs (`pip install -e .`) for all `gaia-*` services.
    - Resolving pathing and build context issues.
    - Adding a volume mount for `gaia-core` to the `gaia-mcp` service in `docker-compose.yml` to ensure the code is available at runtime.
- **Cleanup Steps Added:** The `new_gaia_test.sh` and `new_gaia_start.sh` scripts now include a comprehensive Docker cleanup step to ensure a clean environment for each run.

The application now builds and runs successfully, with all services healthy.

---

## Candidate Container Infrastructure (IMPLEMENTED)

The "-candidate" designation for containers has been fully implemented.

### New Files Created:
- `docker-compose.candidate.yml` - Defines `-candidate` versions of all services
- `candidates/README.md` - Documents the candidate workflow
- `test_candidate.sh` - CLI helper for candidate management

### The "-candidate" Workflow:
1. **Create a "-candidate" Container:** Copy active service to `candidates/gaia-<service>/`
2. **Make Changes:** All new code changes are applied to the candidate container first
3. **Isolated Testing:** Candidate runs on separate ports (6416, 6417, 8767, 8768)
4. **Promote to Active:** Use `./test_candidate.sh <service> --promote` to copy to active

### Port Mapping:
| Service | Active Port | Candidate Port |
|---------|-------------|----------------|
| gaia-core | 6415 | 6416 |
| gaia-web | 6414 | 6417 |
| gaia-mcp | 8765 | 8767 |
| gaia-study | 8766 | 8768 |

### Usage Commands:
```bash
./test_candidate.sh core --start    # Start CPU-only candidate
./test_candidate.sh core --gpu      # Start with GPU
./test_candidate.sh core --logs     # View logs
./test_candidate.sh core --diff     # Compare to active
./test_candidate.sh core --promote  # Promote to active
```

---

## Bicameral Mind Architecture (IMPLEMENTED)

A major new feature has been implemented: the **Bicameral Mind Architecture**. This creates a dual-process cognitive system where CPU Lite and GPU Prime operate as logically segmented components of a unified mind.

### Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                      User Request                                │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    CPU Lite (Generator)                          │
│                    120s timeout window                           │
└─────────────────────────────────────────────────────────────────┘
                              │
              ┌───────────────┴───────────────┐
              │                               │
         Success                         Timeout
              │                               │
              ▼                               ▼
┌──────────────────────┐       ┌──────────────────────────────────┐
│   GPU Prime          │       │   GPU Prime (Generator)           │
│   (Observer)         │       │   + partial CPU response context  │
│   Reviews & Critiques│       └──────────────────────────────────┘
└──────────────────────┘                      │
              │                               ▼
              ▼                 ┌──────────────────────────────────┐
┌──────────────────────┐       │   CPU Lite (Observer)             │
│   Generator reads    │       │   Reviews & Raises Concerns       │
│   critique, revises  │       └──────────────────────────────────┘
│   if needed          │                      │
└──────────────────────┘                      ▼
              │                 ┌──────────────────────────────────┐
              ▼                 │   GPU Prime reads critique,       │
┌──────────────────────┐       │   Approves/Revises                │
│   Stream to User     │       └──────────────────────────────────┘
│   (after approval)   │                      │
└──────────────────────┘                      ▼
                               ┌──────────────────────────────────┐
                               │   Stream to User (after approval) │
                               └──────────────────────────────────┘
```

### Key Features:
1. **CPU/GPU Isolation**: CPU Lite (llama.cpp) handles simple tasks, GPU Prime (vLLM) handles complex ones
2. **120s Timeout Handoff**: If CPU generation takes too long, GPU takes over with partial context
3. **Observer Validation**: Non-generating model critiques responses before user sees them
4. **3-Round Approval Loop**: Generator revises based on critique, max 3 rounds before forced approval
5. **Identity-Focused Critique**: Observer primarily validates:
   - No hallucination
   - Epistemic honesty (admits uncertainty when appropriate)
   - No false confidence
   - Core identity preservation

### New Module Structure:
```
candidates/gaia-core/gaia_core/bicameral/
├── __init__.py           # Module exports
├── schemas.py            # ObserverCritique, GeneratorResponse, BicameralResult
├── generator.py          # 120s timeout, CPU→GPU handoff
├── observer.py           # Identity-focused critique generation
├── approval_loop.py      # 3-round negotiation logic
├── mind.py               # Main BicameralMind orchestrator
└── test_bicameral.py     # Unit tests (ALL PASSING)
```

### Environment Variables:
| Variable | Description | Default |
|----------|-------------|---------|
| `GAIA_BICAMERAL_ENABLED` | Enable bicameral processing | `0` |
| `GAIA_BICAMERAL_CPU_TIMEOUT` | CPU timeout before GPU handoff | `120` |
| `GAIA_BICAMERAL_MAX_ROUNDS` | Max approval rounds | `3` |

### Integration Status:
- ✅ Bicameral module created and tested
- ✅ Integrated into `agent_core.py` (opt-in via env var)
- ✅ Unit tests passing (3/3)
- ⏳ Awaiting real-model testing in candidate container

---

## Codebase Reorganization

### Archive Created:
The old monolithic `gaia-assistant/` directory has been moved to `archive/gaia-assistant-monolith/`:
- Clearly marked as historical reference
- Prevents accidental edits
- Removes confusion about what's active

### Current Active Structure:
```
/gaia/GAIA_Project/
├── archive/
│   ├── README.md
│   └── gaia-assistant-monolith/   # Historical reference only
├── candidates/
│   ├── README.md
│   └── gaia-core/                 # Bicameral testing
├── docker-compose.yml             # Active services
├── docker-compose.override.yml    # Dev overrides
├── docker-compose.candidate.yml   # Candidate services
├── gaia-core/                     # Active - The Brain
├── gaia-web/                      # Active - The Face
├── gaia-study/                    # Active - The Subconscious
├── gaia-mcp/                      # Active - The Hands
├── gaia-common/                   # Active - Shared library
├── gaia-models/                   # Active - Model files
└── knowledge/                     # Active - Knowledge base
```

---

## Next Steps

1. **Test Bicameral with Real Models**: Start the candidate container and test with actual CPU/GPU models
2. **Performance Tuning**: Adjust timeout and round limits based on real-world usage
3. **Observer Prompt Refinement**: Fine-tune the identity validation prompts based on testing
4. **Promotion**: Once validated, promote bicameral changes to active `gaia-core`

---

## Testing Commands

```bash
# Run bicameral unit tests
cd /gaia/GAIA_Project/candidates/gaia-core
python -m gaia_core.bicameral.test_bicameral

# Start candidate container (CPU-only)
./test_candidate.sh core --start

# Start with GPU (after unloading active GPU)
./test_candidate.sh core --gpu

# View candidate logs
./test_candidate.sh core --logs

# Promote to active
./test_candidate.sh core --promote
```
