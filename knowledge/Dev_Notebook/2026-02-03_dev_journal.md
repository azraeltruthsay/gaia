# Dev Journal - 2026-02-03

## Gateway Pattern Implementation: Resolving the ModuleNotFoundError

### The Problem (Resolved)

The `gaia-mcp-candidate` service was failing with:
```
ModuleNotFoundError: No module named 'gaia_core'
```

**Root Cause Analysis:**
- `gaia-mcp` was importing directly from `gaia_core` for study mode and LoRA adapter operations
- The `gaia-mcp` Dockerfile only installed `gaia-common` and `gaia-mcp`, not `gaia-core`
- This violated SOA (Service-Oriented Architecture) boundaries - services shouldn't cross-import

Locations of the violations:
- `gaia-mcp/gaia_mcp/server.py` lines 410, 522, 551
- `gaia-mcp/gaia_mcp/tools.py` - study mode and adapter functions

### The Solution: Full Gateway Pattern

Instead of adding `gaia-core` as a dependency (which would couple the services), we implemented a proper gateway pattern where `gaia-mcp` calls `gaia-study` via HTTP API for all study/adapter operations.

#### Changes Made

1. **New Service Client (`gaia-common/gaia_common/utils/service_client.py`)**
   - HTTP client for inter-service communication
   - Async get/post/delete methods using httpx
   - Pre-configured clients for gaia-study, gaia-core, gaia-mcp
   ```python
   def get_study_client() -> ServiceClient:
       return ServiceClient("gaia-study", default_port=8766, endpoint_env_var="STUDY_ENDPOINT")
   ```

2. **Study Mode API Endpoints (`gaia-study/gaia_study/server.py`)**
   - `POST /study/start` - Start a training session
   - `GET /study/status` - Get current study mode status
   - `POST /study/cancel` - Cancel in-progress training
   - `GET /adapters` - List available LoRA adapters
   - `POST /adapters/load` - Load adapter for generation
   - `POST /adapters/unload` - Unload adapter
   - `GET /adapters/{name}` - Get adapter info
   - `DELETE /adapters/{name}` - Delete adapter

3. **Refactored gaia-mcp (`gaia-mcp/gaia_mcp/server.py` and `tools.py`)**
   - Removed all `from gaia_core...` imports
   - Added `from gaia_common.utils.service_client import get_study_client`
   - All study/adapter tools now use async HTTP calls to gaia-study

4. **Docker Configuration**
   - Updated `candidates/gaia-mcp/Dockerfile` to use `candidates/` paths
   - Added `STUDY_ENDPOINT` environment variable to `docker-compose.candidate.yml`

5. **Dependencies**
   - Added `httpx>=0.25.0` to `gaia-common/pyproject.toml`

#### Architecture After Fix

```
gaia-mcp (The Hands)
    │
    └──HTTP──► gaia-study (The Subconscious)
                    │
                    └──► StudyModeManager
                    └──► LoRA Adapters
```

### Candidate Testing Environment - Verified Working

| Service | Container | Port | Status |
|---------|-----------|------|--------|
| gaia-mcp-candidate | Up | 8767 | healthy |
| gaia-study-candidate | Up | 8768 | healthy |
| gaia-core-candidate | Up | 6416 | healthy |

**Gateway Flow Verified:**
```bash
# study_status via gateway
curl -X POST http://localhost:8767/jsonrpc -d '{"jsonrpc":"2.0","method":"study_status","params":{},"id":1}'
# Returns: {"jsonrpc":"2.0","result":{"state":"idle","progress":0.0,...},"id":1}

# adapter_list via gateway
curl -X POST http://localhost:8767/jsonrpc -d '{"jsonrpc":"2.0","method":"adapter_list","params":{},"id":2}'
# Returns: {"jsonrpc":"2.0","result":{"ok":true,"adapters":[],"count":0},"id":2}
```

---

## Next Major Hurdle: Bidirectional RAG

### Vision: Two-Way Memory Street

Currently GAIA's RAG (Retrieval-Augmented Generation) system is **read-only**:
- GAIA can query the vector store to retrieve relevant documents
- GAIA cannot write new memories based on conversations

**The Goal:** Make the RAG system bidirectional. When a user discusses something important (like "Rupert Roads" or any domain-specific knowledge), GAIA should be able to:

1. **Recognize** that new information is being shared
2. **Extract** the key facts, entities, and relationships
3. **Write** them to persistent storage for future retrieval
4. **Index** them in the vector store for semantic search

### Proposed Architecture

```
User: "Rupert Roads is the main character in my novel..."
                │
                ▼
        ┌───────────────────┐
        │  gaia-core        │
        │  (The Brain)      │
        │  - Recognizes     │
        │    new knowledge  │
        │  - Extracts facts │
        └─────────┬─────────┘
                  │ HTTP API
                  ▼
        ┌───────────────────┐
        │  gaia-study       │
        │  (Subconscious)   │
        │  - Writes memory  │
        │  - Updates index  │
        │  - SOLE WRITER    │
        └───────────────────┘
```

### Key Considerations

1. **Memory Types**
   - Episodic: Conversation events, user interactions
   - Semantic: Facts, entities, relationships
   - Procedural: Learned preferences, patterns

2. **Write Triggers**
   - Explicit: User says "Remember that..."
   - Implicit: GAIA detects important information worth preserving
   - Periodic: Session summaries, learning consolidation

3. **Conflict Resolution**
   - What if new info contradicts existing knowledge?
   - Temporal versioning? Confidence scores?

4. **Privacy & Consent**
   - User control over what gets remembered
   - Ability to forget on request

### Implementation Steps (Future Work)

1. Add `/memory/write` endpoint to gaia-study
2. Create memory extraction pipeline in gaia-core
3. Implement knowledge graph for entity relationships
4. Add memory consolidation background task
5. Build user-facing memory management UI

---

## Session Notes

- Promoted gateway pattern changes from candidates/ to live services
- Discord bot integration enabled for live testing
- SOA boundaries now properly enforced between services
