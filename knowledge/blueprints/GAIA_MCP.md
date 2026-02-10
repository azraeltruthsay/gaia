# GAIA Service Blueprint: `gaia-mcp` (The Hands)

## Role and Overview

`gaia-mcp` is the Multi-tool Control Plane service. It provides a secure, sandboxed environment for `gaia-core` to execute external tools. All tool execution flows through gaia-mcp, which enforces approval workflows for sensitive operations and runs with hardened Linux security settings.

## Container Configuration

**Base Image**: `python:3.11-slim` (runs as non-root user `gaia`)

**Port**: 8765 (live), 8767 (candidate)

**Health Check**: `curl -f http://localhost:8765/health` (30s interval, 30s start_period)

### Security Hardening

```yaml
security_opt:
  - no-new-privileges:true
cap_drop:
  - ALL
cap_add:
  - CHOWN
  - SETGID
  - SETUID
```

### Key Environment Variables

| Variable | Value | Purpose |
|----------|-------|---------|
| `MCP_APPROVAL_REQUIRED` | `true` | Require approval for sensitive tools |
| `MCP_APPROVAL_TTL` | `900` | Approval timeout (15 minutes) |
| `SANDBOX_ROOT` | `/sandbox` | Isolated sandbox directory |
| `GAIA_MCP_BYPASS` | `false` | Bypass approval (security risk) |
| `LOG_LEVEL` | `INFO` | Logging level |

### Volume Mounts

- `./gaia-mcp:/app:rw` — Source code
- `./gaia-common:/gaia-common:ro` — Shared library
- `./knowledge:/knowledge:rw` — Knowledge base (for tool reference)
- `./gaia-models:/models:ro` — Model files for embeddings
- `gaia-sandbox:/sandbox:rw` — Isolated sandbox workspace

## Source Structure

```
gaia-mcp/
├── Dockerfile           # python:3.11-slim, non-root user
├── requirements.txt     # FastAPI, uvicorn, httpx
├── pyproject.toml       # Project metadata
├── gaia_mcp/
│   ├── __init__.py
│   ├── main.py          # FastAPI entry, approval store init, sensitive tools list
│   ├── server.py        # HTTP/JSON-RPC server
│   ├── tools.py         # Tool dispatcher and implementations
│   └── approval.py      # ApprovalStore with challenge-response workflow
└── tests/
    ├── conftest.py
    └── test_*.py
```

## Tool Registry

Tools are dispatched via `tools.py` using the central registry from `gaia_common.utils.tools_registry`:

| Tool | Purpose | Sensitive |
|------|---------|-----------|
| `run_shell` | Sandboxed shell execution | No (uses `safe_execution.run_shell_safe`) |
| `read_file` | Read file contents | No |
| `list_dir`, `list_files`, `list_tree` | Directory operations | No |
| `find_files`, `find_relevant_documents` | Search | No |
| `write_file` | Write to file | Yes (not yet implemented) |
| `ai_write` | LLM-assisted writing | Yes |
| `world_state` | System state snapshot | No |
| `memory_status`, `memory_query` | Knowledge base access | No |
| `memory_rebuild_index` | Rebuild vector index | Yes |
| `fragment_write`, `fragment_read`, `fragment_assemble` | Response fragmentation | No |
| `fragment_list_pending`, `fragment_clear` | Fragment management | No |
| `embed_documents`, `query_knowledge`, `add_document` | Vector indexing | No |

### Sensitive Tools (require approval)

`{ai_write, write_file, run_shell, memory_rebuild_index}`

## Approval Workflow

**`approval.py`** — `ApprovalStore` class:

1. Tool request arrives for a sensitive operation
2. `create_pending()` generates a 5-character alphabetic challenge
3. Human reviews the pending action via API
4. Human reverses the challenge string to approve
5. Approval expires after `MCP_APPROVAL_TTL` (default 900s / 15 min)

Methods: `create_pending()`, `approve_action()`, `get_pending()`, `list_pending_actions()`

## Communication Protocol

- **Inbound**: JSON-RPC requests from `gaia-core` via `MCP_ENDPOINT` (default: `http://gaia-mcp:8765/jsonrpc`)
- **Outbound**: Tool execution results returned as JSON-RPC responses
- `gaia-core` uses `utils/mcp_client.py` as the client

## Interaction with Other Services

- **`gaia-core`** (caller): Receives tool execution requests via JSON-RPC, returns results
- **`gaia-common`** (library): Uses tools registry, safe execution utilities, CognitionPacket protocol
- **External systems**: Tools can interact with file systems, APIs, and databases from within the sandbox
