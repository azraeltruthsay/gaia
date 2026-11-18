# MCP Threat Model

> What gaia-mcp can and cannot do, and how its safety mechanisms work.

## Sensitive Tools (Require Approval)

These tools trigger challenge-response approval before execution:
- `ai_write`, `write_file` — filesystem writes
- `run_shell` — shell command execution
- `memory_rebuild_index` — vector index operations
- `kanka_create_entity`, `kanka_update_entity` — external data writes
- `notebooklm_create_note` — external note creation
- `audio_listen_start` — system audio capture
- `promotion_create_request` — code deployment

## Approval Flow

```
Tool request
  → MCP checks SENSITIVE_TOOLS set
  → Creates pending action in ApprovalStore
  → Generates 5-char alphabetic challenge code
  → Sets 15-minute TTL (MCP_APPROVAL_TTL=900)
  → Waits for human response
  → Human reverses challenge string to confirm
  → Tool executes
```

- In-memory store with thread-safe locking
- Automatic expiry cleanup
- No persistence — approvals lost on container restart

### Bypass Risk
`MCP_BYPASS` environment variable disables ALL approvals. In production, this MUST be unset. Any code that references or sets this variable is a critical finding.

## Capability Constraints

### What MCP Tools CAN Do
- Read from `/knowledge`, `/gaia-common`, `/sandbox` (within size limits)
- Write to `/knowledge`, `/sandbox` (allowlisted paths only)
- Execute whitelisted shell commands only (shell=False, timeout=10s)
- Query vector index (read-only semantic search)
- Fetch URLs (httpx with validation)
- CRUD on Kanka entities and NotebookLM notes (with approval)
- Capture system audio (with approval)

### What MCP Tools CANNOT Do
- Write outside `/knowledge` and `/sandbox`
- Execute arbitrary shell commands (whitelist enforced)
- Access `/app` (service source code) directly
- Escalate privileges (no-new-privileges enforced at container level)
- Create new Linux users/groups
- Access host resources beyond mounted volumes
- Bypass approval TTL (15-minute hard limit, no extension API)
- Access other containers' filesystems (network isolation)

## Rate Limiting

- Kanka API: 25 req/min client-side (below hard cap of 30/min)
- TTL cache on Kanka responses (300s default)
- No general rate limiting on other tools (per-tool timeout only)

## JSON-RPC Interface

- Endpoint: `POST /jsonrpc` (JSON-RPC 2.0)
- Methods: `list_tools()`, `describe_tool(name)`, `execute_tool(name, params)`
- Parameter validation at dispatcher level (type coercion with defaults)
- Missing required fields raise ValueError (not silent defaults)
