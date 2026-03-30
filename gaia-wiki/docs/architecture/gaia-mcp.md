# gaia-mcp — The Hands

Sandboxed tool execution server implementing the Model Context Protocol (MCP).

## Responsibilities

- Execute tool calls from gaia-core in a sandboxed environment
- Enforce approval requirements for dangerous operations
- Provide file system access within the sandbox boundary
- Serve knowledge base queries (blueprint lookups, file reads)

## Security Model

gaia-mcp runs with hardened security:

- `no-new-privileges: true` — prevents privilege escalation
- `cap_drop: ALL` — drops all Linux capabilities
- `cap_add: CHOWN, SETGID, SETUID` — minimum needed for file operations
- Sandbox root at `/sandbox` — isolated from other volumes
- Approval TTL of 900 seconds for sensitive operations

## Endpoints

| Path | Method | Purpose |
|------|--------|---------|
| `/health` | GET | Container health check |
| `/jsonrpc` | POST | MCP JSON-RPC tool execution |

## Configuration

| Env Var | Default | Purpose |
|---------|---------|---------|
| `MCP_APPROVAL_REQUIRED` | `true` | Require approval for tool calls |
| `MCP_APPROVAL_TTL` | `900` | Approval cache duration (seconds) |
| `SANDBOX_ROOT` | `/sandbox` | Tool execution sandbox path |
