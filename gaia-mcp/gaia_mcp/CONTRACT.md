# 📜 GAIA-MCP Module Contract

## 🎭 Role
The **Hands** of GAIA. Responsible for executing sandboxed tools, interfacing with the filesystem, and managing dynamic **Memento Skills**.

## 🔌 API Interface
The primary interface is **JSON-RPC 2.0** over HTTP.

- **Endpoint:** `http://gaia-mcp:8765/jsonrpc`
- **Contract Definition:** [contract.yaml](./contract.yaml)
- **Primary Tools:** `read_file`, `write_file`, `run_shell`, `list_dir`.

## ⚙️ Configuration
Configuration is decentralized and loaded at runtime by `gaia-common`.

- **Source File:** [config.json](./config.json)
- **Key Parameters:**
    - `MCP_LITE_ENABLED`: Enable/disable the tool sidecar.
    - `MCP_LITE_ENDPOINT`: URL for the sidecar API.

## 🧠 Memento Skills (Phase 5g)
GAIA-MCP supports dynamic procedural evolution.

- **Skills Directory:** `./skills/`
- **Management Tools:** `skill_create`, `skill_update`, `skill_read_source`.
- **Workflow:** Skills are hot-reloaded into memory using `importlib.reload`.

## 🛠️ Integration
To integrate with GAIA-MCP from another service:
1. Import `Config` from `gaia_common.config`.
2. Use `config.endpoints["mcp"]` to resolve the service URL.
3. Dispatch JSON-RPC requests to the `/jsonrpc` endpoint.
