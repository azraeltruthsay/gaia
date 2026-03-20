# ⚙️ System Constants & Config Dashboard
---
Type: #system-reference
Status: #live
Last Updated: 2026-03-19
---

## 🌐 Service Endpoints
| Service | Endpoint | Port |
| :--- | :--- | :--- |
| **Core (Brain)** | `http://gaia-core:6415` | 6415 |
| **Nano (Reflex)** | `http://gaia-nano:8080` | 8090→8080 |
| **Prime (Thinker)** | `http://gaia-prime:7777` | 7777 |
| **Web/Discord** | `http://gaia-web:6414` | 6414 |
| **MCP (Tools)** | `http://gaia-mcp:8765/jsonrpc` | 8765 |
| **Study (Learning)** | `http://gaia-study:8766` | 8766 |
| **Audio (Senses)** | `http://gaia-audio:8080` | 8080 |
| **Orchestrator** | `http://gaia-orchestrator:6410` | 6410 |
| **Doctor (Immune)** | `http://gaia-doctor:6419` | 6419 |
| **Monkey (Chaos)** | `http://gaia-monkey:6420` | 6420 |
| **Wiki (Docs)** | `http://gaia-wiki:8080` | 8080 (internal) |
| **Dozzle (Logs)** | `http://dozzle:8080` | 9999 |

## 🧠 Model Pool
- **Thinker/Prime:** `Huihui-Qwen3-8B-GAIA-Prime-adaptive` (vLLM, GPU)
  - *Context:* 16,384 tokens
  - *GPU Memory:* 85% utilization
  - *LoRA:* Enabled (max 4 adapters, rank 64)
- **Core/Operator:** `Qwen3-8B-abliterated-Q4_K_M.gguf` (embedded llama-server in gaia-core, CPU)
  - *Context:* 8,192 tokens
  - *Endpoint:* `http://localhost:8092`
- **Nano/Reflex:** `Qwen3.5-0.8B-Abliterated-Q8_0.gguf` (llama-server on gaia-nano, GPU primary)
  - *Context:* 2,048 tokens
  - *Endpoint:* `http://gaia-nano:8080`
- **Oracle:** `gpt-4o-mini` (OpenAI API)
- **Groq Fallback:** `llama-3.3-70b-versatile` (Groq API)
- **Embedding:** `all-MiniLM-L6-v2` (sentence-transformers, gaia-study)

## 📂 Knowledge & Paths
| Resource | Path |
| :--- | :--- |
| **Knowledge Base** | `/knowledge` |
| **Model Storage** | `/models` |
| **Warm Pool** | `/mnt/gaia_warm_pool` (tmpfs) |
| **Shared State** | `/shared` (gaia-shared volume) |
| **Logs** | `/logs` |
| **Identity File** | `[[system_reference/core_identity.json|core_identity.json]]` |
| **Cheat Sheet** | `[[system_reference/cheat_sheet.json|cheat_sheet.json]]` |
| **Constants** | `gaia-common/gaia_common/constants/gaia_constants.json` |

## 🛡️ Security & MCP
- **MCP Lite:** `ENABLED`
- **Security Scan:** `ENABLED` (prompt injection, PII redaction, secrets, vulnerability detection)
- **Allowed Shell Commands:** `ls`, `cat`, `echo`, `pwd`, `python`, `find`, `grep`, `du`, `df`...
- **Write Tools:** `DISABLED` (Human approval required for file writes)

## 🎭 Persona Defaults
- **Name:** `prime`
- **Identity:** `GAIA - General Artisanal Intelligence Architecture`
- **Temperature:** `0.7`

---
🔙 [[00_Index.md|Master Index]]
