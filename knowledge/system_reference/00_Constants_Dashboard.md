# ⚙️ System Constants & Config Dashboard
---
Type: #system-reference
Status: #live
Last Updated: 2026-03-01
---

## 🌐 Service Endpoints
| Service | Endpoint |
| :--- | :--- |
| **Core (Brain)** | `http://gaia-core:6415` |
| **Web/Discord** | `http://gaia-web:6414` |
| **Prime (LLM)** | `http://gaia-prime:7777` |
| **MCP (Tools)** | `http://gaia-mcp:8765/jsonrpc` |
| **Study (Learning)** | `http://gaia-study:8766` |
| **Audio (Senses)** | `http://gaia-audio:8080` |
| **Orchestrator** | `http://gaia-orchestrator:6410` |

## 🧠 Model Pool
- **GPU Prime:** `Qwen3-8B-abliterated-AWQ` (vLLM)
  - *Context:* 16,384 tokens
  - *GPU Memory:* 85% utilization
- **Lite:** `Qwen3-8B-abliterated-Q4_K_M.gguf` (Llama.cpp)
  - *Context:* 32,000 tokens
- **Fallback:** `llama-3.3-70b-versatile` (Groq API)

## 📂 Knowledge & Paths
| Resource | Path |
| :--- | :--- |
| **Knowledge Base** | `/knowledge` |
| **Model Storage** | `/models` |
| **Logs** | `/logs` |
| **Identity File** | `[[system_reference/core_identity.json|core_identity.json]]` |
| **Cheat Sheet** | `[[system_reference/cheat_sheet.json|cheat_sheet.json]]` |

## 🛡️ Security & MCP
- **MCP Lite:** `ENABLED`
- **Allowed Shell Commands:** `ls`, `cat`, `echo`, `pwd`, `python`, `find`, `grep`, `du`, `df`...
- **Write Tools:** `DISABLED` (Human approval required for file writes)

## 🎭 Persona Defaults
- **Name:** `prime`
- **Identity:** `GAIA - General Artisanal Intelligence Architecture`
- **Temperature:** `0.7`

---
🔙 [[00_Index.md|Master Index]]
