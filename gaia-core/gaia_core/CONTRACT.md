# 📜 GAIA-CORE Module Contract

## 🎭 Role
The **Brain** of GAIA. Central cognitive loop, LLM routing, and reasoning engine. Orchestrates the "Reason-Act-Reflect" loop and manages embedded CPU inference.

## 🔌 API Interface
- **Endpoint:** `http://gaia-core:6415`
- **Protocol:** REST/HTTP
- **Contract Definition:** [contract.yaml](./contract.yaml)
- **Primary Endpoint:** `/process_packet` (accepts `CognitionPacket`, returns NDJSON stream).

## ⚙️ Configuration
- **Source File:** [config.json](./config.json)
- **Key Parameters:**
    - `llm_backend`: Current active thinking backend (e.g., prime, core).
    - `max_tokens_core`: Token limit for the central reasoning pass.
    - `TASK_INSTRUCTIONS`: The primary prompt registry for all cognitive modes.

## 🛠️ Integration
The central hub for all services. Dispatch a `CognitionPacket` to `/process_packet` to initiate reasoning.
