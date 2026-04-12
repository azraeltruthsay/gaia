# 📜 GAIA-PRIME Module Contract

## 🎭 Role
The **Voice** of GAIA. High-performance GPU inference for large models (8B+). Supports LoRA adapters and deep thought snapshots via KV cache.

## 🔌 API Interface
- **Endpoint:** `http://gaia-prime:7777`
- **Protocol:** OpenAI-compatible REST/HTTP
- **Contract Definition:** [contract.yaml](./contract.yaml)
- **Key Endpoints:** `/v1/chat/completions`, `/thought/compose`.

## ⚙️ Configuration
- **Source File:** [config.json](./config.json)
- **Key Parameters:**
    - `MODEL_CONFIGS`: Specialized configuration for prime-tier models (Gemma-4-26B).

## 🛠️ Integration
The primary provider of high-fidelity reasoning for `gaia-core`. Supports advanced thought composition.
