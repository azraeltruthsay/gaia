# 📜 GAIA-STUDY Module Contract

## 🎭 Role
The **Subconscious** of GAIA. Responsible for QLoRA training, vector indexing, and memory maintenance. Sole writer for the system's vector memory.

## 🔌 API Interface
- **Endpoint:** `http://gaia-study:8766`
- **Protocol:** REST/HTTP
- **Contract Definition:** [contract.yaml](./contract.yaml)
- **Key Endpoints:** `/study/start`, `/index/query`, `/adapters`.

## ⚙️ Configuration
- **Source File:** [config.json](./config.json)
- **Key Parameters:**
    - `KNOWLEDGE_BASES`: Definitions for specialized RAG sources.
    - `LORA_CONFIG`: Parameters for autonomous training sessions.

## 🛠️ Integration
Provides RAG context for `gaia-core` and manages the continuous learning and fine-tuning pipelines.
