# 📜 GAIA-NANO Module Contract

## 🎭 Role
The **Reflex** of GAIA. Ultra-fast triage classifier for intent detection, transcript refinement, and activation monitoring.

## 🔌 API Interface
- **Endpoint:** `http://gaia-nano:8080`
- **Protocol:** OpenAI-compatible REST/HTTP
- **Contract Definition:** [contract.yaml](./contract.yaml)

## ⚙️ Configuration
- **Source File:** Environment Variables
- **Key Parameters:**
    - `NANO_DEVICE`: GPU or CPU selection.

## 🛠️ Integration
Called by `gaia-core` for rapid triage and by `gaia-audio` for cleaning raw speech-to-text outputs.
