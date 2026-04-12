# 📜 GAIA-ORCHESTRATOR Module Contract

## 🎭 Role
The **Coordinator** of GAIA. Authority for GPU lifecycle, container management, and transition state machines.

## 🔌 API Interface
- **Endpoint:** `http://gaia-orchestrator:6410`
- **Protocol:** REST/HTTP
- **Contract Definition:** [contract.yaml](./contract.yaml)
- **Key Endpoints:** `/gpu/acquire`, `/lifecycle/transition`, `/containers/swap`.

## ⚙️ Configuration
- **Source File:** [config.json](./config.json)
- **Key Parameters:**
    - `MODEL_REGISTRY`: List of available models and their resource requirements.
    - `WARM_POOL`: Configuration for pre-seeded inference containers.

## 🛠️ Integration
Services must acquire GPU leases via `/gpu/acquire` before intensive inference or training operations.
