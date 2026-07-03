# 📜 GAIA-NANO Module Contract

## 🎭 Role
**DEPRECATED** (Sovereign Duality) — formerly the **Reflex** of GAIA, an ultra-fast triage classifier. The Reflex tier was removed; Core handles all triage. The container is now an `alpine/socat` passthrough forwarding `:8080` → `gaia-core:8092` (Core's embedded engine) to preserve the DNS name.

## 🔌 API Interface
- **Endpoint:** `http://gaia-nano:8080` (transparently proxied to Core's embedded GAIA Engine)
- **Protocol:** OpenAI-compatible REST/HTTP
- **Contract Definition:** [contract.yaml](./contract.yaml)

## ⚙️ Configuration
- **Source File:** `docker-compose.yml` (socat command only — no model, no env config)

## 🛠️ Integration
Legacy callers (`gaia-core` triage paths, `gaia-audio` transcript refinement) still resolve `gaia-nano:8080` and are served by Core's embedded engine. To restore a real Nano tier, replace the proxy with the original engine container.
