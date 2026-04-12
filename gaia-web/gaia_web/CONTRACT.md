# 📜 GAIA-WEB Module Contract

## 🎭 Role
The **Face** of GAIA. Acts as the primary dashboard, API gateway, and Discord bridge. Routes user input to `gaia-core` and proxies administrative requests to backend services.

## 🔌 API Interface
- **Endpoint:** `http://gaia-web:6414`
- **Protocol:** REST/HTTP
- **Contract Definition:** [contract.yaml](./contract.yaml)
- **Primary Endpoint:** `/process_user_input` (accepts user text, returns NDJSON stream).

## ⚙️ Configuration
- **Source File:** [config.json](./config.json)
- **Key Parameters:**
    - `INTEGRATIONS.discord`: Discord bot token and channel IDs.
    - `INTEGRATIONS.webhooks`: External notification endpoints.

## 🛠️ Integration
The primary entry point for users and external systems. Other backend services use `gaia-web` to report presence updates or send notifications via the Discord bridge.
