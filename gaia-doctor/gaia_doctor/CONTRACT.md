# 📜 GAIA-DOCTOR Module Contract

## 🎭 Role
The **Immune System** of GAIA. Persistent High Availability (HA) watchdog for monitoring service health and system dissonance.

## 🔌 API Interface
- **Endpoint:** `http://gaia-doctor:6419`
- **Protocol:** REST/HTTP
- **Contract Definition:** [contract.yaml](./contract.yaml)
- **Key Endpoints:** `/status`, `/alarms`, `/cognitive/run`.

## ⚙️ Configuration
- **Source File:** Environment Variables
- **Key Parameters:**
    - `POLL_INTERVAL`: Frequency of health checks.
    - `FAILURE_THRESHOLD`: Error count before triggering alarm.

## 🛠️ Integration
Monitors all service health. Reports alarms and system dissonance to `gaia-web` and `gaia-monkey`.
