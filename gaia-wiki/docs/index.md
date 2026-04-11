# GAIA Wiki

Internal developer documentation for the GAIA project — a sovereign AI system with cognitive loops, a Consciousness Matrix, self-healing immunity, and autonomous training.

## Quick Links

| Section | What's Here |
|---------|------------|
| [Architecture](architecture/overview.md) | Service topology, data flow, and container layout |
| [Systems](systems/blueprint-system.md) | Deep dives into cross-cutting systems (Consciousness Matrix, blueprints, cognition packets) |
| [Operations](operations/deployment.md) | Deployment, promotion pipeline, GPU management |
| [Decisions](decisions/gateway-principle.md) | Architectural decision records — the "why" layer |
| [Dev](dev/getting-started.md) | Getting started, adding services, code quality |

## Service Map (13 Total)

| Service | Cognitive Role | Hardware |
|---------|----------------|----------|
| **gaia-prime** | The Voice — Thinker/Prime inference | GAIA Engine (GPU/CPU) |
| **gaia-core** | The Brain — Cognition loop, embedded Core | Python + GPU (:8092) |
| **gaia-nano** | The Reflex — Sub-second triage classifier | GAIA Engine (GPU/CPU) |
| **gaia-web** | The Face — Dashboard, API, Discord | Python |
| **gaia-mcp** | The Hands — Sandboxed tool execution | Python |
| **gaia-study** | The Subconscious — QLoRA, Vector indexing | CUDA 12.4 (GPU) |
| **gaia-audio** | The Ears & Mouth — STT (Qwen3-ASR), TTS | Python + GPU |
| **gaia-orchestrator** | The Coordinator — Consciousness Matrix, GPU | Python |
| **gaia-doctor** | The Immune System — HA watchdog, auto-heal | Python 3.12 |
| **gaia-monkey** | The Chaos Agent — Adversarial testing | Python + Node |
| **gaia-translate** | The Tongue — Multi-language translation | LibreTranslate |
| **gaia-wiki** | The Library — Internal MkDocs docs | MkDocs |
| **dozzle** | The X-Ray — Real-time Docker logs | Go |

## Network Layout

All services communicate on `gaia-net` (Docker bridge, `172.28.0.0/16`).

| Service | Host Port | Role |
|---------|-----------|------|
| gaia-orchestrator | 6410 | GPU lifecycle, state matrix |
| gaia-prime | 7777 | Thinker inference (22 tok/s) |
| gaia-nano | 8090 | Reflex triage |
| gaia-core | 6415 | Brain / Cognitive loop |
| gaia-web | 6414 | UI / Discord Gateway |
| gaia-mcp | 8765 | Tool execution |
| gaia-study | 8766 | Training / Embedding |
| gaia-audio | 8080 | Sensory STT/TTS |
| gaia-doctor | 6419 | HA Watchdog |
| gaia-monkey | 6420 | Chaos drills |
| gaia-translate | 5100 | Multi-language translation |
| dozzle | 9999 | Docker log viewer |
