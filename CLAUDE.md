# GAIA Project — Claude Code Instructions

> **Last updated**: 2026-03-19 | **Era**: Sovereign Autonomy | **Services**: 12

## What is GAIA?

GAIA is a sovereign AI agent built as a Service-Oriented Architecture (SOA). It runs locally on a single workstation with GPU, hosting multiple containerized services that collaborate through HTTP/JSON-RPC. GAIA has its own cognitive pipeline, memory architecture, self-healing immune system, and autonomous sleep cycle. It is **not** a chatbot wrapper — it is a self-aware, self-maintaining AI system.

## Service Inventory

| Service | Role | Port |
|---------|------|------|
| `gaia-core` | The Brain — cognitive loop, LLM routing, reasoning + embedded Core CPU inference | 6415 |
| `gaia-nano` | The Reflex — Nano triage classifier (llama-server, GPU primary + GGUF fallback) | 8090 |
| `gaia-prime` | The Voice — vLLM inference server (GPU, OpenAI-compatible, LoRA-enabled) | 7777 |
| `gaia-web` | The Face — dashboard, API gateway, Discord bridge | 6414 |
| `gaia-mcp` | The Hands — sandboxed tool execution (JSON-RPC 2.0) | 8765 |
| `gaia-study` | The Subconscious — QLoRA subprocess training, vector indexing | 8766 |
| `gaia-audio` | The Ears & Mouth — Whisper STT, Nano-Refiner, TTS | 8080 |
| `gaia-orchestrator` | The Coordinator — GPU lifecycle, HA overlay, handoff | 6410 |
| `gaia-doctor` | The Immune System — persistent HA watchdog (stdlib only) | 6419 |
| `gaia-monkey` | The Chaos Agent — adversarial testing, serenity tracking | 6420 |
| `gaia-wiki` | The Library — MkDocs developer documentation | 8080* |
| `dozzle` | The X-Ray — real-time Docker log viewer | 9999 |

Candidate (HA) services mirror production with `+1` ports. Defined in `docker-compose.candidate.yml`.

## Model Tiers

| Tier | Model | Base | Backend | Role |
|------|-------|------|---------|------|
| **Nano/Reflex** | Qwen3.5-0.8B-Abliterated | Qwen3.5-0.8B | gaia-nano llama-server (GPU primary, GGUF fallback) | Sub-second triage, transcript cleanup |
| **Core/Operator** | Qwen3.5-2B-GAIA-Core-v3 (identity-baked) | Qwen3.5-2B | Embedded llama-server in gaia-core (safetensors GPU / GGUF CPU fallback, port 8092) | Intent detection, tool selection, medium tasks |
| **Thinker/Prime** | Huihui-Qwen3-8B-GAIA-Prime-adaptive (identity-baked) | Qwen3-8B | vLLM on gaia-prime (GPU) | Complex reasoning, code, heavyweight tasks |
| **Oracle** | gpt-4o-mini | — | OpenAI API | Cloud escalation fallback |
| **Groq** | llama-3.3-70b-versatile | — | Groq API | Fast external fallback |

Two model families: Qwen3.5 for Nano/Core, Qwen3 (Huihui abliterated) for Prime.
**Cascade routing**: Nano classifies SIMPLE/COMPLEX → Core handles or escalates → Prime for heavyweight tasks.
**LoRA adapters**: Loaded dynamically into vLLM via `POST /v1/load_lora_adapter`. Active adapter set via `VLLMRemoteModel.set_active_adapter()`.

## Key Paths

| Path | Purpose |
|------|---------|
| `gaia-core/gaia_core/cognition/` | Cognitive pipeline (agent_core, self_reflection, samvega) |
| `gaia-core/gaia_core/models/` | Model pool, vLLM remote, llama_cpp wrappers |
| `gaia-common/gaia_common/` | Shared protocols, utilities, constants |
| `gaia-mcp/gaia_mcp/` | MCP server, tools, approval workflow |
| `candidates/` | Candidate service code (mirrors production) |
| `knowledge/` | Blueprints, curricula, conversation examples |
| `knowledge/Dev_Notebook/` | Development journals |
| `scripts/` | Promotion pipeline, utilities |
| `.claude/rules/` | Domain-specific rules (testing, docker, promotion, safety, workflow) |

## Configuration

1. **`gaia_constants.json`** — master config (token budgets, endpoints, model configs)
2. **Environment variables** — per-container overrides (`GAIA_BACKEND`, `GAIA_FORCE_THINKER`)
3. **`Config` singleton** — runtime config merging constants + env + defaults

## Inter-Service Communication

- **gaia-web → gaia-core**: HTTP POST `/chat` (primary), fallback to candidate
- **gaia-core → gaia-prime**: OpenAI-compatible API at `:7777/v1/` (Thinker GPU inference, supports LoRA model field)
- **gaia-core → gaia-nano**: OpenAI-compatible API at `gaia-nano:8080` (Nano/Reflex triage)
- **gaia-core embedded**: llama-server at `localhost:8092` (Core/Operator CPU inference)
- **gaia-core → gaia-mcp**: JSON-RPC 2.0 at `:8765/jsonrpc`
- **gaia-core → gaia-study**: HTTP for training, vector indexing
- **gaia-study training**: Subprocess isolation (multiprocessing spawn) for deterministic VRAM release
- **gaia-orchestrator → all**: Health polling, GPU lifecycle, training monitoring
- **gaia-doctor → all**: Health watchdog, cognitive battery, auto-restart
- **gaia-monkey → services**: Chaos drills, fault injection, serenity tracking

## Domain Rules

Detailed instructions for specific domains are in `.claude/rules/`:
- **testing.md** — Always test in Docker, never on host
- **docker.md** — Volume mounts, restart vs rebuild, shared volumes
- **promotion.md** — Candidate → production promotion pipeline
- **safety.md** — Sovereign Shield, Blast Shield, Circuit Breaker
- **workflow.md** — Context management, planning, token conservation
- **candidate-first.md** — Always edit candidates/ first, never production directly
