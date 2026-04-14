# GAIA Project — Claude Code Instructions

> **Last updated**: 2026-04-14 | **Era**: Sovereign Duality | **Services**: 12

## What is GAIA?

GAIA is a sovereign AI agent built as a Service-Oriented Architecture (SOA). It runs locally on a single workstation with GPU (RTX 5080, 16GB), hosting multiple containerized services that collaborate through HTTP/JSON-RPC. GAIA has its own cognitive pipeline, memory architecture, self-healing immune system, and autonomous sleep cycle. It is **not** a chatbot wrapper — it is a self-aware, self-maintaining AI system.

## Service Inventory

| Service | Role | Port |
|---------|------|------|
| `gaia-core` | The Brain — cognitive loop, LLM routing, reasoning + embedded Core GPU inference | 6415 |
| `gaia-prime` | The Voice — GAIA Engine inference (CPU/GGUF default, GPU when FOCUSING, LoRA-enabled) | 7777 |
| `gaia-web` | The Face — dashboard, API gateway, Discord bridge | 6414 |
| `gaia-mcp` | The Hands — sandboxed tool execution (JSON-RPC 2.0) | 8765 |
| `gaia-study` | The Subconscious — QLoRA subprocess training, vector indexing | 8766 |
| `gaia-audio` | The Ears & Mouth — STT, TTS | 8080 |
| `gaia-orchestrator` | The Coordinator — GPU lifecycle gearbox, consciousness matrix, handoff | 6410 |
| `gaia-doctor` | The Immune System — persistent HA watchdog (stdlib only) | 6419 |
| `gaia-monkey` | The Chaos Agent — adversarial testing, serenity tracking | 6420 |
| `gaia-wiki` | The Library — MkDocs developer documentation | 8080* |
| `gaia-translate` | The Tongue — multi-language translation (LibreTranslate) | 5000 |
| `dozzle` | The X-Ray — real-time Docker log viewer | 9999 |

Candidate (HA) services mirror production with `+1` ports. Defined in `docker-compose.candidate.yml`.

**Deprecated**: `gaia-nano` (E2B Reflex tier) — removed in Sovereign Duality. Core handles all triage.

## Sovereign Duality — Model Architecture

GAIA uses a **two-tier** Pure Gemma 4 architecture. All models are identity-baked.

| Tier | Model | Base | Backend | VRAM (GPU) | Role |
|------|-------|------|---------|------------|------|
| **Core/Operator** | Gemma4-E4B-GAIA-Core-v1 | google/gemma-4-E4B | GAIA Engine managed (GPU NF4 or CPU GGUF) | ~8.8 GB | Triage, intent, tools, vision, audio, chat |
| **Prime/Sovereign** | Gemma4-26B-A4B-Sovereign-v1 | google/gemma-4-26B-A4B | GAIA Engine managed (GPU Expert Buffered or CPU GGUF) | ~4.6 GB | Deep reasoning, architecture, code, planning |
| **Oracle** | gpt-4o-mini | — | OpenAI API | — | Cloud escalation fallback |
| **Groq** | llama-3.3-70b-versatile | — | Groq API | — | Fast external fallback |

**Routing**: Core handles all requests directly. Prime is loaded on GPU only when deep reasoning is needed (FOCUSING state). The orchestrator manages gear shifts via the consciousness matrix.

**LoRA adapters**: Loaded dynamically via GAIA Engine `POST /adapter/load`. Active adapter set via `POST /adapter/set`.

## The Gearbox (Lifecycle States)

GAIA's GPU lifecycle is a transmission system. States map to "gears":

| Gear | State | Core | Prime | GPU VRAM |
|------|-------|------|-------|----------|
| **P** | PARKED | CPU (GGUF) | Unloaded | ~0 GB |
| **1** | AWAKE | GPU (NF4) | CPU (GGUF) | ~8.8 GB |
| **1+** | LISTENING | GPU (NF4) + Audio | CPU (GGUF) | ~8.8 GB |
| **2** | FOCUSING | CPU (GGUF) | GPU (Buffered) | ~4.6 GB |
| **S** | SLEEP | CPU (GGUF) | Unloaded | ~0 GB |
| **0** | DEEP_SLEEP | Unloaded | Unloaded | ~0 GB |
| **T** | MEDITATION | Unloaded | Unloaded | Study owns GPU |

**Transition flow**: `OFF → PARKED → AWAKE ↔ FOCUSING → PARKED`

The **clutch** is the Neural Handoff protocol: capture prefix cache text before GPU unload, replay into CPU backend after load. Defined in `consciousness_matrix.py`.

## GAIA Inference Engine (Separate Repo)

Both tiers run the **GAIA Engine** — a standalone library in its own GitHub repo at `github.com/azraeltruthsay/gaia-engine` (Apache-2.0). Key capabilities: hidden state polygraph, KV cache thought snapshots, LoRA adapter management, GPU↔CPU migration, vision support, ROME/SAE companion modules, GPU lifecycle state machine.

The Engine Manager provides **dual-backend** inference:
- **Safetensors** (GPU): Native PyTorch + torch.compile + FlashAttention via GAIAEngine worker
- **GGUF** (CPU): llama-server (llama.cpp b8770) via subprocess

**Important**: Engine code lives in `gaia-engine/` (separate git repo), NOT `gaia-common/gaia_common/engine/`. The latter is a backward-compat **shim** that delegates to the `gaia_engine` package. When working on engine issues, use the `/engine` skill which knows the correct workflow (work in gaia-engine/, commit/push to the separate repo, update shim if API changes).

See `contracts/services/gaia-engine.yaml` for the full API contract.

## Key Paths

| Path | Purpose |
|------|---------|
| `gaia-engine/` | GAIA Inference Engine (**separate repo** — `github.com/azraeltruthsay/gaia-engine`) |
| `gaia-common/gaia_common/engine/` | Backward-compat shim for engine (delegates to `gaia_engine` package) |
| `gaia-common/gaia_common/lifecycle/states.py` | Gearbox state definitions, transitions, tier expectations |
| `contracts/` | Inter-service API contracts (YAML per service, connectivity matrix, schemas) |
| `gaia-core/gaia_core/cognition/` | Cognitive pipeline (agent_core, self_reflection, samvega) |
| `gaia-core/gaia_core/models/` | Model pool, remote model clients, inference wrappers |
| `gaia-common/gaia_common/` | Shared protocols, utilities, constants |
| `gaia-mcp/gaia_mcp/` | MCP server, tools, approval workflow |
| `candidates/` | Candidate service code (mirrors production) |
| `knowledge/` | Blueprints, curricula, conversation examples |
| `knowledge/Dev_Notebook/` | Development journals |
| `scripts/` | Promotion pipeline, utilities |
| `.claude/rules/` | Domain-specific rules (testing, docker, promotion, safety, workflow) |

## Configuration

1. **`gaia_constants.json`** — master config (token budgets, endpoints, model configs). Model type is `"managed"` (GAIA Engine).
2. **Environment variables** — per-container overrides (`GAIA_BACKEND`, `GAIA_ENGINE_TIER`)
3. **`Config` singleton** — runtime config merging constants + env + defaults

## Inter-Service Communication

See `contracts/CONNECTIVITY.md` for the full matrix. See `contracts/services/*.yaml` for per-service API contracts.

Summary:
- **gaia-web → gaia-core**: HTTP POST `/process_packet` (primary), fallback to candidate
- **gaia-core → gaia-prime**: GAIA Engine API at `:7777` (Prime CPU/GGUF default, GPU when FOCUSING)
- **gaia-core embedded**: GAIA Engine managed mode at `localhost:8092` (Core GPU inference)
- **gaia-core → gaia-mcp**: JSON-RPC 2.0 at `:8765/jsonrpc`
- **gaia-core → gaia-study**: HTTP for training, vector indexing
- **gaia-orchestrator → all**: Health polling, consciousness matrix, gear shifts, training monitoring
- **gaia-doctor → all**: Health watchdog, cognitive battery, auto-restart

## Domain Rules

Detailed instructions for specific domains are in `.claude/rules/`:
- **testing.md** — Always test in Docker, never on host
- **docker.md** — Volume mounts, restart vs rebuild, shared volumes
- **promotion.md** — Candidate → production promotion pipeline
- **safety.md** — Sovereign Shield, Blast Shield, Circuit Breaker
- **workflow.md** — Context management, planning, token conservation
- **candidate-first.md** — Always edit candidates/ first, never production directly

## Skills (`.claude/commands/`)

- **`/engine`** — Work on the GAIA Inference Engine (separate repo). Use this for ANY engine-related changes: inference bugs, KV cache, polygraph, LoRA, ROME, SAE, lifecycle state machine, managed mode. The skill knows to work in `gaia-engine/`, commit/push to the separate repo, and update the shim + contract if the API surface changes.
