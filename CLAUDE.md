# GAIA Project — Claude Code Instructions

> **Last updated**: 2026-03-04 | **Era**: Sovereign Autonomy | **Services**: 11

## What is GAIA?

GAIA is a sovereign AI agent built as a Service-Oriented Architecture (SOA). It runs locally on a single workstation with GPU, hosting multiple containerized services that collaborate through HTTP/JSON-RPC. GAIA has its own cognitive pipeline, memory architecture, self-healing immune system, and autonomous sleep cycle. It is **not** a chatbot wrapper — it is a self-aware, self-maintaining AI system.

---

## Service Inventory

| Service | Role | Port | Entry Point | Health |
|---------|------|------|-------------|--------|
| `gaia-core` | The Brain — cognitive loop, LLM routing, reasoning | 6415 | `gaia_core.main:app` | `/health` |
| `gaia-web` | The Face — dashboard, API gateway, Discord bridge | 6414 | `gaia_web.main:app` | `/health` |
| `gaia-prime` | The Voice — vLLM inference server (GPU, OpenAI-compatible) | 7777 | `vllm.entrypoints.openai.api_server` | `/health` |
| `gaia-mcp` | The Hands — sandboxed tool execution (JSON-RPC 2.0) | 8765 | `gaia_mcp.main:app` | `/health` |
| `gaia-study` | The Subconscious — QLoRA training, vector indexing (sole writer) | 8766 | `gaia_study.main:app` | `/health` |
| `gaia-audio` | The Ears & Mouth — Whisper STT, Nano-Refiner, TTS | 8080 | `gaia_audio.main:app` | `/health` |
| `gaia-orchestrator` | The Coordinator — GPU lifecycle, HA overlay, handoff | 6410 | `gaia_orchestrator.main:app` | `/health` |
| `gaia-doctor` | The Immune System — persistent HA watchdog (stdlib only) | 6419 | `python doctor.py` | `/health` |
| `gaia-wiki` | The Library — MkDocs developer documentation | 8080* | `mkdocs serve` | `/` |
| `dozzle` | The X-Ray — real-time Docker log viewer | 9999 | upstream image | built-in |

*gaia-wiki is internal-only on `gaia-net`.

**Candidate (HA) services** mirror production with `+1` ports (e.g., `gaia-core-candidate:6416`). Defined in `docker-compose.candidate.yml`.

---

## Model Tiers

| Tier | Model | Backend | Role | Context |
|------|-------|---------|------|---------|
| **Nano** | Qwen2.5-0.5B GGUF | llama_cpp (CPU) | Triage classifier, transcript cleanup | 2K |
| **Lite/Operator** | Qwen3-8B-abliterated-Q4_K_M GGUF | llama_cpp (CPU) | Intent detection, tool selection, fast answers | 4K |
| **Prime/Thinker** | Qwen3-8B-abliterated-AWQ | vLLM (GPU) | Complex reasoning, code, long-form | 24K |
| **Oracle** | gpt-4o-mini | OpenAI API | Cloud escalation fallback | API |
| **Groq** | llama-3.3-70b-versatile | Groq API | Fast external fallback | 128K |

**Cascade routing**: Nano classifies SIMPLE/COMPLEX → Lite handles or escalates → Prime for heavyweight tasks.
**Env overrides**: `GAIA_FORCE_THINKER=1`, `GAIA_FORCE_OPERATOR=1`, `GAIA_BACKEND=gpu_prime`.

---

## Cognitive Pipeline (AgentCore)

The `AgentCore.run_turn()` method in `gaia-core/gaia_core/cognition/agent_core.py` processes each turn through these stages:

1. **Circuit Breaker** — Check for `/shared/HEALING_REQUIRED.lock`; abort if present
2. **Entity Validation** — Fuzzy-correct project nouns (Azrael, GAIA, CognitionPacket, etc.)
3. **Loop Detection** — Initialize loop detection; inject recovery context if resuming from reset
4. **Semantic Probe** — Vector lookup across all collections before intent detection
5. **Persona & KB Selection** — Probe-driven or keyword-fallback persona routing
6. **Model Selection & Cascade** — Nano triage → Lite → Prime escalation logic
7. **Packet Creation** — Build CognitionPacket (GCP v0.3) with all metadata
8. **Knowledge Ingestion** — Auto-detect save/update commands; RAG retrieval
9. **Intent Detection** — NLU classification (chat, code_review, recitation, etc.)
10. **Goal Detection** — Multi-turn goal coherence tracking
11. **Tool Routing** — MCP tool selection and execution if needed
12. **Slim Prompt Fast-Path** — Bypass full pipeline for simple intents
13. **Initial Planning** — LLM generates initial plan
14. **Cognitive Self-Audit** — Post-planning integrity check (optional)
15. **Reflection & Refinement** — Secondary model refines plan (iterative, confidence-gated)
16. **Pre-Generation Safety** — EthicalSentinel / CoreIdentityGuardian check (fail-closed)
17. **Observer Selection** — Pick idle model to monitor generation stream
18. **External Voice** — Stream final response with observer interruption support
19. **Thought Seed Parsing** — Extract knowledge gap markers from output
20. **Session Update** — Persist history, emit telemetry

---

## Key Subsystems

### Sovereign Shield
**File**: `gaia-mcp/gaia_mcp/tools.py`
- `py_compile` gate on all `.py` writes (ai_write, write_file, replace)
- Prevents GAIA from introducing syntax errors during self-repair
- Raises `ValueError("Sovereign Shield: ...")` on compilation failure

### Immune System 2.0
**File**: `gaia-common/gaia_common/utils/immune_system.py`
- **Passive**: Log scanning with weighted error triage (SyntaxError=4.0, ModuleNotFoundError=3.0)
- **Proactive MRI**: Module import checks, artifact existence, py_compile + ruff syntax scans
- **Adaptive polling**: 30-300s interval based on health score (sicker = more frequent)
- **[PROD]/[CAND] tagging**: Distinguishes production vs candidate issues
- **States**: STABLE (≤2), MINOR NOISE (≤8), IRRITATED (≤25), CRITICAL (>25)

### Cascade Routing
**File**: `gaia-core/gaia_core/cognition/agent_core.py`
- `_nano_triage()` — 0.5B model classifies SIMPLE vs COMPLEX (32 max tokens, temp 0.1)
- `_assess_complexity()` — Heuristic escalation (code/philosophy/system keywords)
- `_should_escalate_to_thinker()` — Lite → Prime escalation markers

### Spinal Routing (Output Routing)
**File**: `gaia-common/gaia_common/protocols/cognition_packet.py`
- `OutputDestination` enum: CLI, WEB, DISCORD, API, WEBHOOK, LOG, BROADCAST, AUDIO
- `OutputRouting` with primary + secondary destinations, suppress_echo, addressed_to_gaia
- Supports multi-destination output (e.g., Discord + Web simultaneously)

### Proprioception
**Files**: `telemetric_senses.py`, `temporal_state_manager.py`
- **Biological clock**: Sleep duration tracking, wake/sleep cycle summary
- **Atmospheric pressure**: CPU%, memory%, disk%, GPU stats via `GAIAStatus`
- **File change detection**: Watches .py/.md/.json across service dirs
- **Temporal state baking**: KV cache snapshots for Lite model (interview past states)

### Temperature Monitoring
**File**: `scripts/temp_monitor.py`
- CPU temp from `/sys/class/thermal/thermal_zone0/temp`
- GPU temp from `nvidia-smi`
- 30s polling, 10-minute rolling history → `logs/temp_history.json`

### Sleep Cycle
**File**: `gaia-core/gaia_core/cognition/sleep_task_scheduler.py`
- Priority-based autonomous maintenance during SLEEPING state
- P1: auto_as_built_update, conversation_curation
- P2: samvega_introspection
- P3: blueprint_validation, code_evolution, promotion_readiness, initiative_cycle
- P4: code_review, knowledge_research
- P5: wiki_doc_regen, adversarial_resilience_drill

### HA Mesh
**Files**: `service_client.py`, `health_watchdog.py`
- `ServiceClient`: Retry-with-backoff, automatic HA failover to candidate
- `HealthWatchdog`: 30s polling, 2-failure threshold, session sync (live → candidate)
- States: ACTIVE, DEGRADED, FAILOVER_ACTIVE, FAILED
- Maintenance mode: `/shared/ha_maintenance` flag disables failover

### Circuit Breaker
- `HEALING_REQUIRED.lock` in `/shared/` — created on fatal loop threshold
- AgentCore checks at turn start; refuses to process if lock exists
- Manual intervention required to clear

### Zero-Trust Identity
- Docker secrets at `/run/secrets/` for API keys
- No secrets in env vars or config files

### Golden Thread
- Auto-generates `AS_BUILT_LATEST.md` during sleep cycle
- Living documentation of current system state

---

## Memory Architecture

| Layer | Component | File | Purpose |
|-------|-----------|------|---------|
| **Active** | SessionManager | `memory/session_manager.py` | Per-session conversation history, auto-archive at 20 msgs |
| **Archive** | Summarizer + Archiver | `memory/conversation/` | LLM-powered summaries, keyword extraction, long-term storage |
| **Semantic** | SemanticCodex | `memory/semantic_codex.py` | In-memory compressed knowledge sidecar with hot-reload |
| **Vector** | VectorIndexer | `utils/vector_indexer.py` | MiniLM-L6-v2 embeddings, 512-token chunks, JSON-persisted |
| **Emotional** | Samvega | `cognition/samvega.py` | Error learning artifacts (user corrections, confidence mismatches) |
| **Generative** | Thought Seeds | `cognition/thought_seed.py` | Knowledge gaps, ideas for autonomous exploration |

---

## MCP Tool System

**Server**: `gaia-mcp:8765` (JSON-RPC 2.0)

### Tool Categories (~70+ tools)
- **File I/O**: read_file, write_file, ai_write, list_dir, list_tree, find_files
- **Shell**: run_shell (whitelisted commands)
- **Memory**: memory_query, memory_rebuild_index, index_document
- **Knowledge Bases**: find_relevant_documents, query_knowledge, add_document
- **Fragments**: fragment_write/read/assemble (long-response overflow handling)
- **Study/LoRA**: study_start, adapter_load/unload/list
- **Web**: web_search (DuckDuckGo), web_fetch
- **Kanka.io**: Campaign/entity CRUD for world-building
- **NotebookLM**: Notebook interaction, audio generation
- **Audio**: inbox management, listen start/stop
- **Diagnostics**: introspect_logs, world_state, describe_tool

### Approval Workflow
1. Sensitive tools (ai_write, write_file, run_shell, memory_rebuild_index) trigger approval
2. `ApprovalStore` generates 5-char challenge code
3. Human provides reversed code to approve
4. TTL: 900 seconds (configurable via `MCP_APPROVAL_TTL`)

### Blast Shield
Deterministic pre-flight safety (independent of LLM reasoning):
- **run_shell**: Blocks `rm -rf`, `sudo`, `mkfs`, `dd`, `> /dev/sd`
- **write_file**: Blocks `/etc`, `/boot`, `.ssh` paths

---

## Container Architecture

### Volume Mounts
- `gaia-common` is **volume-mounted** (`:ro`) into all containers — file changes are on disk immediately
- Production containers mount `./gaia-core:/app:rw`, etc.
- Candidate containers mount `./candidates/gaia-core:/app:rw`

### When to Restart vs Rebuild
- **Restart only** (`docker restart <service>`): Source file changes (.py), config changes
- **Rebuild required** (`docker compose build <service>`): Dockerfile changes, new dependencies
- **Always restart** after any code change — Python bytecache needs clearing

### Shared Volumes
- `gaia-shared` → `/shared/` — session state, packets, locks
- `gaia-sandbox` → `/sandbox/` — MCP tool execution isolation
- `/models/` — model weights (read-only for most services)
- `/knowledge/` — knowledge bases, blueprints, curricula (read-only)
- `/vector_store/` — semantic index (read-only except gaia-study)

---

## Testing

**CRITICAL**: Never run pytest on the host (Python 3.14 lacks project dependencies). Always use Docker:

```bash
# gaia-core tests
docker compose exec -T gaia-core python -m pytest <path> -v --tb=short

# gaia-common tests (via gaia-core container)
docker compose exec -T gaia-core python -m pytest /gaia-common/tests/<file> -v --tb=short

# gaia-web tests
docker compose exec -T gaia-web python -m pytest <path> -v --tb=short
```

---

## Promotion Process

**Flow**: Candidate-only development → Pre-flight checks → Grammar tests (ruff/mypy/pytest in Docker) → 16-test cognitive smoke battery → Promote (dependency order) → Post-promotion health → Session sanitization → Dev journal → Flatten SOA → Git commit/push

**Master script**: `./scripts/promote_pipeline.sh` (dry-run with `--dry-run`)

**Dependency order**: gaia-common → gaia-core → gaia-web → gaia-mcp → gaia-study → gaia-orchestrator

**Vital Organs** (require extra scrutiny): `main.py`, `agent_core.py`, `mcp_client.py`, `tools.py`, `immune_system.py`

---

## Key Paths

| Path | Purpose |
|------|---------|
| `gaia-core/gaia_core/cognition/` | Cognitive pipeline (agent_core, self_reflection, samvega, etc.) |
| `gaia-core/gaia_core/models/` | Model pool, vLLM, llama_cpp, API wrappers |
| `gaia-core/gaia_core/memory/` | Session manager, semantic codex, conversation archiver |
| `gaia-common/gaia_common/` | Shared protocols, utilities, constants |
| `gaia-mcp/gaia_mcp/` | MCP server, tools, approval workflow |
| `candidates/` | Candidate service code (mirrors production structure) |
| `knowledge/` | Blueprints, curricula, conversation examples, seeds |
| `knowledge/Dev_Notebook/` | Development journals |
| `knowledge/blueprints/` | Architectural blueprints per service |
| `scripts/` | Promotion pipeline, temp monitor, utilities |
| `logs/` | Service logs, immune status, temp history |
| `/shared/` (in containers) | Session state, packets, HA locks |

---

## Configuration Hierarchy

1. **`gaia_constants.json`** — master config (token budgets, endpoints, model configs, KB definitions)
2. **Environment variables** — per-container overrides (`GAIA_BACKEND`, `GAIA_FORCE_THINKER`, etc.)
3. **`Config` singleton** — runtime config object merging constants + env + defaults

### Key Config Values
- Token budgets: Full=8192, Medium=4096, Minimal=2048
- Reflection max tokens: 4096
- Prime context: 24576 tokens, 0.80 GPU utilization
- Embedding model: all-MiniLM-L6-v2
- Docker network: `gaia-net` (bridge, 172.28.0.0/16)

---

## Inter-Service Communication

- **gaia-web → gaia-core**: HTTP POST `/chat` (primary), fallback to candidate
- **gaia-core → gaia-prime**: OpenAI-compatible API at `:7777/v1/`
- **gaia-core → gaia-mcp**: JSON-RPC 2.0 at `:8765/jsonrpc`
- **gaia-core → gaia-study**: HTTP POST for training requests, vector indexing
- **gaia-orchestrator → all**: Health polling, GPU lifecycle, handoff orchestration
- **gaia-doctor → all**: Independent health monitoring, container restart automation

---

## Specialist Agent Mesh

### Claude Agents (`.claude/agents/`)
- **CodeMind** — Structural code review (contract compliance, dependency correctness)
- **Sentinel** — Security review (7 trust boundaries, injection vectors, secrets)
- **Alignment** — Service contract alignment (scaffolded)
- **Blueprint** — Blueprint validation (scaffolded)
- **Study** — Training suitability (scaffolded)
- **Service Scaffold** — Service generation (scaffolded)

### Gemini Skills (`.gemini/skills/`)
Mirror of Claude agents for Gemini CLI. Activate via `Activate codemind`, etc.

---

## Current State (March 2026)

**Architectural era**: Sovereign Autonomy — GAIA manages its own health, self-repairs with safety gates, and maintains autonomous sleep-cycle maintenance.

**Recent major additions** (Feb 28 – Mar 4):
- Immune System 2.0 with MRI diagnostics and adaptive polling
- Sovereign Shield (py_compile gate on all writes)
- Cascade Routing (Nano triage → Lite → Prime)
- Proprioception (biological clock, atmospheric pressure, file change sensing)
- Temperature monitoring daemon
- HA Mesh with health watchdog and session sync
- Localized timezone support (dual UTC/local display)
- Circuit breaker (HEALING_REQUIRED.lock)
- Zero-Trust Identity (Docker secrets)
- Golden Thread (auto AS_BUILT during sleep)
