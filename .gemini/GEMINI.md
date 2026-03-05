# GAIA Project — Gemini CLI Instructions

> **Last updated**: 2026-03-04 | **Era**: Sovereign Autonomy | **Services**: 11

## Project Overview

GAIA is a sovereign AI agent built as a Service-Oriented Architecture (SOA). It runs locally on a single workstation with GPU, hosting 11 containerized services that collaborate through HTTP/JSON-RPC. GAIA has its own cognitive pipeline, memory architecture, self-healing immune system, and autonomous sleep cycle.

## Specialist Skills

Activate specialist agents via Gemini CLI:

| Skill | Command | Status | Domain |
|-------|---------|--------|--------|
| CodeMind | `Activate codemind` | Full | Structural code review, contract compliance |
| Sentinel | `Activate sentinel` | Full | Security review, 7 trust boundaries |
| Alignment | `Activate alignment` | Scaffolded | Service contract alignment |
| Blueprint | `Activate blueprint` | Scaffolded | Blueprint validation |
| Study | `Activate study` | Scaffolded | Training suitability review |
| UX Designer | `Activate ux-designer` | Scaffolded | Creative design |
| Service Scaffold | `Activate service-scaffold` | Scaffolded | New service generation |

Skill definitions: `.gemini/skills/<name>/SKILL.md`

---

## Service Inventory

| Service | Role | Port | Health |
|---------|------|------|--------|
| `gaia-core` | Cognitive loop, LLM routing, reasoning | 6415 | `/health` |
| `gaia-web` | Dashboard, API gateway, Discord bridge | 6414 | `/health` |
| `gaia-prime` | vLLM inference server (GPU, OpenAI-compatible) | 7777 | `/health` |
| `gaia-mcp` | Sandboxed tool execution (JSON-RPC 2.0) | 8765 | `/health` |
| `gaia-study` | QLoRA training, vector indexing (sole writer) | 8766 | `/health` |
| `gaia-audio` | Whisper STT, Nano-Refiner, TTS | 8080 | `/health` |
| `gaia-orchestrator` | GPU lifecycle, HA overlay, handoff | 6410 | `/health` |
| `gaia-doctor` | Persistent HA watchdog (stdlib only) | 6419 | `/status`, `/irritations` |
| `gaia-wiki` | MkDocs developer documentation | 8080* | `/` |
| `dozzle` | Real-time Docker log viewer | 9999 | built-in |

Candidate (HA) services mirror production with +1 ports. Defined in `docker-compose.candidate.yml`.

---

## Model Tiers

| Tier | Model | Backend | Role |
|------|-------|---------|------|
| **Nano** | Qwen2.5-0.5B GGUF | llama_cpp (CPU) | Triage, transcript cleanup |
| **Lite** | Qwen3-8B-abliterated-Q4_K_M GGUF | llama_cpp (CPU) | Intent detection, fast answers |
| **Prime** | Qwen3-8B-abliterated-AWQ | vLLM (GPU) | Complex reasoning, code |
| **Oracle** | gpt-4o-mini | OpenAI API | Cloud fallback |
| **Groq** | llama-3.3-70b-versatile | Groq API | Fast external fallback |

**Cascade**: Nano → Lite → Prime (complexity-based escalation).

---

## Cognitive Pipeline (20 stages)

`gaia-core/gaia_core/cognition/agent_core.py` — `AgentCore.run_turn()`:

1. **Circuit Breaker**: `/shared/HEALING_REQUIRED.lock` check.
2. **Entity Validation**: Fuzzy noun correction for stable cross-references.
3. **Loop Detection**: Multi-turn similarity check & recovery resets.
4. **Semantic Probe**: Pre-intent vector lookup across all KBs to drive persona selection.
5. **Persona & KB Selection**: Probe-driven (contextual) with keyword fallback.
6. **Model Selection**: Cascade logic (Nano triage → Lite → Prime).
7. **CognitionPacket Creation**: GCP v0.3 initialization with session lineage.
8. **Knowledge Ingestion**: Detects save commands/long info dumps; offers to index.
9. **Intent Detection**: NLU classification of system tasks (Lite model).
10. **Goal Detection**: Identifies overarching user goals persisting across turns.
11. **Tool Routing (GCP)**: Early selector for MCP tools based on intent.
12. **Slim Prompt Path**: High-speed bypass for low-complexity/tool-listing tasks.
13. **Initial Planning**: Proactive Codex loading + first-pass reasoning.
14. **Cognitive Self-Audit**: Quality/safety review of the initial plan.
15. **Reflection & Refinement**: Iterative confidence-gated refinement (reflector model).
16. **Pre-Generation Safety**: EthicalSentinel (Identity Guardian) fail-closed check.
17. **Observer Selection**: Picks idle model to monitor generation stream.
18. **External Voice**: Streaming generation with interruption handling.
    - *Think-tag Recovery*: Retries with direct-answer instruction if output is only `<think>` blocks.
    - *Lite Fallback*: Auto-escalates to Lite (CPU) if Prime (GPU) fails.
19. **Post-Stream Review**: Observer validation (detects confabulated file paths).
20. **Output Routing & Persistence**: Spinal routing to destination + Oracle fact learning.

---

## Key Subsystems

### Security & Safety
- **Sovereign Shield**: `py_compile` gate on all `.py` writes — forces compilation check before saving to prevent syntax-driven lobotomization.
- **Blast Shield**: Deterministic pre-flight safety in MCP; blocks `rm -rf`, `sudo`, and sensitive path writes.
- **Epistemic Gate**: Confidence-gated check that triggers "honest ignorance" if RAG fails for domain-specific queries.
- **Identity Guardian**: Hard-coded constraints in `AgentCore` that block generation if persona traits are violated.

### Cognition & Memory
- **Semantic Probe**: Uses embeddings to find the "domain center" of a query before the LLM even sees it.
- **Oracle Persistence**: Responses from `oracle` models (GPT-4o) are automatically saved as local facts to reduce cloud dependency over time.
- **Fragmentation/Rehydration**: MCP tools for chunking responses that exceed context windows, with seamless reassembly.
- **Samvega**: Captures "irritants" (errors/corrections) to drive future self-correction.

---

## MCP Tools (~70+)

Server: `gaia-mcp:8765` (JSON-RPC 2.0)

Categories: File I/O, Shell, Memory, Knowledge Bases, Fragments, Study/LoRA, Web, Kanka.io, NotebookLM, Audio, Diagnostics, Promotion.

Sensitive tools require human approval (5-char challenge-response).

---

## Testing

**Always test in Docker containers** (host Python 3.14 lacks dependencies):

```bash
docker compose exec -T gaia-core python -m pytest <path> -v --tb=short
docker compose exec -T gaia-core python -m pytest /gaia-common/tests/<file> -v --tb=short
docker compose exec -T gaia-web python -m pytest <path> -v --tb=short
```

---

## Promotion Process

Candidate → Pre-flight → Grammar (ruff/mypy/pytest) → 16-test cognitive smoke → Promote (dependency order) → Post-promotion health → Dev journal → Git commit.

**Master script**: `./scripts/promote_pipeline.sh`

**Vital Organs**: `main.py`, `agent_core.py`, `mcp_client.py`, `tools.py`, `immune_system.py`

---

## Key Paths

| Path | Purpose |
|------|---------|
| `gaia-core/gaia_core/cognition/` | Cognitive pipeline |
| `gaia-core/gaia_core/models/` | Model pool, backends |
| `gaia-core/gaia_core/memory/` | Session, codex, archiver |
| `gaia-common/gaia_common/` | Shared protocols, utilities |
| `gaia-mcp/gaia_mcp/` | MCP server, tools, approval |
| `candidates/` | Candidate service code |
| `knowledge/` | Blueprints, curricula, seeds |
| `scripts/` | Promotion, monitoring, utilities |

---

## Container Architecture

- `gaia-common` volume-mounted (`:ro`) into all containers — changes on disk immediately
- **Restart** (`docker restart <svc>`): Source file changes
- **Rebuild** (`docker compose build <svc>`): Dockerfile/dependency changes
- Always restart after code changes (Python bytecache)

## Configuration

1. `gaia_constants.json` → env vars → `Config` singleton
2. Token budgets: Full=8192, Medium=4096, Minimal=2048
3. Docker network: `gaia-net` (172.28.0.0/16)

## Inter-Service Communication

- `gaia-web → gaia-core`: HTTP POST `/chat` (+ candidate fallback)
- `gaia-core → gaia-prime`: OpenAI API at `:7777/v1/`
- `gaia-core → gaia-mcp`: JSON-RPC 2.0 at `:8765/jsonrpc`
- `gaia-core → gaia-study`: HTTP for training/indexing
- `gaia-orchestrator → all`: Health polling, GPU lifecycle
- `gaia-doctor → all`: Independent health monitoring
