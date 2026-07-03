# gaia-core — The Brain

Cognitive loop and reasoning engine. Core runs its own embedded GAIA Engine instance (managed mode, port 8092 inside the container) with the **Gemma4-E4B-GAIA-Core-v1** model. In AWAKE state, Core runs on GPU (NF4, ~8.8GB VRAM). In FOCUSING or SLEEP states, Core demotes to CPU (GGUF) or is unloaded entirely.

## Routing (Sovereign Duality)

Core is the first of two tiers — the old Nano triage tier is deprecated (gaia-nano is now a passthrough to Core's embedded engine):

1. **Core** (Gemma4-E4B) handles all triage plus operational tasks: intent detection, tool selection, vision, audio, chat
2. **Prime** (Qwen3-VL-8B) handles heavyweight tasks: complex reasoning, code generation, planning

Core triages and handles or escalates; Prime handles the hard stuff.

## Responsibilities

- Receive CognitionPackets from gaia-web
- Manage session history and conversation context
- Build prompts with persona traits, prime.md checkpoints, and session summaries
- Run embedded GAIA Engine for operational inference (intent, tool selection, medium tasks)
- Escalate to gaia-prime for heavyweight inference when needed
- Fall back to cloud (Groq only — OpenAI is retired) when local models are unavailable
- Run the Observer for post-execution side-effect verification
- Coordinate tool calls via gaia-mcp
- Manage the sleep/wake cycle state machine

## Key Components

| Component | Path | Role |
|-----------|------|------|
| AgentCore | `gaia_core/cognition/agent_core.py` | Main cognitive loop — processes turns, yields token events |
| ModelPool | `gaia_core/models/model_pool.py` | Model registry — routes `prime`/`lite` roles to backends |
| VLLMRemoteModel | `gaia_core/models/vllm_remote_model.py` | HTTP client for gaia-prime inference with retry + cloud fallback |
| SessionManager | `gaia_core/memory/session_manager.py` | Thread-safe session persistence to `/shared/sessions.json` |
| PrimeCheckpointManager | `gaia_core/cognition/prime_checkpoint.py` | Writes `prime.md` cognitive state on sleep/shutdown |
| LiteJournal | `gaia_core/cognition/lite_journal.py` | Running introspective journal (`Lite.md`) on heartbeat ticks |
| SleepCycleLoop | `gaia_core/cognition/sleep_cycle_loop.py` | Sleep/wake state machine with idle monitoring |
| Observer | `gaia_core/cognition/observer.py` | Post-execution verification of side effects |

## Embedded Engine

Core has an embedded GAIA Engine running in managed mode on port 8092 (inside the container). The engine manager spawns a worker subprocess that owns the CUDA context:

- **AWAKE**: Core model loaded on GPU (NF4, ~8.8GB VRAM), handles operational inference
- **FOCUSING**: Core may be demoted to CPU (GGUF fallback) to free VRAM for Prime
- **SLEEP/DEEP_SLEEP**: Core model unloaded, GPU memory fully freed

The managed mode architecture (see `gaia-engine`) provides zero-GPU standby — the manager process has no CUDA context until a model is loaded.

## Endpoints

| Path | Method | Purpose |
|------|--------|---------|
| `/health` | GET | Container health check |
| `/process_packet` | POST | Main entry — process a CognitionPacket |
| `/status` | GET | Cognitive system status |
| `/cognition/checkpoint` | POST | Write prime.md + Lite.md checkpoints |
| `/gpu/status` | GET | GPU state (active/sleeping) |
| `/gpu/release` | POST | Put gaia-prime to sleep |
| `/gpu/reclaim` | POST | Wake gaia-prime |
| `/sleep/status` | GET | Sleep cycle state machine status |
| `/sleep/wake` | POST | Send wake signal |

## Configuration

| Env Var | Default | Purpose |
|---------|---------|---------|
| `PRIME_ENDPOINT` | `http://gaia-prime:7777` | Prime inference server |
| `MCP_ENDPOINT` | `http://gaia-mcp:8765/jsonrpc` | Tool execution server |
| `STUDY_ENDPOINT` | `http://gaia-study:8766` | Background processing |
| `GAIA_BACKEND` | `auto` | Inference backend mode |
| `GAIA_AUTOLOAD_MODELS` | `1` | Load models on startup vs lazy (orchestrator loads) |
| `GAIA_ALLOW_PRIME_LOAD` | `1` | Allow prime model loading |
| `GROQ_API_KEY` | (empty) | Cloud fallback API key |
