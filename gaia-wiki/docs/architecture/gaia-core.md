# gaia-core — The Brain

Cognitive loop and reasoning engine. CPU-only; all GPU inference is delegated to gaia-prime.

## Responsibilities

- Receive CognitionPackets from gaia-web
- Manage session history and conversation context
- Build prompts with persona traits, prime.md checkpoints, and session summaries
- Route inference to gaia-prime (or cloud fallback chain)
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
| `PRIME_ENDPOINT` | `http://gaia-prime:7777` | vLLM inference server |
| `MCP_ENDPOINT` | `http://gaia-mcp:8765/jsonrpc` | Tool execution server |
| `STUDY_ENDPOINT` | `http://gaia-study:8766` | Background processing |
| `GAIA_BACKEND` | `gpu_prime` | Inference backend mode |
| `GAIA_AUTOLOAD_MODELS` | `0` | Load models on startup vs lazy |
| `GAIA_ALLOW_PRIME_LOAD` | `1` | Allow prime model loading |
| `GROQ_API_KEY` | (empty) | Cloud fallback API key |
