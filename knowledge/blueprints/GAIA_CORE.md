# GAIA Service Blueprint: `gaia-core` (The Brain)

## Role and Overview

`gaia-core` is the cognitive engine of the GAIA system. It processes user input through a multi-step reasoning pipeline: intent detection, knowledge enhancement, LLM inference, tool routing, self-reflection, and response assembly. In v0.3, gaia-core runs **CPU-only** and delegates all GPU inference to `gaia-prime` via HTTP.

## Container Configuration

**Base Image**: `python:3.11-slim` (CPU-only, no CUDA)

**Port**: 6415 (live), 6416 (candidate)

**Health Check**: `curl -f http://localhost:6415/health` (30s interval, 3 retries)

**Startup**: `uvicorn gaia_core.main:app --host 0.0.0.0 --port 6415`

**Dependencies**: Waits for `gaia-prime` (healthy) and `gaia-mcp` (healthy) before starting.

### Key Environment Variables

| Variable | Value | Purpose |
|----------|-------|---------|
| `GAIA_BACKEND` | `gpu_prime` | Selects remote vLLM inference backend |
| `GAIA_FORCE_CPU` | `1` | Prevents local GPU usage |
| `N_GPU_LAYERS` | `0` | No GPU layers for llama.cpp fallback |
| `PRIME_ENDPOINT` | `http://gaia-prime:7777` | Remote vLLM server address |
| `PRIME_MODEL` | `/models/Claude` | Model path on gaia-prime |
| `GROQ_API_KEY` | from `.env` | Groq API fallback (free tier) |
| `MCP_ENDPOINT` | `http://gaia-mcp:8765/jsonrpc` | Tool execution endpoint |
| `STUDY_ENDPOINT` | `http://gaia-study:8766` | Knowledge/embedding service |
| `GAIA_AUTOLOAD_MODELS` | `0` | Lazy model loading on first use |

### Volume Mounts

- `./gaia-core:/app:rw` — Source code (editable in dev)
- `./gaia-common:/gaia-common:ro` — Shared library
- `./knowledge:/knowledge:ro` — Knowledge base
- `./gaia-models:/models:ro` — Model files (GGUF for lite backend)
- `gaia-shared:/shared:rw` — Inter-service state

## Source Structure

```
gaia_core/
├── main.py                        # FastAPI entry point, AIManagerShim, lifespan
├── config.py                      # Config singleton, loads gaia_constants.json
├── gaia_constants.json            # Master config (models, features, task instructions)
├── cognition/                     # Core reasoning pipeline
│   ├── agent_core.py              # Main Reason-Act-Reflect loop
│   ├── cognition_packet.py        # Local packet extensions (v0.3)
│   ├── cognitive_dispatcher.py    # Dispatch tool/execution results
│   ├── knowledge_enhancer.py      # Inject knowledge context
│   ├── self_reflection.py         # Response quality review
│   ├── self_review_worker.py      # Async quality assurance
│   ├── tool_selector.py           # MCP tool selection (confidence scoring)
│   ├── loop_detector.py           # Repetitive pattern detection
│   ├── loop_recovery.py           # Recovery strategies (reset, hint, switch model)
│   ├── external_voice.py          # External model integration
│   ├── adapter_trigger_system.py  # LoRA adapter triggers
│   ├── thought_seed.py            # Seed thoughts for reasoning
│   ├── topic_manager.py           # Topic modeling
│   ├── telemetric_senses.py       # System state telemetry
│   └── nlu/                       # Natural Language Understanding
│       ├── intent_detection.py
│       └── intent_service.py
├── models/                        # Inference backend implementations
│   ├── model_pool.py              # Singleton accessor
│   ├── _model_pool_impl.py        # ModelPool class (lazy imports, multi-backend)
│   ├── model_manager.py           # Lifecycle management
│   ├── vllm_remote_model.py       # HTTP client to gaia-prime (GPU offload)
│   ├── vllm_model.py              # In-process vLLM (legacy, disabled in v0.3)
│   ├── groq_model.py              # Groq API fallback
│   ├── oracle_model.py            # OpenAI GPT API wrapper
│   ├── gemini_model.py            # Google Gemini API wrapper
│   ├── hf_model.py                # HuggingFace transformers
│   ├── mcp_proxy_model.py         # MCP tool integration
│   ├── dev_model.py               # Development/debug model
│   ├── vector_store.py            # Vector DB read access
│   └── tts.py                     # Text-to-speech
├── memory/                        # Session and long-term memory
│   ├── session_manager.py         # Thread-safe JSON-backed sessions
│   ├── memory_manager.py          # Memory orchestration
│   ├── semantic_codex.py          # In-memory symbol index (YAML/JSON/MD)
│   ├── codex_writer.py            # Persist codex entries
│   ├── knowledge_integrity.py     # Fact validation
│   ├── priority_manager.py        # Memory retention prioritization
│   └── conversation/              # Conversation-specific memory
│       ├── summarizer.py          # Compress long conversations
│       ├── keywords.py            # Keyword extraction
│       └── archiver.py            # Archive old sessions
├── behavior/                      # Persona management
│   ├── persona_manager.py         # Load/manage personas
│   ├── persona_adapter.py         # Adapt to context
│   ├── persona_switcher.py        # Dynamic selection per request
│   └── persona_writer.py          # Persist persona state
├── ethics/                        # Safety and governance
│   ├── core_identity_guardian.py   # Identity integrity checks
│   ├── ethical_sentinel.py        # Ethical decision-making
│   └── consent_protocol.py        # Consent and approval tracking
├── pipeline/                      # Cognitive pipeline orchestration
│   ├── pipeline.py                # Main pipeline function
│   ├── manager.py                 # Routing and context assembly
│   ├── primitives.py              # Built-in functions (shell, read, write, vector_query)
│   ├── llm_wrappers.py            # Backend interface wrappers
│   ├── bootstrap.py               # Full initialization
│   └── minimal_bootstrap.py       # Minimal startup
├── utils/                         # Utilities
│   ├── prompt_builder.py          # Token-budget-aware prompt assembly
│   ├── stream_observer.py         # Real-time output validation/interruption
│   ├── output_router.py           # Route responses to destinations
│   ├── resource_monitor.py        # GPU monitoring (pynvml, for status only)
│   ├── mcp_client.py              # JSON-RPC client to gaia-mcp
│   ├── packet_builder.py          # CognitionPacket construction
│   └── world_state.py             # System state snapshots
└── integrations/
    └── discord_connector.py       # Discord bot integration (legacy)
```

## FastAPI Endpoints

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/` | GET | Root info, lists available endpoints |
| `/health` | GET | Health check for container orchestration |
| `/status` | GET | System status (model availability, persona) |
| `/process_packet` | POST | Main inference — accepts CognitionPacket, returns completed packet |

## Cognitive Pipeline

The core reasoning loop in `AgentCore.run_turn()`:

1. **Packet Reception** — FastAPI `/process_packet` endpoint
2. **Session Context** — Load conversation history and persona
3. **Intent Detection** — Determine user intent (`nlu/intent_detection.py`)
4. **Primitive Routing** — Check for built-in functions (shell, read, vector_query)
5. **Knowledge Enhancement** — Inject relevant context (`knowledge_enhancer.py`)
6. **Prompt Assembly** — Build LLM prompt within token budget (`prompt_builder.py`)
7. **Reasoning** — Call primary LLM backend (gpu_prime -> groq -> lite fallback)
8. **Stream Observation** — Real-time output validation (`stream_observer.py`)
9. **Tool Selection** — Route to MCP tools if needed (`tool_selector.py`)
10. **Self-Reflection** — Review and refine response (`self_reflection.py`)
11. **Output Routing** — Deliver to web, Discord, CLI, or API

## Inference Backends

| Backend | Config Key | Class | Connection |
|---------|-----------|-------|------------|
| Remote vLLM | `gpu_prime` | `VLLMRemoteModel` | HTTP to gaia-prime:7777 |
| Groq API | `groq_fallback` | `GroqAPIModel` | HTTPS to api.groq.com |
| Local GGUF | `lite` | llama-cpp-python | In-process CPU |
| OpenAI | `oracle_openai` | `GPTAPIModel` | HTTPS to api.openai.com |
| Gemini | `oracle_gemini` | `GeminiAPIModel` | HTTPS to Google API |

**VLLMRemoteModel** (`vllm_remote_model.py`):
- HTTP client to `PRIME_ENDPOINT` (default: `http://gaia-prime:7777`)
- OpenAI-compatible API: `/v1/chat/completions`
- Streaming via Server-Sent Events (SSE)
- LoRA adapter selection via `set_active_adapter()`
- 120s request timeout, connection pooling via `requests.Session()`

**GroqAPIModel** (`groq_model.py`):
- Available models: llama-3.3-70b, mixtral-8x7b, gemma2-9b
- Free tier: 6K-20K tokens/min rate limits
- Context windows: 8K-128K tokens

## Configuration System

**`config.py`**: Singleton `Config` dataclass. Loads from `gaia_constants.json`. Key attributes:
- `MODEL_CONFIGS`: Dict of backend configurations
- `llm_backend`: Default backend selector (overridable via `GAIA_BACKEND`)
- Token budgets: `full` (8192), `medium` (4096), `minimal` (2048)
- Feature toggles, paths, model directories

**`gaia_constants.json`** (480+ lines): Master runtime configuration including:
- 8 model configurations (gpu_prime, lite, observer, oracle_gemini, oracle_openai, groq_fallback, prime)
- Task instruction templates (6 types)
- Observer settings (check_frequency, thresholds)
- Loop detection config (tool_threshold: 3, output_threshold: 0.95)
- Tool routing config (confidence_threshold: 0.7)
- LoRA adapter config (3 tiers: global/user/session, max 4 adapters)
- Fragmentation settings (continuation_threshold: 0.85, max 5 fragments)
- Integration placeholders (Discord webhook/token loaded from env, not hardcoded)

## Memory Architecture

Three-tier memory hierarchy:

1. **Short-term**: Session history (`session_manager.py`) — JSON-backed, 20 messages before summarization
2. **Mid-term**: Semantic codex (`semantic_codex.py`) — In-memory symbol index from YAML/JSON/MD files with hot-reload
3. **Long-term**: Vector store — Read-only access via `VectorClient`; `gaia-study` is sole writer

## Dependencies

**Runtime**: fastapi, uvicorn, pydantic, httpx, requests, regex, discord.py, llama-cpp-python, groq
**Shared**: gaia-common (protocols, utils, config)
**Dev**: pytest, pytest-asyncio, ruff, mypy
