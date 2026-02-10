# GAIA Library Blueprint: `gaia-common`

## Role and Overview

`gaia-common` is a shared Python library consumed by all GAIA services. It provides common protocols, data structures, utilities, and configuration management. It is not a running service — it is installed as an editable package (`pip install -e`) or made available via `PYTHONPATH` in each service container.

## Installation Methods

- **Dockerfile**: `COPY gaia-common /gaia-common && pip install -e /gaia-common/`
- **docker-compose**: Volume mount `./gaia-common:/gaia-common:ro` + `PYTHONPATH=/app:/gaia-common`
- **Local development**: `pip install -e ./gaia-common/`

## Source Structure

```
gaia-common/
├── pyproject.toml                    # Package metadata
├── setup.py                          # Setuptools configuration
├── requirements.txt                  # Core dependencies
├── gaia_common/
│   ├── __init__.py
│   ├── config.py                     # Config singleton, loads gaia_constants.json
│   ├── protocols/                    # Core data structures
│   │   └── cognition_packet.py       # CognitionPacket v0.3 (central protocol)
│   ├── constants/
│   │   └── gaia_constants.json       # Shared constants reference
│   ├── base/                         # Abstract base classes
│   │   ├── persona.py               # Persona data structures
│   │   └── identity.py              # Identity guardianship protocols
│   ├── integrations/                 # External service integrations
│   │   ├── discord.py               # DiscordConfig, env var + constants loading
│   │   └── discord_connector.py     # Discord bot connection utilities
│   └── utils/                        # Utilities (55+ files)
│       ├── service_client.py         # HTTP client for inter-service communication
│       ├── vector_client.py          # Vector store read-only access
│       ├── logging_setup.py          # Centralized logging configuration
│       ├── chat_logger.py            # Conversation logging
│       ├── heartbeat_logger.py       # Health check log compression
│       ├── install_health_check_filter # Health check log filter
│       ├── packet_utils.py           # CognitionPacket manipulation
│       ├── packet_templates.py       # Packet construction templates
│       ├── safe_execution.py         # Sandboxed code execution
│       ├── tools_registry.py         # Central tool registry (used by gaia-mcp)
│       ├── world_state.py            # System state tracking
│       ├── vector_indexer.py         # Vector store operations
│       ├── knowledge_index.py        # Knowledge base indexing
│       ├── gaia_rescue_helper.py     # Rescue/fallback utilities
│       ├── code_analyzer/            # Document analysis pipeline
│       │   ├── file_scanner.py
│       │   ├── language_detection.py
│       │   ├── structure_extraction.py
│       │   └── llm_analysis.py
│       └── background/              # Background task management
│           ├── task_queue.py
│           └── idle_monitor.py
```

## Key Components

### CognitionPacket (v0.3)

The central data structure for inter-service communication. Defined in `protocols/cognition_packet.py`.

**Sections**:
- **Header**: datetime, session_id, packet_id, persona, origin, routing, model config
- **Content**: original_prompt, system_prompt, task_instructions, data_fields
- **Context**: conversation_history, knowledge_base, constraints, safety_profile
- **Reasoning**: thoughts, planning, execution, reflection logs
- **Response**: candidate text, confidence, finish_reason
- **Governance**: safety_profile_id, approval_required, tool_execution_status
- **Metrics**: token_usage, latency, inference_time, model_name
- **Status**: state (initialized/processing/completed/aborted), error_message, warnings

**Key Enums**: `PersonaRole`, `SystemTask`, `TargetEngine`, `OutputDestination`, `Origin`

### Config (`config.py`)

Singleton configuration class that loads from `gaia_constants.json`. Provides:
- Model configurations and backend selection
- Path management (models, knowledge, personas, LoRA adapters)
- Feature toggles and runtime parameters

### DiscordConfig (`integrations/discord.py`)

Loads Discord settings from both `gaia_constants.json` and environment variables. Env vars override constants:
- `DISCORD_BOT_TOKEN` overrides `constants.INTEGRATIONS.discord.bot_token`
- `DISCORD_WEBHOOK_URL` overrides `constants.INTEGRATIONS.discord.webhook_url`

### Utilities

- **`service_client.py`**: HTTP client with retry logic for inter-service calls
- **`vector_client.py`**: Read-only vector store access (respects sole-writer pattern)
- **`safe_execution.py`**: `run_shell_safe()` for sandboxed command execution
- **`tools_registry.py`**: Central TOOLS dict consumed by `gaia-mcp`
- **`install_health_check_filter`**: Suppresses repetitive health check access logs

## Dependencies

**Runtime**: pydantic, dataclasses-json, requests, httpx >=0.25.0
**Dev**: pytest, ruff, mypy

## Consumed By

Every GAIA service imports from `gaia-common`:
- **`gaia-core`**: CognitionPacket, config, vector_client, packet_utils
- **`gaia-web`**: CognitionPacket, health_check_filter
- **`gaia-mcp`**: tools_registry, safe_execution, CognitionPacket
- **`gaia-study`**: config, vector_indexer
- **`gaia-orchestrator`**: health_check_filter
