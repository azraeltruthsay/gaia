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
- **Header**: session_id, packet_id, persona (identity_id, persona_id, role, tone_hint, safety_profile_id, traits), routing (target_engine, allow_parallel, priority, deadline_iso, queue_id), model (name, provider, context_window_tokens, max_output_tokens, response_buffer_tokens, temperature, top_p, seed, stop, tool_permissions, allow_tools), origin, datetime, sub_id, parent_packet_id, lineage, output_routing, operational_status.
- **Intent**: user_intent, system_task, confidence, tags.
- **Context**: session_history_ref (type, value), cheatsheets, constraints (max_tokens, time_budget_ms, safety_mode, policies), relevant_history_snippet, available_mcp_tools.
- **Content**: original_prompt, data_fields, attachments, timeline (Temporal Context support), intent (optional override).
- **Reasoning**: reflection_log, sketchpad, evaluations.
- **Response**: candidate, confidence, stream_proposal, tool_calls, sidecar_actions.
- **Governance**: safety (execution_allowed, allowed_commands_whitelist_id, dry_run), signatures, audit, privacy.
- **Metrics**: token_usage (prompt_tokens, completion_tokens, total_tokens, projected_tokens), latency_ms, cost_estimate, errors, resources, semantic_probe.
- **Status**: finalized, state (PacketState), next_steps, observer_trace.

**Robustness**: All sections use `default_factory` patterns to ensure 100% successful deserialization from partial JSON payloads.

**Key Enums**: `PersonaRole`, `SystemTask`, `TargetEngine`, `OutputDestination`, `Origin`, `PacketState`

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
