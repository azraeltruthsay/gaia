# GAIA Coding Idioms

> Conventions observed across the GAIA codebase. Flag deviations as warnings unless they have functional impact.

## Naming

- **Modules & functions**: `snake_case` — `agent_core.py`, `detect_intent()`, `process_execution_results()`
- **Classes**: `PascalCase` — `CognitionPacket`, `ServiceClient`, `LoopRecoveryManager`
- **Constants**: `UPPER_SNAKE_CASE` at module level — `HISTORY_SUMMARY_THRESHOLD`, `SENSITIVE_TOOLS`
- **Private symbols**: `_leading_underscore` — `_format_retrieved_session_context()`, `_THINK_TAG_PATTERN`
- **Loggers**: `logging.getLogger("GAIA.ServiceName.ModuleName")` — hierarchical, always at module scope outside classes

## Data Structure Patterns

**Two coexisting patterns — both are correct in their domain:**

| Layer | Pattern | Serialization | Used For |
|-------|---------|---------------|----------|
| Protocol (CognitionPacket) | `@dataclass_json` + `@dataclass` | `.to_dict()` / `.from_dict()` | Message passing, packet flow |
| Model (Blueprint, config) | Pydantic `BaseModel` | `.model_dump(mode="json")` | Validation, API schemas, configs |

**Anti-pattern**: Mixing the two — e.g., using Pydantic for packet sub-models or dataclass_json for API schemas.

## Import Ordering

1. Standard library (`os`, `sys`, `logging`, `json`, `pathlib`, `typing`)
2. Third-party (`fastapi`, `pydantic`, `httpx`, `regex`, `dataclasses_json`)
3. GAIA packages (`from gaia_common.protocols...`, `from gaia_core.cognition...`)
4. Local module imports

**Guarded imports** for heavy/optional deps (torch, llama_cpp, CUDA libs):
```python
try:
    import torch
except ImportError:
    torch = None  # system continues in degraded mode
```

## Logging

- Logger: `logger = logging.getLogger("GAIA.AgentCore")` — at module top, outside classes
- Format: UTC ISO timestamps via `UTCFormatter` in `gaia_common.utils.logging_setup`
- Health checks auto-suppressed: `HealthCheckFilter` strips `/health`, `/ready`, `/live` noise
- Ring-buffer: `LevelRingHandler` routes by level to per-level FIFO files in `/logs/{service}/{LEVEL}`
- **Anti-pattern**: `print()` statements, unstructured logging, logging inside hot loops

## Async Patterns

- HTTP: `httpx.AsyncClient` with context manager
- Inter-service: `ServiceClient` from `gaia_common` with `.post_with_retry()` for fallback
- Routes: `async def handler(request: Request)` with `request.app.state` for app-level state
- Parallel: `asyncio.gather()` for independent tasks, `await` for sequential deps
- **Anti-pattern**: Blocking calls (`requests.get()`, `time.sleep()`) in async context

## Error Handling

- **Guarded imports**: Heavy deps wrapped in try/except, set to `None` if missing
- **Fallback endpoints**: `ServiceClient.post_with_retry()` — 3x exponential backoff, then fallback URL
- **Resilience**: `gaia_common.utils.resilience` provides circuit breaker and retry decorators
- **Graceful degradation**: Features with `required=False` in blueprints use rule-based checks instead of LLM
- **Anti-pattern**: Bare `except:`, swallowing exceptions without logging, retrying without backoff

## Common Anti-Patterns to Watch For

1. **Duplicate EXECUTE directives** — Must check `packet.tool_routing.execution_status == EXECUTED` before emitting
2. **Lite model echo** — 3B model pattern-matches instead of comprehending; suppress tool syntax examples when tool already executed
3. **Stale model promotion** — After reflection with Prime, final generation must use that Prime, not swap back to Lite
4. **Candidate/live desync** — Changes in `candidates/` not synced to production paths; volume mounts only see one
5. **Missing model availability checks** — Always check if model is loaded before using; `StreamObserver` accepts `llm=None`
