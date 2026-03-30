# GAIA Error Code Reference

> **Source of truth**: `gaia-common/gaia_common/errors.py`
> **Last regenerated**: 2026-03-20
> **Total codes**: 62

This document is generated from the error registry. Do not edit manually — update `errors.py` and regenerate.

## Format

```
GAIA-{SERVICE}-{NNN}
```

- **SERVICE**: COMMON, CORE, WEB, MCP, STUDY, DOCTOR, ORCH, AUDIO, ENGINE, MONKEY
- **NNN**: Three-digit code within service range
- **Level convention**: 001-009 CRITICAL, 010-099 ERROR/WARNING

## Categories

| Category | Description |
|----------|-------------|
| MODEL | Model loading, inference, adapters |
| SAFETY | Guardian, sentinel, identity, security |
| NETWORK | Inter-service communication |
| CONFIG | Configuration and environment |
| MEMORY | KV cache, thought snapshots, session state |
| TOOL | MCP tool execution |
| IDENTITY | Identity violation |
| LOOP | Loop detection and recovery |
| RESOURCE | VRAM, disk, memory exhaustion |
| INTERNAL | Unexpected internal errors |

## Service Code Ranges

| Prefix | Service | Range |
|--------|---------|-------|
| GAIA-COMMON | gaia-common (shared library) | 001–049 |
| GAIA-CORE | gaia-core (cognitive engine) | 001–299 |
| GAIA-WEB | gaia-web (gateway/Discord) | 001–099 |
| GAIA-MCP | gaia-mcp (tool execution) | 001–099 |
| GAIA-STUDY | gaia-study (training/indexing) | 001–099 |
| GAIA-DOCTOR | gaia-doctor (immune system) | 001–099 |
| GAIA-ORCH | gaia-orchestrator | 001–099 |
| GAIA-AUDIO | gaia-audio (STT/TTS) | 001–099 |
| GAIA-ENGINE | GAIA Inference Engine | 001–099 |
| GAIA-MONKEY | gaia-monkey (chaos engine) | 001–099 |

---

## GAIA-COMMON (Shared Library)

| Code | Level | Category | Message | Hint |
|------|-------|----------|---------|------|
| GAIA-COMMON-001 | ERROR | RESOURCE | Insufficient memory | Not enough RAM/swap for the requested operation. Free memory or reduce model size. |
| GAIA-COMMON-002 | WARNING | INTERNAL | JSON formatter error | Failed to format a log record as JSON. Check for non-serialisable objects. |

## GAIA-CORE (Cognitive Engine)

### Critical (001–009)

| Code | Level | Category | Message | Hint |
|------|-------|----------|---------|------|
| GAIA-CORE-001 | CRITICAL | LOOP | Healing lock active | HEALING_REQUIRED.lock exists — a previous fatal loop triggered the circuit breaker. Investigate logs, then remove the lock file to resume. |
| GAIA-CORE-002 | CRITICAL | INTERNAL | Circuit breaker tripped | Too many consecutive failures. The system has stopped processing to prevent cascading damage. Check service health and recent error logs. |
| GAIA-CORE-003 | CRITICAL | CONFIG | Startup failure | gaia-core failed to initialise. Check model availability and config. |

### Cognitive (010–049)

| Code | Level | Category | Retryable | Message | Hint |
|------|-------|----------|-----------|---------|------|
| GAIA-CORE-010 | ERROR | IDENTITY | | Identity violation detected | The guardian flagged an identity-compromising prompt or response. Review the input for jailbreak patterns. |
| GAIA-CORE-011 | ERROR | SAFETY | | Guardian exception | The identity guardian raised an exception during evaluation. Check guardian config and model availability. |
| GAIA-CORE-012 | ERROR | SAFETY | | Ethical sentinel block | The ethical sentinel blocked the response. Review safety thresholds. |
| GAIA-CORE-015 | ERROR | MODEL | Yes | Forward-to-model failed | Failed to get a response from the selected model. Check model health and endpoint connectivity. |
| GAIA-CORE-016 | ERROR | MODEL | | Model response empty | The model returned an empty or whitespace-only response. May indicate prompt issues or model overload. |
| GAIA-CORE-020 | ERROR | INTERNAL | | Plan execution error | An error occurred while executing a multi-step plan. Check the plan structure and tool availability. |
| GAIA-CORE-025 | WARNING | LOOP | | Loop detected | Repetitive output pattern detected. Recovery strategies are being applied. |
| GAIA-CORE-026 | ERROR | LOOP | | Loop recovery failed | All loop recovery strategies exhausted. The turn will be terminated. |
| GAIA-CORE-030 | ERROR | MODEL | | Generation empty after retries | Model returned empty output even after retry attempts. |
| GAIA-CORE-035 | WARNING | INTERNAL | | Self-reflection error | The self-reflection module raised an exception. Continuing without reflection. |
| GAIA-CORE-040 | WARNING | INTERNAL | | Intent detection failed | Could not determine user intent. Falling back to direct generation. |
| GAIA-CORE-045 | WARNING | INTERNAL | | Knowledge enhancement failed | Failed to enhance the packet with knowledge context. |

### Models (050–099)

| Code | Level | Category | Retryable | Message | Hint |
|------|-------|----------|-----------|---------|------|
| GAIA-CORE-050 | ERROR | MODEL | | Model load failed | Failed to load a model into the pool. Check model path, VRAM, and dependencies. |
| GAIA-CORE-055 | ERROR | NETWORK | Yes | Prime model unreachable | Cannot connect to the GAIA Engine Prime server (port 7777). Check that gaia-prime is running. |
| GAIA-CORE-060 | ERROR | MODEL | | Nano model unreachable | The Nano (triage) model failed to load or respond. Triage will be skipped. |
| GAIA-CORE-065 | ERROR | MODEL | | No suitable model available | No model in the pool matches the routing criteria. Check model tier config and pool health. |
| GAIA-CORE-070 | WARNING | MODEL | | LoRA adapter load failed | Failed to load LoRA adapter into the GAIA Engine. Check adapter path and compatibility. |
| GAIA-CORE-075 | ERROR | MODEL | Yes | Inference stream interrupted | Streaming inference from model was interrupted mid-generation. Common causes: GPU OOM, vLLM crash, httpx timeout, network drop. Check gaia-prime logs and GPU memory. Doctor will auto-detect via irritation patterns. |
| GAIA-CORE-076 | ERROR | MODEL | Yes | Model swap failed during debate | Failed to swap between Core and Thinker models during council debate. The partial response from the previous model will be used. Check model pool health and VRAM availability. |
| GAIA-CORE-077 | WARNING | LOOP | | Loop detector disabled | Loop detection observer failed to initialize or check. GAIA may not detect repetitive output. Check loop_recovery.py imports. |
| GAIA-CORE-078 | ERROR | MODEL | Yes | Plan generation failed | Forward-to-model call failed during planning stage. Check model availability and VRAM. The user will see a plan-error message. |
| GAIA-CORE-079 | ERROR | INTERNAL | | Attachment ingestion failed | Failed to process a user-uploaded attachment. Check file size limits, supported formats, and disk space. |
| GAIA-CORE-080 | ERROR | MODEL | Yes | Model stream error | Error during model streaming in ExternalVoice. The generation was aborted. Check model server health and VRAM. |

### Session (100–149)

| Code | Level | Category | Message | Hint |
|------|-------|----------|---------|------|
| GAIA-CORE-100 | WARNING | INTERNAL | Session not found | Requested session ID does not exist. It may have expired or never been created. |
| GAIA-CORE-105 | WARNING | INTERNAL | History corrupt or unparseable | Session history could not be loaded. The session will start fresh. |
| GAIA-CORE-110 | ERROR | RESOURCE | Session save failed | Failed to persist session state to disk. |

### MCP/Tools (150–199)

| Code | Level | Category | Retryable | Message | Hint |
|------|-------|----------|-----------|---------|------|
| GAIA-CORE-150 | ERROR | NETWORK | Yes | MCP connection failed | Cannot reach the MCP tool server. Check that gaia-mcp is running on port 8765. |
| GAIA-CORE-155 | WARNING | TOOL | Yes | Tool execution timeout | A tool call exceeded the timeout. Consider increasing the timeout or simplifying the operation. |
| GAIA-CORE-160 | ERROR | TOOL | | Tool result parse error | Failed to parse the JSON-RPC result from a tool call. |

### Sleep (200–249)

| Code | Level | Category | Message | Hint |
|------|-------|----------|---------|------|
| GAIA-CORE-200 | ERROR | INTERNAL | Sleep transition failed | The system failed to enter sleep mode. Check sleep task scheduler state. |
| GAIA-CORE-205 | ERROR | INTERNAL | Wake signal failed | Failed to wake the system from sleep. Manual restart may be needed. |

## GAIA-WEB (Gateway / Discord)

| Code | Level | Category | Message | Hint |
|------|-------|----------|---------|------|
| GAIA-WEB-001 | ERROR | NETWORK | Core service unreachable | gaia-web cannot reach gaia-core. Check that gaia-core is running on port 6415. |
| GAIA-WEB-005 | WARNING | NETWORK | Candidate fallback activated | Primary core is down; routing to candidate. This is degraded operation. |
| GAIA-WEB-010 | ERROR | NETWORK | Discord dispatch failed | Failed to send a message to Discord. Check bot token and channel permissions. |
| GAIA-WEB-015 | WARNING | NETWORK | Discord connection lost | The Discord WebSocket connection dropped. Reconnection will be attempted. |
| GAIA-WEB-020 | WARNING | SAFETY | Security scan blocked request | The security scanner blocked an incoming request. Review the scan report. |
| GAIA-WEB-030 | ERROR | NETWORK | SSE stream error | Server-Sent Events stream encountered an error. The client may need to reconnect. |
| GAIA-WEB-035 | WARNING | SAFETY | Consent validation failed | User consent could not be validated. Request rejected. |
| GAIA-WEB-040 | ERROR | NETWORK | Yes | Discord message send failed | Failed to send response to Discord channel or DM. User's message was processed but the reply was lost. Check Discord API rate limits and bot permissions. |
| GAIA-WEB-045 | ERROR | NETWORK | | Voice channel join failed | Failed to join Discord voice channel. Check bot voice permissions and channel availability. |
| GAIA-WEB-050 | ERROR | INTERNAL | | Voice processing error | Voice processing loop crashed or utterance handling failed. Voice channel may become unresponsive. Check gaia-audio health. |
| GAIA-WEB-055 | ERROR | INTERNAL | | Speech playback failed | TTS output could not be played in voice channel. Check audio encoding, FFmpeg, and voice connection state. |
| GAIA-WEB-060 | ERROR | NETWORK | Yes | Transcription failed | Speech-to-text request to gaia-audio failed. User's voice input was lost. Check gaia-audio service health. |

## GAIA-MCP (Tool Execution)

| Code | Level | Category | Retryable | Message | Hint |
|------|-------|----------|-----------|---------|------|
| GAIA-MCP-001 | ERROR | TOOL | | Tool not found | The requested tool is not registered in the MCP server. |
| GAIA-MCP-010 | WARNING | SAFETY | | Blast Shield blocked | The Blast Shield blocked a dangerous command (rm -rf, sudo, etc.). |
| GAIA-MCP-015 | WARNING | SAFETY | | Approval denied | The tool call requires human approval which was denied. |
| GAIA-MCP-020 | ERROR | TOOL | Yes | Tool execution crashed | The tool raised an unhandled exception during execution. |
| GAIA-MCP-025 | ERROR | SAFETY | | py_compile gate failed | Sovereign Shield: the written Python file has syntax errors. Write was blocked. |
| GAIA-MCP-030 | ERROR | SAFETY | | Sandbox path violation | Tool tried to access a path outside the sandbox. |

## GAIA-DOCTOR (Immune System)

| Code | Level | Category | Message | Hint |
|------|-------|----------|---------|------|
| GAIA-DOCTOR-001 | ERROR | NETWORK | Service unreachable | gaia-doctor cannot reach a monitored service. Auto-restart may be triggered. |
| GAIA-DOCTOR-010 | ERROR | RESOURCE | Restart threshold exceeded | A service has been restarted too many times. Manual investigation required. |
| GAIA-DOCTOR-020 | WARNING | NETWORK | Elasticsearch query failed | Failed to query the ELK stack for log analysis. Check ES connectivity. |
| GAIA-DOCTOR-025 | WARNING | INTERNAL | Cognitive test failure | One or more cognitive battery tests failed. Review test results. |

## GAIA-STUDY (Training / Indexing)

| Code | Level | Category | Message | Hint |
|------|-------|----------|---------|------|
| GAIA-STUDY-001 | ERROR | RESOURCE | Training subprocess crashed | The QLoRA training subprocess exited with an error. Check VRAM and training config. |
| GAIA-STUDY-010 | ERROR | INTERNAL | Vector indexing failed | Failed to build or update the semantic vector index. |

## GAIA-ORCH (Orchestrator)

| Code | Level | Category | Message | Hint |
|------|-------|----------|---------|------|
| GAIA-ORCH-001 | ERROR | RESOURCE | GPU lifecycle error | Failed to manage GPU allocation or deallocation. |
| GAIA-ORCH-010 | ERROR | NETWORK | Handoff failed | Service handoff between primary and candidate failed. |

## GAIA-AUDIO (STT / TTS)

| Code | Level | Category | Message | Hint |
|------|-------|----------|---------|------|
| GAIA-AUDIO-001 | ERROR | MODEL | STT transcription failed | Whisper speech-to-text failed. Check audio input and model availability. |
| GAIA-AUDIO-010 | ERROR | MODEL | TTS synthesis failed | Text-to-speech synthesis failed. |

## GAIA-ENGINE (Inference Engine)

| Code | Level | Category | Retryable | Message | Hint |
|------|-------|----------|-----------|---------|------|
| GAIA-ENGINE-001 | CRITICAL | MODEL | | Model load failed | GAIA Engine failed to load the model from disk. Check model path, disk space, and VRAM. |
| GAIA-ENGINE-005 | ERROR | MODEL | | Model not loaded | Inference requested but no model is loaded. Send POST /model/load first. |
| GAIA-ENGINE-010 | ERROR | MODEL | Yes | Generation failed | Model generation raised an exception. Check input format and token limits. |
| GAIA-ENGINE-015 | WARNING | MODEL | | Empty generation | Model produced zero tokens. May indicate prompt issues or context overflow. |
| GAIA-ENGINE-020 | ERROR | RESOURCE | | KV cache allocation failed | Static KV cache could not be allocated. Likely insufficient VRAM for the requested sequence length. |
| GAIA-ENGINE-025 | WARNING | RESOURCE | | KV cache overflow | Input exceeded the pre-allocated KV cache max_seq_len. Truncation or reallocation needed. |
| GAIA-ENGINE-030 | ERROR | MEMORY | | Thought snapshot failed | ThoughtHold failed to save or resume a KV cache snapshot. Check /shared/thoughts directory. |
| GAIA-ENGINE-035 | WARNING | MEMORY | | Prefix cache miss | SegmentedKVManager segment hash mismatch — cache invalidated and rebuilt. |
| GAIA-ENGINE-040 | ERROR | MODEL | | LoRA adapter load failed | Failed to load a LoRA adapter via PEFT. Check adapter path, rank compatibility, and VRAM. |
| GAIA-ENGINE-045 | WARNING | MODEL | | LoRA adapter not found | Requested adapter name is not loaded. Load it first via POST /adapter/load. |
| GAIA-ENGINE-050 | ERROR | RESOURCE | | Device migration failed | GPU↔CPU migration raised an exception. Model may be in inconsistent state. |
| GAIA-ENGINE-055 | WARNING | INTERNAL | | Polygraph capture failed | HiddenStatePolygraph failed to capture activations. Generation continues without introspection. |
| GAIA-ENGINE-060 | ERROR | MODEL | Yes | Vision processing failed | Multimodal image processing raised an exception. Check image format and processor availability. |
| GAIA-ENGINE-065 | WARNING | MODEL | | Vision processor not available | Model was loaded without vision support but received an image input. |
| GAIA-ENGINE-070 | WARNING | INTERNAL | | torch.compile failed | Model compilation with reduce-overhead mode failed. Falling back to eager execution. |
| GAIA-ENGINE-075 | WARNING | INTERNAL | | SAE atlas recording failed | Failed to record SAE atlas baseline activations. |
| GAIA-ENGINE-080 | ERROR | MODEL | | ROME edit failed | ROME weight edit operation failed. Model weights unchanged. |

## GAIA-MONKEY (Chaos Engine)

| Code | Level | Category | Message | Hint |
|------|-------|----------|---------|------|
| GAIA-MONKEY-001 | ERROR | INTERNAL | Chaos drill failed | A chaos drill could not be executed. Check target service connectivity. |
| GAIA-MONKEY-010 | ERROR | RESOURCE | Serenity state write failed | Failed to update serenity.json. Check /shared/doctor/ permissions. |
| GAIA-MONKEY-015 | WARNING | RESOURCE | Meditation flag error | Failed to read or write defensive_meditation.json. |
| GAIA-MONKEY-020 | WARNING | INTERNAL | PromptFoo suite failed | A PromptFoo red-team test suite failed to execute. |

---

## Exception Hierarchy

```
GaiaError (base — carries error_code, auto-lookups hint from registry)
├── GaiaConfigError    — CONFIG category errors
├── GaiaModelError     — MODEL category errors
├── GaiaSafetyError    — SAFETY/IDENTITY category errors
├── GaiaToolError      — TOOL category errors
└── GaiaNetworkError   — NETWORK category errors
```

**Location**: `gaia-common/gaia_common/exceptions.py`
**Logging helper**: `gaia_common.utils.error_logging.log_gaia_error(logger, code, detail)`
**JSON integration**: `gaia_common.utils.json_formatter` enriches log records with `error_code`, `error_hint`, `error_category`

## Usage

```python
from gaia_common.utils.error_logging import log_gaia_error

# Log a registered error
log_gaia_error(logger, "GAIA-ENGINE-010", "Timeout after 30s on /v1/chat/completions")

# Raise a typed exception
from gaia_common.exceptions import GaiaModelError
raise GaiaModelError("GAIA-ENGINE-001", detail="Model path not found: /models/foo")
```
