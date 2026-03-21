"""Structured error registry for the GAIA SOA.

Every error gets a unique code (``GAIA-{SERVICE}-{NNN}``) plus a human-readable
hint and a severity level.  Services import the pre-registered definitions and
call ``lookup()`` at log-time to enrich log lines with searchable fields.

This module is **stdlib-only** so gaia-doctor (which avoids third-party deps)
can import it safely.

Service code ranges
-------------------
+-------------+-----------------+-----------+
| Prefix      | Service         | Range     |
+=============+=================+===========+
| GAIA-CORE   | gaia-core       | 001 – 299 |
| GAIA-WEB    | gaia-web        | 001 – 099 |
| GAIA-MCP    | gaia-mcp        | 001 – 099 |
| GAIA-STUDY  | gaia-study      | 001 – 099 |
| GAIA-DOCTOR | gaia-doctor     | 001 – 099 |
| GAIA-ORCH   | gaia-orchestr.  | 001 – 099 |
| GAIA-AUDIO  | gaia-audio      | 001 – 099 |
| GAIA-ENGINE | gaia engine lib | 001 – 099 |
| GAIA-MONKEY | gaia-monkey     | 001 – 099 |
| GAIA-COMMON | gaia-common     | 001 – 049 |
+-------------+-----------------+-----------+

Level convention
~~~~~~~~~~~~~~~~
- CRITICAL (001-009): system cannot continue
- ERROR    (010-099): operation failed, system continues
- WARNING  (100-199): degraded but functional
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Optional


# ---------------------------------------------------------------------------
# Category enum
# ---------------------------------------------------------------------------

class ErrorCategory(Enum):
    MODEL = "model"
    SAFETY = "safety"
    NETWORK = "network"
    CONFIG = "config"
    MEMORY = "memory"
    TOOL = "tool"
    IDENTITY = "identity"
    LOOP = "loop"
    RESOURCE = "resource"
    INTERNAL = "internal"


# ---------------------------------------------------------------------------
# Error definition
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class GaiaErrorDef:
    """Immutable descriptor for a single GAIA error."""

    code: str               # e.g. "GAIA-CORE-001"
    message: str            # short summary
    hint: str               # human-readable remediation
    level: int              # logging level constant (logging.ERROR etc.)
    category: ErrorCategory
    is_retryable: bool = False


# ---------------------------------------------------------------------------
# Registry internals
# ---------------------------------------------------------------------------

_REGISTRY: Dict[str, GaiaErrorDef] = {}


def register(
    code: str,
    message: str,
    hint: str,
    level: int = logging.ERROR,
    category: ErrorCategory = ErrorCategory.INTERNAL,
    is_retryable: bool = False,
) -> GaiaErrorDef:
    """Register a new error definition.  Raises ValueError on duplicate codes."""
    if code in _REGISTRY:
        raise ValueError(f"Duplicate error code: {code}")
    defn = GaiaErrorDef(code=code, message=message, hint=hint, level=level, category=category, is_retryable=is_retryable)
    _REGISTRY[code] = defn
    return defn


def lookup(code: str) -> Optional[GaiaErrorDef]:
    """Look up an error definition by code.  Returns None if not found."""
    return _REGISTRY.get(code)


def all_errors() -> Dict[str, GaiaErrorDef]:
    """Return a shallow copy of the full registry."""
    return dict(_REGISTRY)


# ===================================================================
# Pre-registered error definitions
# ===================================================================

# --- GAIA-COMMON (001-049) -------------------------------------------

register("GAIA-COMMON-001", "Insufficient memory",
         "Not enough RAM/swap for the requested operation. Free memory or reduce model size.",
         logging.ERROR, ErrorCategory.RESOURCE)

register("GAIA-COMMON-002", "JSON formatter error",
         "Failed to format a log record as JSON. Check for non-serialisable objects.",
         logging.WARNING, ErrorCategory.INTERNAL)

# --- GAIA-CORE: Critical (001-009) ------------------------------------

register("GAIA-CORE-001", "Healing lock active",
         "HEALING_REQUIRED.lock exists — a previous fatal loop triggered the circuit breaker. "
         "Investigate logs, then remove the lock file to resume.",
         logging.CRITICAL, ErrorCategory.LOOP)

register("GAIA-CORE-002", "Circuit breaker tripped",
         "Too many consecutive failures. The system has stopped processing to prevent cascading damage. "
         "Check service health and recent error logs.",
         logging.CRITICAL, ErrorCategory.INTERNAL)

register("GAIA-CORE-003", "Startup failure",
         "gaia-core failed to initialise. Check model availability and config.",
         logging.CRITICAL, ErrorCategory.CONFIG)

# --- GAIA-CORE: Cognitive (010-049) -----------------------------------

register("GAIA-CORE-010", "Identity violation detected",
         "The guardian flagged an identity-compromising prompt or response. "
         "Review the input for jailbreak patterns.",
         logging.ERROR, ErrorCategory.IDENTITY)

register("GAIA-CORE-011", "Guardian exception",
         "The identity guardian raised an exception during evaluation. "
         "Check guardian config and model availability.",
         logging.ERROR, ErrorCategory.SAFETY)

register("GAIA-CORE-012", "Ethical sentinel block",
         "The ethical sentinel blocked the response. Review safety thresholds.",
         logging.ERROR, ErrorCategory.SAFETY)

register("GAIA-CORE-015", "Forward-to-model failed",
         "Failed to get a response from the selected model. "
         "Check model health and endpoint connectivity.",
         logging.ERROR, ErrorCategory.MODEL, is_retryable=True)

register("GAIA-CORE-016", "Model response empty",
         "The model returned an empty or whitespace-only response. "
         "May indicate prompt issues or model overload.",
         logging.ERROR, ErrorCategory.MODEL)

register("GAIA-CORE-020", "Plan execution error",
         "An error occurred while executing a multi-step plan. "
         "Check the plan structure and tool availability.",
         logging.ERROR, ErrorCategory.INTERNAL)

register("GAIA-CORE-025", "Loop detected",
         "Repetitive output pattern detected. Recovery strategies are being applied.",
         logging.WARNING, ErrorCategory.LOOP)

register("GAIA-CORE-026", "Loop recovery failed",
         "All loop recovery strategies exhausted. The turn will be terminated.",
         logging.ERROR, ErrorCategory.LOOP)

register("GAIA-CORE-030", "Generation empty after retries",
         "Model returned empty output even after retry attempts.",
         logging.ERROR, ErrorCategory.MODEL)

register("GAIA-CORE-035", "Self-reflection error",
         "The self-reflection module raised an exception. Continuing without reflection.",
         logging.WARNING, ErrorCategory.INTERNAL)

register("GAIA-CORE-040", "Intent detection failed",
         "Could not determine user intent. Falling back to direct generation.",
         logging.WARNING, ErrorCategory.INTERNAL)

register("GAIA-CORE-045", "Knowledge enhancement failed",
         "Failed to enhance the packet with knowledge context.",
         logging.WARNING, ErrorCategory.INTERNAL)

# --- GAIA-CORE: Models (050-099) --------------------------------------

register("GAIA-CORE-050", "Model load failed",
         "Failed to load a model into the pool. Check model path, VRAM, and dependencies.",
         logging.ERROR, ErrorCategory.MODEL)

register("GAIA-CORE-055", "Prime model unreachable",
         "Cannot connect to the GAIA Engine Prime server (port 7777). "
         "Check that gaia-prime is running.",
         logging.ERROR, ErrorCategory.NETWORK, is_retryable=True)

register("GAIA-CORE-060", "Nano model unreachable",
         "The Nano (triage) model failed to load or respond. Triage will be skipped.",
         logging.ERROR, ErrorCategory.MODEL)

register("GAIA-CORE-065", "No suitable model available",
         "No model in the pool matches the routing criteria. "
         "Check model tier config and pool health.",
         logging.ERROR, ErrorCategory.MODEL)

register("GAIA-CORE-070", "LoRA adapter load failed",
         "Failed to load LoRA adapter into the GAIA Engine. Check adapter path and compatibility.",
         logging.WARNING, ErrorCategory.MODEL)

register("GAIA-CORE-075", "Inference stream interrupted",
         "Streaming inference from model was interrupted mid-generation. "
         "Common causes: GPU OOM, vLLM crash, httpx timeout, network drop. "
         "Check gaia-prime logs and GPU memory. Doctor will auto-detect via irritation patterns.",
         logging.ERROR, ErrorCategory.MODEL, is_retryable=True)

register("GAIA-CORE-076", "Model swap failed during debate",
         "Failed to swap between Core and Thinker models during council debate. "
         "The partial response from the previous model will be used. "
         "Check model pool health and VRAM availability.",
         logging.ERROR, ErrorCategory.MODEL, is_retryable=True)

register("GAIA-CORE-077", "Loop detector disabled",
         "Loop detection observer failed to initialize or check. "
         "GAIA may not detect repetitive output. Check loop_recovery.py imports.",
         logging.WARNING, ErrorCategory.LOOP)

register("GAIA-CORE-078", "Plan generation failed",
         "Forward-to-model call failed during planning stage. "
         "Check model availability and VRAM. The user will see a plan-error message.",
         logging.ERROR, ErrorCategory.MODEL, is_retryable=True)

register("GAIA-CORE-079", "Attachment ingestion failed",
         "Failed to process a user-uploaded attachment. "
         "Check file size limits, supported formats, and disk space.",
         logging.ERROR, ErrorCategory.INTERNAL)

register("GAIA-CORE-080", "Model stream error",
         "Error during model streaming in ExternalVoice. "
         "The generation was aborted. Check model server health and VRAM.",
         logging.ERROR, ErrorCategory.MODEL, is_retryable=True)

# --- GAIA-CORE: Session (100-149) -------------------------------------

register("GAIA-CORE-100", "Session not found",
         "Requested session ID does not exist. It may have expired or never been created.",
         logging.WARNING, ErrorCategory.INTERNAL)

register("GAIA-CORE-105", "History corrupt or unparseable",
         "Session history could not be loaded. The session will start fresh.",
         logging.WARNING, ErrorCategory.INTERNAL)

register("GAIA-CORE-110", "Session save failed",
         "Failed to persist session state to disk.",
         logging.ERROR, ErrorCategory.RESOURCE)

# --- GAIA-CORE: MCP/Tools (150-199) -----------------------------------

register("GAIA-CORE-150", "MCP connection failed",
         "Cannot reach the MCP tool server. Check that gaia-mcp is running on port 8765.",
         logging.ERROR, ErrorCategory.NETWORK, is_retryable=True)

register("GAIA-CORE-155", "Tool execution timeout",
         "A tool call exceeded the timeout. Consider increasing the timeout or simplifying the operation.",
         logging.WARNING, ErrorCategory.TOOL, is_retryable=True)

register("GAIA-CORE-160", "Tool result parse error",
         "Failed to parse the JSON-RPC result from a tool call.",
         logging.ERROR, ErrorCategory.TOOL)

# --- GAIA-CORE: Sleep (200-249) ---------------------------------------

register("GAIA-CORE-200", "Sleep transition failed",
         "The system failed to enter sleep mode. Check sleep task scheduler state.",
         logging.ERROR, ErrorCategory.INTERNAL)

register("GAIA-CORE-205", "Wake signal failed",
         "Failed to wake the system from sleep. Manual restart may be needed.",
         logging.ERROR, ErrorCategory.INTERNAL)

# --- GAIA-WEB (001-099) -----------------------------------------------

register("GAIA-WEB-001", "Core service unreachable",
         "gaia-web cannot reach gaia-core. Check that gaia-core is running on port 6415.",
         logging.ERROR, ErrorCategory.NETWORK)

register("GAIA-WEB-005", "Candidate fallback activated",
         "Primary core is down; routing to candidate. This is degraded operation.",
         logging.WARNING, ErrorCategory.NETWORK)

register("GAIA-WEB-010", "Discord dispatch failed",
         "Failed to send a message to Discord. Check bot token and channel permissions.",
         logging.ERROR, ErrorCategory.NETWORK)

register("GAIA-WEB-015", "Discord connection lost",
         "The Discord WebSocket connection dropped. Reconnection will be attempted.",
         logging.WARNING, ErrorCategory.NETWORK)

register("GAIA-WEB-020", "Security scan blocked request",
         "The security scanner blocked an incoming request. Review the scan report.",
         logging.WARNING, ErrorCategory.SAFETY)

register("GAIA-WEB-030", "SSE stream error",
         "Server-Sent Events stream encountered an error. The client may need to reconnect.",
         logging.ERROR, ErrorCategory.NETWORK)

register("GAIA-WEB-035", "Consent validation failed",
         "User consent could not be validated. Request rejected.",
         logging.WARNING, ErrorCategory.SAFETY)

register("GAIA-WEB-040", "Discord message send failed",
         "Failed to send response to Discord channel or DM. "
         "User's message was processed but the reply was lost. "
         "Check Discord API rate limits and bot permissions.",
         logging.ERROR, ErrorCategory.NETWORK, is_retryable=True)

register("GAIA-WEB-045", "Voice channel join failed",
         "Failed to join Discord voice channel. "
         "Check bot voice permissions and channel availability.",
         logging.ERROR, ErrorCategory.NETWORK)

register("GAIA-WEB-050", "Voice processing error",
         "Voice processing loop crashed or utterance handling failed. "
         "Voice channel may become unresponsive. Check gaia-audio health.",
         logging.ERROR, ErrorCategory.INTERNAL)

register("GAIA-WEB-055", "Speech playback failed",
         "TTS output could not be played in voice channel. "
         "Check audio encoding, FFmpeg, and voice connection state.",
         logging.ERROR, ErrorCategory.INTERNAL)

register("GAIA-WEB-060", "Transcription failed",
         "Speech-to-text request to gaia-audio failed. "
         "User's voice input was lost. Check gaia-audio service health.",
         logging.ERROR, ErrorCategory.NETWORK, is_retryable=True)

# --- GAIA-MCP (001-099) -----------------------------------------------

register("GAIA-MCP-001", "Tool not found",
         "The requested tool is not registered in the MCP server.",
         logging.ERROR, ErrorCategory.TOOL)

register("GAIA-MCP-010", "Blast Shield blocked",
         "The Blast Shield blocked a dangerous command (rm -rf, sudo, etc.).",
         logging.WARNING, ErrorCategory.SAFETY)

register("GAIA-MCP-015", "Approval denied",
         "The tool call requires human approval which was denied.",
         logging.WARNING, ErrorCategory.SAFETY)

register("GAIA-MCP-020", "Tool execution crashed",
         "The tool raised an unhandled exception during execution.",
         logging.ERROR, ErrorCategory.TOOL, is_retryable=True)

register("GAIA-MCP-025", "py_compile gate failed",
         "Sovereign Shield: the written Python file has syntax errors. Write was blocked.",
         logging.ERROR, ErrorCategory.SAFETY)

register("GAIA-MCP-030", "Sandbox path violation",
         "Tool tried to access a path outside the sandbox.",
         logging.ERROR, ErrorCategory.SAFETY)

# --- GAIA-DOCTOR (001-099) --------------------------------------------

register("GAIA-DOCTOR-001", "Service unreachable",
         "gaia-doctor cannot reach a monitored service. Auto-restart may be triggered.",
         logging.ERROR, ErrorCategory.NETWORK)

register("GAIA-DOCTOR-010", "Restart threshold exceeded",
         "A service has been restarted too many times. Manual investigation required.",
         logging.ERROR, ErrorCategory.RESOURCE)

register("GAIA-DOCTOR-020", "Elasticsearch query failed",
         "Failed to query the ELK stack for log analysis. Check ES connectivity.",
         logging.WARNING, ErrorCategory.NETWORK)

register("GAIA-DOCTOR-025", "Cognitive test failure",
         "One or more cognitive battery tests failed. Review test results.",
         logging.WARNING, ErrorCategory.INTERNAL)

# --- GAIA-STUDY (001-099) ---------------------------------------------

register("GAIA-STUDY-001", "Training subprocess crashed",
         "The QLoRA training subprocess exited with an error. Check VRAM and training config.",
         logging.ERROR, ErrorCategory.RESOURCE)

register("GAIA-STUDY-010", "Vector indexing failed",
         "Failed to build or update the semantic vector index.",
         logging.ERROR, ErrorCategory.INTERNAL)

# --- GAIA-ORCH (001-099) ----------------------------------------------

register("GAIA-ORCH-001", "GPU lifecycle error",
         "Failed to manage GPU allocation or deallocation.",
         logging.ERROR, ErrorCategory.RESOURCE)

register("GAIA-ORCH-010", "Handoff failed",
         "Service handoff between primary and candidate failed.",
         logging.ERROR, ErrorCategory.NETWORK)

# --- GAIA-AUDIO (001-099) ---------------------------------------------

register("GAIA-AUDIO-001", "STT transcription failed",
         "Whisper speech-to-text failed. Check audio input and model availability.",
         logging.ERROR, ErrorCategory.MODEL)

register("GAIA-AUDIO-010", "TTS synthesis failed",
         "Text-to-speech synthesis failed.",
         logging.ERROR, ErrorCategory.MODEL)

# --- GAIA-ENGINE (001-099) --------------------------------------------

register("GAIA-ENGINE-001", "Model load failed",
         "GAIA Engine failed to load the model from disk. Check model path, disk space, and VRAM.",
         logging.CRITICAL, ErrorCategory.MODEL)

register("GAIA-ENGINE-005", "Model not loaded",
         "Inference requested but no model is loaded. Send POST /model/load first.",
         logging.ERROR, ErrorCategory.MODEL)

register("GAIA-ENGINE-010", "Generation failed",
         "Model generation raised an exception. Check input format and token limits.",
         logging.ERROR, ErrorCategory.MODEL, is_retryable=True)

register("GAIA-ENGINE-015", "Empty generation",
         "Model produced zero tokens. May indicate prompt issues or context overflow.",
         logging.WARNING, ErrorCategory.MODEL)

register("GAIA-ENGINE-020", "KV cache allocation failed",
         "Static KV cache could not be allocated. Likely insufficient VRAM for the requested sequence length.",
         logging.ERROR, ErrorCategory.RESOURCE)

register("GAIA-ENGINE-025", "KV cache overflow",
         "Input exceeded the pre-allocated KV cache max_seq_len. Truncation or reallocation needed.",
         logging.WARNING, ErrorCategory.RESOURCE)

register("GAIA-ENGINE-030", "Thought snapshot failed",
         "ThoughtHold failed to save or resume a KV cache snapshot. Check /shared/thoughts directory.",
         logging.ERROR, ErrorCategory.MEMORY)

register("GAIA-ENGINE-035", "Prefix cache miss",
         "SegmentedKVManager segment hash mismatch — cache invalidated and rebuilt.",
         logging.WARNING, ErrorCategory.MEMORY)

register("GAIA-ENGINE-040", "LoRA adapter load failed",
         "Failed to load a LoRA adapter via PEFT. Check adapter path, rank compatibility, and VRAM.",
         logging.ERROR, ErrorCategory.MODEL)

register("GAIA-ENGINE-045", "LoRA adapter not found",
         "Requested adapter name is not loaded. Load it first via POST /adapter/load.",
         logging.WARNING, ErrorCategory.MODEL)

register("GAIA-ENGINE-050", "Device migration failed",
         "GPU↔CPU migration raised an exception. Model may be in inconsistent state.",
         logging.ERROR, ErrorCategory.RESOURCE)

register("GAIA-ENGINE-055", "Polygraph capture failed",
         "HiddenStatePolygraph failed to capture activations. Generation continues without introspection.",
         logging.WARNING, ErrorCategory.INTERNAL)

register("GAIA-ENGINE-060", "Vision processing failed",
         "Multimodal image processing raised an exception. Check image format and processor availability.",
         logging.ERROR, ErrorCategory.MODEL, is_retryable=True)

register("GAIA-ENGINE-065", "Vision processor not available",
         "Model was loaded without vision support but received an image input.",
         logging.WARNING, ErrorCategory.MODEL)

register("GAIA-ENGINE-070", "torch.compile failed",
         "Model compilation with reduce-overhead mode failed. Falling back to eager execution.",
         logging.WARNING, ErrorCategory.INTERNAL)

register("GAIA-ENGINE-075", "SAE atlas recording failed",
         "Failed to record SAE atlas baseline activations.",
         logging.WARNING, ErrorCategory.INTERNAL)

register("GAIA-ENGINE-080", "ROME edit failed",
         "ROME weight edit operation failed. Model weights unchanged.",
         logging.ERROR, ErrorCategory.MODEL)

# --- GAIA-MONKEY (001-099) --------------------------------------------

register("GAIA-MONKEY-001", "Chaos drill failed",
         "A chaos drill could not be executed. Check target service connectivity.",
         logging.ERROR, ErrorCategory.INTERNAL)

register("GAIA-MONKEY-010", "Serenity state write failed",
         "Failed to update serenity.json. Check /shared/doctor/ permissions.",
         logging.ERROR, ErrorCategory.RESOURCE)

register("GAIA-MONKEY-015", "Meditation flag error",
         "Failed to read or write defensive_meditation.json.",
         logging.WARNING, ErrorCategory.RESOURCE)

register("GAIA-MONKEY-020", "PromptFoo suite failed",
         "A PromptFoo red-team test suite failed to execute.",
         logging.WARNING, ErrorCategory.INTERNAL)
