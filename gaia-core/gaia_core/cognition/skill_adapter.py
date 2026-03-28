"""
gaia-core/gaia_core/cognition/skill_adapter.py — Intent-Driven Skill Adapter Manager

Loads and unloads GGUF LoRA adapters on Prime/Core based on detected intent.
Adapters are NOT always-on — they're loaded when needed and unloaded after.

Adapter registry maps intents to adapter paths. The cognitive pipeline calls
`ensure_adapter(intent)` before generation and `release_adapter()` after.

Example flow:
  1. Intent detected: "code_generation"
  2. skill_adapter.ensure_adapter("code_generation") → loads code_skill_v1
  3. Prime generates with adapter active
  4. skill_adapter.release_adapter() → unloads (optional, can keep loaded)
"""

import logging
import os
from typing import Dict, Optional
from urllib.request import Request, urlopen
from urllib.error import URLError
import json

logger = logging.getLogger("GAIA.SkillAdapter")

PRIME_ENDPOINT = os.environ.get("PRIME_ENDPOINT", "http://gaia-prime:7777")
ADAPTER_SCALE = float(os.environ.get("CPU_ADAPTER_SCALE", "0.5"))

# Registry: intent patterns → adapter GGUF paths
# Multiple intents can map to the same adapter.
_ADAPTER_REGISTRY: Dict[str, str] = {}

# Currently loaded adapter (to avoid redundant load/unload)
_current_adapter: Optional[str] = None


def register_adapter(intent: str, adapter_path: str) -> None:
    """Register an adapter for an intent pattern."""
    _ADAPTER_REGISTRY[intent] = adapter_path
    logger.info("Registered adapter: %s → %s", intent, adapter_path)


def _init_registry():
    """Initialize the adapter registry from environment or defaults."""
    global _ADAPTER_REGISTRY

    # Code skill adapter — loaded for code-related intents
    code_adapter = os.environ.get(
        "CODE_SKILL_ADAPTER",
        "/models/lora_adapters/tier3_session/code_skill_v1/adapter.gguf",
    )
    if code_adapter:
        for intent in ("code_generation", "code_review", "tool_routing",
                        "coding", "debug", "programming"):
            _ADAPTER_REGISTRY[intent] = code_adapter

    # Future: add more skill adapters here
    # e.g., "creative_writing" → writing_skill adapter
    #        "math" → math_skill adapter

    if _ADAPTER_REGISTRY:
        logger.info("Skill adapter registry: %d intents registered", len(_ADAPTER_REGISTRY))


def resolve_adapter_for_intent(intent: str) -> Optional[str]:
    """Find the adapter path for a given intent. Returns None if no match."""
    if not _ADAPTER_REGISTRY:
        _init_registry()

    # Exact match first
    if intent in _ADAPTER_REGISTRY:
        return _ADAPTER_REGISTRY[intent]

    # Partial match — check if any registered intent is a substring
    intent_lower = intent.lower()
    for registered_intent, path in _ADAPTER_REGISTRY.items():
        if registered_intent in intent_lower or intent_lower in registered_intent:
            return path

    return None


def ensure_adapter(intent: str, endpoint: str = None) -> bool:
    """Load the appropriate adapter for the given intent if not already active.

    Returns True if an adapter is now active, False if no adapter needed.
    """
    global _current_adapter

    adapter_path = resolve_adapter_for_intent(intent)
    if not adapter_path:
        return False

    # Already loaded
    if _current_adapter == adapter_path:
        return True

    # Unload current if different
    if _current_adapter:
        release_adapter(endpoint)

    # Load new adapter
    ep = endpoint or PRIME_ENDPOINT
    try:
        payload = json.dumps({
            "adapter_path": adapter_path,
            "scale": ADAPTER_SCALE,
        }).encode()
        req = Request(f"{ep}/adapter/load", data=payload,
                      headers={"Content-Type": "application/json"})
        with urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read())
            if data.get("ok"):
                _current_adapter = adapter_path
                logger.info("Skill adapter loaded: %s (intent=%s, scale=%.2f)",
                            adapter_path.split("/")[-2], intent, ADAPTER_SCALE)
                return True
            else:
                logger.warning("Adapter load failed: %s", data)
                return False
    except (URLError, Exception) as e:
        logger.warning("Adapter load error: %s", e)
        return False


def release_adapter(endpoint: str = None) -> None:
    """Unload the current adapter (if any)."""
    global _current_adapter

    if not _current_adapter:
        return

    ep = endpoint or PRIME_ENDPOINT
    try:
        req = Request(f"{ep}/adapter/unload", data=b"{}",
                      headers={"Content-Type": "application/json"})
        with urlopen(req, timeout=10) as resp:
            pass
        logger.info("Skill adapter unloaded")
    except Exception:
        pass
    _current_adapter = None


def get_status() -> dict:
    """Return current adapter status."""
    return {
        "active_adapter": _current_adapter,
        "registry_size": len(_ADAPTER_REGISTRY),
        "scale": ADAPTER_SCALE,
    }
