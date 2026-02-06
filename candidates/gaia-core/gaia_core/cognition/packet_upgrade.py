"""Non-destructive CognitionPacket upgrader (adds new GCP fields if missing)."""
from __future__ import annotations
from typing import Any, Dict, List

def _ensure(obj: object, name: str, default):
    """Sets a default value for an attribute if it's missing or None."""
    if not hasattr(obj, name) or getattr(obj, name) is None:
        setattr(obj, name, default)

def _ensure_slots(dct: Dict[str, Any], keys: List[str]):
    """Ensures specific keys exist in a dictionary with a default of None."""
    for k in keys:
        dct.setdefault(k, None)

def upgrade_packet(packet, config) -> object:
    """
    Idempotent: call on any existing packet object to ensure the new fields exist.
    Never mutates existing non-empty values; only fills missing pieces.
    This prepares the packet for advanced GCP features.
    """
    # --- Budget / version ---
    _ensure(packet, "protocol_version", getattr(config, "GCP_PROTOCOL_VERSION", "1.0.0"))
    _ensure(packet, "estimated_tokens", {"prompt": 0, "budget": int(getattr(config, "PACKET_BUDGET", 4096))})

    # --- Thoughts vs Data ---
    if not hasattr(packet, "cot") or not isinstance(getattr(packet, "cot"), dict):
        packet.cot = {"t1": None, "t2": None, "t3": None, "t4": None, "t5": None}
    else:
        _ensure_slots(packet.cot, ["t1","t2","t3","t4","t5"])

    # The existing CognitionPacket already has a 'scratch' attribute.
    # We just need to ensure the data slots are there.
    if hasattr(packet, "scratch") and isinstance(getattr(packet, "scratch"), dict):
        for c in "ABCDE": packet.scratch.setdefault(f"data{c}", None)
    else:
         _ensure(packet, "scratch", {f"data{c}": None for c in "ABCDE"})

    if not hasattr(packet, "cheats") or not isinstance(getattr(packet, "cheats"), dict):
        packet.cheats = {c: None for c in ["A","B","C","D","E"]}
    else:
        for c in ["A","B","C","D","E"]: packet.cheats.setdefault(c, None)

    # --- Persistence / Safety ---
    _ensure(packet, "sketchpad_refs", [])
    _ensure(packet, "observer_state", {"status": "pending", "notes": []})

    # --- Actions / Response ---
    _ensure(packet, "proposed_actions", {"plan": "", "directives": []})
    if "plan" not in packet.proposed_actions: packet.proposed_actions["plan"] = ""
    if "directives" not in packet.proposed_actions: packet.proposed_actions["directives"] = []
    _ensure(packet, "proposed_response", "")

    # --- Trigger metadata (for future phases like self-audit) ---
    _ensure(packet, "trigger_id", None)
    _ensure(packet, "trigger_reason", None)
    _ensure(packet, "audit_scope", {})

    return packet