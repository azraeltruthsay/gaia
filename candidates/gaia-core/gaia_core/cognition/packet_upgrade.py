"""Non-destructive CognitionPacket upgrader — DEPRECATED.

This module was part of the v0.2 → v0.3 migration path. The attributes it
sets (cot, scratch, cheats, proposed_actions, etc.) do not exist on the v0.3
CognitionPacket dataclass. Callers (prompt_builder.py) wrap calls in
try/except so this is safe to make a no-op.
"""
from __future__ import annotations
import logging

logger = logging.getLogger("GAIA.PacketUpgrade")

def _ensure(obj: object, name: str, default):
    """Legacy helper — no longer used."""
    pass

def _ensure_slots(dct, keys):
    """Legacy helper — no longer used."""
    pass

def upgrade_packet(packet, config) -> object:
    """No-op: v0.3 CognitionPacket fields are set by the dataclass constructor."""
    return packet
