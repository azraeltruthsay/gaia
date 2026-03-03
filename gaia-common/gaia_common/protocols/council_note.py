"""
Generative Council Protocol (GCP) - Council Note Schema.

Defines the structure for internal multi-perspective reasoning, 
temporal handoffs, and agent-to-agent feedback.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from dataclasses_json import dataclass_json

@dataclass_json
@dataclass
class CouncilNote:
    """A record of an internal observation or handoff between personas."""
    note_id: str
    session_id: str
    author_persona: str
    target_persona: str
    
    # Core content
    subject: str
    body: str
    
    # Metadata for temporal grounding
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    urgency: int = 3  # 1 (critical) to 5 (info)
    
    # Epistemic grounding
    confidence: float = 1.0
    source_packet_id: Optional[str] = None
    
    # Handled status
    consumed: bool = False
    consumed_at: Optional[str] = None
    
    # Structured context (optional)
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass_json
@dataclass
class CouncilMeeting:
    """A collection of notes and votes representing a multi-agent decision."""
    meeting_id: str
    topic: str
    notes: List[CouncilNote] = field(default_factory=list)
    verdict: Optional[str] = None
    status: str = "active" # active, resolved, archived
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
