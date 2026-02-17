"""
CognitionPacket – dynamic state for GAIA’s self-reflection loop.

Schema:
  prompt:         str            # original user prompt
  persona:        str            # active persona ID
  identity:       dict           # core identity fields
  instructions:   List[str]      # system prompts
  history:        List[dict]     # recent chat messages

  reflection_count: int           # increments each iteration
  thoughts:         List[dict]    # chain-of-thought entries
  scratch:         Dict[str, Any] # dynamic slots: dataA...dataE
  user_approval:    bool          # flag for shell cmd approval
"""

from __future__ import annotations
from typing import Any, Dict, List
import json
import uuid
from datetime import datetime, timezone
from app.config import Config

class CognitionPacket:
    def __init__(
        self,
        session_id: str,
        packet_id: str,
        time_date: str,
        packet_type: str,
        intent: str,
        intent_confidence: float,
        identity: str,
        persona: str,
        contextual_instructions: str,
        prompt: str,
        history: List[Dict[str, Any]],
        reflection: str,
        reflection_confidence: float,
        execution: str,
        execution_confidence: float,
        response: str,
        response_confidence: float,
        data_fields: Dict[str, Any],
        sub_packet_id: str = None,
        config: Config = None,
    ):
        self.session_id = session_id
        self.packet_id = packet_id
        self.sub_packet_id = sub_packet_id
        self.time_date = time_date
        self.packet_type = packet_type
        self.intent = intent
        self.intent_confidence = intent_confidence
        self.identity = identity
        self.persona = persona
        self.contextual_instructions = contextual_instructions
        self.prompt = prompt
        self.history = history
        self.reflection = reflection
        self.reflection_confidence = reflection_confidence
        self.execution = execution
        self.execution_confidence = execution_confidence
        self.response = response
        self.response_confidence = response_confidence
        self.data_fields = data_fields

    def to_json(self) -> str:
        # Create a dictionary representation of the object
        d = self.__dict__.copy()
        # Remove non-serializable fields if necessary
        if 'config' in d:
            del d['config']
        return json.dumps(d, indent=2)

    @staticmethod
    def from_json(data: str | Dict) -> CognitionPacket:
        if isinstance(data, str):
            data = json.loads(data)
        return CognitionPacket(**data)

# Packet factory
def create_packet(config: Config, prompt: str, session_id: str, history: List[Dict[str, Any]], persona_instructions: List[str]) -> CognitionPacket:
    now = datetime.now(timezone.utc)
    return CognitionPacket(
        session_id=session_id,
        packet_id=str(uuid.uuid4()),
        sub_packet_id=None,
        time_date=now.isoformat(),
        packet_type="Inquiry", # Placeholder
        intent="", # To be filled by intent detection
        intent_confidence=0.0,
        identity="GAIA",
        persona=config.persona_name,
        contextual_instructions="\n".join(persona_instructions),
        prompt=prompt,
        history=history,
        reflection="",
        reflection_confidence=0.0,
        execution="",
        execution_confidence=0.0,
        response="",
        response_confidence=0.0,
        data_fields={},
        config=config,
    )
