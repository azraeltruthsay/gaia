"""
Persona base classes and data structures for GAIA services.

This module defines the abstract interfaces and data structures for persona
management that all GAIA services share.

Usage:
    from gaia_common.base import PersonaData, PersonaManagerProtocol
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable


@dataclass
class PersonaData:
    """
    Represents a persona's configuration data.

    This is the canonical structure for persona definitions loaded from JSON.
    All services that work with personas should use this structure.
    """
    name: str
    identity_id: str = "GAIA"
    role: str = "Default"
    tone_hint: Optional[str] = None
    safety_profile_id: Optional[str] = None
    traits: Dict[str, Any] = field(default_factory=dict)
    knowledge_base_name: Optional[str] = None
    description: Optional[str] = None
    system_prompt_additions: Optional[str] = None
    enabled: bool = True

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PersonaData":
        """Create a PersonaData instance from a dictionary."""
        return cls(
            name=data.get("name", "unknown"),
            identity_id=data.get("identity_id", "GAIA"),
            role=data.get("role", "Default"),
            tone_hint=data.get("tone_hint"),
            safety_profile_id=data.get("safety_profile_id"),
            traits=data.get("traits", {}),
            knowledge_base_name=data.get("knowledge_base_name"),
            description=data.get("description"),
            system_prompt_additions=data.get("system_prompt_additions"),
            enabled=data.get("enabled", True),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "name": self.name,
            "identity_id": self.identity_id,
            "role": self.role,
            "tone_hint": self.tone_hint,
            "safety_profile_id": self.safety_profile_id,
            "traits": self.traits,
            "knowledge_base_name": self.knowledge_base_name,
            "description": self.description,
            "system_prompt_additions": self.system_prompt_additions,
            "enabled": self.enabled,
        }


@runtime_checkable
class PersonaManagerProtocol(Protocol):
    """
    Protocol defining the interface for persona management.

    Services that manage personas should implement this protocol.
    This allows for different implementations (file-based, database, remote)
    while maintaining a consistent interface.
    """

    def load_persona_data(self, name: str) -> Optional[Dict[str, Any]]:
        """
        Load a persona's data by name.

        Args:
            name: The persona name/identifier

        Returns:
            Persona data dictionary or None if not found
        """
        ...

    def list_personas(self) -> List[str]:
        """
        List all available persona names.

        Returns:
            List of persona names
        """
        ...


class BasePersonaManager(ABC):
    """
    Abstract base class for persona managers.

    Provides common functionality and defines the required interface.
    Concrete implementations should handle storage (file, database, etc.).
    """

    @abstractmethod
    def load_persona_data(self, name: str) -> Optional[Dict[str, Any]]:
        """Load a persona's raw data dictionary."""
        pass

    @abstractmethod
    def list_personas(self) -> List[str]:
        """List all available persona names."""
        pass

    def get_persona(self, name: str) -> Optional[PersonaData]:
        """
        Load and parse a persona into a PersonaData instance.

        Args:
            name: The persona name/identifier

        Returns:
            PersonaData instance or None if not found
        """
        data = self.load_persona_data(name)
        if data is None:
            return None
        return PersonaData.from_dict(data)


__all__ = [
    "PersonaData",
    "PersonaManagerProtocol",
    "BasePersonaManager",
]
