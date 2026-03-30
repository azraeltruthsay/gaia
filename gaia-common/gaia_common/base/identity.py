"""
Identity and Guardian base classes for GAIA services.

This module defines the abstract interfaces and data structures for
identity management and ethical guardrails that all GAIA services share.

The identity system uses a tiered model:
- Tier I: Immutable core identity (ethical grounding, cannot be overridden)
- Tier II: Role persona (contextual, can vary by task)
- Tier III: Instruction set (ephemeral, session-specific)

Usage:
    from gaia_common.base import IdentityData, IdentityGuardianProtocol
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable


class IdentityTier(Enum):
    """
    Identity tier levels in the layered identity model.
    """
    TIER_I = "tier_i"    # Immutable core identity
    TIER_II = "tier_ii"  # Role persona
    TIER_III = "tier_iii"  # Instruction set


@dataclass
class ImmutableTrait:
    """
    Represents an immutable trait in Tier I identity.
    """
    name: str
    description: str
    enforcement_level: str = "strict"  # strict, advisory


@dataclass
class IdentityData:
    """
    Represents the core identity configuration.

    This is the canonical structure for Tier I identity loaded from JSON.
    """
    name: str = "GAIA"
    version: str = "1.0"
    immutable_traits: Dict[str, ImmutableTrait] = field(default_factory=dict)
    forbidden_phrases: List[str] = field(default_factory=list)
    core_values: List[str] = field(default_factory=list)
    ethical_guidelines: List[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "IdentityData":
        """Create an IdentityData instance from a dictionary."""
        traits = {}
        for key, value in data.get("immutable_traits", {}).items():
            if isinstance(value, dict):
                traits[key] = ImmutableTrait(
                    name=key,
                    description=value.get("description", ""),
                    enforcement_level=value.get("enforcement_level", "strict"),
                )
            else:
                # Simple string value
                traits[key] = ImmutableTrait(name=key, description=str(value))

        return cls(
            name=data.get("name", "GAIA"),
            version=data.get("version", "1.0"),
            immutable_traits=traits,
            forbidden_phrases=data.get("forbidden_phrases", []),
            core_values=data.get("core_values", []),
            ethical_guidelines=data.get("ethical_guidelines", []),
        )


@dataclass
class ValidationResult:
    """
    Result of an identity/ethics validation check.
    """
    valid: bool
    violations: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    tier: IdentityTier = IdentityTier.TIER_I

    @property
    def passed(self) -> bool:
        """Alias for valid."""
        return self.valid


@runtime_checkable
class IdentityGuardianProtocol(Protocol):
    """
    Protocol defining the interface for identity guardians.

    Identity guardians validate prompts and instructions against
    the immutable Tier I identity.
    """

    def validate_prompt_stack(
        self,
        persona_traits: Dict[str, Any],
        instructions: List[str],
        prompt: str,
    ) -> bool:
        """
        Validate a prompt stack against Tier I identity rules.

        Args:
            persona_traits: The current persona's traits
            instructions: List of system instructions
            prompt: The user prompt to validate

        Returns:
            True if valid, False if identity violation detected
        """
        ...


class BaseIdentityGuardian(ABC):
    """
    Abstract base class for identity guardians.

    Provides common functionality for Tier I identity validation.
    Concrete implementations should handle identity loading and
    specific validation rules.
    """

    @abstractmethod
    def load_identity(self) -> Optional[IdentityData]:
        """Load the Tier I identity configuration."""
        pass

    @abstractmethod
    def validate_prompt_stack(
        self,
        persona_traits: Dict[str, Any],
        instructions: List[str],
        prompt: str,
    ) -> bool:
        """Validate a prompt stack against identity rules."""
        pass

    def check_forbidden_phrases(
        self,
        texts: List[str],
        forbidden: List[str],
    ) -> List[str]:
        """
        Check texts for forbidden phrases.

        Args:
            texts: List of text strings to check
            forbidden: List of forbidden phrases

        Returns:
            List of violations found
        """
        violations = []
        for text in texts:
            text_lower = text.lower()
            for phrase in forbidden:
                if phrase.lower() in text_lower:
                    violations.append(f"Forbidden phrase detected: '{phrase}'")
        return violations

    def validate_with_result(
        self,
        persona_traits: Dict[str, Any],
        instructions: List[str],
        prompt: str,
    ) -> ValidationResult:
        """
        Validate and return a detailed result.

        Args:
            persona_traits: The current persona's traits
            instructions: List of system instructions
            prompt: The user prompt to validate

        Returns:
            ValidationResult with details
        """
        valid = self.validate_prompt_stack(persona_traits, instructions, prompt)
        return ValidationResult(
            valid=valid,
            violations=[] if valid else ["Validation failed"],
            tier=IdentityTier.TIER_I,
        )


@runtime_checkable
class EthicalSentinelProtocol(Protocol):
    """
    Protocol defining the interface for ethical sentinels.

    Ethical sentinels perform runtime ethics checking on content
    before it's generated or executed.
    """

    def check_content(self, content: str) -> ValidationResult:
        """
        Check content for ethical concerns.

        Args:
            content: The content to check

        Returns:
            ValidationResult with any concerns
        """
        ...

    def check_action(
        self,
        action_type: str,
        params: Dict[str, Any],
    ) -> ValidationResult:
        """
        Check an action for ethical concerns.

        Args:
            action_type: Type of action (e.g., "write_file", "execute_shell")
            params: Action parameters

        Returns:
            ValidationResult with any concerns
        """
        ...


__all__ = [
    "IdentityTier",
    "ImmutableTrait",
    "IdentityData",
    "ValidationResult",
    "IdentityGuardianProtocol",
    "BaseIdentityGuardian",
    "EthicalSentinelProtocol",
]
