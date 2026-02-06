"""
Base classes for GAIA services.

This module provides abstract base classes and interfaces:
- Persona: Base class for persona definitions and management
- Identity: Base class for identity/guardian components
"""

from .persona import (
    PersonaData,
    PersonaManagerProtocol,
    BasePersonaManager,
)
from .identity import (
    IdentityTier,
    ImmutableTrait,
    IdentityData,
    ValidationResult,
    IdentityGuardianProtocol,
    BaseIdentityGuardian,
    EthicalSentinelProtocol,
)

__all__ = [
    # Persona
    "PersonaData",
    "PersonaManagerProtocol",
    "BasePersonaManager",
    # Identity
    "IdentityTier",
    "ImmutableTrait",
    "IdentityData",
    "ValidationResult",
    "IdentityGuardianProtocol",
    "BaseIdentityGuardian",
    "EthicalSentinelProtocol",
]
