"""
gaia-common: Shared library for GAIA SOA services.

This package provides the foundational components shared across all GAIA services:

- **protocols**: Core data structures (CognitionPacket, GCP schemas)
- **utils**: Shared utilities (logging, packet templates)
- **base**: Abstract base classes (Persona, Identity)

Quick Start:
    from gaia_common.protocols import CognitionPacket, PacketState
    from gaia_common.utils import setup_logging, get_logger
    from gaia_common.base import PersonaData, IdentityGuardianProtocol
"""

__version__ = "0.1.0"

# Submodule exports for convenience
from . import protocols
from . import utils
from . import base

__all__ = [
    "__version__",
    "protocols",
    "utils",
    "base",
]
