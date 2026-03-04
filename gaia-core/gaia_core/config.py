from __future__ import annotations
import logging
from gaia_common.config import Config as CommonConfig

logger = logging.getLogger("GAIA.Config")

class Config(CommonConfig):
    """
    Core-specific configuration wrapper.
    Inherits all authoritative settings and generic properties from gaia-common.
    """
    # Placeholder for core-specific logic if needed in the future
    pass

def get_config() -> Config:
    """Get the core-wrapped authoritative config."""
    return Config.get_instance()
