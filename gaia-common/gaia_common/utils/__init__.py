"""
Shared utilities for GAIA services.

This module provides common functionality used across services:
- Logging: UTC-timestamped logging with consistent formatting
- Packet Templates: Render CognitionPackets for prompts
- VectorClient: Read-only vector query interface
- Helpers: File system and timestamp utilities
- Packet Utils: Safety checking and version migration
"""

from .logging_setup import (
    setup_logging,
    get_logger,
    UTCFormatter,
    HealthCheckFilter,
    install_health_check_filter,
)
from .packet_templates import (
    packet_to_template_dict,
    render_gaia_packet_template,
    MAX_INLINE_LEN,
)
from .vector_client import (
    QueryResult,
    VectorClient,
    VectorClientFactory,
)
from .helpers import (
    safe_mkdir,
    get_timestamp,
    get_timestamp_for_filename,
    ensure_parent_dir,
    normalize_path,
)
from .packet_utils import (
    is_execution_safe,
    upgrade_v2_to_v3_packet,
)
from .heartbeat_logger import HeartbeatLogger, HeartbeatLoggerProxy
from .tools_registry import TOOLS as tools_registry

__all__ = [
    # Logging
    "setup_logging",
    "get_logger",
    "UTCFormatter",
    "HealthCheckFilter",
    "install_health_check_filter",
    # Packet templates
    "packet_to_template_dict",
    "render_gaia_packet_template",
    "MAX_INLINE_LEN",
    # Vector client
    "QueryResult",
    "VectorClient",
    "VectorClientFactory",
    # Helpers
    "safe_mkdir",
    "get_timestamp",
    "get_timestamp_for_filename",
    "ensure_parent_dir",
    "normalize_path",
    # Packet utils
    "is_execution_safe",
    "upgrade_v2_to_v3_packet",
    # Heartbeat logging
    "HeartbeatLogger",
    "HeartbeatLoggerProxy",
    "tools_registry", # Export TOOLS as tools_registry
]
