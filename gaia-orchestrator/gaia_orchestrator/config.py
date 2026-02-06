"""
Configuration management for GAIA Orchestrator.

Loads settings from environment variables and optional YAML config file.
"""

import os
from pathlib import Path
from typing import Optional
from pydantic import Field
from pydantic_settings import BaseSettings
import yaml


class OrchestratorConfig(BaseSettings):
    """Configuration for the GAIA Orchestrator service."""

    # Service settings
    host: str = Field(default="0.0.0.0", description="Host to bind to")
    port: int = Field(default=6410, description="Port to listen on")
    debug: bool = Field(default=False, description="Enable debug mode")

    # State persistence
    state_dir: Path = Field(
        default=Path("/shared/orchestrator"),
        description="Directory for state persistence"
    )
    state_file: str = Field(default="state.json", description="State filename")

    # Service endpoints
    core_url: str = Field(
        default="http://gaia-core:6415",
        description="gaia-core endpoint"
    )
    core_candidate_url: str = Field(
        default="http://gaia-core-candidate:6416",
        description="gaia-core-candidate endpoint"
    )
    web_url: str = Field(
        default="http://gaia-web:6414",
        description="gaia-web endpoint"
    )
    study_url: str = Field(
        default="http://gaia-study:8766",
        description="gaia-study endpoint"
    )
    mcp_url: str = Field(
        default="http://gaia-mcp:8765",
        description="gaia-mcp endpoint"
    )

    # Docker settings
    docker_socket: str = Field(
        default="/var/run/docker.sock",
        description="Docker socket path"
    )
    compose_file_live: Path = Field(
        default=Path("/gaia/GAIA_Project/docker-compose.yml"),
        description="Live stack compose file"
    )
    compose_file_candidate: Path = Field(
        default=Path("/gaia/GAIA_Project/docker-compose.candidate.yml"),
        description="Candidate stack compose file"
    )

    # GPU settings
    gpu_cleanup_threshold_mb: int = Field(
        default=500,
        description="VRAM usage (MB) below which GPU is considered released"
    )
    gpu_cleanup_timeout_seconds: int = Field(
        default=30,
        description="Max time to wait for CUDA cleanup"
    )
    gpu_cleanup_poll_interval: float = Field(
        default=1.0,
        description="Seconds between VRAM checks during cleanup"
    )

    # Handoff settings
    handoff_timeout_seconds: int = Field(
        default=120,
        description="Default timeout for handoff operations"
    )

    # Timeouts
    http_timeout_seconds: float = Field(
        default=30.0,
        description="HTTP client timeout for service calls"
    )

    class Config:
        env_prefix = "ORCHESTRATOR_"
        env_file = ".env"


def load_yaml_config(config_path: Optional[Path] = None) -> dict:
    """Load configuration from YAML file if it exists."""
    if config_path is None:
        config_path = Path(__file__).parent.parent / "config" / "orchestrator.yaml"

    if config_path.exists():
        with open(config_path) as f:
            return yaml.safe_load(f) or {}
    return {}


_config: Optional[OrchestratorConfig] = None


def get_config() -> OrchestratorConfig:
    """Get the singleton configuration instance."""
    global _config
    if _config is None:
        # Load YAML config as defaults, then override with env vars
        yaml_config = load_yaml_config()
        _config = OrchestratorConfig(**yaml_config)
    return _config


def reset_config() -> None:
    """Reset config singleton (useful for testing)."""
    global _config
    _config = None
