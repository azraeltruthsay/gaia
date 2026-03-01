"""Audio service configuration — loads from gaia_constants.json."""

from __future__ import annotations
import logging
from dataclasses import dataclass, field
from gaia_common.config import Config as CommonConfig, get_config as get_common_config

logger = logging.getLogger("GAIA.Audio.Config")

@dataclass
class CloudFallback:
    provider: str = "elevenlabs"
    enabled: bool = False
    api_key_env: str = "ELEVENLABS_API_KEY"

class AudioConfig(CommonConfig):
    """
    Audio-specific configuration wrapper.
    Inherits authoritative settings from gaia-common and adds audio helpers.
    """
    @property
    def audio_cfg(self) -> dict:
        return self.INTEGRATIONS.get("audio", {})

    @property
    def enabled(self) -> bool:
        return self.audio_cfg.get("enabled", True)

    @property
    def endpoint(self) -> str:
        return self.audio_cfg.get("endpoint", "http://gaia-audio:8080")

    @property
    def stt_model(self) -> str:
        return self.audio_cfg.get("stt_model", "base.en")

    @property
    def tts_engine(self) -> str:
        return self.audio_cfg.get("tts_engine", "coqui")

    @property
    def tts_voice(self) -> str | None:
        return self.audio_cfg.get("tts_voice")

    @property
    def sample_rate(self) -> int:
        return self.audio_cfg.get("sample_rate", 16000)

    @property
    def vram_budget_mb(self) -> int:
        return self.audio_cfg.get("vram_budget_mb", 5600)

    @property
    def half_duplex(self) -> bool:
        return self.audio_cfg.get("half_duplex", False)

    @property
    def mute_on_sleep(self) -> bool:
        return self.audio_cfg.get("mute_on_sleep", True)

    @property
    def cloud_fallback(self) -> CloudFallback:
        cloud = self.audio_cfg.get("cloud_fallback", {})
        return CloudFallback(
            provider=cloud.get("provider", "elevenlabs"),
            enabled=cloud.get("enabled", False),
            api_key_env=cloud.get("api_key_env", "ELEVENLABS_API_KEY"),
        )

    # Service endpoints
    @property
    def core_endpoint(self) -> str:
        return self.endpoints.get("core", "http://gaia-core:6415")

    @property
    def web_endpoint(self) -> str:
        return self.endpoints.get("web", "http://gaia-web:6414")

    @property
    def orchestrator_endpoint(self) -> str:
        return self.endpoints.get("orchestrator", "http://gaia-orchestrator:6410")

    @classmethod
    def from_constants(cls) -> AudioConfig:
        """Get the authoritative instance."""
        return cls.get_instance()

def get_config() -> AudioConfig:
    """Get the audio-wrapped authoritative config."""
    return AudioConfig.get_instance()
