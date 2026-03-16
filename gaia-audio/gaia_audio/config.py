"""Audio service configuration — loads from gaia_constants.json."""

from __future__ import annotations
import logging
from dataclasses import dataclass
from gaia_common.config import Config as CommonConfig

logger = logging.getLogger("GAIA.Audio.Config")

@dataclass
class CloudFallback:
    provider: str = "elevenlabs"
    enabled: bool = False
    api_key_env: str = "ELEVENLABS_API_KEY"

class AudioConfig(CommonConfig):
    """
    Audio-specific configuration wrapper.
    Inherits authoritative settings from gaia-common and adds audio-specific properties.
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

    # ── Three-Tier Model Paths ──────────────────────────────────────

    @property
    def listener_model_path(self) -> str:
        return self.audio_cfg.get("listener_model_path") or self.model_path("audio", "stt") or "/models/Qwen3-ASR-0.6B"

    @property
    def listener_device(self) -> str:
        return self.audio_cfg.get("listener_device", "auto")

    @property
    def nano_speaker_model_path(self) -> str:
        return self.audio_cfg.get("nano_speaker_model_path") or self.model_path("audio", "tts_nano") or "/models/Qwen3-TTS-12Hz-0.6B-Base"

    @property
    def prime_speaker_model_path(self) -> str:
        return self.audio_cfg.get("prime_speaker_model_path") or self.model_path("audio", "tts_prime") or "/models/Qwen3-TTS-12Hz-1.7B-Base"

    @property
    def voice_ref_audio(self) -> str | None:
        return self.audio_cfg.get("voice_ref_audio")

    @property
    def voice_ref_text(self) -> str | None:
        return self.audio_cfg.get("voice_ref_text")

    @property
    def tts_auto_threshold(self) -> int:
        """Character count: below = Nano Speaker, above = Prime Speaker."""
        return self.audio_cfg.get("tts_auto_threshold", 200)

    @property
    def prime_speaker_timeout(self) -> int:
        """Seconds to wait for GPU before Nano fallback."""
        return self.audio_cfg.get("prime_speaker_timeout", 30)

    # ── General Audio Settings ──────────────────────────────────────

    @property
    def sample_rate(self) -> int:
        return self.audio_cfg.get("sample_rate", 24000)

    @property
    def vram_budget_mb(self) -> int:
        return self.audio_cfg.get("vram_budget_mb", 16000)

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

    # Note: Service endpoints are now available via base.get_endpoint()

    @classmethod
    def from_constants(cls) -> AudioConfig:
        """Get the authoritative instance."""
        return cls.get_instance()

def get_config() -> AudioConfig:
    """Get the audio-wrapped authoritative config."""
    return AudioConfig.get_instance()
