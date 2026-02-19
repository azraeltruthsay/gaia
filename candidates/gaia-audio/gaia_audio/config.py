"""Audio service configuration — loads from gaia_constants.json."""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field

logger = logging.getLogger("GAIA.Audio.Config")

_DEFAULT_CONSTANTS_PATH = "/app/gaia_common/constants/gaia_constants.json"


@dataclass
class CloudFallback:
    provider: str = "elevenlabs"
    enabled: bool = False
    api_key_env: str = "ELEVENLABS_API_KEY"


@dataclass
class AudioConfig:
    enabled: bool = True
    endpoint: str = "http://gaia-audio:8080"
    stt_model: str = "base.en"
    tts_engine: str = "system"
    tts_voice: str | None = None
    sample_rate: int = 16000
    vram_budget_mb: int = 5600
    half_duplex: bool = True
    mute_on_sleep: bool = True
    cloud_fallback: CloudFallback = field(default_factory=CloudFallback)

    # Service endpoints for integration
    core_endpoint: str = "http://gaia-core:6415"
    web_endpoint: str = "http://gaia-web:6414"
    orchestrator_endpoint: str = "http://gaia-orchestrator:6410"

    @classmethod
    def from_constants(cls, path: str | None = None) -> AudioConfig:
        """Load audio config from gaia_constants.json INTEGRATIONS.audio block."""
        path = path or os.environ.get("GAIA_CONSTANTS_PATH", _DEFAULT_CONSTANTS_PATH)
        try:
            with open(path, encoding="utf-8") as f:
                constants = json.load(f)
            audio_cfg = constants.get("INTEGRATIONS", {}).get("audio", {})
            if not audio_cfg:
                logger.warning("No INTEGRATIONS.audio in constants; using defaults")
                return cls()

            cloud = audio_cfg.get("cloud_fallback", {})
            return cls(
                enabled=audio_cfg.get("enabled", True),
                endpoint=audio_cfg.get("endpoint", cls.endpoint),
                stt_model=audio_cfg.get("stt_model", cls.stt_model),
                tts_engine=audio_cfg.get("tts_engine", cls.tts_engine),
                tts_voice=audio_cfg.get("tts_voice"),
                sample_rate=audio_cfg.get("sample_rate", cls.sample_rate),
                vram_budget_mb=audio_cfg.get("vram_budget_mb", cls.vram_budget_mb),
                half_duplex=audio_cfg.get("half_duplex", cls.half_duplex),
                mute_on_sleep=audio_cfg.get("mute_on_sleep", cls.mute_on_sleep),
                cloud_fallback=CloudFallback(
                    provider=cloud.get("provider", "elevenlabs"),
                    enabled=cloud.get("enabled", False),
                    api_key_env=cloud.get("api_key_env", "ELEVENLABS_API_KEY"),
                ),
            )
        except FileNotFoundError:
            logger.warning("Constants file not found at %s; using defaults", path)
            return cls()
        except Exception:
            logger.error("Failed to load audio config", exc_info=True)
            return cls()
