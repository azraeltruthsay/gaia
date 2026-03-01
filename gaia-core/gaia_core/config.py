from __future__ import annotations
import logging
from typing import Dict, Any, Optional, Tuple
from gaia_common.config import Config as CommonConfig, get_config as get_common_config

logger = logging.getLogger("GAIA.Config")

class Config(CommonConfig):
    """
    Core-specific configuration wrapper.
    Inherits authoritative settings from gaia-common and adds core-specific properties.
    """
    @property
    def KNOWLEDGE_CODEX_DIR(self) -> str:
        return self.KNOWLEDGE_DIR

    @property
    def LORA_ADAPTERS_DIR(self) -> str:
        return f"{self.MODELS_DIR}/lora_adapters"

    @property
    def CODEX_ALLOW_HOT_RELOAD(self) -> bool:
        return True

    # Sleep Cycle helpers
    @property
    def SLEEP_ENABLED(self) -> bool:
        return self.SLEEP_CYCLE.get("enabled", True)

    @property
    def SLEEP_IDLE_THRESHOLD_MINUTES(self) -> int:
        return self.SLEEP_CYCLE.get("idle_threshold_minutes", 30)

    @property
    def SLEEP_ENABLE_QLORA(self) -> bool:
        return self.SLEEP_CYCLE.get("enable_qlora", False)

    @property
    def SLEEP_ENABLE_DREAM(self) -> bool:
        return self.SLEEP_CYCLE.get("enable_dream", False)

    @property
    def SLEEP_TASK_TIMEOUT(self) -> int:
        return self.SLEEP_CYCLE.get("task_timeout_seconds", 600)

    # Temporal Awareness helpers
    @property
    def LITE_JOURNAL_ENABLED(self) -> bool:
        return self.TEMPORAL_AWARENESS.get("enabled", True)

    @property
    def TEMPORAL_STATE_ENABLED(self) -> bool:
        return self.TEMPORAL_AWARENESS.get("enabled", True)

    @property
    def TEMPORAL_BAKE_INTERVAL_TICKS(self) -> int:
        return self.TEMPORAL_AWARENESS.get("bake_interval_ticks", 3)

    @property
    def TEMPORAL_STATE_MAX_FILES(self) -> int:
        return self.TEMPORAL_AWARENESS.get("max_state_files", 5)

    @property
    def TEMPORAL_STATE_BAKE_CONTEXT_TOKENS(self) -> int:
        return self.TEMPORAL_AWARENESS.get("bake_context_tokens", 6000)

    @property
    def TEMPORAL_INTERVIEW_ENABLED(self) -> bool:
        return self.TEMPORAL_AWARENESS.get("interview_enabled", True)

    @property
    def TEMPORAL_INTERVIEW_INTERVAL_TICKS(self) -> int:
        return self.TEMPORAL_AWARENESS.get("interview_interval_ticks", 6)

    @property
    def TEMPORAL_INTERVIEW_ROUNDS(self) -> int:
        return self.TEMPORAL_AWARENESS.get("interview_rounds", 3)

    @property
    def HEARTBEAT_INTERVAL_SECONDS(self) -> int:
        return 1200 # Fixed for now or could move to SYSTEM

    @property
    def HEARTBEAT_ENABLED(self) -> bool:
        return True

    @property
    def use_oracle(self) -> bool:
        return self.constants.get("use_oracle", False)

    def get_persona_instructions(self) -> str:
        """Return default persona instructions from constants or fallback."""
        return (
            self.constants.get("persona_defaults", {}).get("instructions")
            or "You are GAIA, a General Artisanal Intelligence. Assist with integrity and care."
        )

    def get_model_name(self, model_alias: str) -> str:
        return self.MODEL_CONFIGS.get(model_alias, {}).get("model")

    def _load_cheat_sheet(self):
        """Loads the cheat sheet JSON file."""
        import json
        path = f"{self.KNOWLEDGE_DIR}/system_reference/cheat_sheet.json"
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}

def get_config() -> Config:
    """Get the core-wrapped authoritative config."""
    return Config.get_instance()
