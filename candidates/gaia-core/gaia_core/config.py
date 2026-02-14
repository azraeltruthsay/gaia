from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, Tuple
import json
import os
import logging
from pathlib import Path

logger = logging.getLogger("GAIA.Config")

@dataclass
class Config:
    """
    A simplified configuration class for gaia-core.
    Values should be injected at runtime or loaded from gaia_constants.json.
    """
    # Model configurations
    MODEL_CONFIGS: Dict[str, Any] = field(default_factory=dict)

    # Paths
    MODELS_DIR: str = "/models"
    MODEL_DIR: str = "/models"  # Alias for MODELS_DIR (legacy compatibility)
    KNOWLEDGE_DIR: str = "/knowledge"
    KNOWLEDGE_CODEX_DIR: str = "/knowledge"
    SYSTEM_REF_DIR: str = "/knowledge/system_reference"
    PERSONAS_DIR: str = "/knowledge/personas"
    LOGS_DIR: str = "/logs"
    HISTORY_DIR: str = field(default_factory=lambda: os.getenv("HISTORY_DIR", "/shared/history"))
    LORA_ADAPTERS_DIR: str = "/models/lora_adapters"
    SHARED_DIR: str = field(default_factory=lambda: os.getenv("SHARED_DIR", "/shared"))
    EMBEDDING_MODEL_PATH: Optional[str] = None
    identity_file_path: str = "/knowledge/system_reference/core_identity.json"
    system_reference_path: str = "/knowledge/system_reference"
    cheat_sheet_path: Path = Path("/knowledge/system_reference/cheat_sheet.json")
    cheat_sheet: dict = field(default_factory=dict)
    CODEX_FILE_EXTS: Tuple[str, ...] = field(default_factory=lambda: (".yaml", ".yml", ".json", ".md"))
    CODEX_ALLOW_HOT_RELOAD: bool = True

    # LLM backend settings
    llm_backend: str = "vllm"
    n_gpu_layers: int = -1  # -1 means all layers on GPU
    temperature: float = 0.4
    top_p: float = 0.95
    max_tokens: int = 4096
    MAX_TOKENS: int = 4096
    max_tokens_lite: int = 16000
    RESPONSE_BUFFER: int = 768

    # Feature flags
    use_oracle: bool = False

    # Sleep cycle settings (loaded from gaia_constants.json SLEEP_CYCLE section)
    SLEEP_ENABLED: bool = True
    SLEEP_IDLE_THRESHOLD_MINUTES: int = 5
    SLEEP_CHECKPOINT_DIR: str = "/shared/sleep_state"
    SLEEP_ENABLE_QLORA: bool = False
    SLEEP_ENABLE_DREAM: bool = False
    SLEEP_TASK_TIMEOUT: int = 600

    # Tool and primitive settings
    primitives: list[str] = field(default_factory=lambda: ["read", "write", "vector_query", "shell"])
    SAFE_EXECUTE_FUNCTIONS: list[str] = field(default_factory=list)

    # Runtime constants (loaded from gaia_constants.json or similar)
    constants: Dict[str, Any] = field(default_factory=dict)

    # Singleton instance
    _instance: Optional[Config] = None

    def __post_init__(self):
        """Load constants from gaia_constants.json if available."""
        self._load_constants()
        self.cheat_sheet_path = Path(self.SYSTEM_REF_DIR) / "cheat_sheet.json"
        self.cheat_sheet = self._load_cheat_sheet()

    def _load_constants(self):
        """Load configuration from gaia_constants.json.

        Canonical location is gaia-common/constants/gaia_constants.json.
        The gaia-common copy is checked first so that all services share
        one source of truth for feature flags and system prompts.
        """
        # Try multiple paths — prefer the canonical gaia-common location
        possible_paths = [
            # Live container: gaia-common mounted at /gaia-common
            "/gaia-common/gaia_common/constants/gaia_constants.json",
            # Candidate container: gaia-common mounted at /app/gaia-common
            "/app/gaia-common/gaia_common/constants/gaia_constants.json",
            os.path.join(os.path.dirname(__file__), "..", "gaia-common", "gaia_common", "constants", "gaia_constants.json"),
            "/app/gaia_common/constants/gaia_constants.json",
            os.path.join(os.path.dirname(__file__), "gaia_constants.json"),
            "/app/gaia_core/gaia_constants.json",
            "/app/gaia_constants.json",
            os.path.expanduser("~/gaia_constants.json"),
        ]

        for path in possible_paths:
            if os.path.exists(path):
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    self.constants = data
                    # Extract MODEL_CONFIGS if present
                    if "MODEL_CONFIGS" in data:
                        self.MODEL_CONFIGS = data["MODEL_CONFIGS"]
                    # Extract other common settings
                    if "llm_backend" in data:
                        self.llm_backend = data["llm_backend"]
                    if "SAFE_EXECUTE_FUNCTIONS" in data:
                        self.SAFE_EXECUTE_FUNCTIONS = data["SAFE_EXECUTE_FUNCTIONS"]
                    if "max_tokens" in data:
                        self.max_tokens = data["max_tokens"]
                    if "max_tokens_lite" in data:
                        self.max_tokens_lite = data["max_tokens_lite"]
                    if "RESPONSE_BUFFER" in data:
                        self.RESPONSE_BUFFER = data["RESPONSE_BUFFER"]
                    if "model_paths" in data and "Embedding" in data["model_paths"]:
                        self.EMBEDDING_MODEL_PATH = data.get("model_paths", {}).get("Embedding", os.getenv("EMBEDDING_MODEL_PATH"))
                    if "CODEX_FILE_EXTS" in data:
                        self.CODEX_FILE_EXTS = tuple(data["CODEX_FILE_EXTS"])
                    # Sleep cycle settings
                    sleep_cfg = data.get("SLEEP_CYCLE", {})
                    if sleep_cfg:
                        self.SLEEP_ENABLED = sleep_cfg.get("enabled", self.SLEEP_ENABLED)
                        self.SLEEP_IDLE_THRESHOLD_MINUTES = sleep_cfg.get("idle_threshold_minutes", self.SLEEP_IDLE_THRESHOLD_MINUTES)
                        self.SLEEP_CHECKPOINT_DIR = sleep_cfg.get("checkpoint_dir", self.SLEEP_CHECKPOINT_DIR)
                        self.SLEEP_ENABLE_QLORA = sleep_cfg.get("enable_qlora", self.SLEEP_ENABLE_QLORA)
                        self.SLEEP_ENABLE_DREAM = sleep_cfg.get("enable_dream", self.SLEEP_ENABLE_DREAM)
                        self.SLEEP_TASK_TIMEOUT = sleep_cfg.get("task_timeout_seconds", self.SLEEP_TASK_TIMEOUT)
                    logger.info(f"Loaded GAIA constants from {path}")
                    return
                except Exception as e:
                    logger.warning(f"Failed to load constants from {path}: {e}")

        logger.warning("No gaia_constants.json found; using defaults")

    def get_api_key(self, provider: str) -> str:
        import os
        return os.getenv(f"{provider.upper()}_API_KEY")

    def get_persona_instructions(self) -> str:
        """Return default persona instructions from constants or fallback."""
        return (
            self.constants.get("persona_defaults", {}).get("instructions")
            or "You are GAIA, a General Artisanal Intelligence. Assist with integrity and care."
        )

    def get_model_name(self, model_alias: str) -> str:
        return self.MODEL_CONFIGS.get(model_alias, {}).get("model")

    @classmethod
    def get_instance(cls) -> Config:
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def _load_cheat_sheet(self):
        """Loads the cheat sheet JSON file."""
        try:
            with open(self.cheat_sheet_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except FileNotFoundError:
            logger.warning(f"Cheat sheet not found at {self.cheat_sheet_path}. Returning empty dict.")
            return {}
        except json.JSONDecodeError as e:
            logger.error(f"Failed to decode cheat sheet JSON from {self.cheat_sheet_path}: {e}")
            return {}
        except Exception as e:
            logger.error(f"Failed to load cheat sheet from {self.cheat_sheet_path}: {e}")
            return {}


def get_config() -> Config:
    """Get the singleton Config instance."""
    return Config.get_instance()
