from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Tuple
import json
import os
import logging
from pathlib import Path

logger = logging.getLogger("GAIA.Config")

@dataclass
class Config:
    """
    Authoritative configuration loader for all GAIA services.
    Loads from gaia_constants.json and applies environment variable overrides.
    """
    # ── Master Dictionary ──────────────────────────────────────────
    constants: Dict[str, Any] = field(default_factory=dict)

    # ── Model Settings ──────────────────────────────────────────────
    MODEL_CONFIGS: Dict[str, Any] = field(default_factory=dict)
    llm_backend: str = "gpu_prime"
    n_gpu_layers: int = -1
    temperature: float = 0.7
    top_p: float = 0.95
    max_tokens: int = 8192
    max_tokens_lite: int = 32000
    RESPONSE_BUFFER: int = 768

    # ── System Paths ────────────────────────────────────────────────
    MODELS_DIR: str = "/models"
    KNOWLEDGE_DIR: str = "/knowledge"
    LOGS_DIR: str = "/logs"
    SHARED_DIR: str = "/shared"
    SLEEP_CHECKPOINT_DIR: str = "/shared/sleep_state"
    HISTORY_DIR: str = "/shared/history"
    IDENTITY_FILE: str = "/knowledge/system_reference/core_identity.json"
    CHEAT_SHEET_FILE: str = "/knowledge/system_reference/cheat_sheet.json"
    EMBEDDING_MODEL_PATH: str = "/models/all-MiniLM-L6-v2"

    # ── Service Endpoints ───────────────────────────────────────────
    endpoints: Dict[str, str] = field(default_factory=lambda: {
        "core": "http://gaia-core:6415",
        "web": "http://gaia-web:6414",
        "prime": "http://gaia-prime:7777",
        "mcp": "http://gaia-mcp:8765/jsonrpc",
        "study": "http://gaia-study:8766",
        "audio": "http://gaia-audio:8080",
        "orchestrator": "http://gaia-orchestrator:6410"
    })

    # ── Timeouts ────────────────────────────────────────────────────
    timeouts: Dict[str, float] = field(default_factory=lambda: {
        "HTTP_DEFAULT": 30.0,
        "HTTP_QUICK": 5.0,
        "LLM_PLANNING": 60.0,
        "LLM_REFLECTION": 90.0,
        "LLM_AUDIT": 15.0,
        "MCP_DEFAULT": 20.0,
        "SLEEP_BOOT_PRIME": 180.0,
        "SLEEP_STUDY_CLEANUP": 300.0
    })

    # ── Feature Flags & Sub-configs ────────────────────────────────
    SLEEP_CYCLE: Dict[str, Any] = field(default_factory=dict)
    TEMPORAL_AWARENESS: Dict[str, Any] = field(default_factory=dict)
    FRAGMENTATION: Dict[str, Any] = field(default_factory=dict)
    INTEGRATIONS: Dict[str, Any] = field(default_factory=dict)
    SAFE_EXECUTE_FUNCTIONS: List[str] = field(default_factory=list)
    CODEX_FILE_EXTS: Tuple[str, ...] = field(default_factory=lambda: (".md", ".yaml", ".yml", ".json"))

    # ── Generic GAIA Properties (Consolidated) ──────────────────────
    @property
    def KNOWLEDGE_CODEX_DIR(self) -> str:
        return self.KNOWLEDGE_DIR

    @property
    def LORA_ADAPTERS_DIR(self) -> str:
        return f"{self.MODELS_DIR}/lora_adapters"

    @property
    def PERSONAS_DIR(self) -> str:
        return f"{self.KNOWLEDGE_DIR}/personas"

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
        return 1200 # Default

    @property
    def HEARTBEAT_ENABLED(self) -> bool:
        return True

    @property
    def identity_file_path(self) -> str:
        return self.IDENTITY_FILE

    @property
    def system_reference_path(self) -> str:
        return f"{self.KNOWLEDGE_DIR}/system_reference"

    @property
    def cheat_sheet_path(self) -> str:
        return self.CHEAT_SHEET_FILE

    @property
    def cheat_sheet(self) -> Dict[str, Any]:
        return self._load_cheat_sheet()

    def _load_cheat_sheet(self) -> Dict[str, Any]:
        """Loads the cheat sheet JSON file."""
        if os.path.exists(self.CHEAT_SHEET_FILE):
            try:
                with open(self.CHEAT_SHEET_FILE, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass
        return {}

    @property
    def use_oracle(self) -> bool:
        return self.constants.get("use_oracle", False)

    @property
    def max_tokens_operator(self) -> int:
        return self.constants.get("max_tokens_operator") or self.max_tokens_lite

    @property
    def LITE_IS_OPERATOR(self) -> bool:
        """Formal role shift: Lite is now the Cognitive Operator."""
        return True

    def get_persona_instructions(self) -> str:
        """Return default persona instructions from constants or fallback."""
        return (
            self.constants.get("persona_defaults", {}).get("instructions")
            or "You are GAIA, a General Artisanal Intelligence. Assist with integrity and care."
        )

    def get_model_name(self, model_alias: str) -> str:
        return self.MODEL_CONFIGS.get(model_alias, {}).get("model")

    # Singleton instance
    _instance: Optional[Config] = None
    _last_mtime: float = 0.0
    _source_path: Optional[str] = None
    _last_check_time: float = 0.0

    def __post_init__(self):
        """Initialize by loading from file and then applying environment overrides."""
        self._load_from_json()
        self._apply_env_overrides()

    def _load_from_json(self):
        """Load configuration from gaia_constants.json using hierarchical search."""
        possible_paths = [
            # Inside containers (GAIA_CONSTANTS_PATH env or canonical mounts)
            os.environ.get("GAIA_CONSTANTS_PATH", ""),
            "/gaia-common/gaia_common/constants/gaia_constants.json",
            "/app/gaia_common/constants/gaia_constants.json",
            # Host-side: relative to project layout
            os.path.join(os.path.dirname(__file__), "constants", "gaia_constants.json"),
            "gaia-common/gaia_common/constants/gaia_constants.json",
        ]

        data = {}
        found_path = None
        import time
        for path in possible_paths:
            if path and os.path.exists(path):
                try:
                    self._last_mtime = os.path.getmtime(path)
                    self._source_path = path
                    self._last_check_time = time.monotonic()
                    with open(path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    found_path = path
                    break
                except Exception as e:
                    logger.warning(f"Failed to load constants from {path}: {e}")

        if not data:
            logger.warning("No gaia_constants.json found; using hardcoded defaults")
            return

        self.constants = data
        logger.info(f"Loaded GAIA constants from {found_path}")

        # Map top-level keys
        self.MODEL_CONFIGS = data.get("MODEL_CONFIGS", self.MODEL_CONFIGS)
        self.llm_backend = data.get("llm_backend", self.llm_backend)
        self.max_tokens = data.get("max_tokens", self.max_tokens)
        self.max_tokens_lite = data.get("max_tokens_lite", self.max_tokens_lite)
        self.n_gpu_layers = data.get("n_gpu_layers", self.n_gpu_layers)
        self.SAFE_EXECUTE_FUNCTIONS = data.get("SAFE_EXECUTE_FUNCTIONS", self.SAFE_EXECUTE_FUNCTIONS)
        
        # Map sections
        self.SLEEP_CYCLE = data.get("SLEEP_CYCLE", {})
        self.TEMPORAL_AWARENESS = data.get("TEMPORAL_AWARENESS", {})
        self.FRAGMENTATION = data.get("fragmentation", {})
        self.INTEGRATIONS = data.get("INTEGRATIONS", {})
        self.endpoints.update(data.get("SERVICE_ENDPOINTS", {}))
        self.timeouts.update(data.get("TIMEOUTS", {}))

        # Map System Paths
        sys_cfg = data.get("SYSTEM", {})
        self.MODELS_DIR = sys_cfg.get("MODELS_DIR", self.MODELS_DIR)
        self.KNOWLEDGE_DIR = sys_cfg.get("KNOWLEDGE_DIR", self.KNOWLEDGE_DIR)
        self.LOGS_DIR = sys_cfg.get("LOGS_DIR", self.LOGS_DIR)
        self.SHARED_DIR = sys_cfg.get("SHARED_DIR", self.SHARED_DIR)
        self.SLEEP_CHECKPOINT_DIR = sys_cfg.get("SLEEP_CHECKPOINT_DIR", self.SLEEP_CHECKPOINT_DIR)
        self.HISTORY_DIR = sys_cfg.get("HISTORY_DIR", self.HISTORY_DIR)
        self.IDENTITY_FILE = sys_cfg.get("IDENTITY_FILE", self.IDENTITY_FILE)
        self.CHEAT_SHEET_FILE = sys_cfg.get("CHEAT_SHEET_FILE", self.CHEAT_SHEET_FILE)

    def _apply_env_overrides(self):
        """Apply high-priority environment variable overrides."""
        # 1. Backend & Inference
        self.llm_backend = os.getenv("GAIA_BACKEND", self.llm_backend)
        
        # 2. Remote vLLM override (Standardizes PRIME_ENDPOINT usage)
        prime_endpoint = os.getenv("PRIME_ENDPOINT")
        if prime_endpoint:
            self.endpoints["prime"] = prime_endpoint
            if "gpu_prime" in self.MODEL_CONFIGS:
                self.MODEL_CONFIGS["gpu_prime"]["endpoint"] = prime_endpoint
                self.MODEL_CONFIGS["gpu_prime"]["type"] = "vllm_remote"

        # 3. Port/Endpoint overrides
        for svc in self.endpoints:
            env_key = f"GAIA_{svc.upper()}_ENDPOINT"
            self.endpoints[svc] = os.getenv(env_key, self.endpoints[svc])

        # 4. Path overrides
        self.MODELS_DIR = os.getenv("MODELS_DIR", self.MODELS_DIR)
        self.KNOWLEDGE_DIR = os.getenv("KNOWLEDGE_DIR", self.KNOWLEDGE_DIR)
        
        # 5. Model Window overrides (critical for VRAM management)
        vllm_max = os.getenv("VLLM_MAX_MODEL_LEN")
        if vllm_max and "gpu_prime" in self.MODEL_CONFIGS:
            self.MODEL_CONFIGS["gpu_prime"]["max_model_len"] = int(vllm_max)

    def get_endpoint(self, service: str) -> str:
        """Get the endpoint for a specific service."""
        return self.endpoints.get(service, "")

    def get_timeout(self, key: str, default: float = 30.0) -> float:
        """Get a specific timeout value."""
        return float(self.timeouts.get(key, default))

    def get_api_key(self, provider: str) -> Optional[str]:
        """
        VouchCore Pattern: Get API key from system-level identity or environment.
        Prioritizes Docker secrets (/run/secrets) over environment variables.
        """
        # 1. Try Docker secrets first
        secret_path = Path(f"/run/secrets/{provider.lower()}_api_key")
        if secret_path.exists():
            try:
                return secret_path.read_text().strip()
            except Exception as e:
                logger.error(f"Failed to read Docker secret for {provider}: {e}")

        # 2. Fallback to environment variables
        return os.getenv(f"{provider.upper()}_API_KEY")

    def refresh_if_needed(self):
        """Check if the source file has changed and reload if necessary."""
        if not self._source_path or not os.path.exists(self._source_path):
            return
        
        import time
        now = time.monotonic()
        # Only check every 10 seconds to avoid excessive disk I/O
        if now - self._last_check_time < 10.0:
            return
            
        self._last_check_time = now
        try:
            current_mtime = os.path.getmtime(self._source_path)
            if current_mtime > self._last_mtime:
                logger.info(f"Detected change in {self._source_path}. Reloading config...")
                self._load_from_json()
                self._apply_env_overrides()
        except Exception as e:
            logger.error(f"Failed to check for config updates: {e}")

    @classmethod
    def get_instance(cls) -> Config:
        """Get the singleton Config instance."""
        if cls._instance is None:
            cls._instance = cls()
        else:
            # Auto-check for reload on access (debounced to 10s internally)
            cls._instance.refresh_if_needed()
        return cls._instance

def get_config() -> Config:
    """Get the singleton Config instance."""
    return Config.get_instance()
