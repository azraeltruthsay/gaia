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
    INTEGRATIONS: Dict[str, Any] = field(default_factory=dict)
    SAFE_EXECUTE_FUNCTIONS: List[str] = field(default_factory=list)
    CODEX_FILE_EXTS: Tuple[str, ...] = field(default_factory=lambda: (".md", ".yaml", ".yml", ".json"))

    # Singleton instance
    _instance: Optional[Config] = None

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
        for path in possible_paths:
            if path and os.path.exists(path):
                try:
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

    @classmethod
    def get_instance(cls) -> Config:
        """Get the singleton Config instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

def get_config() -> Config:
    """Get the singleton Config instance."""
    return Config.get_instance()
