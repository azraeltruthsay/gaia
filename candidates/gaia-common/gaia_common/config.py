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
    llm_backend: str = "prime"
    n_gpu_layers: int = -1
    temperature: float = 0.7
    top_p: float = 0.95
    max_tokens: int = 8192
    max_tokens_lite: int = 8192
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

    # ── Inference Endpoints (engine-level, not service-level) ──────
    inference_endpoints: Dict[str, str] = field(default_factory=lambda: {
        "nano": "http://gaia-nano:8080",
        "core": "http://gaia-core:8092",
        "prime": "http://gaia-prime:7777"
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

    # Singleton instance
    _instance: Optional[Config] = None
    _last_mtime: float = 0.0
    _source_path: Optional[str] = None
    _last_check_time: float = 0.0

    def __post_init__(self):
        """Initialize by loading from file and then applying environment overrides."""
        self._load_service_configs()
        self._load_from_json()
        self._apply_env_overrides()

    # ── Decentralized Config (Phase 5-C, Proposal 07) ─────────────────

    def _load_service_configs(self):
        """Load per-service config.json files into self.constants.

        Each service can define its own config.json with service-specific
        constants. These are loaded BEFORE gaia_constants.json so the
        monolith can still act as a global emergency override.

        Search order per service:
          1. /app/config.json (inside container, current service)
          2. GAIA_SERVICE_CONFIG env var
          3. Peer service configs from project root (cross-service aggregation)
        """
        aggregated = {}
        loaded_from = []

        # 1. Current service's own config.json
        own_config_paths = [
            "/app/config.json",
            os.environ.get("GAIA_SERVICE_CONFIG", ""),
        ]
        for path in own_config_paths:
            if path and os.path.exists(path):
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        own_data = json.load(f)
                    aggregated.update(own_data)
                    loaded_from.append(path)
                    break
                except Exception as e:
                    logger.warning("Failed to load service config from %s: %s", path, e)

        # 2. Aggregate peer service configs (project root layout)
        #    Only in development mode or when GAIA_PROJECT_ROOT is set.
        project_root = os.environ.get("GAIA_PROJECT_ROOT", "")
        if not project_root:
            # Try to infer from common layout
            for candidate in [
                "/gaia/GAIA_Project",
                os.path.join(os.path.dirname(__file__), "..", ".."),
            ]:
                if os.path.isdir(candidate):
                    project_root = candidate
                    break

        if project_root:
            service_dirs = [
                "gaia-core", "gaia-mcp", "gaia-prime",
                "gaia-study", "gaia-web", "gaia-audio",
                "gaia-orchestrator",
            ]
            for svc in service_dirs:
                svc_config = os.path.join(project_root, svc, "config.json")
                if os.path.exists(svc_config):
                    try:
                        with open(svc_config, "r", encoding="utf-8") as f:
                            svc_data = json.load(f)
                        aggregated.update(svc_data)
                        loaded_from.append(svc_config)
                    except Exception as e:
                        logger.debug("Failed to load %s: %s", svc_config, e)

            # Also check candidates/ for dev mode
            for svc in service_dirs:
                cand_config = os.path.join(project_root, "candidates", svc, "config.json")
                if os.path.exists(cand_config):
                    try:
                        with open(cand_config, "r", encoding="utf-8") as f:
                            cand_data = json.load(f)
                        # Candidate configs override production
                        aggregated.update(cand_data)
                        loaded_from.append(cand_config)
                    except Exception as e:
                        logger.debug("Failed to load %s: %s", cand_config, e)

        if aggregated:
            self.constants = aggregated
            logger.info(
                "Loaded %d decentralized config keys from %d service configs",
                len(aggregated), len(loaded_from),
            )

    def _load_from_json(self):
        """Load configuration from gaia_constants.json (global override).

        Runs AFTER _load_service_configs so the monolith can override
        any per-service value. This is the "emergency patch" path —
        as services migrate their constants, gaia_constants.json shrinks
        toward containing only global/shared values.
        """
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
            if not self.constants:
                logger.warning("No gaia_constants.json found and no service configs loaded; using hardcoded defaults")
            return

        # Merge: global constants override per-service values
        self.constants.update(data)
        self._model_registry = self.constants.get("MODEL_REGISTRY", {})
        logger.info(f"Loaded GAIA constants from {found_path} (global override layer)")

        # Map top-level keys from merged constants (service configs + monolith)
        c = self.constants
        self.MODEL_CONFIGS = c.get("MODEL_CONFIGS", self.MODEL_CONFIGS)
        self.llm_backend = c.get("llm_backend", self.llm_backend)
        self.max_tokens = c.get("max_tokens", self.max_tokens)
        self.max_tokens_lite = c.get("max_tokens_lite", self.max_tokens_lite)
        self.n_gpu_layers = c.get("n_gpu_layers", self.n_gpu_layers)
        self.SAFE_EXECUTE_FUNCTIONS = c.get("SAFE_EXECUTE_FUNCTIONS", self.SAFE_EXECUTE_FUNCTIONS)

        # Map sections
        self.SLEEP_CYCLE = c.get("SLEEP_CYCLE", {})
        self.TEMPORAL_AWARENESS = c.get("TEMPORAL_AWARENESS", {})
        self.FRAGMENTATION = c.get("fragmentation", {})
        self.INTEGRATIONS = c.get("INTEGRATIONS", {})
        self.endpoints.update(c.get("SERVICE_ENDPOINTS", {}))
        self.inference_endpoints.update(c.get("INFERENCE_ENDPOINTS", {}))
        self.timeouts.update(c.get("TIMEOUTS", {}))

        # Derive EMBEDDING_MODEL_PATH from registry if available
        emb = self._model_registry.get("embedding")
        if emb and isinstance(emb, str):
            self.EMBEDDING_MODEL_PATH = emb

        # Map System Paths (Defensively skip read-only properties)
        sys_cfg = c.get("SYSTEM", {})
        for key, value in sys_cfg.items():
            prop = getattr(self.__class__, key, None)
            if isinstance(prop, property):
                logger.debug(f"Config: skipping property '{key}' in auto-mapping")
                continue
            if hasattr(self, key):
                setattr(self, key, value)

    def model_path(self, role: str, variant: str = "merged") -> str:
        """Lookup a model path from MODEL_REGISTRY.

        Examples::

            model_path("prime", "merged")  -> "/models/Qwen3.5-4B-Abliterated-merged"
            model_path("nano", "gguf")     -> "/models/Qwen3.5-0.8B-Abliterated-Q8_0.gguf"
            model_path("embedding")        -> "/models/all-MiniLM-L6-v2"
            model_path("lora_adapters")    -> "/models/lora_adapters"
            model_path("audio", "stt")     -> "/models/Qwen3-ASR-0.6B"
        """
        entry = self._model_registry.get(role, {})
        if isinstance(entry, str):
            return entry
        return entry.get(variant, "")

    def _apply_env_overrides(self):
        """Apply high-priority environment variable overrides."""
        # 1. Backend & Inference — normalize legacy names to canonical
        backend = os.getenv("GAIA_BACKEND", self.llm_backend)
        _BACKEND_COMPAT = {"gpu_prime": "prime", "cpu_prime": "prime", "thinker": "prime"}
        self.llm_backend = _BACKEND_COMPAT.get(backend, backend)

        # 2. Remote vLLM override (Standardizes PRIME_ENDPOINT usage)
        prime_endpoint = os.getenv("PRIME_ENDPOINT")
        if prime_endpoint:
            self.endpoints["prime"] = prime_endpoint
            if "prime" in self.MODEL_CONFIGS:
                self.MODEL_CONFIGS["prime"]["endpoint"] = prime_endpoint
                self.MODEL_CONFIGS["prime"]["type"] = "vllm_remote"

        # 3. Port/Endpoint overrides
        for svc in self.endpoints:
            env_key = f"GAIA_{svc.upper()}_ENDPOINT"
            self.endpoints[svc] = os.getenv(env_key, self.endpoints[svc])

        # 4. Path overrides
        self.MODELS_DIR = os.getenv("MODELS_DIR", self.MODELS_DIR)
        self.KNOWLEDGE_DIR = os.getenv("KNOWLEDGE_DIR", self.KNOWLEDGE_DIR)

        # 5. Model Window overrides (critical for VRAM management)
        vllm_max = os.getenv("VLLM_MAX_MODEL_LEN")
        if vllm_max and "prime" in self.MODEL_CONFIGS:
            self.MODEL_CONFIGS["prime"]["max_model_len"] = int(vllm_max)

    def get_endpoint(self, service: str) -> str:
        """Get the endpoint for a specific service."""
        return self.endpoints.get(service, "")

    def get_inference_endpoint(self, tier: str) -> str:
        """Get the inference engine endpoint for a tier (nano/core/prime)."""
        return self.inference_endpoints.get(tier, "")

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
