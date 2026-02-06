# === model_pool.py (extended) ===

# Guarded import for llama_cpp to avoid blowing up dev shells that lack native libs
try:
    from llama_cpp import Llama
except Exception:
    Llama = None

from gaia_core.config import get_config, Config
import logging
from gaia_core.behavior.persona_manager import PersonaManager # New import
from gaia_core.behavior.persona_adapter import PersonaAdapter # New import
from gaia_core.utils.resource_monitor import ResourceMonitor # New import
import multiprocessing
import threading
import traceback, sys
# Guard heavy model class imports. Importing these modules at import-time can
# trigger torch/CUDA initialization (or fail when system CUDA/toolkit mismatch
# exists). We attempt to import them lazily; if unavailable we set to None and
# allow the rest of the system to continue operating in a degraded mode.
GPTAPIModel = None
GeminiAPIModel = None
HFModel = None
MCPProxyModel = None
VLLMChatModel = None
try:
    from .dev_model import DevModel
except Exception:
    DevModel = None
try:
    from .oracle_model import GPTAPIModel as _GPTAPIModel
    GPTAPIModel = _GPTAPIModel
except Exception:
    GPTAPIModel = None
try:
    from .gemini_model import GeminiAPIModel as _GeminiAPIModel
    GeminiAPIModel = _GeminiAPIModel
except Exception:
    GeminiAPIModel = None
try:
    from .hf_model import HFModel as _HFModel
    HFModel = _HFModel
except Exception:
    HFModel = None
try:
    from .mcp_proxy_model import MCPProxyModel as _MCPProxyModel
    MCPProxyModel = _MCPProxyModel
except Exception:
    MCPProxyModel = None
try:
    from .vllm_model import VLLMChatModel as _VLLMChatModel
    VLLMChatModel = _VLLMChatModel
except Exception:
    VLLMChatModel = None
try:
    from .groq_model import GroqAPIModel as _GroqAPIModel
    GroqAPIModel = _GroqAPIModel
except Exception:
    GroqAPIModel = None
import json, time, os
from datetime import datetime
from typing import List

# --- resolver imports (added) ----------------------------------------------
import subprocess, shlex
from pathlib import Path
try:
    import yaml  # optional; only needed if a manifest exists
except Exception:
    yaml = None
try:
    import pynvml
except Exception:
    pynvml = None
# ----------------------------------------------------------------------------

SentenceTransformer = None


def _get_sentence_transformer():
    """Lazy importer for sentence_transformers to avoid heavy imports at module load."""
    global SentenceTransformer
    if SentenceTransformer is None:
        try:
            from sentence_transformers import SentenceTransformer as _ST
            SentenceTransformer = _ST
        except Exception:
            # Try a lightweight fallback using huggingface/transformers if available.
            try:
                from transformers import AutoTokenizer, AutoModel
                import torch

                class HFEmbeddingWrapper:
                    """Simple adapter that provides an `encode` method similar to
                    sentence_transformers.SentenceTransformer using mean pooling.
                    This is a fallback when sentence-transformers is not installed.
                    """
                    def __init__(self, model_ref, device='cpu'):
                        self.device = device if isinstance(device, str) else ('cuda' if torch.cuda.is_available() else 'cpu')
                        self.tokenizer = AutoTokenizer.from_pretrained(model_ref)
                        self.model = AutoModel.from_pretrained(model_ref).to(self.device)

                    def _mean_pooling(self, model_output, attention_mask):
                        token_embeddings = model_output[0]  # first element is last_hidden_state
                        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
                        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
                        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
                        return sum_embeddings / sum_mask

                    def encode(self, texts, show_progress_bar=False, convert_to_numpy=True):
                        # Accept list[str] or str
                        if isinstance(texts, str):
                            texts = [texts]
                        enc = self.tokenizer(texts, padding=True, truncation=True, return_tensors='pt')
                        enc = {k: v.to(self.device) for k, v in enc.items()}
                        with torch.no_grad():
                            out = self.model(**enc)
                        pooled = self._mean_pooling(out, enc['attention_mask'])
                        if convert_to_numpy:
                            return pooled.cpu().numpy()
                        return pooled

                SentenceTransformer = HFEmbeddingWrapper
            except Exception:
                SentenceTransformer = False
    return SentenceTransformer if SentenceTransformer is not False else None


logger = logging.getLogger("GAIA.ModelPool")


# --- model path resolver (added, non-breaking) -----------------------------
def _read_manifest(path: Path) -> dict:
    if not path.exists():
        return {}
    if yaml is None:
        raise RuntimeError("model_manifest.yaml present but PyYAML is missing")
    return yaml.safe_load(path.read_text())


def _ensure_download(role: str, spec: dict, models_dir: Path, scripts_dir: Path, allow_autosetup: bool) -> Path:
    """Call scripts/download_models.py --role <role> if allowed; return target path."""
    rel = spec.get("path", "")
    out = (models_dir / rel)
    if out.exists() or not allow_autosetup:
        return out
    dl = scripts_dir / "download_models.py"
    if not dl.exists():
        return out
    cmd = [os.sys.executable, str(dl), "--role", role]
    try:
        subprocess.check_call(cmd)
    except Exception as e:
        logger.warning(f"[resolver] download for role={role} failed: {e}")
    return out


def resolve_model_paths(config: Config) -> dict:
    """
    Returns {'prime': '/abs/prime.gguf', 'lite': '/abs/lite.gguf', ...}
    Priority: env vars -> manifest -> as-is.
    """
    out = {}
    models_dir = Path(config.MODELS_DIR)
    manifest_path = models_dir / "model_manifest.yaml"
    
    # env overrides win
    env_map = {"prime": os.getenv("GAIA_PRIME_GGUF"), "lite": os.getenv("GAIA_LITE_GGUF")}
    for role, val in env_map.items():
        if val:
            out[role] = str(Path(val).expanduser().resolve())

    # manifest fallback
    if "prime" not in out or "lite" not in out:
        if manifest_path.exists() and yaml is not None:
            mf = _read_manifest(manifest_path)
            for role, spec in mf.get("roles", {}).items():
                if role in out:
                    continue
                candidate = models_dir / spec.get("path", "")
                out[role] = str(candidate.resolve())
    return out
# ---------------------------------------------------------------------------


def _get_gpu_free_total_bytes() -> tuple:
    """Return (free_bytes, total_bytes) for GPU 0 if available.
    Prefer pynvml when installed; otherwise parse `nvidia-smi` output.
    Returns (None, None) when values cannot be determined.
    """
    # Try pynvml first
    try:
        if pynvml is not None:
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
            free = int(mem.free)
            total = int(mem.total)
            try:
                pynvml.nvmlShutdown()
            except Exception:
                pass
            return free, total
    except Exception:
        pass

    # Fallback: nvidia-smi query
    try:
        out = subprocess.check_output([
            "nvidia-smi",
            "--query-gpu=memory.free,memory.total",
            "--format=csv,nounits,noheader"
        ], universal_newlines=True)
        line = out.strip().splitlines()[0]
        free_mb, total_mb = [int(x.strip()) for x in line.split(',')]
        return free_mb * 1024 * 1024, total_mb * 1024 * 1024
    except Exception:
        return None, None


def _choose_initial_n_gpu(desired_n_gpu: int, free_bytes: int | None) -> int:
    """Choose a conservative initial n_gpu_layers value based on free VRAM.

    Rules (tunable):
      - free < 2 GiB -> 0 (CPU-only)
      - 2 <= free < 4 GiB -> 1
      - 4 <= free < 8 GiB -> 2
      - free >= 8 GiB -> min(desired, 4) (allow up to 4)
    If free_bytes is None, return desired_n_gpu (no information).
    """
    if free_bytes is None:
        return desired_n_gpu
    free_gib = free_bytes / (1024 ** 3)
    if free_gib < 2:
        return 0
    if free_gib < 4:
        return min(desired_n_gpu, 1) if desired_n_gpu > 0 else 0
    if free_gib < 8:
        return min(desired_n_gpu, 2) if desired_n_gpu > 0 else 0
    # >=8 GiB
    return min(desired_n_gpu, 4) if desired_n_gpu > 0 else 0

class ModelPool:
    def __init__(self, config: Config = None):
        self.config = config or get_config()
        self.models = {}
        self.model_status = {}  # new: track each model's current role
        self.persona_manager = PersonaManager(self.config.PERSONAS_DIR)  # Initialize PersonaManager
        self.active_persona_obj = None  # Store the active PersonaAdapter object
        self.resource_monitor = ResourceMonitor.get_instance() # New line
        # Event to signal when the embedding model has finished loading (success or failure)
        self._embed_ready = threading.Event()
        self._embed_load_status = 'not_started'  # one of: not_started, loading, loaded, failed
        # persistent state for model roles (avoids re-discovery each run)
        try:
            import json
            self.MODEL_STATE_FILE = Path(self.config.LOGS_DIR) / "model_pool_state.json"
        except Exception:
            self.MODEL_STATE_FILE = None
        # If requested, pre-warm the embedder synchronously at startup for deterministic tests
        try:
            self._prewarm_embed_requested = os.getenv("GAIA_PREWARM_EMBED", "0") == "1"
        except Exception:
            self._prewarm_embed_requested = False
        try:
            self._prime_guard_override = os.getenv("GAIA_ALLOW_PRIME_LOAD", "0") == "1"
        except Exception:
            self._prime_guard_override = False

    def register_dev_model(self, name: str):
        """
        Registers a dev model that prints prompts to the console.
        """
        if DevModel is None:
            logger.error("DevModel not available, cannot register dev model.")
            return
        self.models[name] = DevModel(name=name)
        self.model_status[name] = "idle"
        # Set the dev model as the prime model to ensure it is selected.
        self.models["prime"] = self.models[name]
        self.model_status["prime"] = "idle"
        logger.info(f"✅ Registered dev model '{name}' and promoted it to 'prime'")

    def enable_prime_load(self):
        """Allow guarded prime loads (used by rescue shell / prime probe)."""
        self._prime_guard_override = True
        try:
            os.environ.setdefault("GAIA_ALLOW_PRIME_LOAD", "1")
        except Exception:
            pass

    def load_models(self, use_oracle=False):
        logger.info("--- ENTERING load_models ---")
        if getattr(self, "_models_loaded", False):
            try:
                import traceback
                stack = "".join(traceback.format_stack(limit=6)[:-1])
            except Exception:
                stack = "<stack unavailable>"
            logger.info("ModelPool.load_models() called again; skipping. Caller stack:\\n%s", stack)
            return

        self._auto_set_gpu_layers()
        self._start_embed_loader()
        self._apply_env_model_overrides()

        ordered_keys = self._ordered_model_keys()
        for model_name in ordered_keys:
            self._load_model_entry(model_name, use_oracle=use_oracle)

        self._promote_prime_aliases()

        try:
            self._models_loaded = True
        except Exception:
            pass
        logger.info("--- LEAVING load_models ---")

    def load_prime_only(self, force: bool = False) -> bool:
        """Load only the prime (vLLM) models when permitted by guards."""
        if not self._prime_guard_allows(force):
            logger.warning("Prime load blocked: GAIA_ALLOW_PRIME_LOAD not enabled")
            return False
        self._apply_env_model_overrides()
        prime_candidates = [name for name in ('gpu_prime', 'prime', 'cpu_prime') if name in self.config.MODEL_CONFIGS]
        loaded = False
        for name in prime_candidates:
            loaded |= self._load_model_entry(name, use_oracle=False, force=force)
            if 'prime' in self.models or 'gpu_prime' in self.models:
                break
        self._promote_prime_aliases()
        return bool(self.models.get('prime') or self.models.get('gpu_prime') or self.models.get('cpu_prime'))

    def _prime_guard_allows(self, force: bool = False) -> bool:
        if force:
            return True
        try:
            if os.getenv("GAIA_ALLOW_PRIME_LOAD_FORCE", "0") == "1":
                return True
        except Exception:
            pass
        if getattr(self, "_prime_guard_override", False):
            return True
        try:
            return os.getenv("GAIA_ALLOW_PRIME_LOAD", "0") == "1"
        except Exception:
            return False

    def _auto_set_gpu_layers(self):
        try:
            import torch
            if torch.cuda.is_available():
                free_bytes, total_bytes = _get_gpu_free_total_bytes()
                if total_bytes:
                    est_layers = int(total_bytes / (150 * 1024 * 1024))
                    env_cap = int(os.getenv("GAIA_MAX_N_GPU_LAYERS", "64"))
                    max_layers = min(est_layers, env_cap)
                    existing = int(getattr(self.config, 'n_gpu_layers', 0) or 0)
                    chosen = min(existing, max_layers) if existing > 0 else max_layers
                    try:
                        self.config.n_gpu_layers = int(chosen)
                    except Exception:
                        pass
                    logger.info(
                        "[GAIA] Computed gpu_layer_cap=%s (est=%s, env_cap=%s); using n_gpu_layers=%s",
                        max_layers,
                        est_layers,
                        env_cap,
                        chosen,
                    )
        except Exception as e:
            logger.warning(f"[GAIA] Could not auto-set n_gpu_layers: {e}")

    def _start_embed_loader(self):
        def _load_embed():
            logger.info("--- ENTERING _load_embed ---")
            try:
                logger.info("⚙️ Loading Embedding model")
                embed_on_gpu = os.getenv("GAIA_EMBED_ON_GPU", "0") == "1"
                st_cls = _get_sentence_transformer()
                if st_cls is None:
                    raise RuntimeError("sentence_transformers unavailable")
                embed_model = st_cls(self.config.EMBEDDING_MODEL_PATH) if embed_on_gpu else st_cls(self.config.EMBEDDING_MODEL_PATH, device='cpu')
                self.models["embed"] = embed_model
                self.model_status["embed"] = "idle"
                if getattr(self, '_prewarm_embed_requested', False):
                    try:
                        logger.info("🌡️ Pre-warming embedder as requested by GAIA_PREWARM_EMBED=1")
                        embed_model.encode(["prewarm"], show_progress_bar=False)
                        logger.info("✅ Embedder pre-warmed and ready")
                    except Exception as exc:
                        logger.error(f"❌ Pre-warm embedder failed: {exc}")
                self._embed_load_status = 'loaded'
                logger.info("--- LEAVING _load_embed (SUCCESS) ---")
            except Exception as exc:
                traceback.print_exc(file=sys.stderr)
                logger.error(f"❌ Failed to load Embedding model: {exc}")
                self._embed_load_status = 'failed'
                logger.info("--- LEAVING _load_embed (FAILURE) ---")
            finally:
                try:
                    self._embed_ready.set()
                except Exception:
                    pass

        logger.info("--- ENTERING _start_embed_loader ---")
        try:
            self._embed_ready.clear()
        except Exception:
            pass
        # If an embedder is already loaded (e.g., via prewarm_embed), just mark it ready.
        if 'embed' in self.models and self.models['embed'] is not None:
            try:
                self._embed_load_status = 'loaded'
                self._embed_ready.set()
            except Exception:
                pass
            logger.info("--- LEAVING _start_embed_loader (already loaded) ---")
            return
        _load_embed() # Call directly for synchronous loading and direct error reporting
        logger.info("--- LEAVING _start_embed_loader ---")

    def _apply_env_model_overrides(self):
        try:
            use_env_models = os.getenv("GAIA_USE_ENV_MODELS", "1") == "1"
        except Exception:
            use_env_models = True
        if not use_env_models:
            return
        try:
            # Support explicit HF prime model for vLLM via GAIA_PRIME_HF_MODEL
            prime_hf = os.getenv("GAIA_PRIME_HF_MODEL")
            prime_path = os.getenv("GAIA_PRIME_GGUF")
            lite_path = os.getenv("GAIA_LITE_GGUF")
            observer_hf = os.getenv("GAIA_OBSERVER_HF_MODEL")

            # If an HF prime is provided, prefer configuring it as a vLLM-backed gpu_prime.
            # This ensures vLLM is used for HF directories (e.g., Qwen3) instead of llama_cpp.
            if prime_hf:
                self.config.MODEL_CONFIGS["gpu_prime"] = {
                    "type": "vllm",
                    "model": prime_hf,
                    "path": prime_hf,
                    "enabled": True,
                }
                # Also alias 'prime' to gpu_prime so other code finds it under 'prime'
                self.config.MODEL_CONFIGS["prime"] = {"alias": "gpu_prime", "enabled": True}
                # When an HF prime is explicitly provided, prefer vLLM and disable
                # any CPU-based prime loads for now to avoid attempting to load
                # HF directories with llama_cpp (which expects GGUF files).
                try:
                    if "cpu_prime" in self.config.MODEL_CONFIGS:
                        self.config.MODEL_CONFIGS["cpu_prime"]["enabled"] = False
                        logger.info("GAIA_PRIME_HF_MODEL set: disabling cpu_prime to force vLLM usage for prime")
                except Exception:
                    logger.exception("Failed to disable cpu_prime when GAIA_PRIME_HF_MODEL is set")

            # Backwards-compatible: if a GGUF prime path is provided, register as cpu_prime fallback
            if prime_path and "gpu_prime" not in self.config.MODEL_CONFIGS:
                self.config.MODEL_CONFIGS["cpu_prime"] = {
                    "type": "local",
                    "path": prime_path,
                    "enabled": True,
                }
                self.config.MODEL_CONFIGS["prime"] = {"alias": "cpu_prime", "enabled": True}

            if lite_path:
                self.config.MODEL_CONFIGS["lite"] = {
                    "type": "local",
                    "path": lite_path,
                    "enabled": True,
                }
            if observer_hf:
                self.config.MODEL_CONFIGS["observer"] = {
                    "type": "hf",
                    "model": observer_hf,
                    "enabled": True,
                }
        except Exception:
            logger.exception("Failed to apply GAIA_* model overrides")

    def _ordered_model_keys(self) -> List[str]:
        preferred_order = ['gpu_prime', 'prime', 'lite', 'observer', 'cpu_prime']
        ordered_keys: List[str] = []
        for key in preferred_order:
            if key in self.config.MODEL_CONFIGS:
                ordered_keys.append(key)
        for key in self.config.MODEL_CONFIGS.keys():
            if key not in ordered_keys:
                ordered_keys.append(key)
        return ordered_keys

    def _load_model_entry(self, model_name: str, use_oracle: bool = False, force: bool = False) -> bool:
        model_config = self.config.MODEL_CONFIGS.get(model_name)
        if not isinstance(model_config, dict):
            logger.error(f"⚠️ Skipping model '{model_name}': invalid config type: {type(model_config)}")
            return False
        alias_target = model_config.get('alias')
        if alias_target:
            logger.debug("Skipping alias-only config '%s' -> '%s'", model_name, alias_target)
            return False
        if force:
            self.models.pop(model_name, None)
            self.model_status.pop(model_name, None)
        if not force and model_name in self.models:
            logger.info("Model '%s' already present in pool; skipping load", model_name)
            return False
        if not model_config.get('enabled', True):
            return False

        model_type = model_config.get('type')
        try:
            if model_type == 'local':
                global Llama
                if Llama is None:
                    from llama_cpp import Llama as _L
                    Llama = _L
                desired_n_gpu = int(getattr(self.config, 'n_gpu_layers', 0) or 0)
                free_bytes, total_bytes = _get_gpu_free_total_bytes()
                if free_bytes is None:
                    initial_try = desired_n_gpu
                    logger.info("GPU pre-check: unable to query free VRAM; proceeding with requested n_gpu_layers=%s", desired_n_gpu)
                else:
                    initial_try = _choose_initial_n_gpu(desired_n_gpu, free_bytes)
                    logger.info(
                        "GPU pre-check: free_vram=%.2f GiB total=%.2f GiB -> initial n_gpu_layers=%s",
                        (free_bytes / (1024**3)),
                        (total_bytes / (1024**3)) if total_bytes else 0.0,
                        initial_try,
                    )
                attempt_layers = []
                if desired_n_gpu > 0:
                    candidates = [initial_try, desired_n_gpu, max(1, desired_n_gpu // 2), max(1, desired_n_gpu // 4), 0]
                    seen = set()
                    for c in candidates:
                        if c not in seen:
                            attempt_layers.append(c)
                            seen.add(c)
                else:
                    attempt_layers = [0]
                # Force lite to stay on CPU to avoid competing with gpu_prime and to
                # keep intent detection lightweight and reliable.
                if model_name == "lite":
                    attempt_layers = [0]
                model_path = model_config.get('path') or (self.config.MODEL_CONFIGS.get(model_config.get('alias', ''), {}) or {}).get('path')
                if not model_path:
                    raise RuntimeError(f"Model '{model_name}' has no 'path' configured (config: {model_config})")
                last_exc = None
                chat_format = model_config.get("chat_format")
                # If the GGUF template is stripped, force a simple chat format to avoid empty prompts.
                if not chat_format and model_config.get("strip_chat_template"):
                    chat_format = "chatml"

                for n_try in attempt_layers:
                    try:
                        # Expand context for Operator (lite) without forcing Thinker to use the same window.
                        ctx_tokens = getattr(self.config, "max_tokens_lite", self.config.max_tokens) if model_name == "lite" else self.config.max_tokens
                        self.models[model_name] = Llama(
                            model_path=model_path,
                            n_gpu_layers=n_try,
                            n_ctx=ctx_tokens,
                            n_threads=getattr(self.config, 'n_threads', None) or multiprocessing.cpu_count(),
                            stream=True,
                            verbose=False,
                            chat_format=chat_format,
                        )
                        logger.info("✅ %s loaded with n_gpu_layers=%s: %s", model_name, n_try, self.models[model_name])
                        break
                    except Exception as e:
                        last_exc = e
                        logger.warning("Model load attempt failed for %s with n_gpu_layers=%s: %s", model_name, n_try, e)
                        if n_try == attempt_layers[-1]:
                            logger.error(f"❌ Failed to load {model_name} model: {e}")
                if last_exc and model_name not in self.models:
                    return False
                try:
                    model_obj = self.models.get(model_name)
                    if model_obj:
                        model_path = getattr(model_obj, 'model_path', getattr(model_obj, 'model', 'unknown'))
                        n_gpu = getattr(model_obj, 'n_gpu_layers', getattr(self.config, 'n_gpu_layers', 'unknown'))
                        logger.info("MODEL_DIAG: name=%s path=%s n_gpu_layers=%s", model_name, model_path, n_gpu)
                    else:
                        logger.error("MODEL_DIAG: %s not present after load attempts", model_name)
                except Exception:
                    logger.exception("MODEL_DIAG: failed to log diagnostics for %s", model_name)
            elif model_type == 'api' and use_oracle:
                provider = model_config.get("provider", "openai")
                logger.info(f"🔹 Loading {model_name} model (provider={provider})")
                if provider == "gemini":
                    if GeminiAPIModel is None:
                        raise RuntimeError("GeminiAPIModel unavailable (missing dependency)")
                    api_key = self.config.get_api_key("google")
                    self.models[model_name] = GeminiAPIModel(
                        model_config.get("model") or "gemini-1.5-flash",
                        api_key=api_key,
                    )
                else:
                    if GPTAPIModel is None:
                        raise RuntimeError("GPTAPIModel unavailable (missing dependency)")
                    # Pass the model alias so the client uses the correct configured model
                    self.models[model_name] = GPTAPIModel(self.config, model_alias=model_name)
            elif model_type == 'api':
                return False
            elif model_type == 'hf':
                logger.info(f"🔹 Loading HF model {model_name}")
                try:
                    import torch
                    has_cuda = torch.cuda.is_available()
                    dtype = torch.float16 if has_cuda else None
                    device_map = 'auto' if has_cuda else None
                except Exception:
                    dtype = None
                    device_map = None
                model_ref = model_config.get('path') or model_config.get('model')
                self.models[model_name] = HFModel(
                    model_ref,
                    local_path=model_config.get('path'),
                    device_map=device_map,
                    torch_dtype=dtype,
                )
            elif model_type == 'mcp':
                logger.info(f"🔹 Registering MCP proxy model {model_name}")
                self.models[model_name] = MCPProxyModel(self.config, role_name=model_name)
            elif model_type == 'vllm':
                logger.info(f"🔹 Loading vLLM model {model_name}")
                gpu_info = _get_gpu_free_total_bytes()
                self.models[model_name] = VLLMChatModel(model_config, self.config, gpu_info=gpu_info)
            elif model_type == 'groq':
                if GroqAPIModel is None:
                    logger.warning("GroqAPIModel unavailable (groq package not installed)")
                    return False
                logger.info(f"🔹 Loading Groq API model {model_name}")
                api_key = os.getenv("GROQ_API_KEY")
                if not api_key:
                    logger.warning(f"GROQ_API_KEY not set; skipping {model_name}")
                    return False
                model_id = model_config.get("model", "llama-3.3-70b-versatile")
                self.models[model_name] = GroqAPIModel(model_name=model_id, api_key=api_key)
            else:
                if model_type and model_type != 'api':
                    logger.warning("Unknown model type '%s' for %s; skipping", model_type, model_name)
                return False
        except Exception as e:
            logger.error(f"❌ Failed to load {model_name} model: {e}")
            return False

        self.model_status[model_name] = 'idle'
        return True

    def _promote_prime_aliases(self):
        if "prime" not in self.models and "cpu_prime" in self.models:
            logger.info("Promoting cpu_prime to 'prime' because no dedicated prime model was loaded")
            self.models["prime"] = self.models["cpu_prime"]
            self.model_status["prime"] = self.model_status.get("cpu_prime", "idle")
        try:
            import torch
            if torch.cuda.is_available() and "gpu_prime" in self.models:
                logger.info("🔼 Promoting gpu_prime to 'prime' for GPU inference")
                self.models["prime"] = self.models["gpu_prime"]
                self.model_status["prime"] = "idle"
                try:
                    setattr(self.config, "llm_backend", "gpu_prime")
                    os.environ.setdefault("GAIA_BACKEND", "gpu_prime")
                    logger.info("GAIA backend set to 'gpu_prime' by default because CUDA is available")
                except Exception:
                    pass
        except Exception:
            pass

    def wait_for_embed(self, timeout: float = None):
        """Block up to `timeout` seconds for the embed model to finish loading.
        Returns the embedding model instance if available, otherwise None.
        If timeout is None, wait indefinitely.
        """
        try:
            self._embed_ready.wait(timeout=timeout)
        except Exception:
            pass
        return self.models.get('embed')

    def get_embed_model(self, timeout: float = 0, lazy_load: bool = True):
        """Convenience: if timeout>0, wait up to timeout seconds for embed; else return current embed or None.

        If lazy_load=True (default) and embed is not loaded, will attempt to load it via prewarm_embed().
        """
        if timeout and timeout > 0:
            return self.wait_for_embed(timeout=timeout)

        embed = self.models.get('embed')
        if embed is None and lazy_load:
            # Try to load the embed model
            try:
                logger.info("🔄 Lazy loading embed model on demand...")
                if self.prewarm_embed(timeout=30):
                    embed = self.models.get('embed')
            except Exception as e:
                logger.error(f"Failed to lazy load embed model: {e}")

        return embed

    def prewarm_embed(self, timeout: int = 10) -> bool:
        """Synchronous helper to ensure the embedding model is loaded and warmed.

        Returns True if embedder is loaded and warmed, False otherwise.
        """
        if 'embed' in self.models and self.models['embed'] is not None:
            try:
                self.models['embed'].encode(["prewarm"], show_progress_bar=False)
                self._embed_load_status = 'loaded'
                self._embed_ready.set()
                return True
            except Exception as e:
                traceback.print_exc(file=sys.stderr)
                logger.error(f"prewarm_embed encode failed: {e}")
                self._embed_load_status = 'failed'
                self._embed_ready.set()
                return False

        # If embed model not present, attempt to load quickly
        embed_on_gpu = os.getenv("GAIA_EMBED_ON_GPU", "0") == "1"
        st_cls = _get_sentence_transformer()
        if st_cls is None:
            raise RuntimeError("sentence_transformers unavailable")
        # If the configured embedding path doesn't exist, attempt common
        # alternatives (container /models mount and configured MODEL_DIR)
        emb_path = Path(str(self.config.EMBEDDING_MODEL_PATH)) if self.config.EMBEDDING_MODEL_PATH else None
        tried = []
        if emb_path is None or not emb_path.exists():
            # Build candidate paths to try
            try:
                from gaia_core.config import get_config
                _cfg = get_config()
                MODELS_DIR = _cfg.MODELS_DIR
                MODEL_DIR = _cfg.MODEL_DIR
                candidates = []
                if emb_path is not None:
                    # If emb_path was absolute under /models, prefer the rest
                    if str(emb_path).startswith("/models/"):
                        rest = str(emb_path)[len("/models/"):]
                        candidates.append(Path(MODELS_DIR) / rest)
                        candidates.append(Path("/models") / rest)
                        candidates.append(Path(MODEL_DIR) / rest)
                # Also include explicit MODELS_DIR and /models locations
                candidates.append(Path(MODELS_DIR) / (emb_path.name if emb_path else ""))
                candidates.append(Path("/models") / (emb_path.name if emb_path else ""))
            except Exception:
                candidates = [Path("/models") / (emb_path.name if emb_path else "")]
            for c in candidates:
                tried.append(str(c))
                try:
                    if c.exists():
                        emb_path = c
                        break
                except Exception:
                    pass
        if emb_path is None:
            raise RuntimeError(f"Embedding model path could not be resolved. Tried: {tried}")
        if embed_on_gpu:
            self.models['embed'] = st_cls(str(emb_path))
        else:
            self.models['embed'] = st_cls(str(emb_path), device='cpu')
        try:
            self.models['embed'].encode(["prewarm"], show_progress_bar=False)
            self._embed_load_status = 'loaded'
            self._embed_ready.set()
            return True
        except Exception as e:
            logger.error(f"prewarm_embed initial load failed: {e}")
            self._embed_load_status = 'failed'
            self._embed_ready.set()
            return False

    def ensure_model_loaded(self, name: str, force: bool = False) -> bool:
        """
        Ensure a specific model is loaded. Returns True if model is available after call.

        This is the lazy loading mechanism - if GAIA_AUTOLOAD_MODELS=0, models won't load
        at startup but will load on-demand when first requested.
        """
        logger.warning(f"[LAZY_LOAD] ensure_model_loaded called for '{name}', force={force}")
        logger.warning(f"[LAZY_LOAD] current pool keys: {list(self.models.keys())}")
        logger.warning(f"[LAZY_LOAD] MODEL_CONFIGS keys: {list(self.config.MODEL_CONFIGS.keys()) if hasattr(self.config, 'MODEL_CONFIGS') else 'NO MODEL_CONFIGS'}")

        # Already loaded?
        if not force and name in self.models and self.models[name] is not None:
            logger.warning(f"[LAZY_LOAD] '{name}' already in pool, returning True")
            return True

        # Special handling for embed model (not in MODEL_CONFIGS, uses prewarm_embed)
        if name == 'embed':
            logger.info("🔄 Lazy loading embed model on demand...")
            try:
                return self.prewarm_embed(timeout=30)
            except Exception as e:
                logger.error(f"Failed to lazy load embed model: {e}")
                return False

        # Check if model is configured
        if name not in self.config.MODEL_CONFIGS:
            # Check for role aliases (prime -> gpu_prime/cpu_prime)
            if name == 'prime':
                for candidate in ('gpu_prime', 'cpu_prime'):
                    if candidate in self.config.MODEL_CONFIGS:
                        if self.ensure_model_loaded(candidate, force=force):
                            self._promote_prime_aliases()
                            return 'prime' in self.models
            logger.warning(f"Model '{name}' not configured in MODEL_CONFIGS")
            return False

        # Apply env overrides before loading (in case not called during startup)
        self._apply_env_model_overrides()

        # For prime models, check guard
        if name in ('gpu_prime', 'prime', 'cpu_prime'):
            if not self._prime_guard_allows(force):
                logger.warning(f"Prime model '{name}' load blocked: GAIA_ALLOW_PRIME_LOAD not enabled")
                return False

        # Load the model
        logger.info(f"🔄 Lazy loading model '{name}' on demand...")
        success = self._load_model_entry(name, use_oracle=False, force=force)

        # Promote aliases if needed
        if success and name in ('gpu_prime', 'cpu_prime'):
            self._promote_prime_aliases()

        return name in self.models and self.models[name] is not None

    def get(self, name: str, lazy_load: bool = True):
        """
        Get a model by name. If lazy_load=True (default), will attempt to load
        the model on-demand if it's not already in the pool.
        """
        model = self.models.get(name)
        if model is None and lazy_load:
            # Attempt lazy loading
            if self.ensure_model_loaded(name):
                model = self.models.get(name)
        if model is None:
            logger.error(f"❌ Requested model '{name}' not found in pool! Pool keys: {list(self.models.keys())}")
            logger.error(f"DEBUG: Model for '{name}' is None. Type of model: {type(model)}, Value of model: {model}")
        return model

    def get_model_for_role(self, role: str, lazy_load: bool = True):
        """
        Resolve a logical role name (prime, lite, observer, gpu_prime, cpu_prime, etc.)
        to an actual model instance in the pool. This centralizes the policy: Prime
        should preferentially use a GPU-backed model (gpu_prime) if present; otherwise
        fall back to cpu_prime. Lite/Observer can be mapped to the same underlying
        model by environment configuration if desired.

        If lazy_load=True (default), will attempt to load the model on-demand if
        it's not already in the pool.
        """
        # Handle distracted state
        if self.resource_monitor.is_distracted() and role == 'prime':
            logger.warning("System is distracted, falling back to 'lite' model for 'prime' role.")
            return self.get_model_for_role('lite', lazy_load=lazy_load)

        # Dev model override
        if "azrael" in self.models:
            return self.models["azrael"]
        # Direct hit
        if role in self.models:
            return self.models[role]

        # Role alias resolution: consult config.MODEL_CONFIGS for alias mapping
        cfg = self.config.MODEL_CONFIGS.get(role, {}) if self.config and hasattr(self.config, 'MODEL_CONFIGS') else {}
        alias = cfg.get('alias')
        if alias and alias in self.models:
            return self.models[alias]

        # Prime preference: prefer gpu_prime when available
        if role == 'prime':
            if 'prime' in self.models:
                return self.models['prime']
            if 'gpu_prime' in self.models:
                return self.models['gpu_prime']
            if 'cpu_prime' in self.models:
                return self.models['cpu_prime']

        # For lite or observer, allow env var to force sharing via GAIA_SHARE_LITE_WITH
        try:
            import os
            share_lite_with = os.getenv('GAIA_SHARE_LITE_WITH')
            if role in ('lite', 'observer') and share_lite_with:
                if share_lite_with in self.models:
                    return self.models[share_lite_with]
        except Exception:
            pass

        # --- LAZY LOADING ---
        # If we haven't found the model yet and lazy_load is enabled, try loading it
        if lazy_load:
            # Try to load the specific role
            if self.ensure_model_loaded(role):
                if role in self.models:
                    return self.models[role]
                # Check for promoted alias (e.g., prime -> gpu_prime)
                if role == 'prime':
                    if 'prime' in self.models:
                        return self.models['prime']
                    if 'gpu_prime' in self.models:
                        return self.models['gpu_prime']
                    if 'cpu_prime' in self.models:
                        return self.models['cpu_prime']

            # If role is an alias, try loading the aliased model
            if alias and self.ensure_model_loaded(alias):
                if alias in self.models:
                    return self.models[alias]

        # Fallback: first available model
        for name in ('prime', 'lite', 'cpu_prime', 'gpu_prime'):
            if name in self.models:
                return self.models[name]
        # Last resort: any model
        if self.models:
            return next(iter(self.models.values()))
        return None

    def list_models(self):
        return list(self.models.keys())

    def set_status(self, name: str, status: str):
        if name in self.models:
            self.model_status[name] = status
            logger.info(f"🔄 Model '{name}' status set to '{status}'")

    def get_idle_model(self, exclude=[]):
        for name, status in self.model_status.items():
            if status == "idle" and name not in exclude:
                return name
        return None

    def acquire_model(self, name: str, lazy_load: bool = True):
        """
        Acquire a model by name (marks it as busy). If lazy_load=True (default),
        will attempt to load the model on-demand if it's not already in the pool.
        """
        if name not in self.models and lazy_load:
            self.ensure_model_loaded(name)
        if name in self.models:
            self.set_status(name, "busy")
            return self.models[name]
        return None

    def release_model(self, name: str):
        if name in self.models:
            self.set_status(name, "idle")

    def release_model_for_role(self, role: str):
        name = self._resolve_model_name_for_role(role)
        if name:
            self.release_model(name)

    def _resolve_model_name_for_role(self, role: str) -> str | None:
        name = None
        if role in self.models:
            name = role
        else:
            cfg = self.config.MODEL_CONFIGS.get(role, {}) if self.config and hasattr(self.config, 'MODEL_CONFIGS') else {}
            alias = cfg.get('alias')
            if alias and alias in self.models:
                name = alias
        if not name:
            if role == 'prime':
                if 'prime' in self.models:
                    name = 'prime'
                elif 'gpu_prime' in self.models:
                    name = 'gpu_prime'
                elif 'cpu_prime' in self.models:
                    name = 'cpu_prime'
            else:
                share = None
                try:
                    import os
                    share = os.getenv('GAIA_SHARE_LITE_WITH') if role in ('lite', 'observer') else None
                except Exception:
                    share = None
                if share and share in self.models:
                    name = share
        return name

    def acquire_model_for_role(self, role: str, lazy_load: bool = True):
        """Resolve role to a model name and acquire it (mark busy). Returns the model instance or None.

        If lazy_load=True (default), will attempt to load the model on-demand if
        it's not already in the pool.

        For prime roles, implements a fallback chain: gpu_prime -> groq_fallback -> oracle_openai
        """
        name = self._resolve_model_name_for_role(role)

        # If no model resolved, try lazy loading first
        if not name and lazy_load:
            if self.ensure_model_loaded(role):
                name = self._resolve_model_name_for_role(role)

        # FALLBACK CHAIN: If primary model unavailable for prime roles, try fallbacks
        if not name or name not in self.models:
            if role in ('prime', 'gpu_prime', 'cpu_prime'):
                fallback_chain = ['groq_fallback', 'oracle_openai', 'oracle_gemini']
                for fallback in fallback_chain:
                    if fallback in self.models:
                        logger.warning(f"🔄 Using {fallback} as fallback for {role}")
                        self.set_status(fallback, "busy")
                        return self.models[fallback]
                    # Try lazy loading the fallback
                    if self.ensure_model_loaded(fallback):
                        if fallback in self.models:
                            logger.warning(f"🔄 Loaded {fallback} as fallback for {role}")
                            self.set_status(fallback, "busy")
                            return self.models[fallback]
                logger.error(f"No fallback available for {role}")
                return None

        if not name:
            logger.error("ModelPool.acquire_model_for_role: no model resolved for role '%s'", role)
            return None

        model = self.models.get(name)

        # If model not in pool, try lazy loading
        if not model and lazy_load:
            if self.ensure_model_loaded(name):
                model = self.models.get(name)

        if not model:
            logger.error("ModelPool.acquire_model_for_role: resolved model '%s' missing from pool", name)
            return None
        self.set_status(name, "busy")
        return model

    def forward_to_model(self, role: str, messages: list, release: bool = True, **kwargs):
        """Utility used by AgentCore/tests to run a short chat completion via role lookup."""
        name = self._resolve_model_name_for_role(role)
        if not name or name not in self.models:
            raise ValueError(f"Model '{role}' not found in pool.")
        model = self.models[name]
        self.set_status(name, "busy")
        try:
            if hasattr(model, "create_chat_completion"):
                result = model.create_chat_completion(messages=messages, **kwargs)
            elif callable(model):
                result = model(messages)
            else:
                raise ValueError(f"Model '{name}' does not support chat completions")
            return result
        finally:
            if release:
                self.release_model(name)

    def get_active_persona(self) -> PersonaAdapter:
        """Returns the currently active PersonaAdapter object."""
        return self.active_persona_obj
    
    def set_persona(self, persona):
        """
        Make the given persona object discoverable to code that calls
        `model_pool.get_active_persona()` or inspects `model_pool.persona_name`.
        """
        if isinstance(persona, dict):
            self.active_persona_obj = PersonaAdapter(persona, self.config)
        elif isinstance(persona, PersonaAdapter):
            self.active_persona_obj = persona
        else:
            logger.error(f"Invalid persona type: {type(persona)}")
            return

        # Best-effort human label
        self.persona_name = getattr(self.active_persona_obj, "name", getattr(self.active_persona_obj, "id", "unknown"))
        logging.getLogger(__name__).debug(f"🔄 Active persona set to {self.persona_name}")
    
    def complete(self, name: str, prompt: str, max_tokens: int = 128, temperature: float = 0.2) -> str:
         """
         Convenience method to run a short blocking completion using create_completion.
         Intended for Observer-style short prompt checks.
         """
         model = self.models.get(name)
         if not model:
             logger.error(f"Model '{name}' not found for complete() call.")
             return ""
         try:
             result = model.create_completion(
                 prompt=prompt,
                 max_tokens=max_tokens,
                 temperature=temperature,
                 stop=["\n"]
             )
             return result["choices"][0]["text"].strip()
         except Exception as e:
             logger.warning(f"⚠️ ModelPool.complete() failed for '{name}': {e}")
             return ""

    def shutdown(self) -> None:
        """Best-effort shutdown for model backends (vLLM, llama.cpp, API stubs)."""
        try:
            logger.info("ModelPool.shutdown: starting")
            for name, model in list(self.models.items()):
                try:
                    # Prefer explicit shutdown/close if available.
                    if hasattr(model, "shutdown") and callable(getattr(model, "shutdown")):
                        model.shutdown()
                    elif hasattr(model, "close") and callable(getattr(model, "close")):
                        model.close()
                    elif hasattr(model, "llm") and hasattr(model.llm, "shutdown") and callable(model.llm.shutdown):
                        model.llm.shutdown()
                except Exception:
                    logger.debug("ModelPool.shutdown: failed to close model %s", name, exc_info=True)
            self.models.clear()
            self.model_status.clear()
            logger.info("ModelPool.shutdown: complete")
        except Exception:
            logger.debug("ModelPool.shutdown: unexpected error", exc_info=True)