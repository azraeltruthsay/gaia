import faulthandler, os
import multiprocessing
import sys

# Ensure multiprocessing uses the 'spawn' start method early so CUDA / torch
# initialisation inside child processes won't encounter the "Cannot re-init
# CUDA in forked subprocess" error when vLLM launches worker processes.
try:
    multiprocessing.set_start_method('spawn', force=True)
except Exception:
    # ignore if already set or unsupported; best-effort
    pass

# Allow the rescue/assistant process to load Prime by default. This must be
# explicitly enabled for other processes. Operators who want to permit prime
# loads in this process can override the environment before invoking the
# script; otherwise set here so an interactive rescue session can load Prime.


_model_pool_ref = None


def _resolve_model_pool():
    """Lazily import model_pool only when needed to avoid heavy startup cost."""
    global _model_pool_ref
    if _model_pool_ref is None:
        try:
            from gaia_core.models.model_pool import model_pool as _mp
        except Exception as exc:
            raise RuntimeError("model_pool import failed") from exc
        _model_pool_ref = _mp
        # Opt-in prime-load enablement for rescue shell; ignore if unsupported
        try:
            _model_pool_ref.enable_prime_load()
        except Exception:
            pass
        # Rebind the module-level symbol so later references see the real pool
        globals()["model_pool"] = _model_pool_ref
    return _model_pool_ref


class _ModelPoolLazyProxy:
    """Proxy that resolves the real model_pool object on first attribute access."""

    __slots__ = ()

    def __getattr__(self, item):
        return getattr(_resolve_model_pool(), item)

    def __repr__(self):
        return repr(_resolve_model_pool())


# Expose a proxy so legacy references (`model_pool.*`) keep working without an
# eager import. After the first attribute read the global is rebound to the real
# object.
model_pool = _ModelPoolLazyProxy()
# ensure logs dir exists
base_log_dir = os.path.join(os.path.dirname(__file__), "logs")
try:
    os.makedirs(base_log_dir, exist_ok=True)
    log_dir = base_log_dir
except PermissionError:
    # Fall back to a writable temporary logs directory to avoid startup crash
    tmp_dir = os.path.join("/tmp", "gaia_logs")
    try:
        os.makedirs(tmp_dir, exist_ok=True)
        log_dir = tmp_dir
        print(f"[warn] permission denied creating {base_log_dir}; using fallback {tmp_dir}")
    except Exception:
        # Last resort: use current working directory
        log_dir = os.getcwd()
        print(f"[warn] failed to create fallback logs dir; using cwd={log_dir}")
# open a file to record native stack traces
fh = open(os.path.join(log_dir, "faulthandler.log"), "w")
faulthandler.enable(fh, all_threads=True)

import argparse
import code
import importlib
import logging
import subprocess
import time
import sys
import threading
from datetime import datetime
from gaia_core.cognition.self_review_worker import run_review_with_prompt
from typing import Dict, Any

# GAIA internal modules
from gaia_core.behavior.persona_adapter import PersonaAdapter
from gaia_core.behavior.persona_manager import PersonaManager
from gaia_core.cognition.agent_core import AgentCore
from gaia_core.utils.output_router import _strip_think_tags_robust as strip_think_tags
from gaia_core.cognition import topic_manager
from gaia_core.cognition.cognition_packet import CognitionPacket
from gaia_core.config import Config
from gaia_common.utils.logging_setup import setup_logging
from gaia_core.ethics.core_identity_guardian import CoreIdentityGuardian
from gaia_core.ethics.ethical_sentinel import EthicalSentinel
from gaia_core.utils import gaia_rescue_helper as helper
from gaia_core.memory.session_manager import SessionManager
from gaia_core.models.model_manager import get_manager
import json, logging as _logging
_startup_logger = _logging.getLogger("GAIA.Rescue.startup")
# Immediate startup diagnostic: optionally attempt to load critical models.
#
# Historically we attempted a conservative autoload (prime then others) at
# import time. In practice this can unexpectedly initialize CUDA in the
# parent process and cause vLLM worker subprocesses to fail with the
# "Cannot re-initialize CUDA in forked subprocess" error. To avoid that
# class of failures, autoload is now opt-in: set GAIA_AUTOLOAD_MODELS=1 to
# opt back into the old behavior. Operators running an interactive rescue
# session should prefer explicitly loading Prime from the shell so it can
# be done with full spawn/isolation semantics.
try:
    autoload = os.getenv("GAIA_AUTOLOAD_MODELS", "0") == "1"
    if not autoload:
        _startup_logger.warning("[startup-diagnostic] autoload disabled (GAIA_AUTOLOAD_MODELS!=1); skipping automatic prime/lite autoload to avoid CUDA/fork hazards")
    else:
        # Defer actual autoload actions to the CLI `main()` so any spawn-based
        # loaders are invoked after interpreter bootstrap (avoids spawn/bootstrapping
        # race conditions when modules are imported).
        _startup_logger.warning("[startup-diagnostic] GAIA_AUTOLOAD_MODELS=1 detected; autoload will be performed from main() to ensure spawn is safe")
except Exception:
    _startup_logger.exception("[startup-diagnostic] Unexpected failure during startup diagnostics")

# Model presence checks and autoload enforcement happen inside main() after we
# know whether the operator requested GAIA_AUTOLOAD_MODELS=1.
from gaia_common.utils.vector_indexer import embed_gaia_reference, vector_query
from gaia_core.utils.dev_matrix_analyzer import DevMatrixAnalyzer
# from gaia_common.utils.training_utils import (
#     check_for_training_delta,
#     get_next_model_version,
#     convert_to_gguf,
#     update_training_log,
#     get_base_model_name,
# )
from gaia_core.utils import mcp_client
import json

from gaia_core.memory.semantic_codex import SemanticCodex
faulthandler.enable()  # dumps C‑level traceback on seg‑fault

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
config = Config()
LOG_DIR = config.LOGS_DIR
# Initialize centralized logging (UTC timestamps)
setup_logging(log_dir=LOG_DIR, level=logging.INFO)
logger = logging.getLogger("GAIA.Rescue")

# Runtime toggle: enable observer LLM checks when the operator explicitly
# sets GAIA_ENABLE_OBSERVER_LLM=1 in the environment. This avoids changing
# the constants file and allows dynamic enabling in containers.
try:
    if os.getenv("GAIA_ENABLE_OBSERVER_LLM", "0") == "1":
        # Set the constants entry so StreamObserver.observe() will call the
        # observer LLM instead of skipping LLM checks.
        config.constants["OBSERVER_USE_LLM"] = True
        logger.warning("Runtime override: OBSERVER_USE_LLM enabled via GAIA_ENABLE_OBSERVER_LLM=1")
except Exception:
    logger.exception("Failed to apply GAIA_ENABLE_OBSERVER_LLM runtime override")




# Silence noisy deps that spam INFO
for noisy in ("llama_index", "llama_cpp", "transformers", "sentence_transformers", "huggingface_hub"):
    logging.getLogger(noisy).setLevel(logging.WARNING)
logging.getLogger("GAIA.AgentCore").setLevel(logging.INFO) # Set to INFO to capture continuation prompt
logging.getLogger("GAIA.VLLM").setLevel(logging.INFO)


# =============================================================================
#  Module-level function for Discord ProcessPoolExecutor (must be picklable)
# =============================================================================
def _run_agent_core_in_process(_content, _session_id, _destination, _source, _metadata):
    """
    Run AgentCore in a subprocess. This function must be at module level
    to be picklable by ProcessPoolExecutor.
    """
    # Import strip_think_tags locally to avoid circular import issues
    from gaia_core.utils.output_router import _strip_think_tags_robust as strip_think_tags

    _ai = MinimalAIManager()  # Re-initialize in child process
    _agent_core = AgentCore(_ai, ethical_sentinel=_ai.ethical_sentinel)
    _full_response = ""
    for event in _agent_core.run_turn(
        _content,
        session_id=_session_id,
        destination=_destination,
        source=_source,
        metadata=_metadata
    ):
        et = event.get("type")
        val = event.get("value")
        if et == "token":
            _clean_val = strip_think_tags(val) if isinstance(val, str) else str(val)
            _full_response += _clean_val
    return _full_response


# =============================================================================
#  MinimalAIManager – a super‑light wrapper around GAIA's cognition stack
# =============================================================================
class MinimalAIManager:
    """Lightweight manager for GAIA rescue shell."""

    # --------------------------------------------------------------------- init
    def __init__(self):
        # ------------------------------------------------------------------ cfg
        self.config: Config = model_pool.config  # reuse global Config instance
        # Back‑compat shim for legacy UPPER‑CASE attrs consumed by older helpers
        self.config.MAX_TOKENS = getattr(self.config, "max_tokens", 4096)
        self.config.RESPONSE_BUFFER = self.config.constants.get("RESPONSE_BUFFER", 512)

        # ---------------------------------------------------------------- models
        self.llm = None
        self.lite_llm = None
        # Try to eagerly obtain an embedding model from the model pool so the
        # ConversationSummarizer (used by SessionManager) has an embedder ready
        # and doesn't need to wait for a lazy fetch during summarization.
        self.embed_model = None
        try:
            # model_pool is a lazy proxy that will resolve on first access.
            # Some model_pool implementations accept a `timeout` kwarg; guard
            # against mismatched signatures by falling back to `get('embed')`.
            try:
                self.embed_model = model_pool.get_embed_model(timeout=5)
            except TypeError:
                # Older or alternate implementations may not accept timeout
                self.embed_model = model_pool.get_embed_model()
            except AttributeError:
                # If get_embed_model isn't available, try direct lookup
                self.embed_model = getattr(model_pool, 'get', lambda k: None)('embed')
        except Exception:
            # Non-fatal: leave embed_model as None and allow SessionManager to
            # attempt its own fallback logic (it already tries model_pool).
            logger.debug("Could not eagerly acquire embed_model for rescue session; proceeding without it.", exc_info=True)

        # ---------------------------------------------------------------- state
        self.status: Dict[str, Any] = {"boot_time": datetime.utcnow().isoformat()}
        self.helper = helper  # hot‑reloadable helper limb
        self.topic_cache_path = "app/shared/topic_cache.json"

        # ------------------------------------------------------------- ethics
        self.identity_guardian = CoreIdentityGuardian(config=self.config)
        self.ethical_sentinel = EthicalSentinel(identity_guardian=self.identity_guardian)

        # ---------------------------------------------------------------- sess
        # Pass the (possibly None) embed_model into SessionManager so the
        # ConversationSummarizer receives an explicit embedder when available.
        self.session_manager = SessionManager(config=self.config, llm=self.llm, embed_model=self.embed_model)

        # ---------------------------------------------------------------- pool / persona
        self.model_pool = model_pool  # reuse already‑loaded pool
        self.persona_manager = PersonaManager(self.config.PERSONAS_DIR)
        persona = self.persona_manager.load_persona_data("dev")
        if persona and hasattr(self.model_pool, "set_persona"):
            try:
                self.model_pool.set_persona(persona)
            except Exception:  # pragma: no cover – not fatal in rescue mode
                logger.warning("ModelPool.set_persona() failed; continuing without persona binding.")

        # ------------------------------------------------------------------
        # Snapshot short-circuit: when the capability mapper runs GAIA under
        # `trace`, it sets GAIA_CAPABILITY_SNAPSHOT=1 so we don’t drop into
        # the interactive shell.
        # ------------------------------------------------------------------
        if os.getenv("GAIA_CAPABILITY_SNAPSHOT") == "1":
            logger.info("Capability snapshot mode: skipping interactive shell")
            return  # 🚪 exit early

    # --------------------------------------------------------------------- init
    def initialize(self, persona_name: str = "dev") -> None:
        """Load the specified persona or fall back to an emergency stub."""
        try:
            data = self.persona_manager.load_persona_data(persona_name) or {}
            if not data:
                logger.error("❌ Could not load '%s' persona; using fallback.", persona_name)
                data = {
                    "name": "gaia-dev-emergency",
                    "description": "Emergency fallback persona.",
                    "template": "You are GAIA-Dev, operating in a minimal rescue shell.",
                    "instructions": [
                        "Primary persona file is missing. Operate with caution.",
                    ],
                }
            self.active_persona = PersonaAdapter(data, self.config)
            logger.info("✅ Loaded %s persona '%s' via PersonaManager.", "default" if persona_name == "dev" else "", self.active_persona.name)
        except Exception as exc:  # pragma: no cover
            logger.exception("Persona init failed: %s", exc)

    def shutdown(self) -> None:
        """Best-effort shutdown hook for MinimalAIManager used by the rescue shell.

        Currently this primarily ensures the shared `model_pool` is explicitly
        shut down so vLLM engine processes are terminated under our control
        rather than during interpreter finalization.
        """
        try:
            if getattr(self, 'model_pool', None):
                try:
                    self.model_pool.shutdown()
                except Exception:
                    logger.debug("MinimalAIManager.shutdown: model_pool.shutdown() failed", exc_info=True)
        except Exception:
            logger.debug("MinimalAIManager.shutdown: unexpected error", exc_info=True)

    # ------------------------------------------------------------------- reload
    def reload(self, module_name: str) -> None:
        """Hot-reload a module; handy while tinkering inside the shell."""
        try:
            if module_name.endswith(".py"):
                module_name = module_name[:-3]
            if module_name.startswith("app/"):
                module_name = module_name.replace("/", ".")

            module_path = module_name.replace("app.", "app/").replace(".", "/") + ".py"
            if not os.path.exists(module_path):
                logger.error("❌ File not found: %s", module_path)
                return

            module = importlib.import_module(module_name)
            importlib.reload(module)

            if "gaia_rescue_helper" in module_name:
                self.helper = importlib.import_module("app.utils.gaia_rescue_helper")
            logger.info("✅ Reloaded module: %s", module_name)
        except Exception as exc:  # pragma: no cover
            logger.exception("Reload failed: %s", exc)

    # --------------------------------------------------------------------- I/O
    def read(self, filepath: str) -> None:
        """Log file contents to console."""
        try:
            with open(filepath, "r", encoding="utf-8") as fh:
                content = fh.read()
            logger.info("📄 %s\n---\n%s\n---", filepath, content)
        except Exception as exc:  # pragma: no cover
            logger.error("❌ Read error: %s", exc)

    def write(self, filepath: str, content: str) -> None:
        """Write file with timestamped backup."""
        try:
            if os.path.exists(filepath):
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                backup = f"{filepath}.bak.{ts}"
                os.rename(filepath, backup)
                logger.info("🗃️  Backup created: %s", backup)

            with open(filepath, "w", encoding="utf-8") as fh:
                fh.write(content)
            logger.info("✅ Wrote %s", filepath)
        except Exception as exc:  # pragma: no cover
            logger.error("❌ Write error: %s", exc)

    # --------------------------------------------------------- safe execute
    def execute(self, command: str) -> None:
        """Run a whitelisted shell command."""
        safe_cmds = self.config.SAFE_EXECUTE_FUNCTIONS
        if not any(command.strip().startswith(c) for c in safe_cmds):
            logger.error("❌ Unsafe command blocked: %s (allowed: %s)", command, safe_cmds)
            return

        try:
            res = subprocess.run(command, shell=True, check=True, capture_output=True, text=True, timeout=10)
            out, err = res.stdout.strip(), res.stderr.strip()
            if out:
                logger.info("🖥️  STDOUT\n---\n%s\n---", out)
            if err:
                logger.warning("⚠️  STDERR\n---\n%s\n---", err)
            self.status["last_exec"] = {"success": True, "command": command, "stdout": out, "stderr": err}
        except subprocess.CalledProcessError as exc:
            logger.error("❌ Exit code %s\n%s", exc.returncode, exc.stderr)
        except subprocess.TimeoutExpired:
            logger.error("❌ Command timed out after 10 s: %s", command)
        except Exception as exc:  # pragma: no cover
            logger.exception("Exec failed: %s", exc)

    # ----------------------------------------------------------- topic helpers
    def add_topic(self, topic: Dict[str, Any]) -> None:
        """Primitive to add a new topic to GAIA's internal thought cache."""
        topic_manager.add_topic(self.topic_cache_path, topic)

    def resolve_topic(self, topic_id: str) -> bool:
        """Primitive to mark a topic as resolved."""
        return topic_manager.resolve_topic(self.topic_cache_path, topic_id)

    def update_topic(self, topic_id: str, updates: Dict[str, Any]) -> bool:
        """Primitive to update an existing topic's metadata."""
        return topic_manager.update_topic(self.topic_cache_path, topic_id, updates)

    # ------------------------------ Semantic Codex helpers (Rescue Shell)
    def codex_get(self, symbol: str) -> None:
        """Print the body for a codex symbol (e.g., '§CHEATSHEET/PRIMITIVES')."""
        entry = SemanticCodex.instance(self.config).get(symbol if symbol.startswith("§") else f"§{symbol}")
        if entry:
            logger.info(f"[codex] {entry.symbol} ({entry.scope})")
            print(entry.body)
        else:
            logger.info(f"[codex] not found: {symbol}")
            print("[NOT FOUND]")

    def codex_search(self, query: str) -> None:
        """List matching codex symbols."""
        matches = SemanticCodex.instance(self.config).search(query)
        for e in matches:
            print(f"{e.symbol}  —  {e.title}")
        if not matches:
            print("[no matches]")

    def codex_reload(self) -> None:
        """Hot reload codex files if changed on disk."""
        changed = SemanticCodex.instance(self.config).hot_reload()
        print("reloaded" if changed else "no change")

# =============================================================================
#  Interactive chat loop
# =============================================================================

def rescue_chat_loop(ai: MinimalAIManager, session_id: str) -> None:
    """Interactive chat session that streams AgentCore events for a given session."""
    print(f"\n💬 Entering GAIA Rescue chat mode for session: '{session_id}'\n👉 Use '<<<' and '>>>' on new lines for multi‑line input.\n")

    # Backwards‑compatible convenience: if callers pass `ai=None` (common in
    # quick tests), create a MinimalAIManager fallback so the interactive loop
    # still functions instead of raising an AttributeError when accessing
    # `ai.ethical_sentinel` below.
    if ai is None:
        logger.warning("rescue_chat_loop called with ai=None; instantiating MinimalAIManager fallback.")
        try:
            ai = MinimalAIManager()
            try:
                ai.initialize()  # best-effort init (persona 'dev' by default)
            except Exception:
                logger.exception("MinimalAIManager.initialize() failed during rescue fallback; proceeding with uninitialized manager.")
        except Exception:
            logger.exception("Failed to instantiate MinimalAIManager fallback; subsequent AgentCore construction may fail.")

    # Debug: show model pool state inside the running rescue loop/process
    try:
        logger.warning("[MODEL_POOL DEBUG] entering rescue_chat_loop: model_pool id=%s keys=%s", id(model_pool), list(getattr(model_pool, 'models', {}).keys()))
    except Exception:
        logger.exception("[MODEL_POOL DEBUG] failed to inspect model_pool at loop start")
    # If the model pool is empty, attempt a best-effort synchronous load so
    # interactive rescue sessions can function without a separate autoload
    # step. This keeps the interactive experience smooth while honoring the
    # GAIA_AUTOLOAD_MODELS opt-in behavior.
    try:
        if not getattr(model_pool, 'models', {}):
            import os as _os
            if _os.getenv('GAIA_AUTOLOAD_MODELS', '0') == '1':
                logger.info("Model pool empty at interactive loop start; attempting synchronous load_models() because GAIA_AUTOLOAD_MODELS=1")
                try:
                    model_pool.load_models()
                    logger.info("Model pool loaded during rescue_chat_loop; keys=%s", list(getattr(model_pool, 'models', {}).keys()))
                except Exception as _e:
                    logger.warning("Synchronous model_pool.load_models() during rescue loop failed: %s", _e)
            else:
                logger.info("Model pool empty and GAIA_AUTOLOAD_MODELS!=1; interactive features that require LLMs will be limited.")
    except Exception:
        logger.exception("Error while attempting to bootstrap model_pool in rescue_chat_loop")
    # Pass the sentinel to the agent core
    agent_core = AgentCore(ai, ethical_sentinel=ai.ethical_sentinel)

    # Initial pre-user validation prompt: send a friendly identity/availability
    # check so interactive sessions always get an early validation that the
    # responder model is up, knows GAIA's identity, and can respond naturally.
    # This is a lightweight sanity-check and will be skipped when no LLM is
    # available in the pool.
    try:
        # Skip if pool is empty
        if list(getattr(model_pool, 'models', {}).keys()):
            startup_check_prompt = (
                "Hello there! Please introduce yourself briefly in a friendly, natural voice "
                "as GAIA (General Artisanal Intelligence). Say you are in development mode and "
                "that you're ready to assist and collaborate. Keep it short (one or two sentences)."
            )
            logger.info("Rescue.startup: running pre-user identity/availability check")
            print("\n🔎 Performing model availability check...\nGAIA (self-check) > ", end="", flush=True)
            for event in agent_core.run_turn(startup_check_prompt, session_id=session_id, destination="cli_check"):
                et = event.get("type")
                val = event.get("value")
                if et == "token":
                    # If token is a dict with structured fields, try to show response text
                    try:
                        if isinstance(val, dict) and val.get("response_to_user"):
                            print(val.get("response_to_user"), end="", flush=True)
                        else:
                            print(val, end="", flush=True)
                    except Exception:
                        print(val, end="", flush=True)
            print("\n— model availability check complete —\n")
        else:
            logger.info("Rescue.startup: model pool empty; skipping pre-user availability check")
    except Exception:
        logger.exception("Rescue.startup: pre-user availability check failed; continuing to interactive loop")

    approval_pending = None
    validation_phrase = None
    last_command = None

    while True:
        try:
            if approval_pending:
                logger.info("Approval requested for pending command")
                logger.debug("[DEBUG] Approval requested command_len=%d", len(last_command or ""))
                user_input = input(f"Type the validation phrase to approve execution ('{validation_phrase}'): ").strip()
                if user_input == validation_phrase:
                    print("✅ Approved. Executing command...")
                    logger.info("Approval granted for pending command")
                    logger.debug("[DEBUG] Approval granted command_len=%d", len(last_command or ""))
                    # Call agent_core with approval flag
                    for event in agent_core.run_turn(last_command, session_id=session_id, destination="cli_chat", execute_approved=True):
                        et = event.get("type")
                        val = event.get("value")
                        if et == "token":
                            print(val, end="", flush=True)
                            try:
                                logger.info("Approved command executed")
                                logger.debug("[DEBUG] Command executed output_len=%d", len(str(val)))
                            except Exception:
                                logger.debug("[DEBUG] Command executed output_len=unknown")
                    approval_pending = None
                    validation_phrase = None
                    last_command = None
                else:
                    print("❌ Incorrect phrase. Command not approved.")
                    logger.info("Approval denied for pending command")
                    logger.debug("[DEBUG] Approval denied command_len=%d", len(last_command or ""))
                    approval_pending = None
                    validation_phrase = None
                    last_command = None
                continue

            prompt = input("You > ").strip()
            logger.info("Prompt received")
            logger.debug("[DEBUG] Prompt received len=%d", len(prompt))
            if prompt.lower() in {"exit", "quit"}:
                print("\n👋 Exiting chat mode.")
                logger.info("User exited chat mode.")
                break

            if prompt == "<<<":
                print("🔽 Multi-line mode (type >>> to send).")
                lines = []
                while (line := input()) != ">>>":
                    lines.append(line)
                prompt = "\n".join(lines)
                logger.info("Multi-line prompt received")
                logger.debug("[DEBUG] Multi-line prompt received len=%d", len(prompt))

            if not prompt:
                logger.info("Empty prompt received; skipping.")
                continue

            intent_str = "other"
            if os.getenv("GAIA_BACKEND") != "azrael":
                from gaia_core.cognition.nlu.intent_service import detect_intent
                # Prefer using ai.lite_llm if present; otherwise try to acquire an idle 'lite' model
                lite_model = ai.lite_llm
                acquired_name = None
                try:
                    if lite_model is None:
                        try:
                            # Acquire a lite model from the model pool for intent detection
                            lite_model = ai.model_pool.acquire_model_for_role('lite')
                            acquired_name = 'lite'
                        except Exception:
                            lite_model = None

                    plan = detect_intent(prompt, ai.config, lite_llm=lite_model, full_llm=ai.llm)
                    intent_str = getattr(plan, "intent", "other")
                    normalized_prompt = prompt.lower()
                    if intent_str == "read_file":
                        if not any(keyword in normalized_prompt for keyword in ("file", " read", "open ", "cat ", "path", "/", "log")):
                            intent_str = "other"
                    elif intent_str == "write_file":
                        if not any(keyword in normalized_prompt for keyword in ("write", "save", "append", "update", "file", "path")):
                            intent_str = "other"
                    try:
                        plan.intent = intent_str
                        plan.read_only = intent_str in {"read_file", "explain_file", "explain_symbol"}
                    except Exception:
                        pass
                except Exception:
                    logger.exception("Intent detection failed or no model available; defaulting to 'other'.")
                    intent_str = "other"
                finally:
                    # Release the acquired lite model when we are done
                    try:
                        if acquired_name:
                            ai.model_pool.release_model_for_role(acquired_name)
                    except Exception:
                        pass
            else:
                logger.info("GAIA_BACKEND is 'azrael', skipping intent detection.")

            if intent_str in {"mark_task_complete", "reflect"}:
                print("[Intent: Self-Review] Routing to dev_matrix review...")
                from gaia_core.cognition.self_review_worker import run_review_with_prompt
                approval_id, diff = run_review_with_prompt(prompt)
                print(f"\nTo approve, run: gaia_rescue.py --approve {approval_id}")
                continue
            print("GAIA > ", end="", flush=True)

            t_loop_start = time.perf_counter()

            full_response = ""
            # NEW PACKET-DRIVEN FLOW
            for event in agent_core.run_turn(prompt, session_id=session_id):
                print(f"EVENT: {event}", file=sys.stderr)
                et = event.get("type")
                val = event.get("value")

                if et == "token":
                    # Check for validation phrase in the output
                    if isinstance(val, dict) and val.get("validation_phrase"):
                        validation_phrase = val["validation_phrase"]
                        last_command = prompt
                        approval_pending = True
                        logger.info(f"Validation phrase detected: '{validation_phrase}' for command: '{last_command}'")
                        print(strip_think_tags(val["response_to_user"]), end="", flush=True)
                        break
                    clean_val = strip_think_tags(val) if isinstance(val, str) else val
                    print(clean_val, end="", flush=True)
                    full_response += clean_val if isinstance(clean_val, str) else str(clean_val)
                elif et == "interruption_start":
                    print(f"\n\n--- 🔔 INTERRUPT: {event.get('reason', 'Reason not provided.')} ---")
                    print("🤔 Engaging self-reflection to generate a corrected response…\n")
                    logger.info(f"Interrupt detected: {event.get('reason', 'Reason not provided.')}")
                elif et == "correction_start":
                    print("GAIA (Corrected) > ", end="", flush=True)
                    logger.info("Correction started.")
                elif et == "action_failure":
                    logger.error(f"Action failed: {event.get('command')}, Error: {event.get('error')}")
                    print(f"\n❌ Action failed: {event.get('command')}")

            t_loop_end = time.perf_counter()
            logger.info(f"gaia_rescue: run_turn loop took {t_loop_end - t_loop_start:.2f}s")
            print()  # newline for next prompt

        except (KeyboardInterrupt, EOFError):
            print("\n👋 Exiting chat mode.")
            break
        except Exception:  # pragma: no cover
            logger.exception("Chat loop error: %s", exc)
            print(f"\n❌ Error: {exc}")

# =============================================================================
#  Non-interactive single turn runner
# =============================================================================
def run_single_turn_non_interactive(ai: MinimalAIManager, prompt: str, session_id: str) -> None:
    """Runs a single turn of GAIA conversation non-interactively."""
    logger.info(f"Running single non-interactive turn for session: '{session_id}' with prompt: '{prompt}'")
    agent_core = AgentCore(ai, ethical_sentinel=ai.ethical_sentinel)
    for event in agent_core.run_turn(prompt, session_id=session_id):
        print(f"EVENT: {event}", file=sys.stderr)
        # We only care about the final output for non-interactive mode
        if event.get("type") == "token":
            value = event.get("value", "")
            print(strip_think_tags(value), end="", flush=True)
    print() # Ensure a newline at the end


# =============================================================================
#  Discord Integration
# =============================================================================
def start_discord_listener(ai: MinimalAIManager = None, session_id_prefix: str = "discord"):
    """
    Start the Discord bot listener that routes messages to AgentCore.
    Handles both channel mentions and DMs.

    Args:
        ai: MinimalAIManager instance (will be created if None)
        session_id_prefix: Prefix for channel-based sessions

    Returns:
        The DiscordConnector instance, or None if startup failed
    """
    try:
        from gaia_core.integrations.discord_connector import DiscordConnector, DiscordConfig
        from gaia_common.utils.destination_registry import register_connector

        # Initialize AI manager if not provided
        if ai is None:
            ai = MinimalAIManager()

        agent_core = AgentCore(ai, ethical_sentinel=ai.ethical_sentinel)

        # Create Discord connector
        config = DiscordConfig.from_env()
        if not config.bot_token:
            logger.error("Cannot start Discord listener: DISCORD_BOT_TOKEN not set")
            return None

        connector = DiscordConnector(config)

        connector = DiscordConnector(config)

        def handle_discord_message(content: str, author_id: str, metadata: dict):
            """Callback for incoming Discord messages (channels and DMs)."""
            is_dm = metadata.get("is_dm", False)
            session_id = metadata.get("session_id", f"{session_id_prefix}_{author_id}")
            source = "discord_dm" if is_dm else "discord_channel"

            logger.info(f"Discord: Received {'DM' if is_dm else 'channel message'} from {metadata.get('author_name', author_id)}")

            # Run AgentCore in a separate process to avoid blocking the Discord bot
            import concurrent.futures

            full_response = ""
            try:
                with concurrent.futures.ProcessPoolExecutor(max_workers=1) as executor:
                    # Use the module-level _run_agent_core_in_process function (picklable)
                    future = executor.submit(_run_agent_core_in_process, content, session_id, "discord", source, metadata)
                    full_response = future.result(timeout=180)  # Add a timeout for safety (3 minutes)
            except concurrent.futures.TimeoutError:
                logger.error("Discord: AgentCore.run_turn process timed out.")
                full_response = "I'm sorry, I took too long to respond. Please try again."
            except Exception:
                logger.exception("Discord: AgentCore.run_turn process failed.")
                full_response = "I encountered an error processing your message. Please try again."

            # Strip think tags from the COMPLETE response (not token-by-token)
            # This handles multi-token tags like <think>..content..</think>
            full_response = strip_think_tags(full_response)

            # If stripping left us with nothing, provide a fallback
            if not full_response.strip():
                logger.warning("Discord: Response empty after stripping think tags")
                full_response = "I processed your request but my response was incomplete. Please try again."

            # Optional: Check response quality for leaked meta-content
            try:
                from gaia_core.utils.stream_observer import StreamObserver
                quality_issue = StreamObserver.check_response_quality(full_response, content)
                if quality_issue and quality_issue.level in ("CAUTION", "BLOCK"):
                    logger.warning(f"Discord: Response quality issue: {quality_issue.reason}")
                    # Try stripping again more aggressively
                    full_response = strip_think_tags(full_response)
            except Exception:
                pass  # Quality check is optional, don't fail on errors

            # Send response back via Discord
            try:
                from gaia_common.protocols.cognition_packet import DestinationTarget

                # Always include user_id for DMs to enable direct messaging
                # For channel messages, we still include channel_id
                target = DestinationTarget(
                    destination=None,  # Will be determined by connector
                    channel_id=metadata.get("channel_id"),
                    user_id=author_id,  # Always include - connector will decide how to use it
                    metadata=metadata
                )
                logger.debug(f"Discord: Sending response (is_dm={is_dm}, user_id={author_id}, channel_id={metadata.get('channel_id')})")
                
                # Use the connector to send the message
                connector.send(full_response, target)
                
                logger.info(f"Discord: Sent response to {'DM' if is_dm else 'channel'}")
            except Exception:
                logger.exception("Discord: Failed to send response")
        
        # Register callback and start listener
        connector.set_message_callback(handle_discord_message)

        # Register with destination registry for output routing
        try:
            register_connector(connector)
            logger.info("Discord connector registered with DestinationRegistry")
        except Exception:
            logger.warning("Failed to register Discord connector with DestinationRegistry")

        if connector.start_bot_listener():
            logger.info("Discord bot listener started successfully")
            return connector
        else:
            logger.error("Discord bot listener failed to start")
            return None

    except ImportError as e:
        logger.error(f"Discord integration not available: {e}")
        return None
    except Exception:
        logger.exception("Failed to start Discord listener")
        return None


    # Setup file logging
    log_dir = os.path.join(os.path.dirname(__file__), "..", "logs")
    try:
        os.makedirs(log_dir, exist_ok=True)
        file_path = os.path.join(log_dir, "gaia_rescue.log")
    except PermissionError:
        # Fall back to a writable temp directory when the configured logs
        # location is not writable (common inside restrictive containers).
        import tempfile
        try:
            fallback = tempfile.mkdtemp(prefix="gaia_logs_")
            logger.warning("Permission denied creating %s; falling back to %s", log_dir, fallback)
            file_path = os.path.join(fallback, "gaia_rescue.log")
        except Exception:
            # Last-resort fallback to /tmp
            logger.exception("Failed to create fallback temp log dir; falling back to /tmp")
            file_path = "/tmp/gaia_rescue.log"
    except Exception:
        # Last-resort fallback to /tmp
        logger.exception("Unexpected error creating log dir %s; falling back to /tmp", log_dir)
        file_path = "/tmp/gaia_rescue.log"
    file_handler = logging.FileHandler(file_path, mode="a")
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s %(name)s %(levelname)s %(message)s")
    file_handler.setFormatter(formatter)
    logging.getLogger().addHandler(file_handler)

# =============================================================================
#  CLI entry‑point
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description="GAIA Rescue Shell")
    parser.add_argument("--session-id", type=str, default="cli_default_session", help="Session ID to use/continue.")
    parser.add_argument("--persona", type=str, default="dev", help="Persona to load (default: dev).")
    parser.add_argument("--prompt", type=str, help="Run a single prompt and exit.")
    parser.add_argument("--use-oracle", action="store_true", help="Enable the use of the Oracle model.")
    parser.add_argument("--register-dev-model", action="store_true", help="Register the dev model on startup.")
    parser.add_argument("--reuse-packet", type=str, help="Path to JSON packet to reuse")
    parser.add_argument("--reflect-packet", type=str, help="Path to packet for reflection")
    parser.add_argument("--set", nargs='*', help="Packet field updates in key=value format")
    parser.add_argument("--review", action="store_true", help="Run a self-review pass and request approval for any proposed dev_matrix updates")
    parser.add_argument("--approve", type=str, help="Non-interactive approval string to submit (the reversed challenge) when using --review")
    parser.add_argument("--nl-prompt", type=str, help="Natural-language prompt for dev_matrix review")
    parser.add_argument("--single-turn-prompt", type=str, help="Run a single prompt non-interactively and exit.")
    parser.add_argument("--discord", action="store_true", help="Start Discord bot listener (requires DISCORD_BOT_TOKEN)")
    parser.add_argument("--discord-only", action="store_true", help="Run only Discord bot (no interactive CLI)")
    # Study Mode / LoRA Adapter commands (Commented out for refactoring)
    # parser.add_argument("--study", type=str, nargs='+', metavar='DOC', help="Start study mode to learn from documents")
    # parser.add_argument("--study-name", type=str, default=None, help="Name for the adapter being trained")
    # parser.add_argument("--study-tier", type=int, default=3, choices=[1, 2, 3], help="Adapter tier (1=global, 2=user, 3=session)")
    # parser.add_argument("--study-pillar", type=str, default="general", help="GAIA pillar alignment")
    # parser.add_argument("--study-triggers", type=str, nargs='*', help="Keywords to trigger this adapter")
    # parser.add_argument("--adapter-list", action="store_true", help="List available LoRA adapters")
    # parser.add_argument("--adapter-info", type=str, metavar="NAME", help="Get info about a specific adapter")
    # parser.add_argument("--adapter-delete", type=str, metavar="NAME", help="Delete an adapter")
    parser.add_argument("--embed-knowledge-base", type=str, help="Embed documents for a specific knowledge base.")
    args = parser.parse_args()
    SESSION_ID = args.session_id

    import os
    # If operator requested oracle mode, force backend selection to oracle_openai
    # so intent/routing won't pick gpu_prime.
    if getattr(args, "use_oracle", False):
        os.environ["GAIA_BACKEND"] = "oracle_openai"

    ai = MinimalAIManager()
    mgr = get_manager()
    # Note: ensure_prime_loaded() is called later in the autoload block (lines 889-890)
    # to avoid loading prime unconditionally. Prime should only load when autoload=1.

    # Graceful shutdown hooks (SIGINT/SIGTERM/atexit) to stop model backends cleanly.
    try:
        import signal
        import atexit

        def _graceful_shutdown(signum=None, _frame=None):
            try:
                logger.info("Graceful shutdown initiated%s", f" (signal {signum})" if signum else "")
                if getattr(ai, "shutdown", None):
                    ai.shutdown()
            except Exception:
                logger.debug("Graceful shutdown hook failed", exc_info=True)
            if signum is not None:
                raise SystemExit(0)

        if os.getenv("GAIA_GRACEFUL_SHUTDOWN", "1") != "0":
            signal.signal(signal.SIGINT, _graceful_shutdown)
            signal.signal(signal.SIGTERM, _graceful_shutdown)
            atexit.register(_graceful_shutdown)
    except Exception:
        logger.debug("Failed to register graceful shutdown hooks", exc_info=True)
    
    # Allow operator to force which model provides observer/lite roles via env
    observer_role = os.getenv("GAIA_OBSERVER_ROLE")
    if observer_role:
        # If the operator requests 'lite' to be used for observer, set the sharing env var
        if observer_role in ("lite", "prime", "gpu_prime"):
            os.environ.setdefault("GAIA_SHARE_LITE_WITH", observer_role)
    # Ensure 'prime' config is set as alias for 'gpu_prime' if CUDA is available,
    # but default backend stays Operator (lite) unless explicitly overridden.
    try:
        import torch
        if torch.cuda.is_available() and "gpu_prime" in ai.model_pool.config.MODEL_CONFIGS:
            ai.model_pool.config.MODEL_CONFIGS["prime"] = {"alias": "gpu_prime", "enabled": True, "type": "local"}
            # Do not override GAIA_BACKEND unless operator explicitly sets it.
        logger.info("Loading gpu_prime and lite models at startup (deferred to autoload or explicit request).")
    except Exception as e:
        logger.error(f"Model loading failed: {e}")
    ai.initialize(args.persona)

    # Load models only if explicitly enabled or if the /models directory
    # contains files. This keeps the rescue shell fast on hosts that do not
    # provide local model binaries (common in development or when models are
    # provided via external mounts).
    models_dir = os.environ.get("MODELS_DIR", "/models")
    autoload = os.environ.get("GAIA_AUTOLOAD_MODELS", "0") == "1"
    models_present = False
    try:
        if os.path.isdir(models_dir):
            for f in os.listdir(models_dir):
                fp = os.path.join(models_dir, f)
                if os.path.isfile(fp) and os.path.getsize(fp) > 0:
                    models_present = True
                    break
    except Exception:
        models_present = False

    # Only auto-load models when GAIA_AUTOLOAD_MODELS is explicitly enabled.
    # Previously we would also load when model files were present; that leads
    # to surprise loads when users simply `docker run` into the image. Require
    # an explicit opt-in so `bash` stays fast and quiet by default.
    if autoload:
        logger.info("GAIA_AUTOLOAD_MODELS=1; loading models from %s", models_dir)
        try:
            logger.warning("[MODEL_POOL DEBUG] before load: id=%s keys=%s", id(model_pool), list(getattr(model_pool, 'models', {}).keys()))
        except Exception:
            logger.exception("[MODEL_POOL DEBUG] failed to inspect model_pool before load")
        # First, attempt to ensure 'prime' is loaded via the ModelManager which
        # will prefer an in-process load but fall back to a spawn-based loader
        # when necessary. Calling this from main() (instead of at import time)
        # avoids the multiprocessing bootstrap race.
        # Skip prime autoload when operator forces a non-prime backend (e.g., lite)
        try:
            backend_env = os.getenv('GAIA_BACKEND', '').strip().lower()
            skip_prime = os.getenv('GAIA_SKIP_PRIME_LOAD', '0') == '1'
            if backend_env in ('lite', 'observer') or skip_prime:
                logger.info("Skipping ensure_prime_loaded because GAIA_BACKEND=%s or GAIA_SKIP_PRIME_LOAD=1", backend_env or '<unset>')
            else:
                mgr = get_manager()
                res = mgr.ensure_prime_loaded(force=False, timeout=180)
                logger.info("ensure_prime_loaded result: %s", res)
        except Exception:
            logger.exception("ensure_prime_loaded() raised an exception")

        # Then load remaining models (lite/observer/embed) into the pool.
        model_pool.load_models(args.use_oracle)
        try:
            logger.warning("[MODEL_POOL DEBUG] after load: id=%s keys=%s", id(model_pool), list(getattr(model_pool, 'models', {}).keys()))
        except Exception:
            logger.exception("[MODEL_POOL DEBUG] failed to inspect model_pool after load")
        # If requested, enforce that critical models are present and fail fast
        try:
            allow_missing = os.getenv('GAIA_ALLOW_MISSING_MODELS', '0') == '1'
            fail_on_missing = os.getenv('GAIA_FAIL_ON_MISSING_MODELS', '0') == '1'
            missing = []
            if 'prime' not in model_pool.models:
                missing.append('prime')
            if 'lite' not in model_pool.models:
                missing.append('lite')
            if missing:
                msg = f"Critical models missing after autoload: {missing}."
                if fail_on_missing and not allow_missing:
                    logger.error(msg + " Exiting due to GAIA_FAIL_ON_MISSING_MODELS=1.")
                    sys.exit(2)
                else:
                    logger.warning(msg + " Continuing because GAIA_ALLOW_MISSING_MODELS is set or fail-on-missing not enabled.")
        except Exception:
            logger.exception("Error while validating autoloaded models")
    else:
        if models_present:
            logger.info("Model files exist at %s but GAIA_AUTOLOAD_MODELS=0; skipping automatic load. Set GAIA_AUTOLOAD_MODELS=1 or pass --register-dev-model/other flags to load models.", models_dir)
        else:
            logger.info("No model files found at %s and GAIA_AUTOLOAD_MODELS not set; skipping model load in rescue mode.", models_dir)

    # Optional keepalive to keep EngineCore warm. Enable by setting GAIA_KEEPALIVE_SECONDS>0.
    # GAIA_KEEPALIVE_MODEL selects which role to ping (default: gpu_prime).
    def _start_keepalive():
        try:
            interval = float(os.getenv("GAIA_KEEPALIVE_SECONDS", "0"))
        except Exception:
            interval = 0
        if interval <= 0:
            return
        model_name = os.getenv("GAIA_KEEPALIVE_MODEL", "gpu_prime")
        def _loop():
            while True:
                try:
                    msgs = [{"role": "user", "content": "ping"}]
                    model_pool.forward_to_model(model_name, messages=msgs, max_tokens=1, temperature=0.0, top_p=1.0)
                except Exception:
                    pass
                time.sleep(interval)
        t = threading.Thread(target=_loop, name="gaia-keepalive", daemon=True)
        t.start()
    _start_keepalive()

    if args.register_dev_model:
        model_pool.register_dev_model("azrael")

    if args.embed_knowledge_base:
        from gaia_common.utils.vector_indexer import VectorIndexer
        VectorIndexer.instance(args.embed_knowledge_base).build_index_from_docs()
        sys.exit(0)

    if args.reuse_packet:
        with open(args.reuse_packet, "r") as f:
            raw = json.load(f)
            packet = CognitionPacket.from_dict(raw)
        updated_packet = AgentCore(ai).reflect_on_packet(packet, dict(s.split("=", 1) for s in args.set or []))
        for chunk in AgentCore(ai).run_turn_from_packet(updated_packet, session_id=SESSION_ID):
            print(chunk.get("content", ""), end="", flush=True)
        sys.exit()

    if args.reflect_packet:
        with open(args.reflect_packet, "r") as f:
            raw = json.load(f)
            packet = CognitionPacket.from_dict(raw)
        reflected = AgentCore(ai).reflect_on_packet(packet, dict(s.split("=", 1) for s in args.set or []))
        print("\n--- Reflected Packet ---\n")
        print(reflected.to_json())
        sys.exit()

    # Reset the session state at the start of a rescue run to ensure a clean slate.
    # This prevents state from a previous run from polluting the current one.
    logger.info(f"Clearing session '{SESSION_ID}' for a clean test run.")
    ai.session_manager.reset_session(SESSION_ID)

    analyzer = DevMatrixAnalyzer(ai.config)
    analyzer.analyze_and_update()

    def _run_slim_prompt(prompt: str, model_name: str = None):
        """Bypass plan/reflect for single-shot prompts while still providing identity + MCP context."""
        model_role = model_name or os.getenv("GAIA_BACKEND") or "gpu_prime"
        mcp_summary = ai.config.constants.get("mcp_capabilities_summary") or ai.config.constants.get("mcp_capabilities") or ""
        system = (
            "You are GAIA - General Artisanal Intelligence Architecture. "
            "Answer the user directly and succinctly. "
            f"MCP Body: {mcp_summary}"
        )
        # Try to use a raw completion to avoid chat templates from qwen_chat.
        try:
            model_obj = ai.model_pool.get_model_for_role(model_role)
        except Exception:
            model_obj = None
        prompt_text = f"{system}\n\nUser: {prompt}\nAssistant:"
        res = None
        if model_obj and hasattr(model_obj, "create_completion"):
            try:
                res = model_obj.create_completion(
                    prompt_text,
                    max_tokens=256,
                    temperature=ai.config.temperature,
                    top_p=ai.config.top_p,
                    stream=False,
                    stop=None,
                )
            except Exception:
                res = None
        if res is None:
            # Fallback to chat format via forward_to_model (single call only)
            messages = [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ]
            res = ai.model_pool.forward_to_model(
                model_role,
                messages=messages,
                max_tokens=256,
                temperature=ai.config.temperature,
                top_p=ai.config.top_p,
            )
        try:
            text = res["choices"][0]["message"]["content"]
        except Exception:
            text = str(res)
        print(text, flush=True)
        return

    # --- Self-review & approval flow (CLI helper)
    if args.review:
        # Run a single review pass
        from gaia_core.cognition import self_review_worker
        print("Running self-review...")
        res = self_review_worker.run_review_once()
        if not res:
            print("No actionable self-review proposals were generated.")
            sys.exit(0)

        action_id = res.get("action_id")
        challenge = res.get("challenge")
        created_at = res.get("created_at")
        expiry = res.get("expiry")
        proposal = res.get("proposal") if isinstance(res, dict) else None
        print(f"Pending action_id={action_id} challenge={challenge} created_at={created_at} expiry={expiry}")
        if proposal:
            print("\n--- Proposal (for human review) ---\n")
            # Visual codeblock-like print for CLI; keep it simple and safe
            print("```diff")
            print(proposal)
            print("```")

        # Determine approval string (reversed challenge expected)
        approval_input = args.approve
        if not approval_input:
            print("Type the reversed challenge to approve the pending action, or press Enter to cancel:")
            user_in = input().strip()
            if not user_in:
                print("Approval cancelled by user.")
                sys.exit(1)
            approval_input = user_in

        # Try to submit via MCP HTTP endpoint first
        try:
            if ai.config.constants.get("MCP_LITE_ENDPOINT"):
                print("Submitting approval via MCP endpoint...")
                r = mcp_client.approve_action_via_mcp(action_id, approval_input)
                if not r.get("ok"):
                    print(f"MCP approval failed: {r.get('error')}. Falling back to local approval if available.")
                    raise RuntimeError("MCP approval failed")
                print("Approval submitted via MCP. Result:")
                print(r.get("result"))
                sys.exit(0)
        except Exception:
            # The local approval logic has been removed.
            # The new architecture uses an HTTP client to communicate with the MCP.
            pass

    # # --- Automated Fine-Tuning Check (Temporarily Disabled) ---
    # # needs_training, token_count, delta_path = check_for_training_delta(ai.config)
    # # if needs_training:
    # #     print(f"\n✨ New training data found! (~{token_count} tokens)")
    # #     response = input("Do you want to fine-tune the model now? (y/n): ").lower()
    # #     if response == 'y':
    # #         base_model_name = get_base_model_name(ai.config)
    # #         full_retrain = base_model_name not in open(os.path.join(ai.config.KNOWLEDGE_DIR, "training_log.json")).read()
    # #         new_model_version = get_next_model_version(ai.config, full_retrain)
    # #         
    # #         training_process = subprocess.run([
    # #             "python", "app/models/fine_tune_gaia.py",
    # #             "--dataset", delta_path,
    # #             "--base-model", "gpt2", # Or another suitable base model from HF
    # #             "--output-name", new_model_version
    # #         ])

    # #         if training_process.returncode == 0:
    # #             final_model_path = os.path.join(ai.config.LORA_ADAPTERS_DIR, new_model_version, "final")
    # #             gguf_output_path = os.path.join(ai.config.LORA_ADAPTERS_DIR, f"{new_model_version}.gguf")
    # #             convert_to_gguf(final_model_path, gguf_output_path)
    # #             
    # #             with open(delta_path, 'r') as f:
    # #                 new_entries = [json.loads(line) for line in f]
    # #             update_training_log(ai.config, new_entries, new_model_version)
    # #             print(f"✅ Fine-tuning complete. New model saved as {new_model_version}.gguf")
    # #         else:
    # #             print("❌ Fine-tuning process failed.")

    # --- Study Mode / LoRA Adapter CLI commands ---
    # if args.adapter_list:
    #     pass

    # if args.adapter_info:
    #     pass

    # if args.adapter_delete:
    #     pass

    # Study mode disabled - requires gaia-study service via HTTP
    # if args.study:
    #     try:
    #         import asyncio
    #         from gaia_core.cognition.study_mode_manager import StudyModeManager, TrainingConfig, TrainingConfig
    #         constants_path = os.path.join(os.path.dirname(__file__), "app", "gaia_constants.json")
    #         with open(constants_path) as f:
    #             constants = json.load(f)
    #         study_config = constants.get("STUDY_MODE", {})
    #         lora_config = constants.get("LORA_CONFIG", {})
    #         model_configs = constants.get("MODEL_CONFIGS", {})
    #
    #         # Set base model path if available
    #         gpu_prime = model_configs.get("gpu_prime", {})
    #         if gpu_prime.get("path"):
    #             study_config["base_model_path"] = gpu_prime["path"]
    #
    #         adapter_dir = lora_config.get("adapter_dir", "/models/lora_adapters")
    #         manager = StudyModeManager(study_config, adapter_base_dir=adapter_dir)
    #
    #         # Generate adapter name if not provided
    #         adapter_name = args.study_name
    #         if not adapter_name:
    #             # Generate from first document name
    #             from pathlib import Path
    #             first_doc = Path(args.study[0]).stem
    #             adapter_name = f"{first_doc}_adapter"
    #             adapter_name = adapter_name.lower().replace(" ", "_").replace("-", "_")
    #
    #         print(f"\n📚 Starting Study Mode")
    #         print(f"   Adapter name: {adapter_name}")
    #         print(f"   Documents: {', '.join(args.study)}")
    #         print(f"   Tier: {args.study_tier}")
    #         print(f"   Pillar: {args.study_pillar}")
    #         print()
    #
    #         config = TrainingConfig(
    #             adapter_name=adapter_name,
    #             tier=args.study_tier,
    #             pillar=args.study_pillar,
    #             source_documents=args.study,
    #             description=f"Learned from: {', '.join(args.study)}",
    #             activation_triggers=args.study_triggers or [],
    #         )
    #
    #         async def run_training():
    #             return await manager.start_training(config)
    #
    #         print("🔄 Training in progress...")
    #         result = asyncio.run(run_training())
    #
    #         if result.success:
    #             print(f"\n✅ Training complete!")
    #             print(f"   Adapter path: {result.adapter_path}")
    #             print(f"   Final loss: {result.final_loss:.4f}" if result.final_loss else "")
    #             print(f"   Steps: {result.training_steps}")
    #             print(f"   Duration: {result.duration_seconds:.1f}s")
    #             print(f"   Samples processed: {result.samples_processed}")
    #         else:
    #             print(f"\n❌ Training failed: {result.error_message}")
    #
    #     except Exception as e:
    #         logger.exception("Study mode failed")
    #         print(f"❌ Error: {e}")
    #     sys.exit(0)

    # Discord bot mode
    discord_connector = None
    if args.discord or args.discord_only:
        print("\n🤖 Starting Discord bot listener...")
        discord_connector = start_discord_listener(ai, session_id_prefix=SESSION_ID)
        if discord_connector:
            print("✅ Discord bot is running. GAIA will respond to @mentions and DMs.")
            if args.discord_only:
                print("Running in Discord-only mode. Press Ctrl+C to exit.")
                try:
                    # Keep the main thread alive
                    while True:
                        time.sleep(1)
                except KeyboardInterrupt:
                    print("\n👋 Shutting down Discord bot...")
                    discord_connector.stop_bot_listener()
                    sys.exit(0)
        else:
            print("❌ Failed to start Discord bot. Check DISCORD_BOT_TOKEN is set.")
            if args.discord_only:
                sys.exit(1)

    if args.single_turn_prompt:
        run_single_turn_non_interactive(ai, args.single_turn_prompt, session_id=SESSION_ID)
        sys.exit(0) # Exit after single turn
    elif args.prompt:
        if os.getenv("GAIA_SLIM_PROMPT", "0") == "1":
            _run_slim_prompt(args.prompt)
            return
        try:
            logger.warning("[MODEL_POOL DEBUG] at single-prompt run: model_pool id=%s keys=%s", id(model_pool), list(getattr(model_pool, 'models', {}).keys()))
        except Exception:
            logger.exception("[MODEL_POOL DEBUG] failed to inspect model_pool before single-prompt run")
        agent_core = AgentCore(ai, ethical_sentinel=ai.ethical_sentinel)
        for event in agent_core.run_turn(args.prompt, session_id=SESSION_ID):
            print(f"EVENT: {event}", file=sys.stderr)
            value = event.get("value", "")
            print(strip_think_tags(value) if isinstance(value, str) else value, end="", flush=True)
        print()
    else:
        discord_status = "running" if discord_connector else "not started (use --discord flag)"
        print(
            "\n🧠 GAIA Rescue Shell initialized.\n"
            f"   Session ID: {SESSION_ID}\n"
            f"   Discord: {discord_status}\n\n"
            "Diagnostics & direct interaction available.\n\n"
            "• rescue_chat_loop()         - start interactive chat for the current session\n"
            "• ai.read('path') / ai.write - file ops\n"
            "• ai.execute('ls -l')        - safe shell\n"
            "• ai.helper.*                - helper utilities\n"
            f"• ai.session_manager.reset_session('{SESSION_ID}') - clear this session's history\n"
            "• reload('app.utils.gaia_rescue_helper') - hot-reload helper\n"
            "• start_discord_listener(ai) - start Discord bot manually\n"
            "• exit() or Ctrl-D           - quit\n"
        )
        # After interactive shell returns, perform best-effort shutdown of AI resources
        # NOTE: calling ai.shutdown() here will clear the shared model pool which
        # may terminate vLLM engine processes while you're still debugging.
        # Default behavior: do NOT shut down automatically. Set
        # GAIA_AUTO_SHUTDOWN=1 in the environment to opt in to automatic
        # shutdown after the interactive shell exits.
        try:
            if os.getenv("GAIA_AUTO_SHUTDOWN", "0") == "1":
                if getattr(ai, 'shutdown', None):
                    ai.shutdown()
            else:
                logger.info("Skipping automatic ai.shutdown() after interactive shell; set GAIA_AUTO_SHUTDOWN=1 to enable")
        except Exception:
            logger.debug("ai.shutdown() failed after interactive shell", exc_info=True)

        code.interact(
            local={
                "ai": ai,
                "rescue_chat_loop": lambda: rescue_chat_loop(ai, SESSION_ID),
                "status": lambda: print(ai.status),
                "reload": ai.reload,
                "vector_query": vector_query,
                "embed_reference": embed_gaia_reference,
                # [ADDED][CODEX] handy one-liners for shell testing
                "codex_get": ai.codex_get,
                "codex_search": ai.codex_search,
                "codex_reload": ai.codex_reload,
                "register_dev_model": lambda: model_pool.register_dev_model("azrael"),
                # Discord integration
                "start_discord_listener": lambda: start_discord_listener(ai, SESSION_ID),
                "discord_connector": discord_connector,
            }
        )

    if args.review:
        # ...existing review logic...
        pass
    if args.nl_prompt:
        result = run_review_with_prompt(args.nl_prompt)
        # Backwards-compatible: the helper may return (approval_id, diff) or
        # the extended form (approval_id, diff, proposal, created_at, expiry).
        approval_id = None
        diff = None
        proposal = None
        created_at = None
        expiry = None
        if isinstance(result, (list, tuple)):
            if len(result) >= 2:
                approval_id, diff = result[0], result[1]
            if len(result) >= 3:
                proposal = result[2]
            if len(result) >= 4:
                created_at = result[3]
            if len(result) >= 5:
                expiry = result[4]
        elif isinstance(result, dict):
            approval_id = result.get("action_id")
            diff = result.get("diff") or result.get("proposal")
            proposal = result.get("proposal")
            created_at = result.get("created_at")
            expiry = result.get("expiry")

        print(f"\nTo approve, run: gaia_rescue.py --approve {approval_id}")
        if proposal:
            print("\n--- Proposal (for human review) ---\n")
            print("```diff")
            print(proposal)
            print("```")
        if created_at or expiry:
            print(f"Created: {created_at}  Expiry: {expiry}")
        sys.exit(0)

if __name__ == "__main__":
    main()
