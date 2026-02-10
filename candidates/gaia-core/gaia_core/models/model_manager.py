"""ModelManager: a small spine to query model status, ensure models are loaded
and provide a safe spawn-based fallback for loading Prime (vLLM) when direct
in-process load fails due to CUDA/multiprocessing issues.

This module aims to centralize model lifecycle checks and provide a single
surface for other parts of the app (runserver, gaia_rescue) to request model
availability and run simple calls.
"""
from __future__ import annotations

import logging
import multiprocessing
import time
from typing import Any, Dict, Optional

logger = logging.getLogger("GAIA.ModelManager")


def _model_manager_child_loader(q, force_flag):
    """Module-level helper used as spawn target so it is picklable.

    The child process imports a fresh model_pool and attempts to enable and
    load the prime model. Results are placed on the provided queue.
    """
    try:
        from gaia_core.models.model_pool import model_pool as child_pool
        try:
            child_pool.enable_prime_load()
        except Exception:
            pass
        results = {"ok": True}
        try:
            ok = child_pool.load_prime_only(force=force_flag)
        except TypeError:
            ok = child_pool.load_prime_only(True)
        results["loaded"] = bool(ok)
        # Best-effort: also initialize lite model for observer if configured
        try:
            lite_ok = False
            if hasattr(child_pool, 'load_lite_only'):
                lite_ok = bool(child_pool.load_lite_only())
            results["lite_loaded"] = lite_ok
        except Exception:
            results["lite_loaded"] = False
        q.put(results)
    except Exception as e:
        import traceback as _tb

        q.put({"ok": False, "error": str(e), "trace": _tb.format_exc()})


class ModelManager:
    """Singleton-ish manager around the existing model_pool.

    It does not itself host models; instead it delegates to app.models.model_pool
    but adds spawn-aware helpers and a consistent status API.
    """

    _instance: Optional["ModelManager"] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        # lazy import to avoid triggering heavy imports at module import time
        self._model_pool = None
        self._last_load_error: Optional[str] = None

    def _get_pool(self):
        if self._model_pool is None:
            try:
                        from gaia_core.models.model_pool import get_model_pool  # local import

                        # Use the lazy accessor to avoid importing heavy model libs at
                        # module-import time. get_model_pool() returns the proxied
                        # singleton (or None on failure).
                        self._model_pool = get_model_pool()
            except Exception as e:
                logger.exception("Failed to import model_pool: %s", e)
                self._last_load_error = str(e)
                self._model_pool = None
        return self._model_pool

    def ensure_prime_loaded(self, force: bool = False, timeout: int = 120) -> Dict[str, Any]:
        """Ensure 'prime' (GPU-backed) model is resident.

        Strategy:
        1) Try to enable prime load and call load_prime_only() in-process.
        2) If that fails with multiprocessing/CUDA re-init style problems, spawn a
           fresh interpreter using multiprocessing.get_context('spawn') and run
           the loader there. Return a dict describing success and any error/trace.
        """
        pool = self._get_pool()
        if pool is None:
            return {"ok": False, "error": "model_pool unavailable", "trace": self._last_load_error}

        # Decide whether to attempt an in-process load or prefer the spawn
        # fallback. If the current multiprocessing start method is not
        # 'spawn' or CUDA appears already initialized in this interpreter,
        # prefer spawn first to avoid the "Cannot re-initialize CUDA in forked
        # subprocess" error.
        prefer_spawn = False
        try:
            import multiprocessing as _mp
            start_method = _mp.get_start_method(allow_none=True)
            if start_method != "spawn":
                prefer_spawn = True
        except Exception:
            # if we can't determine, be conservative and prefer spawn
            prefer_spawn = True

        # Honor an explicit environment override to force spawn-based loader.
        try:
            import os as _os
            if _os.getenv("GAIA_FORCE_SPAWN", "0") == "1":
                prefer_spawn = True
                logger.info("ModelManager: GAIA_FORCE_SPAWN=1; forcing spawn-based loader")
        except Exception:
            pass

        try:
            # Avoid importing torch at module import time; only import when
            # necessary to check CUDA state.
            import importlib

            try:
                torch_spec = importlib.util.find_spec("torch")
                if torch_spec is not None:
                    torch = importlib.import_module("torch")
                    # If CUDA has already been initialized in this process,
                    # an in-process vLLM init is risky.
                    try:
                        if getattr(torch.cuda, "is_initialized", lambda: False)():
                            prefer_spawn = True
                    except Exception:
                        # be conservative if we cannot query
                        prefer_spawn = True
            except Exception:
                # If any torch import issue arises, prefer spawn fallback
                prefer_spawn = True
        except Exception:
            prefer_spawn = True

        if prefer_spawn:
            logger.info("ModelManager: preferring spawn-based loader (start_method != 'spawn' or CUDA initialized)")
        else:
            logger.info("ModelManager: attempting in-process prime load first")

        if not prefer_spawn:
            # Attempt in-process fast path
            try:
                logger.info("ModelManager: enabling prime load in-process")
                try:
                    pool.enable_prime_load()
                except Exception:
                    # ignore if shim not present
                    logger.debug("enable_prime_load() not present or failed (ignored)")
                # Many model_pool.load_prime_only implementations accept force kw
                try:
                    ok = pool.load_prime_only(force=force)
                except TypeError:
                    # old signature used positional True
                    ok = pool.load_prime_only(True)

                if ok:
                    logger.info("ModelManager: prime loaded in-process")
                    return {"ok": True, "loaded": True, "method": "in-process"}
            except Exception as e:
                # Capture exception and attempt spawn fallback
                import traceback

                trace = traceback.format_exc()
                logger.warning("In-process prime load failed; attempting spawn fallback: %s", e)
                self._last_load_error = str(e)

        # Spawn fallback: use a fresh interpreter created via spawn context to
        # attempt model load. This avoids fork/CUDA re-init problems.
        try:
            ctx = multiprocessing.get_context("spawn")
            q = ctx.Queue()

            # Use the module-level child loader which is picklable by spawn.
            p = ctx.Process(target=_model_manager_child_loader, args=(q, force))
            p.start()
            p.join(timeout)
            if p.is_alive():
                logger.warning("Spawned loader still alive after timeout; terminating")
                p.terminate()
                p.join(5)
            result = None
            try:
                if not q.empty():
                    result = q.get_nowait()
            except Exception:
                result = None

            if result is None:
                return {"ok": False, "error": "spawned loader produced no result"}
            if result.get("ok"):
                return {"ok": True, "loaded": result.get("loaded"), "method": "spawn"}
            return {"ok": False, "error": result.get("error"), "trace": result.get("trace")}
        except Exception as e:
            logger.exception("Spawn fallback failed: %s", e)
            return {"ok": False, "error": str(e)}

    def call_model(self, role: str, *args, **kwargs) -> Dict[str, Any]:
        """Call a model by role (e.g., 'prime' or 'lite'). Returns a dict with
        either 'result' or 'error'. This is a thin helper that delegates to
        the underlying model object returned by model_pool.get_model_for_role.
        """
        pool = self._get_pool()
        if pool is None:
            return {"ok": False, "error": "model_pool unavailable"}
        try:
            model = pool.get_model_for_role(role)
            if model is None:
                return {"ok": False, "error": f"role {role} not found in pool"}
            # Prefer a chat API if present
            if hasattr(model, "create_chat_completion"):
                out = model.create_chat_completion(*args, **kwargs)
                return {"ok": True, "result": out}
            # Try a generic generate() API
            if hasattr(model, "generate"):
                out = model.generate(*args, **kwargs)
                return {"ok": True, "result": out}
            return {"ok": False, "error": "model has no known call API"}
        except Exception as e:
            import traceback

            logger.exception("Error calling model %s: %s", role, e)
            return {"ok": False, "error": str(e), "trace": traceback.format_exc()}


def get_manager() -> ModelManager:
    return ModelManager()
