"""
KV Cache Manager — periodic checkpoint persistence for llama-server instances.

Tracks inference activity per role (reflex/nano, core) and periodically saves
KV cache state to disk via the llama-server /slots API.  Restores on startup
to warm caches after container restart.

Each role maps to a VLLMRemoteModel-compatible endpoint that supports the
save/restore slot API (llama-server with --slot-save-path).
"""

from __future__ import annotations

import logging
import os
import threading
import time
from typing import Dict, Optional

logger = logging.getLogger("GAIA.KVCacheManager")

# Checkpoint filename per role (written into the slot-save-path directory)
_CHECKPOINT_FILENAMES: Dict[str, str] = {
    "reflex": "reflex_checkpoint",
    "core": "core_checkpoint",
}

# Default checkpoint interval in seconds
_DEFAULT_INTERVAL = 300  # 5 minutes


class KVCacheManager:
    """Manages periodic KV cache checkpoints for llama-server instances."""

    def __init__(self, model_pool, interval: int = _DEFAULT_INTERVAL) -> None:
        self._model_pool = model_pool
        self._interval = interval

        # Track inference counts since last checkpoint per role
        self._inference_counts: Dict[str, int] = {"reflex": 0, "core": 0}
        self._lock = threading.Lock()

        # Background checkpoint thread
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def notify_inference(self, role: str) -> None:
        """Record that an inference was performed for the given role."""
        with self._lock:
            if role in self._inference_counts:
                self._inference_counts[role] += 1

    def save_all(self) -> Dict[str, bool]:
        """Save KV cache for all roles that had inference activity."""
        results: Dict[str, bool] = {}
        with self._lock:
            counts = dict(self._inference_counts)

        for role, count in counts.items():
            if count == 0:
                results[role] = True  # Nothing to save
                continue
            model = self._get_model_for_role(role)
            if model is None:
                logger.debug("KVCacheManager: no model for role '%s', skipping save", role)
                results[role] = False
                continue
            filename = _CHECKPOINT_FILENAMES.get(role, f"{role}_checkpoint")
            ok = model.save_kv_cache(filename)
            results[role] = ok
            if ok:
                with self._lock:
                    self._inference_counts[role] = 0
                logger.info("KV cache saved for role '%s' (filename=%s)", role, filename)

        return results

    def restore_all(self) -> Dict[str, bool]:
        """Restore KV cache for all configured roles."""
        results: Dict[str, bool] = {}
        for role in _CHECKPOINT_FILENAMES:
            model = self._get_model_for_role(role)
            if model is None:
                logger.debug("KVCacheManager: no model for role '%s', skipping restore", role)
                results[role] = False
                continue
            if not model.supports_kv_cache:
                logger.info("KV cache restore skipped for '%s': server doesn't support slot API", role)
                results[role] = False
                continue
            filename = _CHECKPOINT_FILENAMES[role]
            ok = model.restore_kv_cache(filename)
            results[role] = ok
            if ok:
                logger.info("KV cache restored for role '%s' (filename=%s)", role, filename)
        return results

    def start(self) -> None:
        """Start the background checkpoint thread."""
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._checkpoint_loop,
            name="kv-cache-checkpoint",
            daemon=True,
        )
        self._thread.start()
        logger.info("KVCacheManager: background checkpoint thread started (interval=%ds)", self._interval)

    def stop(self) -> None:
        """Stop the background checkpoint thread and do a final save."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=10)
        logger.info("KVCacheManager: background thread stopped")

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _checkpoint_loop(self) -> None:
        """Background loop: save checkpoints every N seconds if inference occurred."""
        while not self._stop_event.wait(self._interval):
            try:
                has_activity = False
                with self._lock:
                    has_activity = any(c > 0 for c in self._inference_counts.values())
                if has_activity:
                    self.save_all()
            except Exception:
                logger.warning("KVCacheManager: checkpoint loop error", exc_info=True)

    def _get_model_for_role(self, role: str):
        """Resolve a VLLMRemoteModel instance for the given role.

        - 'reflex' → the Nano endpoint model (models["reflex"])
        - 'core'   → the Core/Lite CPU endpoint model (models["core"])
        """
        try:
            model_key = role  # role names match model pool keys
            models = getattr(self._model_pool, "models", {})
            model = models.get(model_key)
            if model is not None and hasattr(model, "save_kv_cache"):
                return model
        except Exception:
            logger.debug("KVCacheManager: could not resolve model for role '%s'", role, exc_info=True)
        return None


# Module-level singleton (set during startup)
_manager: Optional[KVCacheManager] = None


def get_kv_cache_manager() -> Optional[KVCacheManager]:
    """Return the global KVCacheManager instance, or None if not initialized."""
    return _manager


def init_kv_cache_manager(model_pool, interval: int = _DEFAULT_INTERVAL) -> KVCacheManager:
    """Initialize and return the global KVCacheManager singleton."""
    global _manager
    _manager = KVCacheManager(model_pool, interval=interval)
    return _manager
