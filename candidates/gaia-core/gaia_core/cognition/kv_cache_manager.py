"""
KV Cache Manager — periodic checkpoint persistence and pressure-based
auto-compaction for llama-server instances.

Tracks inference activity per role (reflex/nano, core) and periodically saves
KV cache state to disk via the llama-server /slots API.  Restores on startup
to warm caches after container restart.

Pressure monitoring: polls cache fill ratio every N seconds and auto-compacts
(save checkpoint → erase slot) before the cache hits 100%, preventing
context-exceeded crashes.  The erased slot is rebuilt transparently on the
next inference from the session manager's sliding window.

Each role maps to a VLLMRemoteModel-compatible endpoint that supports the
save/restore slot API (llama-server with --slot-save-path).
"""

from __future__ import annotations

import logging
import time
import threading
from typing import Dict, List, Optional

logger = logging.getLogger("GAIA.KVCacheManager")

# Checkpoint filename per role (written into the slot-save-path directory)
_CHECKPOINT_FILENAMES: Dict[str, str] = {
    "reflex": "reflex_checkpoint",
    "core": "core_checkpoint",
}

# Default intervals
_DEFAULT_CHECKPOINT_INTERVAL = 300  # 5 minutes
_DEFAULT_COMPACT_INTERVAL = 30     # 30 seconds

# Default pressure thresholds
_DEFAULT_COMPACT_THRESHOLD = 0.80
_DEFAULT_CRITICAL_THRESHOLD = 0.95

# Max compaction log entries to retain
_MAX_COMPACTION_LOG = 50


class KVCacheManager:
    """Manages periodic KV cache checkpoints and pressure-based auto-compaction."""

    def __init__(
        self,
        model_pool,
        checkpoint_interval: int = _DEFAULT_CHECKPOINT_INTERVAL,
        compact_interval: int = _DEFAULT_COMPACT_INTERVAL,
        compact_threshold: float = _DEFAULT_COMPACT_THRESHOLD,
        critical_threshold: float = _DEFAULT_CRITICAL_THRESHOLD,
    ) -> None:
        self._model_pool = model_pool
        self._checkpoint_interval = checkpoint_interval
        self._compact_interval = compact_interval
        self._compact_threshold = compact_threshold
        self._critical_threshold = critical_threshold

        # Track inference counts since last checkpoint per role
        self._inference_counts: Dict[str, int] = {"reflex": 0, "core": 0}
        self._lock = threading.Lock()

        # Background threads
        self._stop_event = threading.Event()
        self._checkpoint_thread: Optional[threading.Thread] = None
        self._compact_thread: Optional[threading.Thread] = None

        # Compaction event log
        self._compaction_log: List[dict] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def notify_inference(self, role: str) -> None:
        """Record that an inference was performed for the given role."""
        with self._lock:
            if role in self._inference_counts:
                self._inference_counts[role] += 1

    def get_cache_pressure(self, role: str) -> float:
        """Return the KV cache pressure (0.0-1.0) for a given role.

        Returns -1.0 if the model doesn't support pressure queries.
        """
        model = self._get_model_for_role(role)
        if model is None or not hasattr(model, "get_cache_pressure"):
            return -1.0
        try:
            return model.get_cache_pressure()
        except Exception:
            logger.debug("Failed to get cache pressure for '%s'", role, exc_info=True)
            return -1.0

    def get_all_pressures(self) -> Dict[str, float]:
        """Return cache pressure for all configured roles."""
        return {role: self.get_cache_pressure(role) for role in _CHECKPOINT_FILENAMES}

    def compact(self, role: str, reason: str = "manual") -> bool:
        """Save a timestamped checkpoint then erase the KV cache slot.

        Returns True if compaction succeeded.
        """
        model = self._get_model_for_role(role)
        if model is None:
            logger.warning("compact: no model for role '%s'", role)
            return False

        pressure_before = -1.0
        if hasattr(model, "get_cache_pressure"):
            pressure_before = model.get_cache_pressure()

        # Save a timestamped checkpoint before erasing
        ts = int(time.time())
        checkpoint_name = f"{role}_pre_compact_{ts}"
        saved = False
        if hasattr(model, "save_kv_cache"):
            saved = model.save_kv_cache(checkpoint_name)
            if saved:
                logger.info("Pre-compact checkpoint saved: %s", checkpoint_name)

        # Erase the slot
        erased = False
        if hasattr(model, "erase_slot"):
            erased = model.erase_slot()
        else:
            logger.warning("compact: model for '%s' does not support erase_slot", role)

        # Log the compaction event
        event = {
            "role": role,
            "reason": reason,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "epoch": ts,
            "pressure_before": round(pressure_before, 4) if pressure_before >= 0 else None,
            "checkpoint_saved": saved,
            "checkpoint_name": checkpoint_name if saved else None,
            "erased": erased,
        }
        with self._lock:
            self._compaction_log.append(event)
            if len(self._compaction_log) > _MAX_COMPACTION_LOG:
                self._compaction_log = self._compaction_log[-_MAX_COMPACTION_LOG:]

        if erased:
            logger.info(
                "KV cache compacted for '%s' (reason=%s, pressure_before=%.2f)",
                role, reason, pressure_before,
            )
        else:
            logger.warning("KV cache compaction FAILED for '%s' (reason=%s)", role, reason)

        return erased

    def get_compaction_log(self) -> List[dict]:
        """Return the recent compaction event log."""
        with self._lock:
            return list(self._compaction_log)

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
        """Start background checkpoint and compact threads."""
        if self._checkpoint_thread is not None and self._checkpoint_thread.is_alive():
            return
        self._stop_event.clear()

        self._checkpoint_thread = threading.Thread(
            target=self._checkpoint_loop,
            name="kv-cache-checkpoint",
            daemon=True,
        )
        self._checkpoint_thread.start()

        self._compact_thread = threading.Thread(
            target=self._compact_loop,
            name="kv-cache-compact",
            daemon=True,
        )
        self._compact_thread.start()

        logger.info(
            "KVCacheManager: started (checkpoint=%ds, compact=%ds, threshold=%.0f%%, critical=%.0f%%)",
            self._checkpoint_interval, self._compact_interval,
            self._compact_threshold * 100, self._critical_threshold * 100,
        )

    def stop(self) -> None:
        """Stop background threads and do a final save."""
        self._stop_event.set()
        if self._checkpoint_thread is not None:
            self._checkpoint_thread.join(timeout=10)
        if self._compact_thread is not None:
            self._compact_thread.join(timeout=10)
        logger.info("KVCacheManager: background threads stopped")

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _checkpoint_loop(self) -> None:
        """Background loop: save checkpoints every N seconds if inference occurred."""
        while not self._stop_event.wait(self._checkpoint_interval):
            try:
                has_activity = False
                with self._lock:
                    has_activity = any(c > 0 for c in self._inference_counts.values())
                if has_activity:
                    self.save_all()
            except Exception:
                logger.warning("KVCacheManager: checkpoint loop error", exc_info=True)

    def _compact_loop(self) -> None:
        """Background loop: poll cache pressure and auto-compact when thresholds are exceeded."""
        while not self._stop_event.wait(self._compact_interval):
            try:
                for role in _CHECKPOINT_FILENAMES:
                    pressure = self.get_cache_pressure(role)
                    if pressure < 0:
                        continue  # slot info unavailable

                    if pressure >= self._critical_threshold:
                        logger.warning(
                            "KV cache CRITICAL for '%s': %.1f%% — emergency compacting",
                            role, pressure * 100,
                        )
                        self.compact(role, reason=f"critical ({pressure:.0%})")
                    elif pressure >= self._compact_threshold:
                        logger.info(
                            "KV cache high for '%s': %.1f%% — auto-compacting",
                            role, pressure * 100,
                        )
                        self.compact(role, reason=f"threshold ({pressure:.0%})")
                    else:
                        logger.debug("KV cache '%s': %.1f%%", role, pressure * 100)
            except Exception:
                logger.warning("KVCacheManager: compact loop error", exc_info=True)

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


def init_kv_cache_manager(model_pool, **kwargs) -> KVCacheManager:
    """Initialize and return the global KVCacheManager singleton.

    Reads KV_CACHE config block from gaia_constants.json if available,
    with kwargs overrides.
    """
    global _manager

    # Load config from constants
    config = {}
    try:
        from gaia_core.config import get_config
        cfg = get_config()
        raw = getattr(cfg, "_raw_constants", None) or {}
        config = raw.get("KV_CACHE", {})
    except Exception:
        logger.debug("Could not load KV_CACHE config from constants", exc_info=True)

    params = {
        "checkpoint_interval": config.get("checkpoint_interval_seconds", _DEFAULT_CHECKPOINT_INTERVAL),
        "compact_interval": config.get("compact_interval_seconds", _DEFAULT_COMPACT_INTERVAL),
        "compact_threshold": config.get("compact_threshold", _DEFAULT_COMPACT_THRESHOLD),
        "critical_threshold": config.get("critical_threshold", _DEFAULT_CRITICAL_THRESHOLD),
    }
    params.update(kwargs)

    _manager = KVCacheManager(model_pool, **params)
    return _manager
