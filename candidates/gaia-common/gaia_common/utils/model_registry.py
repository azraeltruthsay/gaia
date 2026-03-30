"""
Model Registry — centralized model lifecycle management.

Tracks all model states (base, adapted, merged, GGUF derivatives),
supports archival for rollback, and provides fast model swapping.

The registry is the single source of truth for model paths across
all GAIA services. It persists to gaia-shared as a versioned JSON file.

Usage:
    from gaia_common.utils.model_registry import ModelRegistry

    registry = ModelRegistry()
    core = registry.get("core")
    core.safetensors_path  # /models/Qwen3.5-2B-GAIA-Core
    core.gguf_path         # /models/Qwen3.5-2B-GAIA-Core-Q8_0.gguf
    core.version           # 3
    core.parent            # Qwen3.5-2B (base model)

    # After ROME edit:
    registry.archive("core")  # saves current as v3
    registry.update("core", safetensors_path="/models/Qwen3.5-2B-GAIA-Core-v4")
    registry.derive_gguf("core")  # re-quantize from new safetensors
"""

import json
import logging
import os
import shutil
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional, List, Dict

logger = logging.getLogger("GAIA.ModelRegistry")

REGISTRY_DIR = Path(os.environ.get("SHARED_DIR", "/shared")) / "model_registry"
REGISTRY_FILE = REGISTRY_DIR / "registry.json"
ARCHIVE_DIR = REGISTRY_DIR / "archive"


@dataclass
class ModelEntry:
    """A single model in the registry."""
    name: str                              # e.g. "core", "nano", "prime"
    role: str                              # "reflex", "operator", "thinker"
    base_model: str                        # e.g. "Qwen3.5-2B"
    safetensors_path: str = ""             # HF model directory (source of truth)
    gguf_path: str = ""                    # Derived GGUF for CPU fallback
    gguf_quantization: str = "Q8_0"        # GGUF quant type
    adapter_path: str = ""                 # Active LoRA adapter (if not merged)
    version: int = 1                       # Incremented on each update
    created_at: float = 0.0               # Unix timestamp
    updated_at: float = 0.0
    parent: str = ""                       # Base model this was derived from
    training_tier: str = ""                # Last training tier applied (0, I, II, III)
    notes: str = ""
    archived_versions: List[str] = field(default_factory=list)


@dataclass
class RegistryState:
    """Complete registry state."""
    models: Dict[str, ModelEntry] = field(default_factory=dict)
    last_updated: float = 0.0
    schema_version: int = 1


class ModelRegistry:
    """Centralized model lifecycle manager."""

    def __init__(self, registry_dir: Optional[Path] = None):
        self._dir = registry_dir or REGISTRY_DIR
        self._file = self._dir / "registry.json"
        self._archive = self._dir / "archive"
        self._state: Optional[RegistryState] = None
        self._ensure_dirs()

    def _ensure_dirs(self):
        self._dir.mkdir(parents=True, exist_ok=True)
        self._archive.mkdir(parents=True, exist_ok=True)

    def _load(self) -> RegistryState:
        if self._state is not None:
            return self._state

        if self._file.exists():
            try:
                data = json.loads(self._file.read_text())
                models = {}
                for name, entry_data in data.get("models", {}).items():
                    models[name] = ModelEntry(**entry_data)
                self._state = RegistryState(
                    models=models,
                    last_updated=data.get("last_updated", 0.0),
                    schema_version=data.get("schema_version", 1),
                )
            except Exception as e:
                logger.warning("Failed to load registry, starting fresh: %s", e)
                self._state = RegistryState()
        else:
            self._state = RegistryState()

        return self._state

    def _save(self):
        state = self._load()
        state.last_updated = time.time()

        data = {
            "schema_version": state.schema_version,
            "last_updated": state.last_updated,
            "models": {name: asdict(entry) for name, entry in state.models.items()},
        }

        # Atomic write
        tmp = self._file.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=2))
        tmp.rename(self._file)
        logger.debug("Registry saved (%d models)", len(state.models))

    def get(self, name: str) -> Optional[ModelEntry]:
        """Get a model entry by name."""
        state = self._load()
        return state.models.get(name)

    def list_models(self) -> Dict[str, ModelEntry]:
        """List all registered models."""
        return dict(self._load().models)

    def register(self, entry: ModelEntry) -> None:
        """Register a new model or update an existing one."""
        state = self._load()
        now = time.time()
        entry.updated_at = now
        if entry.created_at == 0.0:
            entry.created_at = now
        state.models[entry.name] = entry
        self._save()
        logger.info("Registered model '%s' v%d at %s", entry.name, entry.version, entry.safetensors_path)

    def update(self, name: str, **kwargs) -> Optional[ModelEntry]:
        """Update specific fields of a model entry. Increments version."""
        state = self._load()
        entry = state.models.get(name)
        if entry is None:
            logger.warning("Cannot update '%s' — not in registry", name)
            return None

        for key, value in kwargs.items():
            if hasattr(entry, key):
                setattr(entry, key, value)

        entry.version += 1
        entry.updated_at = time.time()
        self._save()
        logger.info("Updated model '%s' → v%d", name, entry.version)
        return entry

    def archive(self, name: str) -> Optional[str]:
        """Archive the current version of a model for rollback.

        Saves a snapshot of the registry entry (not the weights themselves —
        those are already on disk). Returns the archive filename.
        """
        state = self._load()
        entry = state.models.get(name)
        if entry is None:
            return None

        archive_name = f"{name}_v{entry.version}_{int(time.time())}.json"
        archive_path = self._archive / archive_name
        archive_path.write_text(json.dumps(asdict(entry), indent=2))

        entry.archived_versions.append(archive_name)
        # Keep only last 10 archives per model
        if len(entry.archived_versions) > 10:
            old = entry.archived_versions.pop(0)
            old_path = self._archive / old
            if old_path.exists():
                old_path.unlink()

        self._save()
        logger.info("Archived model '%s' v%d → %s", name, entry.version, archive_name)
        return archive_name

    def rollback(self, name: str, archive_name: Optional[str] = None) -> Optional[ModelEntry]:
        """Rollback a model to a previous archived version.

        If archive_name is None, rolls back to the most recent archive.
        """
        state = self._load()
        entry = state.models.get(name)
        if entry is None:
            return None

        if archive_name is None:
            if not entry.archived_versions:
                logger.warning("No archived versions for '%s'", name)
                return None
            archive_name = entry.archived_versions[-1]

        archive_path = self._archive / archive_name
        if not archive_path.exists():
            logger.warning("Archive not found: %s", archive_path)
            return None

        archived = json.loads(archive_path.read_text())
        restored = ModelEntry(**archived)
        restored.updated_at = time.time()
        restored.notes = f"Rolled back from v{entry.version} to archived {archive_name}"

        state.models[name] = restored
        self._save()
        logger.info("Rolled back '%s' to %s (v%d)", name, archive_name, restored.version)
        return restored

    def get_active_paths(self, name: str) -> dict:
        """Get the active serving paths for a model.

        Returns dict with safetensors_path, gguf_path, and adapter_path
        for use by the orchestrator and inference servers.
        """
        entry = self.get(name)
        if entry is None:
            return {}
        return {
            "safetensors": entry.safetensors_path,
            "gguf": entry.gguf_path,
            "adapter": entry.adapter_path,
            "version": entry.version,
            "role": entry.role,
        }


def get_registry() -> ModelRegistry:
    """Get the singleton registry instance."""
    global _registry
    if _registry is None:
        _registry = ModelRegistry()
    return _registry


_registry: Optional[ModelRegistry] = None
