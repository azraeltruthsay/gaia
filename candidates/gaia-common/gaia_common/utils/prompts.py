"""
GaiaSpeak — Unified Prompt Library (Phase 5-C, Proposal 12)

Centralizes all system/instruction prompts into versioned YAML templates.
Templates live in knowledge/system_reference/prompts/ and are hot-reloadable
without Docker rebuilds or service restarts.

Usage:
    from gaia_common.utils.prompts import speak

    # Simple template
    prompt = speak("nano_triage")

    # Template with variable injection
    prompt = speak("injection_check", message=user_input[:500])

    # Get raw template (no variable substitution)
    raw = GaiaSpeak.instance().get_raw("epistemic_compact")
"""

from __future__ import annotations

import logging
import os
import threading
import time
from pathlib import Path
from string import Template
from typing import Any, Dict, Optional

logger = logging.getLogger("GAIA.GaiaSpeak")

# Default search paths for prompt YAML files
_DEFAULT_PROMPTS_DIRS = [
    "/knowledge/system_reference/prompts",           # inside containers
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "..",
                 "knowledge", "system_reference", "prompts"),  # relative
]


class GaiaSpeak:
    """Centralized prompt template registry.

    Loads YAML files from the prompts directory, caches them in memory,
    and supports hot-reload when files change on disk.

    Templates use Python string.Template syntax: ${variable_name}.
    Missing variables are left as-is (safe substitution).
    """

    _instance: Optional[GaiaSpeak] = None
    _lock = threading.Lock()

    def __init__(self, prompts_dir: Optional[str] = None):
        self._prompts_dir: Optional[Path] = None
        self._templates: Dict[str, str] = {}
        self._load_time: float = 0.0
        self._file_mtimes: Dict[str, float] = {}

        # Find prompts directory
        search = [prompts_dir] if prompts_dir else _DEFAULT_PROMPTS_DIRS
        for candidate in search:
            if candidate and Path(candidate).is_dir():
                self._prompts_dir = Path(candidate).resolve()
                break

        if self._prompts_dir:
            self._load_all()
        else:
            logger.debug("GaiaSpeak: no prompts directory found (searched %d paths)", len(search))

    @classmethod
    def instance(cls, prompts_dir: Optional[str] = None) -> GaiaSpeak:
        """Get or create the singleton instance."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls(prompts_dir)
        return cls._instance

    # ── Loading ────────────────────────────────────────────────────────

    def _load_all(self) -> None:
        """Load all YAML files from the prompts directory."""
        if not self._prompts_dir:
            return

        try:
            import yaml
        except ImportError:
            # Fallback: parse YAML manually (simple key: | block format)
            self._load_all_simple()
            return

        count = 0
        for yaml_file in sorted(self._prompts_dir.glob("*.yaml")):
            try:
                mtime = yaml_file.stat().st_mtime
                self._file_mtimes[str(yaml_file)] = mtime

                data = yaml.safe_load(yaml_file.read_text(encoding="utf-8"))
                if isinstance(data, dict):
                    for key, value in data.items():
                        if isinstance(value, str):
                            self._templates[key] = value.strip()
                            count += 1
            except Exception:
                logger.debug("GaiaSpeak: failed to load %s", yaml_file, exc_info=True)

        self._load_time = time.monotonic()
        logger.info("GaiaSpeak: loaded %d templates from %s", count, self._prompts_dir)

    def _load_all_simple(self) -> None:
        """Fallback YAML loader for environments without PyYAML.

        Parses simple key: | block format. Handles most GaiaSpeak templates
        without requiring the yaml package.
        """
        if not self._prompts_dir:
            return

        count = 0
        for yaml_file in sorted(self._prompts_dir.glob("*.yaml")):
            try:
                mtime = yaml_file.stat().st_mtime
                self._file_mtimes[str(yaml_file)] = mtime

                text = yaml_file.read_text(encoding="utf-8")
                current_key = None
                current_lines: list = []

                for line in text.splitlines():
                    # Skip comments
                    if line.strip().startswith("#"):
                        continue

                    # New key: detect "key_name: |" or "key_name: value"
                    if not line.startswith(" ") and not line.startswith("\t") and ":" in line:
                        # Save previous key
                        if current_key and current_lines:
                            self._templates[current_key] = "\n".join(current_lines).strip()
                            count += 1

                        parts = line.split(":", 1)
                        key = parts[0].strip()
                        value = parts[1].strip() if len(parts) > 1 else ""

                        if value == "|" or value == ">":
                            current_key = key
                            current_lines = []
                        elif value:
                            self._templates[key] = value
                            count += 1
                            current_key = None
                            current_lines = []
                        else:
                            current_key = None
                            current_lines = []
                    elif current_key is not None:
                        # Continuation of block scalar — strip 2-space indent
                        if line.startswith("  "):
                            current_lines.append(line[2:])
                        else:
                            current_lines.append(line)

                # Save final key
                if current_key and current_lines:
                    self._templates[current_key] = "\n".join(current_lines).strip()
                    count += 1

            except Exception:
                logger.debug("GaiaSpeak: failed to load %s (simple parser)", yaml_file, exc_info=True)

        self._load_time = time.monotonic()
        logger.info("GaiaSpeak: loaded %d templates (simple parser) from %s", count, self._prompts_dir)

    # ── Hot Reload ─────────────────────────────────────────────────────

    def reload_if_changed(self) -> bool:
        """Check for file changes and reload if needed. Returns True if reloaded."""
        if not self._prompts_dir:
            return False

        # Debounce: only check every 30 seconds
        if time.monotonic() - self._load_time < 30.0:
            return False

        changed = False
        for yaml_file in self._prompts_dir.glob("*.yaml"):
            path_str = str(yaml_file)
            try:
                mtime = yaml_file.stat().st_mtime
                if self._file_mtimes.get(path_str) != mtime:
                    changed = True
                    break
            except Exception:
                pass

        if changed:
            logger.info("GaiaSpeak: prompt files changed, reloading")
            self._templates.clear()
            self._load_all()
            return True
        return False

    # ── Template Access ────────────────────────────────────────────────

    def get(self, key: str, **kwargs: Any) -> str:
        """Get a prompt template with variable substitution.

        Uses safe_substitute: missing variables are left as ${name}
        instead of raising an error.

        Args:
            key: Template key (e.g., "nano_triage", "injection_check")
            **kwargs: Variables to inject (e.g., message="user input")

        Returns:
            Rendered prompt string, or empty string if key not found.
        """
        self.reload_if_changed()

        raw = self._templates.get(key, "")
        if not raw:
            logger.debug("GaiaSpeak: template '%s' not found", key)
            return ""

        if kwargs:
            return Template(raw).safe_substitute(**kwargs)
        return raw

    def get_raw(self, key: str) -> str:
        """Get the raw template string without variable substitution."""
        self.reload_if_changed()
        return self._templates.get(key, "")

    def has(self, key: str) -> bool:
        """Check if a template key exists."""
        return key in self._templates

    def keys(self) -> list:
        """List all available template keys."""
        return sorted(self._templates.keys())

    @property
    def template_count(self) -> int:
        return len(self._templates)

    @property
    def prompts_dir(self) -> Optional[Path]:
        return self._prompts_dir


# ── Module-level shortcut ──────────────────────────────────────────────

def speak(key: str, **kwargs: Any) -> str:
    """Get a rendered prompt template.

    Module-level shortcut for GaiaSpeak.instance().get(key, **kwargs).

    Usage:
        from gaia_common.utils.prompts import speak
        prompt = speak("nano_triage")
        prompt = speak("injection_check", message=user_input)
    """
    return GaiaSpeak.instance().get(key, **kwargs)
