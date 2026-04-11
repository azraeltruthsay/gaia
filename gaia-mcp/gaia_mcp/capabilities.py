"""
Capability Engine — Unified Tools + Skills Registry (Phase 5-C, Proposal 03)

Merges static Domain Tools and dynamic Memento Skills into a single
hot-reloadable system. Every action GAIA can take is a 'Limb' — a
Capability object with unified metadata, dispatch, and security.

Architecture:
  - Static Limbs:  loaded from domain_tools.py + tools.py at init
  - Dynamic Limbs: loaded from skills/*.py, hot-reloadable, can override static
  - Security:      Blast Shield + Approval gate applied to ALL limbs
  - Dispatch:      execute_limb(domain, action, params) → result

Usage (from tools.py or mcp_client.py):
    engine = CapabilityEngine(approval_store)
    result = await engine.execute_limb("file", "read", {"path": "/data/x.txt"})

Internal naming: GAIA_LIMB (prepares for global rename from 'tool' to 'limb')
"""

from __future__ import annotations

import ast
import asyncio
import importlib
import importlib.util
import logging
import py_compile
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

logger = logging.getLogger("GAIA.CapabilityEngine")

# Internal naming constant — all new code uses GAIA_LIMB
GAIA_LIMB = "limb"


# ── Capability Metadata ───────────────────────────────────────────────

@dataclass
class Capability:
    """A single action GAIA can perform — whether static or dynamic."""
    domain: str               # e.g. "file", "web", "knowledge"
    action: str               # e.g. "read", "search", "query"
    handler: Callable         # sync or async callable(params) -> result
    source: str               # "static" | "dynamic" | "fabric"
    sensitive: bool = False   # requires approval
    description: str = ""
    schema: Dict = field(default_factory=dict)  # JSON Schema for params
    legacy_name: str = ""     # backward-compat tool name (e.g. "read_file")
    module_path: str = ""     # file path for dynamic skills (hot-reload)
    loaded_at: float = 0.0    # timestamp of last load


# ── Capability Engine ─────────────────────────────────────────────────

class CapabilityEngine:
    """Unified registry for all GAIA actions (Tools + Skills).

    Resolution order for execute_limb(domain, action):
      1. Dynamic override (skill with matching domain.action)
      2. Static domain tool (from domain_tools.py mappings)
      3. Legacy tool name fallback (backward compat)

    Security is centralized: Blast Shield + Approval gate fire for
    every execution, regardless of source.
    """

    def __init__(
        self,
        approval_store=None,
        skills_dir: Optional[str] = None,
        tool_map: Optional[Dict[str, Callable]] = None,
        async_tool_map: Optional[Dict[str, Callable]] = None,
    ):
        self.approval_store = approval_store

        # Registry: (domain, action) -> Capability
        self._registry: Dict[Tuple[str, str], Capability] = {}

        # Legacy name index: legacy_name -> (domain, action)
        self._legacy_index: Dict[str, Tuple[str, str]] = {}

        # Skills directory for hot-reload
        self._skills_dir = Path(skills_dir) if skills_dir else None

        # Static tool maps (injected from tools.py)
        self._tool_map = tool_map or {}
        self._async_tool_map = async_tool_map or {}

        # Track loaded skill modules for hot-reload
        self._skill_modules: Dict[str, Any] = {}

        # Sensitive actions set
        self._sensitive: Set[Tuple[str, str]] = set()

    # ── Initialization ─────────────────────────────────────────────────

    def init_from_domain_tools(self):
        """Load static capabilities from gaia-common domain_tools."""
        try:
            from gaia_common.utils.domain_tools import (
                DOMAIN_TOOLS,
                ACTION_TO_LEGACY,
                SENSITIVE_ACTIONS,
            )
        except ImportError:
            logger.warning("domain_tools not available; static limbs not loaded")
            return

        for domain, domain_cfg in DOMAIN_TOOLS.items():
            if domain.startswith("_"):
                continue
            actions = domain_cfg.get("actions", {})
            for action_name, action_cfg in actions.items():
                legacy = ACTION_TO_LEGACY.get((domain, action_name), "")
                sensitive = (domain, action_name) in SENSITIVE_ACTIONS

                # Resolve handler from tool maps
                handler = self._resolve_static_handler(legacy)

                cap = Capability(
                    domain=domain,
                    action=action_name,
                    handler=handler,
                    source="static",
                    sensitive=sensitive,
                    description=action_cfg.get("description", ""),
                    legacy_name=legacy,
                )

                self._registry[(domain, action_name)] = cap
                if legacy:
                    self._legacy_index[legacy] = (domain, action_name)
                if sensitive:
                    self._sensitive.add((domain, action_name))

        logger.info(
            "CapabilityEngine: loaded %d static limbs across %d domains",
            len(self._registry),
            len(set(k[0] for k in self._registry)),
        )

    def init_skills(self, skills_dir: Optional[str] = None):
        """Load dynamic skills from the skills directory.

        Skills can override static capabilities by registering with
        the same domain.action key.
        """
        if skills_dir:
            self._skills_dir = Path(skills_dir)
        if not self._skills_dir:
            return

        self._skills_dir.mkdir(parents=True, exist_ok=True)
        if str(self._skills_dir) not in sys.path:
            sys.path.insert(0, str(self._skills_dir))

        for py_file in self._skills_dir.glob("*.py"):
            if py_file.name == "__init__.py":
                continue
            self._load_skill(py_file)

    def _load_skill(self, path: Path) -> bool:
        """Load a single skill module and register its capabilities."""
        skill_name = path.stem
        try:
            spec = importlib.util.spec_from_file_location(
                f"gaia_skill_{skill_name}", str(path)
            )
            if not spec or not spec.loader:
                return False

            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            self._skill_modules[skill_name] = module

            if not hasattr(module, "execute"):
                logger.warning("Skill '%s' has no execute() function", skill_name)
                return False

            # Determine domain.action from module metadata or name
            domain = getattr(module, "DOMAIN", "skill")
            action = getattr(module, "ACTION", skill_name)
            sensitive = getattr(module, "SENSITIVE", False)
            description = getattr(module, "DESCRIPTION", module.__doc__ or "")

            key = (domain, action)
            was_override = key in self._registry

            cap = Capability(
                domain=domain,
                action=action,
                handler=module.execute,
                source="dynamic",
                sensitive=sensitive,
                description=description[:200],
                module_path=str(path),
                loaded_at=time.time(),
            )

            self._registry[key] = cap
            if sensitive:
                self._sensitive.add(key)

            if was_override:
                logger.info(
                    "Skill '%s' OVERRIDES static limb %s.%s",
                    skill_name, domain, action,
                )
            else:
                logger.info("Skill '%s' registered as %s.%s", skill_name, domain, action)

            return True
        except Exception:
            logger.exception("Failed to load skill '%s'", skill_name)
            return False

    def hot_reload_skill(self, skill_name: str) -> bool:
        """Hot-reload a single skill by name."""
        if not self._skills_dir:
            return False
        path = self._skills_dir / f"{skill_name}.py"
        if not path.exists():
            logger.warning("Skill file not found: %s", path)
            return False

        # Syntax validation before reload
        try:
            source = path.read_text()
            ast.parse(source)
            py_compile.compile(str(path), doraise=True)
        except (SyntaxError, py_compile.PyCompileError) as e:
            logger.error("Syntax error in skill '%s': %s", skill_name, e)
            return False

        return self._load_skill(path)

    def create_skill(self, skill_name: str, code: str) -> Dict[str, Any]:
        """Create a new dynamic skill from source code."""
        if not self._skills_dir:
            return {"ok": False, "error": "skills directory not configured"}

        # Syntax validation
        try:
            ast.parse(code)
        except SyntaxError as e:
            return {"ok": False, "error": f"Syntax error: {e}"}

        path = self._skills_dir / f"{skill_name}.py"
        if path.exists():
            return {"ok": False, "error": f"Skill '{skill_name}' already exists. Use update_skill."}

        self._skills_dir.mkdir(parents=True, exist_ok=True)
        path.write_text(code)

        if self._load_skill(path):
            return {"ok": True, "skill": skill_name, "path": str(path)}
        return {"ok": False, "error": "Failed to load skill after writing"}

    def update_skill(self, skill_name: str, code: str) -> Dict[str, Any]:
        """Update an existing skill with rollback on failure."""
        if not self._skills_dir:
            return {"ok": False, "error": "skills directory not configured"}

        path = self._skills_dir / f"{skill_name}.py"
        backup = path.with_suffix(".py.bak")

        # Syntax validation
        try:
            ast.parse(code)
        except SyntaxError as e:
            return {"ok": False, "error": f"Syntax error: {e}"}

        # Backup existing
        if path.exists():
            backup.write_text(path.read_text())

        path.write_text(code)

        if self._load_skill(path):
            return {"ok": True, "skill": skill_name, "path": str(path)}

        # Rollback
        if backup.exists():
            path.write_text(backup.read_text())
            self._load_skill(path)
        return {"ok": False, "error": "Load failed after update; rolled back"}

    def _resolve_static_handler(self, legacy_name: str) -> Callable:
        """Look up a handler from the static tool maps."""
        if legacy_name in self._async_tool_map:
            return self._async_tool_map[legacy_name]
        if legacy_name in self._tool_map:
            return self._tool_map[legacy_name]
        # Return a stub that raises — will be replaced if skill overrides
        def _not_implemented(params):
            raise NotImplementedError(
                f"Static handler for '{legacy_name}' not wired into CapabilityEngine"
            )
        return _not_implemented

    # ── Execution ──────────────────────────────────────────────────────

    async def execute_limb(
        self,
        domain: str,
        action: str,
        params: Optional[Dict] = None,
        pre_approved: bool = False,
    ) -> Any:
        """Execute a capability by domain + action.

        Resolution:
          1. Dynamic skill override (if registered)
          2. Static domain tool
          3. Legacy name fallback

        Security: Blast Shield + Approval gate fire for all limbs.
        """
        params = params or {}
        key = (domain, action)

        cap = self._registry.get(key)
        if not cap:
            # Try legacy name as domain (backward compat)
            if domain in self._legacy_index:
                real_domain, real_action = self._legacy_index[domain]
                cap = self._registry.get((real_domain, real_action))
            if not cap:
                raise ValueError(
                    f"Unknown limb: {domain}.{action}. "
                    f"Available domains: {sorted(set(k[0] for k in self._registry))}"
                )

        # ── Security Gate ──
        # 1. Sensitivity check
        if cap.sensitive and not pre_approved:
            raise PermissionError(
                f"Limb '{domain}.{action}' requires explicit approval."
            )

        # 2. Blast Shield (always)
        if self.approval_store:
            try:
                legacy = cap.legacy_name or f"{domain}_{action}"
                self.approval_store.validate_against_blast_shield(legacy, params)
            except (ValueError, PermissionError) as e:
                raise PermissionError(f"Blast Shield block: {e}")

        # ── Dispatch ──
        logger.info(
            "execute_limb: %s.%s [source=%s]%s",
            domain, action, cap.source,
            " (override)" if cap.source == "dynamic" and cap.legacy_name else "",
        )

        handler = cap.handler
        try:
            if asyncio.iscoroutinefunction(handler):
                return await handler(params)
            else:
                return await asyncio.to_thread(handler, params)
        except Exception as e:
            logger.error("Limb %s.%s failed: %s", domain, action, e)
            raise

    async def execute_by_legacy_name(
        self,
        method: str,
        params: Optional[Dict] = None,
        pre_approved: bool = False,
    ) -> Any:
        """Execute by legacy tool name (backward compat bridge).

        Maps legacy names like 'read_file' -> ('file', 'read') -> execute_limb.
        """
        params = params or {}

        # Check if it's a domain name with an action param
        if method in set(k[0] for k in self._registry):
            action = params.pop("action", None)
            if not action:
                actions = [k[1] for k in self._registry if k[0] == method]
                raise ValueError(
                    f"Domain '{method}' requires an 'action' parameter. "
                    f"Available: {sorted(actions)}"
                )
            return await self.execute_limb(method, action, params, pre_approved)

        # Legacy name lookup
        if method in self._legacy_index:
            domain, action = self._legacy_index[method]
            return await self.execute_limb(domain, action, params, pre_approved)

        # Direct skill name (e.g. "system_pulse")
        cap = self._registry.get(("skill", method))
        if cap:
            return await self.execute_limb("skill", method, params, pre_approved)

        raise ValueError(
            f"Unknown method: '{method}'. "
            f"Available domains: {sorted(set(k[0] for k in self._registry))}"
        )

    # ── Introspection ──────────────────────────────────────────────────

    def list_domains(self) -> List[str]:
        """List all registered domains."""
        return sorted(set(k[0] for k in self._registry))

    def list_actions(self, domain: str) -> List[str]:
        """List all actions for a domain."""
        return sorted(k[1] for k in self._registry if k[0] == domain)

    def list_limbs(self, include_dynamic: bool = True) -> List[Dict[str, Any]]:
        """List all registered capabilities with metadata."""
        result = []
        for (domain, action), cap in sorted(self._registry.items()):
            if not include_dynamic and cap.source == "dynamic":
                continue
            result.append({
                "domain": domain,
                "action": action,
                "source": cap.source,
                "sensitive": cap.sensitive,
                "legacy_name": cap.legacy_name,
                "description": cap.description[:100],
            })
        return result

    def list_overrides(self) -> List[Dict[str, str]]:
        """List dynamic skills that override static tools."""
        return [
            {"domain": cap.domain, "action": cap.action, "module": cap.module_path}
            for cap in self._registry.values()
            if cap.source == "dynamic" and cap.legacy_name
        ]

    def get_capability(self, domain: str, action: str) -> Optional[Capability]:
        """Get capability metadata."""
        return self._registry.get((domain, action))

    @property
    def limb_count(self) -> int:
        return len(self._registry)

    @property
    def dynamic_count(self) -> int:
        return sum(1 for c in self._registry.values() if c.source == "dynamic")

    @property
    def static_count(self) -> int:
        return sum(1 for c in self._registry.values() if c.source == "static")
