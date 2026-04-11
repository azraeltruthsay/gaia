import importlib
import importlib.util
import os
import sys
import ast
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Callable

from gaia_common.utils import get_logger

logger = logging.getLogger("GAIA.SkillManager")

class SkillManager:
    """
    Manages dynamic Memento Skills for GAIA.
    Loads and hot-reloads Python modules from the skills/ directory.
    """
    def __init__(self, skills_dir: Optional[str] = None):
        if skills_dir is None:
            # Default to the skills/ package relative to this file
            self.skills_dir = Path(__file__).parent / "skills"
        else:
            self.skills_dir = Path(skills_dir)
        
        self.skills_dir.mkdir(parents=True, exist_ok=True)
        self.registry: Dict[str, Callable] = {}
        self.modules: Dict[str, Any] = {}
        
        # Add skills_dir to sys.path if not present
        if str(self.skills_dir) not in sys.path:
            sys.path.insert(0, str(self.skills_dir))
            
        self.load_all_skills()

    def _validate_syntax(self, source_code: str) -> bool:
        """Verify that the source code has valid Python syntax using AST."""
        try:
            ast.parse(source_code)
            return True
        except SyntaxError as e:
            logger.error(f"Syntax error in skill code: {e}")
            return False

    def load_all_skills(self):
        """Scan the skills directory and load all .py files as skills."""
        for py_file in self.skills_dir.glob("*.py"):
            if py_file.name == "__init__.py":
                continue
            skill_name = py_file.stem
            self.load_skill(skill_name)

    def load_skill(self, skill_name: str) -> bool:
        """Load or reload a single skill by name."""
        file_path = self.skills_dir / f"{skill_name}.py"
        if not file_path.exists():
            logger.warning(f"Skill file not found: {file_path}")
            return False

        try:
            # Check if module already loaded
            module_name = f"gaia_mcp.skills.{skill_name}"
            
            if module_name in sys.modules:
                # Reload existing module
                module = importlib.reload(sys.modules[module_name])
            else:
                # Load new module
                spec = importlib.util.spec_from_file_location(module_name, str(file_path))
                if spec is None:
                    return False
                module = importlib.util.module_from_spec(spec)
                sys.modules[module_name] = module
                spec.loader.exec_module(module)

            # Register the 'execute' function from the module
            if hasattr(module, "execute"):
                self.registry[skill_name] = module.execute
                self.modules[skill_name] = module
                logger.info(f"Successfully loaded skill: {skill_name}")
                return True
            else:
                logger.error(f"Skill '{skill_name}' missing 'execute' function")
                return False

        except Exception as e:
            logger.error(f"Failed to load skill '{skill_name}': {e}")
            return False

    def create_skill(self, skill_name: str, code: str) -> Dict[str, Any]:
        """Create a new skill from source code."""
        if not self._validate_syntax(code):
            return {"ok": False, "error": "Syntax error in Python code."}

        file_path = self.skills_dir / f"{skill_name}.py"
        try:
            with open(file_path, "w") as f:
                f.write(code)
            
            success = self.load_skill(skill_name)
            if success:
                return {"ok": True, "message": f"Skill '{skill_name}' created and loaded."}
            else:
                return {"ok": False, "error": f"Failed to load skill '{skill_name}' after writing."}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    def update_skill(self, skill_name: str, code: str) -> Dict[str, Any]:
        """Update an existing skill's source code and hot-reload it."""
        file_path = self.skills_dir / f"{skill_name}.py"
        if not file_path.exists():
            return {"ok": False, "error": f"Skill '{skill_name}' does not exist."}

        # Backup current version
        bak_path = file_path.with_suffix(".py.bak")
        try:
            with open(file_path, "r") as f:
                old_code = f.read()
            with open(bak_path, "w") as f:
                f.write(old_code)
        except Exception as e:
            logger.warning(f"Failed to create backup for '{skill_name}': {e}")

        result = self.create_skill(skill_name, code)
        
        # If reload failed, attempt rollback
        if not result["ok"]:
            try:
                with open(bak_path, "r") as f:
                    rollback_code = f.read()
                with open(file_path, "w") as f:
                    f.write(rollback_code)
                self.load_skill(skill_name)
                result["message"] = "Update failed; rolled back to previous version."
            except Exception as e:
                logger.error(f"Rollback failed for '{skill_name}': {e}")
        
        return result

    def get_skill_source(self, skill_name: str) -> Optional[str]:
        """Read the source code of a skill."""
        file_path = self.skills_dir / f"{skill_name}.py"
        if not file_path.exists():
            return None
        try:
            with open(file_path, "r") as f:
                return f.read()
        except Exception:
            return None

    def list_skills(self) -> List[str]:
        """List all loaded skill names."""
        return list(self.registry.keys())

    async def execute_limb(self, skill_name: str, params: Dict[str, Any]) -> Any:
        """Execute a dynamic skill."""
        func = self.registry.get(skill_name)
        if not func:
            return {"ok": False, "error": f"Skill '{skill_name}' not found."}
        
        try:
            # Check if the function is a coroutine
            import asyncio
            if asyncio.iscoroutinefunction(func):
                return await func(params)
            else:
                # Wrap synchronous skill in a thread to keep MCP server responsive
                return await asyncio.to_thread(func, params)
        except Exception as e:
            logger.error(f"Execution error in skill '{skill_name}': {e}")
            return {"ok": False, "error": f"Skill execution failed: {e}"}
