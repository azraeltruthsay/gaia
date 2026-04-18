"""
Skill Package — SKILL.md format loader and schema.

Skills are self-contained, versioned units of agent capability stored as
SKILL.md files with YAML frontmatter. Two execution modes:

  KNOWLEDGE: Prompt-only skills (injected as system prompt, like Fabric patterns)
  PLAYBOOK:  Code-backed skills (Python modules with execute() function)

Inspired by Memento-Skills (github.com/Memento-Teams/Memento-Skills).
"""

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger("GAIA.SkillPackage")


@dataclass
class SkillParameter:
    """Parameter definition for a skill."""
    name: str
    type: str = "string"
    required: bool = True
    description: str = ""


@dataclass
class SkillPackage:
    """A loaded SKILL.md skill definition."""
    name: str
    description: str
    version: int = 1
    execution_mode: str = "PLAYBOOK"  # KNOWLEDGE or PLAYBOOK
    legacy_maps_to: Optional[str] = None
    legacy_action: Optional[str] = None
    sensitive: bool = False
    parameters: List[SkillParameter] = field(default_factory=list)
    body: str = ""  # system prompt (KNOWLEDGE) or instructions (PLAYBOOK)
    source_path: Optional[Path] = None
    domain: str = "custom"
    dependencies: List[str] = field(default_factory=list)

    @property
    def is_knowledge(self) -> bool:
        return self.execution_mode.upper() == "KNOWLEDGE"

    @property
    def is_playbook(self) -> bool:
        return self.execution_mode.upper() == "PLAYBOOK"


def _parse_frontmatter(text: str) -> tuple:
    """Parse YAML frontmatter from SKILL.md content.

    Returns (frontmatter_dict, body_text).
    """
    # Match --- delimited frontmatter
    match = re.match(r'^---\s*\n(.*?)\n---\s*\n?(.*)', text, re.DOTALL)
    if not match:
        return {}, text

    fm_text = match.group(1)
    body = match.group(2).strip()

    # Simple YAML parser — avoids PyYAML dependency in gaia-mcp
    fm = {}
    current_key = None
    current_list = None

    for line in fm_text.split("\n"):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue

        # List item under current key
        if stripped.startswith("- ") and current_list is not None:
            item_text = stripped[2:].strip()
            # Check if it's a dict item (name: value)
            if ": " in item_text:
                item_dict = {}
                # Parse inline key: value
                item_dict[item_text.split(": ", 1)[0].strip()] = _parse_value(item_text.split(": ", 1)[1].strip())
                current_list.append(item_dict)
            else:
                current_list.append(_parse_value(item_text))
            continue

        # Nested key under list item
        if line.startswith("    ") and current_list and isinstance(current_list[-1], dict):
            if ": " in stripped:
                k, v = stripped.split(": ", 1)
                current_list[-1][k.strip()] = _parse_value(v.strip())
            continue

        # Top-level key: value
        if ": " in stripped:
            key, val = stripped.split(": ", 1)
            key = key.strip()
            val = val.strip()
            current_key = key
            if val == "" or val == "[]":
                current_list = []
                fm[key] = current_list
            else:
                fm[key] = _parse_value(val)
                current_list = None
        elif stripped.endswith(":"):
            current_key = stripped[:-1].strip()
            current_list = []
            fm[current_key] = current_list

    return fm, body


def _parse_value(val: str):
    """Parse a YAML scalar value."""
    if val.lower() == "true":
        return True
    if val.lower() == "false":
        return False
    if val.lower() in ("null", "~", ""):
        return None
    try:
        return int(val)
    except ValueError:
        pass
    try:
        return float(val)
    except ValueError:
        pass
    # Strip quotes
    if (val.startswith('"') and val.endswith('"')) or \
       (val.startswith("'") and val.endswith("'")):
        return val[1:-1]
    return val


def load_skill_package(path: Path) -> Optional[SkillPackage]:
    """Load a single SKILL.md file into a SkillPackage.

    Args:
        path: Path to SKILL.md file

    Returns:
        SkillPackage or None if parsing fails
    """
    try:
        text = path.read_text(encoding="utf-8")
    except Exception as e:
        logger.warning("Failed to read %s: %s", path, e)
        return None

    fm, body = _parse_frontmatter(text)
    if not fm.get("name"):
        logger.warning("SKILL.md at %s has no 'name' in frontmatter", path)
        return None

    # Parse parameters
    params = []
    for p in fm.get("parameters", []):
        if isinstance(p, dict):
            params.append(SkillParameter(
                name=p.get("name", "input"),
                type=p.get("type", "string"),
                required=p.get("required", True),
                description=p.get("description", ""),
            ))

    return SkillPackage(
        name=fm["name"],
        description=fm.get("description", ""),
        version=fm.get("version", 1),
        execution_mode=fm.get("execution_mode", "PLAYBOOK").upper(),
        legacy_maps_to=fm.get("legacy_maps_to"),
        legacy_action=fm.get("legacy_action"),
        sensitive=fm.get("sensitive", False),
        parameters=params,
        body=body,
        source_path=path,
        domain=fm.get("domain", "custom"),
        dependencies=fm.get("dependencies", []),
    )


def load_all_packages(skills_dir: Path) -> Dict[str, SkillPackage]:
    """Load all SKILL.md files from a directory tree.

    Scans recursively for files matching *.skill.md or SKILL.md.

    Returns:
        Dict mapping skill name to SkillPackage
    """
    packages = {}
    if not skills_dir.exists():
        logger.info("Skills directory %s does not exist", skills_dir)
        return packages

    for path in sorted(skills_dir.rglob("*.skill.md")):
        pkg = load_skill_package(path)
        if pkg:
            if pkg.name in packages:
                logger.warning("Duplicate skill name '%s' at %s (already loaded from %s)",
                               pkg.name, path, packages[pkg.name].source_path)
            packages[pkg.name] = pkg

    # Also check for SKILL.md in subdirectories (Memento convention)
    for path in sorted(skills_dir.rglob("SKILL.md")):
        pkg = load_skill_package(path)
        if pkg and pkg.name not in packages:
            packages[pkg.name] = pkg

    logger.info("Loaded %d skill packages from %s", len(packages), skills_dir)
    return packages
