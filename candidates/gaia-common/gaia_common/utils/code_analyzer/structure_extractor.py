"""
structure_extractor.py

Handles extracting code structure like imports, classes, functions, and variables.
"""

import ast
import logging
from typing import List, Dict

logger = logging.getLogger("GAIA.StructureExtractor")

def extract_structure(code: str, language: str = "python") -> Dict[str, List[Dict]]:
    """
    Extract high-level structure of the code (functions, classes).

    Args:
        code (str): Source code text
        language (str): Programming language (default: python)

    Returns:
        dict: Structure dictionary with line numbers
    """
    if language.lower() != "python":
        logger.warning("⚠️ Structure extraction not implemented for non-Python.")
        return {"functions": [], "classes": []}

    structure = {"functions": [], "classes": []}

    try:
        tree = ast.parse(code)
        lines = code.splitlines()

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                structure["functions"].append({
                    "type": "function",
                    "name": node.name,
                    "line_start": node.lineno - 1,
                    "line_end": _get_end_line(node, lines)
                })
            elif isinstance(node, ast.ClassDef):
                structure["classes"].append({
                    "type": "class",
                    "name": node.name,
                    "line_start": node.lineno - 1,
                    "line_end": _get_end_line(node, lines)
                })

        logger.debug("🧱 Structure parsed from code.")

    except Exception as e:
        logger.error(f"❌ Failed to extract structure: {e}", exc_info=True)

    return structure

def _get_end_line(node, lines: List[str]) -> int:
    """
    Estimate end line by walking to the next sibling or end of file.
    This is a best-effort guess due to AST limitations.
    """
    try:
        start = node.lineno - 1
        indent = len(lines[start]) - len(lines[start].lstrip())
        for i in range(start + 1, len(lines)):
            line = lines[i]
            if line.strip() == "":
                continue
            current_indent = len(line) - len(line.lstrip())
            if current_indent <= indent:
                return i
        return len(lines)
    except Exception:
        return start + 1
