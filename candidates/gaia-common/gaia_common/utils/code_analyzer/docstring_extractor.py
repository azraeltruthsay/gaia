"""
docstring_extractor.py

Handles extracting docstrings and comments from code files.
"""

import ast
import logging

logger = logging.getLogger("GAIA.DocstringExtractor")

def extract_docstrings(code: str, language: str = "python") -> dict:
    """
    Extracts module, function, and class docstrings from Python code.
    Extendable to other languages in future.

    Args:
        code (str): Source code string
        language (str): Programming language (default: python)

    Returns:
        dict: Parsed docstring segments
    """
    if language.lower() != "python":
        logger.warning("⚠️ Docstring extraction not supported for non-Python code yet.")
        return {}

    docstrings = {
        "module": None,
        "functions": [],
        "classes": []
    }

    try:
        tree = ast.parse(code)
        docstrings["module"] = ast.get_docstring(tree)

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                docstrings["functions"].append({
                    "name": node.name,
                    "doc": ast.get_docstring(node)
                })
            elif isinstance(node, ast.ClassDef):
                docstrings["classes"].append({
                    "name": node.name,
                    "doc": ast.get_docstring(node)
                })

        logger.debug("📘 Docstrings extracted.")

    except Exception as e:
        logger.error(f"❌ Failed to extract docstrings: {e}", exc_info=True)

    return docstrings
