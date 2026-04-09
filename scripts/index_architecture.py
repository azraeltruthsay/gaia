#!/usr/bin/env python3
"""
Architectural RAG Extraction Script - "The Self-Map"

Indexes GAIA's service architecture into the code_architecture vector
collection for Stage 0 Neural Grounding and self-knowledge queries.

Three phases:
1. Blueprint generation — service YAML contracts
2. AST extraction — function signatures and docstrings
3. Vector indexing — embed all docs via VectorIndexer

Usage:
    docker exec gaia-study python /gaia/GAIA_Project/scripts/index_architecture.py

Requires: code_architecture entry in KNOWLEDGE_BASES (gaia_constants.json).
Writes to: /knowledge/code_architecture/ (writable base mount in gaia-study).
"""

import ast
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("GAIA.ArchIndex")

# Container-relative paths
PROJECT_ROOT = Path("/gaia/GAIA_Project")
DOC_DIR = Path("/knowledge/code_architecture")
COLLECTION_NAME = "code_architecture"

# All GAIA services (source dirs that exist under PROJECT_ROOT)
SERVICES = [
    "gaia-core", "gaia-web", "gaia-mcp", "gaia-orchestrator",
    "gaia-study", "gaia-audio",
    "gaia-doctor", "gaia-monkey",
    "gaia-common", "gaia-engine",
]


def extract_ast_summaries():
    """Extract function/class signatures and docstrings to markdown files."""
    logger.info("=== Phase 1: AST Extraction ===")
    DOC_DIR.mkdir(parents=True, exist_ok=True)
    file_count = 0

    for service in SERVICES:
        pkg_name = service.replace("-", "_")
        source_root = PROJECT_ROOT / service / pkg_name

        if not source_root.exists():
            # Try candidates/ path
            source_root = PROJECT_ROOT / "candidates" / service / pkg_name
            if not source_root.exists():
                logger.debug("Skipping %s: no source dir found", service)
                continue

        summary_lines = [f"# AST Summary: {service}\n"]
        found_symbols = 0

        for p in sorted(source_root.rglob("*.py")):
            if "__pycache__" in str(p) or "/tests/" in str(p):
                continue

            try:
                tree = ast.parse(p.read_text(encoding="utf-8"))
            except Exception:
                continue

            module_path = p.relative_to(source_root.parent.parent)
            file_symbols = []

            for node in ast.iter_child_nodes(tree):
                if isinstance(node, ast.ClassDef):
                    doc = ast.get_docstring(node) or ""
                    methods = [n.name for n in ast.iter_child_nodes(node)
                               if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
                               and not n.name.startswith("_")]
                    file_symbols.append(
                        f"### class `{node.name}`\n"
                        f"{doc[:200]}\n"
                        f"Methods: {', '.join(methods[:10]) or 'none'}\n"
                    )
                elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    if node.name.startswith("_"):
                        continue
                    try:
                        sig = f"{node.name}({ast.unparse(node.args)})"
                    except Exception:
                        sig = node.name + "(...)"
                    doc = ast.get_docstring(node) or ""
                    file_symbols.append(f"### `{sig}`\n{doc[:200]}\n")

            if file_symbols:
                summary_lines.append(f"## {module_path}\n")
                summary_lines.extend(file_symbols)
                found_symbols += len(file_symbols)

        if found_symbols > 0:
            ast_file = DOC_DIR / f"{service}_ast.md"
            ast_file.write_text("\n".join(summary_lines), encoding="utf-8")
            logger.info("Wrote %s: %d symbols", ast_file.name, found_symbols)
            file_count += 1

    logger.info("AST extraction complete: %d services indexed", file_count)


def index_contracts():
    """Copy contract YAML summaries into the doc dir for indexing."""
    logger.info("=== Phase 2: Contract Indexing ===")
    contracts_dir = PROJECT_ROOT / "contracts" / "services"
    if not contracts_dir.exists():
        logger.warning("No contracts/services/ directory found")
        return

    DOC_DIR.mkdir(parents=True, exist_ok=True)
    count = 0
    for yaml_file in sorted(contracts_dir.glob("*.yaml")):
        content = yaml_file.read_text(encoding="utf-8")
        target = DOC_DIR / f"contract_{yaml_file.stem}.yaml"
        target.write_text(f"# API Contract: {yaml_file.stem}\n\n{content}", encoding="utf-8")
        count += 1

    # Also index the connectivity matrix if it exists
    connectivity = contracts_dir.parent / "CONNECTIVITY.md"
    if connectivity.exists():
        target = DOC_DIR / "connectivity_matrix.md"
        target.write_text(connectivity.read_text(encoding="utf-8"), encoding="utf-8")
        count += 1

    logger.info("Indexed %d contract files", count)


def build_vector_index():
    """Trigger VectorIndexer to build the code_architecture collection."""
    logger.info("=== Phase 3: Vector Indexing ===")
    try:
        from gaia_common.utils.vector_indexer import VectorIndexer
        indexer = VectorIndexer.instance(COLLECTION_NAME)
        indexer.build_index_from_docs()
        logger.info("Vector index built for '%s'", COLLECTION_NAME)
    except Exception as e:
        logger.error("Vector indexing failed: %s", e, exc_info=True)
        # Non-fatal — the docs are still on disk for manual indexing
        logger.info("Documents are available in %s for manual indexing", DOC_DIR)


if __name__ == "__main__":
    try:
        extract_ast_summaries()
        index_contracts()
        build_vector_index()
        logger.info("GAIA Self-Map updated successfully.")
    except Exception as e:
        logger.critical("Self-Map update failed: %s", e, exc_info=True)
        sys.exit(1)
