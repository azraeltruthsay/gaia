"""
chunk_creator.py

Handles creating document chunks from code content for vector database indexing.
"""

import hashlib
import logging
from typing import List, Dict

logger = logging.getLogger("GAIA.ChunkCreator")

def create_chunks(file_path: str, code: str, structure: Dict) -> List[Dict]:
    """
    Break code into vector-storable chunks. Prioritizes functions, classes, and docstrings.

    Args:
        file_path (str): Path to source file
        code (str): Full text of code
        structure (dict): Parsed functions/classes/etc from structure_extractor

    Returns:
        List[Dict]: List of chunk dictionaries with metadata
    """
    chunks = []

    try:
        lines = code.splitlines()
        for item in structure.get("functions", []) + structure.get("classes", []):
            start = item.get("line_start", 0)
            end = item.get("line_end", len(lines))
            snippet = "\n".join(lines[start:end])

            chunk = {
                "file_path": file_path,
                "type": item.get("type"),
                "name": item.get("name"),
                "lines": f"{start+1}-{end}",
                "content": snippet,
                "hash": hashlib.md5(snippet.encode("utf-8")).hexdigest()
            }
            chunks.append(chunk)

        logger.debug(f"🔗 Created {len(chunks)} code chunks for {file_path}")

    except Exception as e:
        logger.error(f"❌ Failed to chunk {file_path}: {e}", exc_info=True)

    return chunks
