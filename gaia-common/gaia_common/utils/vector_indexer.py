"""
Vector Indexer (with local embedding support)
- Uses MiniLM or compatible model for embedding/query.
"""
"""
Vector Indexer (with local embedding support)
- Uses MiniLM or compatible model for embedding/query.

This module intentionally avoids importing sentence_transformers at top-level
because that package pulls in torch/CUDA. All SentenceTransformer imports
are done lazily inside the functions or class methods that actually need them.
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

logger = logging.getLogger("GAIA.VectorIndexer")

EMBED_MODEL_PATH = "/models/all-MiniLM-L6-v2"

# Chunking configuration
DEFAULT_CHUNK_SIZE = 512  # tokens (roughly 4 chars per token)
DEFAULT_CHUNK_OVERLAP = 64  # tokens overlap between chunks

# Lazy import to avoid circular dependencies
# Config will be imported when needed from gaia_core
Config = None


def _chunk_text(text: str, chunk_size: int = DEFAULT_CHUNK_SIZE, overlap: int = DEFAULT_CHUNK_OVERLAP) -> List[Dict[str, Any]]:
    """
    Split text into overlapping chunks for better semantic retrieval.

    Args:
        text: The full document text
        chunk_size: Target size in tokens (approx 4 chars per token)
        overlap: Number of tokens to overlap between chunks

    Returns:
        List of dicts with 'text', 'start_char', 'end_char', 'chunk_idx'
    """
    if not text or not text.strip():
        return []

    # Approximate chars per chunk (4 chars ≈ 1 token)
    char_chunk_size = chunk_size * 4
    char_overlap = overlap * 4

    # If text is smaller than one chunk, return as-is
    if len(text) <= char_chunk_size:
        return [{
            'text': text,
            'start_char': 0,
            'end_char': len(text),
            'chunk_idx': 0
        }]

    chunks = []
    start = 0
    chunk_idx = 0

    while start < len(text):
        end = start + char_chunk_size

        # If not at the end, try to break at a sentence or paragraph boundary
        if end < len(text):
            # Look for paragraph break first (within last 20% of chunk)
            search_start = start + int(char_chunk_size * 0.8)
            para_break = text.rfind('\n\n', search_start, end)
            if para_break > search_start:
                end = para_break + 2  # Include the newlines
            else:
                # Look for sentence break
                for punct in ['. ', '.\n', '! ', '!\n', '? ', '?\n']:
                    sent_break = text.rfind(punct, search_start, end)
                    if sent_break > search_start:
                        end = sent_break + len(punct)
                        break
        else:
            end = len(text)

        chunk_text = text[start:end].strip()
        if chunk_text:
            chunks.append({
                'text': chunk_text,
                'start_char': start,
                'end_char': end,
                'chunk_idx': chunk_idx
            })
            chunk_idx += 1

        # Move start forward, accounting for overlap
        start = end - char_overlap
        if start >= len(text) - char_overlap:
            break

    logger.debug(f"[CHUNK] Split text ({len(text)} chars) into {len(chunks)} chunks")
    return chunks


def _get_config():
    """Lazy load Config to avoid circular imports."""
    global Config
    if Config is None:
        try:
            from gaia_core.config import Config as _Config
            Config = _Config
        except ImportError:
            # Fallback for standalone use
            class Config:
                pass
    return Config()


class VectorIndexer:
    """
    Singleton VectorIndexer wrapper expected by MCPClient.

    Provides a stable `instance()` entrypoint and a `query(query, top_k)` method
    that returns a list of scored results. This wraps the existing helper
    functions so older code can import `VectorIndexer` safely.
    """

    _instances = {}

    def __init__(self, knowledge_base_name: str = "system", model_path: str = EMBED_MODEL_PATH):
        self.knowledge_base_name = knowledge_base_name
        self.model_path = model_path

        global_config = _get_config()
        self.knowledge_bases_config = getattr(global_config, 'constants', {}).get("KNOWLEDGE_BASES", {})

        self.kb_config = self.knowledge_bases_config.get(self.knowledge_base_name)
        if not self.kb_config:
            raise ValueError(f"Knowledge base '{self.knowledge_base_name}' not found in configuration.")

        self.doc_dir = Path(global_config.KNOWLEDGE_DIR) / self.kb_config["doc_dir"]
        self.vector_store_dir = Path(global_config.KNOWLEDGE_DIR) / self.kb_config["vector_store_dir"]
        self.index_path = self.vector_store_dir / "index.json"

        try:
            from sentence_transformers import SentenceTransformer as _ST
        except Exception as e:
            logger.error(f"Failed to import SentenceTransformer: {e}")
            _ST = None
        if _ST is not None:
            try:
                self.model = _ST(self.model_path, device='cpu')
            except Exception as e:
                logger.error(f"Failed to load SentenceTransformer model from {self.model_path}: {e}")
                self.model = None
        else:
            self.model = None
        self.index = self.load_vector_index()

    @classmethod
    def instance(cls, knowledge_base_name: str = "system"):
        print(f"VectorIndexer.instance called with knowledge_base_name: {knowledge_base_name}")
        if knowledge_base_name not in cls._instances:
            cls._instances[knowledge_base_name] = cls(knowledge_base_name)
        return cls._instances[knowledge_base_name]

    def load_vector_index(self) -> Dict[str, Any]:
        logger.error("THIS IS A TEST LOG MESSAGE TO SEE IF THIS FUNCTION IS CALLED AT ALL")
        if self.index_path.exists():
            with open(self.index_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                doc_count = len(data.get("docs", []))
                if doc_count == 0:
                    logger.warning(
                        f"Vector index at '{self.index_path}' exists but is EMPTY. "
                        f"Run build_index_from_docs() to populate it from '{self.doc_dir}'."
                    )
                else:
                    logger.info(f"Loaded vector index for '{self.knowledge_base_name}' with {doc_count} documents.")
                return data
        else:
            logger.warning(
                f"Vector index not found at '{self.index_path}'. "
                f"Run build_index_from_docs() to create it from '{self.doc_dir}'."
            )
            return {"docs": [], "embeddings": []}

    def save_vector_index(self):
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.index_path, "w", encoding="utf-8") as f:
            json.dump(self.index, f)

    def refresh_index(self):
        self.index = self.load_vector_index()

    def build_index_from_docs(self, chunk_size: int = DEFAULT_CHUNK_SIZE, chunk_overlap: int = DEFAULT_CHUNK_OVERLAP) -> bool:
        """Walk the configured doc_dir and (re)build the vector index with chunking.

        Args:
            chunk_size: Target chunk size in tokens (default 512)
            chunk_overlap: Overlap between chunks in tokens (default 64)
        """
        logger.info(f"Building index for '{self.knowledge_base_name}' from '{self.doc_dir}' (chunk_size={chunk_size}, overlap={chunk_overlap})")
        if not self.model:
            raise RuntimeError("Embedding model not available: SentenceTransformer not loaded.")

        docs = []
        embeddings = []
        file_count = 0
        chunk_count = 0

        for doc_file in self.doc_dir.glob("**/*.*"):
            if doc_file.is_file():
                try:
                    with open(doc_file, "r", encoding="utf-8") as f:
                        text = f.read()

                    # Chunk the document
                    chunks = _chunk_text(text, chunk_size, chunk_overlap)
                    file_count += 1

                    for chunk in chunks:
                        doc_entry = {
                            "filename": str(doc_file),
                            "text": chunk['text'],
                            "chunk_idx": chunk['chunk_idx'],
                            "start_char": chunk['start_char'],
                            "end_char": chunk['end_char'],
                            "total_chunks": len(chunks)
                        }
                        docs.append(doc_entry)
                        emb = self.model.encode(chunk['text'])
                        embeddings.append(emb.tolist())
                        chunk_count += 1

                except Exception as e:
                    logger.error(f"Error processing file {doc_file}: {e}")
                    continue

        self.index = {"docs": docs, "embeddings": embeddings}
        self.save_vector_index()
        logger.info(f"[CHUNK] Built index: {file_count} files -> {chunk_count} chunks")
        return True

    def add_document(self, file_path: str, chunk_size: int = DEFAULT_CHUNK_SIZE, chunk_overlap: int = DEFAULT_CHUNK_OVERLAP) -> bool:
        """Add a single document to the vector index with chunking.

        Args:
            file_path: Path to the document file
            chunk_size: Target chunk size in tokens (default 512)
            chunk_overlap: Overlap between chunks in tokens (default 64)
        """
        if not self.model:
            raise RuntimeError("Embedding model not available: SentenceTransformer not loaded.")

        doc_file = Path(file_path)
        if not doc_file.is_file():
            raise FileNotFoundError(f"File not found: {file_path}")

        with open(doc_file, "r", encoding="utf-8") as f:
            text = f.read()

        # Chunk the document
        chunks = _chunk_text(text, chunk_size, chunk_overlap)
        chunk_count = 0

        for chunk in chunks:
            doc_entry = {
                "filename": str(doc_file),
                "text": chunk['text'],
                "chunk_idx": chunk['chunk_idx'],
                "start_char": chunk['start_char'],
                "end_char": chunk['end_char'],
                "total_chunks": len(chunks)
            }
            self.index["docs"].append(doc_entry)
            embedding = self.model.encode(chunk['text'])
            self.index["embeddings"].append(embedding.tolist())
            chunk_count += 1

        self.save_vector_index()
        logger.info(f"[CHUNK] Added document '{file_path}' as {chunk_count} chunks")
        return True

    def query(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Return top_k matches as a list of dicts: {idx, score, filename, text}.

        If the index is empty, returns an empty list.
        """
        if not self.model:
            raise RuntimeError("Embedding model not available: SentenceTransformer not loaded.")
        self.refresh_index()
        if not self.index.get("docs"):
            logger.warning(
                f"Query on '{self.knowledge_base_name}' returned no results: index is empty. "
                f"GAIA will not have knowledge context and may hallucinate. "
                f"Run build_index_from_docs() to populate the index."
            )
            return []
        q_emb = self.model.encode(query)
        import numpy as np
        try:
            from sentence_transformers import util as _util
            sims = [float(_util.pytorch_cos_sim(q_emb, np.array(e))[0][0]) for e in self.index["embeddings"]]
        except Exception:
            # Fallback: cosine via numpy
            def _cos(a, b):
                a = np.array(a)
                b = np.array(b)
                if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
                    return 0.0
                return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
            sims = [_cos(q_emb, e) for e in self.index["embeddings"]]

        idxs = sorted(range(len(sims)), key=lambda i: sims[i], reverse=True)[:top_k]
        results = []
        for i in idxs:
            results.append({
                "idx": i,
                "score": sims[i],
                "filename": self.index["docs"][i].get("filename"),
                "text": self.index["docs"][i].get("text"),
            })
        return results

# DEPRECATED FUNCTIONS
# These functions are deprecated and will be removed in a future version.
# Use the VectorIndexer class instead.

def embed_gaia_reference() -> bool:
    """
    DEPRECATED: Use VectorIndexer.instance("system").build_index_from_docs() instead.
    """
    return VectorIndexer.instance("system").build_index_from_docs()

def vector_query(query: str) -> str:
    """
    DEPRECATED: Use VectorIndexer.instance("system").query(query) instead.
    """
    results = VectorIndexer.instance("system").query(query, top_k=1)
    if results:
        top_result = results[0]
        return f"Top match: {top_result['filename']}\n\n{top_result['text'][:500]}..."
    return "No knowledge indexed."
