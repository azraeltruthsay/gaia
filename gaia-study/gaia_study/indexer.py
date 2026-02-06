"""
Vector Indexer - Document embedding and index management.

This module handles:
- Building vector indexes from document directories
- Adding documents to existing indexes
- Querying indexes for similar content
- Index persistence (JSON format)

This is the SOLE WRITER to the vector store in the SOA architecture.
gaia-core reads via VectorClient (gaia-common).
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from gaia_common.utils import get_logger

logger = get_logger(__name__)


class VectorIndexer:
    """
    Vector index builder and manager.

    This class is the SOLE WRITER to vector indexes in the GAIA SOA.
    Other services should use VectorClient (gaia-common) for read-only access.

    Features:
    - Build indexes from document directories
    - Add individual documents to indexes
    - Query indexes for similarity search
    - Persist indexes to JSON files

    Usage:
        indexer = VectorIndexer("my_knowledge_base")
        indexer.build_index_from_docs()  # Initial build
        indexer.add_document("/path/to/new_doc.md")  # Add single doc
        results = indexer.query("What is GAIA?", top_k=5)
    """

    # Class-level instance cache
    _instances: Dict[str, "VectorIndexer"] = {}

    def __init__(
        self,
        knowledge_base_name: str,
        model_path: Optional[str] = None,
    ):
        """
        Initialize the vector indexer.

        Args:
            knowledge_base_name: Name of the knowledge base
            model_path: Path to embedding model (default from env)
        """
        self.knowledge_base_name = knowledge_base_name
        self.model_path = model_path or os.getenv(
            "EMBEDDING_MODEL_PATH",
            "/models/all-MiniLM-L6-v2"
        )

        # Configure paths
        knowledge_dir = Path(os.getenv("KNOWLEDGE_DIR", "/knowledge"))
        vector_store_dir = Path(os.getenv("VECTOR_STORE_PATH", "/vector_store"))

        # Knowledge base specific paths
        self.doc_dir = knowledge_dir / knowledge_base_name
        self.index_path = vector_store_dir / knowledge_base_name / "index.json"

        # Lazy-loaded model
        self._model = None
        self._index: Optional[Dict[str, Any]] = None

        logger.info(
            f"VectorIndexer initialized for '{knowledge_base_name}' "
            f"doc_dir={self.doc_dir} index_path={self.index_path}"
        )

    @classmethod
    def instance(cls, knowledge_base_name: str) -> "VectorIndexer":
        """Get or create a singleton instance for a knowledge base."""
        if knowledge_base_name not in cls._instances:
            cls._instances[knowledge_base_name] = cls(knowledge_base_name)
        return cls._instances[knowledge_base_name]

    @property
    def model(self):
        """Lazy-load the embedding model."""
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(self.model_path)
                logger.info(f"Loaded embedding model from {self.model_path}")
            except ImportError:
                raise RuntimeError(
                    "sentence-transformers not installed. "
                    "Install with: pip install sentence-transformers"
                )
            except Exception as e:
                raise RuntimeError(f"Failed to load embedding model: {e}")
        return self._model

    @property
    def index(self) -> Dict[str, Any]:
        """Get the loaded index, loading from disk if needed."""
        if self._index is None:
            self._index = self.load_index()
        return self._index

    def load_index(self) -> Dict[str, Any]:
        """Load the vector index from disk."""
        if not self.index_path.exists():
            logger.warning(
                f"Vector index not found at {self.index_path}. "
                f"Run build_index_from_docs() to create it."
            )
            return {"docs": [], "embeddings": []}

        try:
            with open(self.index_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            doc_count = len(data.get("docs", []))
            logger.info(
                f"Loaded vector index for '{self.knowledge_base_name}' "
                f"with {doc_count} documents"
            )
            return data
        except Exception as e:
            logger.error(f"Failed to load vector index: {e}")
            return {"docs": [], "embeddings": []}

    def save_index(self) -> None:
        """Save the current index to disk."""
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.index_path, "w", encoding="utf-8") as f:
            json.dump(self._index, f, indent=2)
        logger.info(f"Saved vector index to {self.index_path}")

    def refresh_index(self) -> None:
        """Reload the index from disk."""
        self._index = self.load_index()

    def build_index_from_docs(self) -> bool:
        """
        Build or rebuild the vector index from the document directory.

        Returns:
            True if successful
        """
        logger.info(
            f"Building index for '{self.knowledge_base_name}' "
            f"from '{self.doc_dir}'"
        )

        if not self.doc_dir.exists():
            raise FileNotFoundError(f"Document directory not found: {self.doc_dir}")

        docs = []
        embeddings = []
        processed = 0
        errors = 0

        # Supported file extensions
        extensions = {".txt", ".md", ".json", ".py", ".yaml", ".yml"}

        for doc_file in self.doc_dir.glob("**/*"):
            if not doc_file.is_file():
                continue
            if doc_file.suffix.lower() not in extensions:
                continue

            try:
                with open(doc_file, "r", encoding="utf-8", errors="replace") as f:
                    text = f.read()

                # Skip empty or very short files
                if len(text.strip()) < 10:
                    continue

                embedding = self.model.encode(text)
                docs.append({
                    "filename": str(doc_file),
                    "text": text,
                    "metadata": {
                        "size": len(text),
                        "extension": doc_file.suffix,
                    }
                })
                embeddings.append(embedding.tolist())
                processed += 1

            except Exception as e:
                logger.error(f"Error processing {doc_file}: {e}")
                errors += 1
                continue

        self._index = {"docs": docs, "embeddings": embeddings}
        self.save_index()

        logger.info(
            f"Index build complete: {processed} documents indexed, "
            f"{errors} errors"
        )
        return True

    def add_document(self, file_path: str) -> bool:
        """
        Add a single document to the index.

        Args:
            file_path: Path to the document file

        Returns:
            True if successful
        """
        doc_file = Path(file_path)
        if not doc_file.is_file():
            raise FileNotFoundError(f"File not found: {file_path}")

        try:
            with open(doc_file, "r", encoding="utf-8", errors="replace") as f:
                text = f.read()

            embedding = self.model.encode(text)

            # Ensure index is loaded
            _ = self.index

            self._index["docs"].append({
                "filename": str(doc_file),
                "text": text,
                "metadata": {
                    "size": len(text),
                    "extension": doc_file.suffix,
                }
            })
            self._index["embeddings"].append(embedding.tolist())

            self.save_index()
            logger.info(f"Added document to index: {file_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to add document {file_path}: {e}")
            raise

    def query(
        self,
        query: str,
        top_k: int = 5,
        min_score: float = 0.0,
    ) -> List[Dict[str, Any]]:
        """
        Query the index for similar documents.

        Args:
            query: Search query text
            top_k: Number of results to return
            min_score: Minimum similarity score

        Returns:
            List of result dicts with idx, score, filename, text
        """
        # Load index if needed
        index = self.index

        if not index.get("docs"):
            logger.warning(
                f"Query on '{self.knowledge_base_name}' returned no results: "
                "index is empty"
            )
            return []

        # Encode query
        query_embedding = self.model.encode(query)

        # Calculate similarities
        similarities = self._compute_similarities(
            query_embedding,
            index["embeddings"]
        )

        # Sort and filter results
        sorted_indices = sorted(
            range(len(similarities)),
            key=lambda i: similarities[i],
            reverse=True
        )

        results = []
        for idx in sorted_indices[:top_k]:
            score = similarities[idx]
            if score < min_score:
                continue

            doc = index["docs"][idx]
            results.append({
                "idx": idx,
                "score": score,
                "filename": doc.get("filename", ""),
                "text": doc.get("text", ""),
            })

        return results

    def _compute_similarities(
        self,
        query_embedding,
        doc_embeddings: List[List[float]],
    ) -> List[float]:
        """Compute cosine similarities."""
        import numpy as np

        query_vec = np.array(query_embedding)

        try:
            from sentence_transformers import util
            doc_matrix = np.array(doc_embeddings)
            sims = util.pytorch_cos_sim(query_vec, doc_matrix)[0]
            return [float(s) for s in sims]
        except Exception:
            pass

        # Fallback: numpy
        def cosine_sim(a, b):
            a = np.array(a)
            b = np.array(b)
            norm_a = np.linalg.norm(a)
            norm_b = np.linalg.norm(b)
            if norm_a == 0 or norm_b == 0:
                return 0.0
            return float(np.dot(a, b) / (norm_a * norm_b))

        return [cosine_sim(query_vec, e) for e in doc_embeddings]

    def doc_count(self) -> int:
        """Get the number of documents in the index."""
        return len(self.index.get("docs", []))

    def get_status(self) -> Dict[str, Any]:
        """Get indexer status information."""
        return {
            "knowledge_base_name": self.knowledge_base_name,
            "doc_dir": str(self.doc_dir),
            "index_path": str(self.index_path),
            "doc_count": self.doc_count(),
            "index_exists": self.index_path.exists(),
            "model_loaded": self._model is not None,
        }
