"""
Vector Client - Read-only interface for vector search.

This module provides a read-only client for querying vector indexes.
The indexes are built and maintained by gaia-study; this client only reads them.

Usage:
    from gaia_common.utils import VectorClient

    client = VectorClient(index_path="/knowledge/vector_store/index.json")
    results = client.query("What is GAIA?", top_k=5)
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class QueryResult:
    """A single result from a vector query."""
    index: int
    score: float
    filename: str
    text: str
    metadata: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "index": self.index,
            "score": self.score,
            "filename": self.filename,
            "text": self.text,
            "metadata": self.metadata,
        }


class VectorClient:
    """
    Read-only client for vector search.

    This client loads pre-built vector indexes and provides query functionality.
    Index building is handled by gaia-study; this is read-only.

    Features:
    - Lazy loading of embedding model (only when query() is called)
    - Simple cosine similarity search
    - Support for multiple knowledge bases

    Example:
        client = VectorClient(
            index_path="/knowledge/system/vector_store/index.json",
            model_path="/models/all-MiniLM-L6-v2"
        )
        results = client.query("How does GAIA handle ethics?", top_k=3)
    """

    def __init__(
        self,
        index_path: Optional[str] = None,
        model_path: Optional[str] = None,
    ):
        """
        Initialize the vector client.

        Args:
            index_path: Path to the vector index JSON file
            model_path: Path to the embedding model (lazy loaded)
        """
        self.index_path = Path(index_path) if index_path else None
        self.model_path = model_path or os.getenv(
            "EMBEDDING_MODEL_PATH",
            "/models/all-MiniLM-L6-v2"
        )
        self._model = None
        self._index: Optional[Dict[str, Any]] = None

    @property
    def model(self):
        """Lazy-load the embedding model."""
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(self.model_path)
                logger.info(f"Loaded embedding model from {self.model_path}")
            except ImportError:
                logger.warning(
                    "sentence-transformers not installed. "
                    "Install with: pip install sentence-transformers"
                )
                raise
            except Exception as e:
                logger.error(f"Failed to load embedding model: {e}")
                raise
        return self._model

    def load_index(self, force: bool = False) -> Dict[str, Any]:
        """
        Load the vector index from disk.

        Args:
            force: If True, reload even if already loaded

        Returns:
            The index dictionary with 'docs' and 'embeddings' keys
        """
        if self._index is not None and not force:
            return self._index

        if self.index_path is None:
            raise ValueError("No index_path configured")

        if not self.index_path.exists():
            logger.warning(f"Vector index not found at {self.index_path}")
            self._index = {"docs": [], "embeddings": []}
            return self._index

        try:
            with open(self.index_path, "r", encoding="utf-8") as f:
                self._index = json.load(f)
            doc_count = len(self._index.get("docs", []))
            logger.info(f"Loaded vector index with {doc_count} documents")
            return self._index
        except Exception as e:
            logger.error(f"Failed to load vector index: {e}")
            self._index = {"docs": [], "embeddings": []}
            return self._index

    def query(
        self,
        query: str,
        top_k: int = 5,
        min_score: float = 0.0,
    ) -> List[QueryResult]:
        """
        Query the vector index for similar documents.

        Args:
            query: The search query text
            top_k: Number of results to return
            min_score: Minimum similarity score (0.0 to 1.0)

        Returns:
            List of QueryResult objects sorted by similarity score
        """
        index = self.load_index()

        if not index.get("docs"):
            logger.warning("Vector index is empty, returning no results")
            return []

        # Encode the query
        try:
            query_embedding = self.model.encode(query)
        except Exception as e:
            logger.error(f"Failed to encode query: {e}")
            return []

        # Calculate similarities
        similarities = self._compute_similarities(
            query_embedding,
            index["embeddings"]
        )

        # Sort and filter
        results = []
        sorted_indices = sorted(
            range(len(similarities)),
            key=lambda i: similarities[i],
            reverse=True
        )

        for idx in sorted_indices[:top_k]:
            score = similarities[idx]
            if score < min_score:
                continue

            doc = index["docs"][idx]
            results.append(QueryResult(
                index=idx,
                score=score,
                filename=doc.get("filename", ""),
                text=doc.get("text", ""),
                metadata=doc.get("metadata", {}),
            ))

        return results

    def _compute_similarities(
        self,
        query_embedding,
        doc_embeddings: List[List[float]],
    ) -> List[float]:
        """Compute cosine similarities between query and documents."""
        import numpy as np

        query_vec = np.array(query_embedding)

        # Try to use sentence-transformers util for efficiency
        try:
            from sentence_transformers import util
            doc_matrix = np.array(doc_embeddings)
            sims = util.pytorch_cos_sim(query_vec, doc_matrix)[0]
            return [float(s) for s in sims]
        except Exception:
            pass

        # Fallback: numpy cosine similarity
        def cosine_sim(a, b):
            a = np.array(a)
            b = np.array(b)
            norm_a = np.linalg.norm(a)
            norm_b = np.linalg.norm(b)
            if norm_a == 0 or norm_b == 0:
                return 0.0
            return float(np.dot(a, b) / (norm_a * norm_b))

        return [cosine_sim(query_vec, e) for e in doc_embeddings]

    def is_loaded(self) -> bool:
        """Check if the index is loaded."""
        return self._index is not None

    def doc_count(self) -> int:
        """Get the number of documents in the index."""
        index = self.load_index()
        return len(index.get("docs", []))

    def get_status(self) -> Dict[str, Any]:
        """Get client status information."""
        index = self.load_index()
        return {
            "index_path": str(self.index_path) if self.index_path else None,
            "model_path": self.model_path,
            "doc_count": len(index.get("docs", [])),
            "model_loaded": self._model is not None,
            "index_loaded": self._index is not None,
        }


class VectorClientFactory:
    """
    Factory for creating VectorClient instances for different knowledge bases.

    Usage:
        factory = VectorClientFactory(
            base_path="/knowledge",
            model_path="/models/all-MiniLM-L6-v2"
        )
        client = factory.get_client("system")
    """

    def __init__(
        self,
        base_path: str = "/knowledge",
        model_path: Optional[str] = None,
    ):
        self.base_path = Path(base_path)
        self.model_path = model_path
        self._clients: Dict[str, VectorClient] = {}

    def get_client(self, knowledge_base_name: str) -> VectorClient:
        """
        Get or create a client for a knowledge base.

        Args:
            knowledge_base_name: Name of the knowledge base

        Returns:
            VectorClient instance
        """
        if knowledge_base_name not in self._clients:
            # Standard path: /knowledge/{kb_name}/vector_store/index.json
            index_path = (
                self.base_path
                / knowledge_base_name
                / "vector_store"
                / "index.json"
            )
            self._clients[knowledge_base_name] = VectorClient(
                index_path=str(index_path),
                model_path=self.model_path,
            )

        return self._clients[knowledge_base_name]

    def list_available(self) -> List[str]:
        """List knowledge bases with available indexes."""
        available = []
        try:
            for entry in self.base_path.iterdir():
                if entry.is_dir():
                    index_path = entry / "vector_store" / "index.json"
                    if index_path.exists():
                        available.append(entry.name)
        except Exception as e:
            logger.warning(f"Error listing knowledge bases: {e}")
        return sorted(available)


__all__ = [
    "QueryResult",
    "VectorClient",
    "VectorClientFactory",
]
