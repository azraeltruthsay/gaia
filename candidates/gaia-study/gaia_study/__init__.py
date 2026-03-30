"""
gaia-study: The Subconscious - Background processing and learning.

This service handles all background and learning operations:
- Vector index building and maintenance (SOLE WRITER)
- Document embedding and semantic search indexing
- LoRA adapter training and management (SOLE WRITER)
- Background task processing
- Conversation summarization
- Knowledge integrity checking

IMPORTANT: This is the SOLE WRITER to:
- /vector_store (read-only for other services)
- /models/adapters (LoRA fine-tuning outputs)

Other services should use gaia-common's VectorClient for read-only access.

Usage:
    # Start the server
    uvicorn gaia_study.main:app --host 0.0.0.0 --port 8766

    # Or run directly
    python -m gaia_study.main

Dependencies:
- gaia-common[vector]: Shared protocols and vector utilities
"""

__version__ = "0.1.0"
__service__ = "gaia-study"

from .indexer import VectorIndexer

__all__ = [
    "__version__",
    "__service__",
    "VectorIndexer",
]
