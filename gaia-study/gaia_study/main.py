"""
GAIA Study Server - Entry Point

Background processing service for GAIA:
- Vector index building and maintenance (SOLE WRITER)
- Document embedding
- Conversation summarization
- LoRA adapter training

Usage:
    uvicorn gaia_study.main:app --host 0.0.0.0 --port 8766
"""

import logging
import os

from fastapi import FastAPI

from gaia_common.utils import setup_logging, get_logger, install_health_check_filter

from .server import create_app
from .indexer import VectorIndexer

# Initialize logging
log_dir = os.getenv("GAIA_LOG_DIR", "/var/log/gaia")
log_level = getattr(logging, os.getenv("LOG_LEVEL", "INFO").upper(), logging.INFO)
setup_logging(log_dir=log_dir, level=log_level, service_name="gaia-study")
install_health_check_filter()

logger = get_logger(__name__)

# Create the FastAPI application
app = create_app()


@app.on_event("startup")
async def startup_event():
    """Initialize service on startup."""
    logger.info("gaia-study starting up...")
    logger.info(f"Knowledge dir: {os.getenv('KNOWLEDGE_DIR', '/knowledge')}")
    logger.info(f"Vector store: {os.getenv('VECTOR_STORE_PATH', '/vector_store')}")
    logger.info(f"Models dir: {os.getenv('MODELS_DIR', '/models')}")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    logger.info("gaia-study shutting down...")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "gaia_study.main:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", "8766")),
        reload=os.getenv("GAIA_ENV", "production") == "development",
    )
