"""
gaia-mcp: The Hands - Sandboxed tool execution service.
"""

import os
import logging
from typing import Dict, Any
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import json

          # Setup logging
try:
    from gaia_common.utils import setup_logging
    setup_logging(log_dir="/logs", level=logging.INFO, service_name="gaia-mcp")
except ImportError:
    logging.basicConfig(level=logging.INFO)

logger = logging.getLogger("GAIA.MCP.API")

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Initializing GAIA MCP...")
    yield
    logger.info("GAIA MCP shutting down...")

app = FastAPI(lifespan=lifespan, title="GAIA MCP API")

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "gaia-mcp"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("gaia_mcp.main:app", host="0.0.0.0", port=8765)
