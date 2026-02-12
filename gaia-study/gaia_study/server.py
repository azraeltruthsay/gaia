"""
GAIA Study Server - FastAPI Application

Background processing API for vector indexing, document management,
and LoRA adapter training (Study Mode).
"""

import logging
import os
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field

from gaia_common.utils import get_logger

from .indexer import VectorIndexer
from .study_mode_manager import StudyModeManager, TrainingConfig, TrainingResult

logger = get_logger(__name__)

# Singleton study mode manager (initialized on first use)
_study_manager: Optional[StudyModeManager] = None


def get_study_manager() -> StudyModeManager:
    """Get or create the singleton StudyModeManager instance."""
    global _study_manager
    if _study_manager is None:
        # Load config from environment or defaults
        adapter_dir = os.getenv("LORA_ADAPTER_DIR", "/models/lora_adapters")

        # Build study config from environment
        study_config = {
            "max_training_time_seconds": int(os.getenv("MAX_TRAINING_TIME", "600")),
            "max_training_samples": int(os.getenv("MAX_TRAINING_SAMPLES", "1000")),
            "max_training_content_kb": int(os.getenv("MAX_TRAINING_CONTENT_KB", "100")),
            "use_real_training": os.getenv("USE_REAL_TRAINING", "true").lower() == "true",
            "base_model_path": os.getenv("BASE_MODEL_PATH", "/models/Claude"),
            "governance": {
                "forbidden_patterns": [
                    "ignore previous instructions",
                    "you are now",
                    "forget your training",
                    "forget your values",
                    "forget your ethics"
                ],
                "max_session_adapters": 3,
                "max_user_adapters": 10
            },
            "qlora_config": {
                "load_in_4bit": True,
                "bnb_4bit_compute_dtype": "bfloat16",
                "bnb_4bit_quant_type": "nf4",
                "bnb_4bit_use_double_quant": True,
            }
        }

        _study_manager = StudyModeManager(study_config, adapter_base_dir=adapter_dir)
        logger.info(f"StudyModeManager initialized with adapter_dir={adapter_dir}")

    return _study_manager


class IndexBuildRequest(BaseModel):
    """Request to build/rebuild a vector index."""
    knowledge_base_name: str
    force_rebuild: bool = False


class DocumentAddRequest(BaseModel):
    """Request to add a document to the index."""
    knowledge_base_name: str
    file_path: str


class QueryRequest(BaseModel):
    """Request to query the vector index."""
    knowledge_base_name: str
    query: str
    top_k: int = 5


# ═══════════════════════════════════════════════════════════════════════════
# Study Mode / LoRA Training Request Models
# ═══════════════════════════════════════════════════════════════════════════

class StudyStartRequest(BaseModel):
    """Request to start a study/training session."""
    adapter_name: str
    documents: List[str] = Field(..., description="List of document paths to learn from")
    tier: int = Field(default=3, ge=1, le=3, description="Adapter tier: 1=global, 2=user, 3=session")
    pillar: str = Field(default="general", description="Knowledge pillar")
    description: str = ""
    max_steps: int = Field(default=100, ge=1, le=1000)
    activation_triggers: List[str] = Field(default_factory=list)
    tags: List[str] = Field(default_factory=list)


class AdapterLoadRequest(BaseModel):
    """Request to load an adapter."""
    adapter_name: str
    tier: int = Field(default=3, ge=1, le=3)


class AdapterDeleteRequest(BaseModel):
    """Request to delete an adapter."""
    adapter_name: str
    tier: int = Field(default=3, ge=1, le=3)


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="GAIA Study Server",
        description="Background processing and vector indexing service",
        version="0.1.0",
    )

    # Indexer instances (lazy loaded per knowledge base)
    indexers: Dict[str, VectorIndexer] = {}

    def get_indexer(knowledge_base_name: str) -> VectorIndexer:
        """Get or create an indexer for a knowledge base."""
        if knowledge_base_name not in indexers:
            indexers[knowledge_base_name] = VectorIndexer(knowledge_base_name)
        return indexers[knowledge_base_name]

    @app.get("/health")
    async def health_check():
        """Health check endpoint."""
        return {"status": "healthy", "service": "gaia-study"}

    @app.get("/status")
    async def get_status():
        """Get service status including loaded indexes."""
        return {
            "service": "gaia-study",
            "loaded_indexes": list(indexers.keys()),
            "index_stats": {
                name: idx.get_status()
                for name, idx in indexers.items()
            }
        }

    @app.post("/index/build")
    async def build_index(
        request: IndexBuildRequest,
        background_tasks: BackgroundTasks
    ):
        """
        Build or rebuild a vector index.

        This is a long-running operation that runs in the background.
        """
        indexer = get_indexer(request.knowledge_base_name)

        def do_build():
            try:
                indexer.build_index_from_docs()
                logger.info(f"Index build completed for {request.knowledge_base_name}")
            except Exception as e:
                logger.error(f"Index build failed: {e}")

        background_tasks.add_task(do_build)

        return {
            "status": "building",
            "knowledge_base_name": request.knowledge_base_name,
            "message": "Index build started in background"
        }

    @app.post("/index/add")
    async def add_document(request: DocumentAddRequest):
        """Add a single document to the index."""
        try:
            indexer = get_indexer(request.knowledge_base_name)
            indexer.add_document(request.file_path)
            return {
                "status": "success",
                "knowledge_base_name": request.knowledge_base_name,
                "file_path": request.file_path,
            }
        except FileNotFoundError as e:
            raise HTTPException(status_code=404, detail=str(e))
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/index/query")
    async def query_index(request: QueryRequest):
        """Query the vector index for similar documents."""
        try:
            indexer = get_indexer(request.knowledge_base_name)
            results = indexer.query(request.query, top_k=request.top_k)
            return {
                "status": "success",
                "query": request.query,
                "results": [
                    {
                        "index": r["idx"],
                        "score": r["score"],
                        "filename": r["filename"],
                        "text_preview": r["text"][:500] if r["text"] else "",
                    }
                    for r in results
                ]
            }
        except Exception as e:
            logger.exception(f"Query failed: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/index/{knowledge_base_name}/status")
    async def get_index_status(knowledge_base_name: str):
        """Get status of a specific index."""
        indexer = get_indexer(knowledge_base_name)
        return indexer.get_status()

    @app.post("/index/{knowledge_base_name}/refresh")
    async def refresh_index(knowledge_base_name: str):
        """Refresh the index from disk."""
        indexer = get_indexer(knowledge_base_name)
        indexer.refresh_index()
        return {
            "status": "refreshed",
            "knowledge_base_name": knowledge_base_name,
            "doc_count": indexer.doc_count(),
        }

    # ═══════════════════════════════════════════════════════════════════════════
    # GPU Handoff Endpoints (called by gaia-orchestrator)
    # ═══════════════════════════════════════════════════════════════════════════

    # Track GPU availability for training
    _gpu_available = {"available": False, "received_at": None}

    @app.post("/study/gpu-ready")
    async def gpu_ready(background_tasks: BackgroundTasks):
        """
        Signal from orchestrator that the GPU is now available for training.

        If a training session is queued, it will be started automatically.
        Otherwise, gaia-study acknowledges GPU availability for future use.
        """
        import time

        _gpu_available["available"] = True
        _gpu_available["received_at"] = time.strftime(
            "%Y-%m-%dT%H:%M:%SZ", time.gmtime()
        )
        logger.info("GPU ready signal received from orchestrator")

        manager = get_study_manager()
        status = manager.get_status()

        if status["state"] in ("idle", "complete", "failed"):
            logger.info("GPU ready: no queued training, standing by")
            return {
                "ok": True,
                "message": "GPU acknowledged, no training queued",
                "gpu_available": True,
            }
        else:
            logger.info(f"GPU ready: training state is '{status['state']}', may proceed")
            return {
                "ok": True,
                "message": f"GPU acknowledged, training state: {status['state']}",
                "gpu_available": True,
            }

    @app.post("/study/gpu-release")
    async def gpu_release():
        """
        Request from orchestrator to release the GPU.

        Cancels any in-progress training, cleans up CUDA resources,
        and acknowledges the release.
        """
        import gc
        import time

        logger.info("GPU release request received from orchestrator")

        # Cancel any in-progress training
        manager = get_study_manager()
        status = manager.get_status()

        if status["state"] not in ("idle", "complete", "failed"):
            logger.info(f"Cancelling in-progress training (state: {status['state']})")
            manager.cancel_training()

        # Clean up CUDA resources
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()
                allocated = torch.cuda.memory_allocated() / 1e9
                logger.info(f"CUDA cache cleared. VRAM allocated: {allocated:.2f} GB")
        except ImportError:
            logger.debug("torch not available for CUDA cleanup")
        except Exception as e:
            logger.warning(f"CUDA cleanup error: {e}")

        _gpu_available["available"] = False
        _gpu_available["released_at"] = time.strftime(
            "%Y-%m-%dT%H:%M:%SZ", time.gmtime()
        )

        return {
            "ok": True,
            "message": "GPU released, CUDA resources cleaned up",
            "gpu_available": False,
        }

    # ═══════════════════════════════════════════════════════════════════════════
    # Study Mode / LoRA Adapter Endpoints
    # ═══════════════════════════════════════════════════════════════════════════

    @app.post("/study/start")
    async def study_start(request: StudyStartRequest, background_tasks: BackgroundTasks):
        """
        Start a study/training session to learn from documents.

        This creates a LoRA adapter trained on the provided documents.
        Training runs in the background.
        """
        try:
            manager = get_study_manager()

            # Check if already training
            status = manager.get_status()
            if status["state"] not in ["idle", "complete", "failed"]:
                raise HTTPException(
                    status_code=409,
                    detail=f"Training already in progress: {status['state']}"
                )

            # Build training config
            config = TrainingConfig(
                adapter_name=request.adapter_name,
                tier=request.tier,
                pillar=request.pillar,
                source_documents=request.documents,
                description=request.description,
                max_steps=request.max_steps,
                activation_triggers=request.activation_triggers,
                tags=request.tags,
            )

            # Start training asynchronously
            async def do_training():
                try:
                    result = await manager.start_training(config)
                    if result.success:
                        logger.info(f"Training completed: {result.adapter_name}")
                    else:
                        logger.error(f"Training failed: {result.error_message}")
                except Exception as e:
                    logger.exception(f"Training error: {e}")

            background_tasks.add_task(do_training)

            return {
                "ok": True,
                "status": "started",
                "adapter_name": request.adapter_name,
                "message": "Training started in background. Use /study/status to monitor progress."
            }

        except HTTPException:
            raise
        except Exception as e:
            logger.exception(f"Failed to start training: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/study/status")
    async def study_status():
        """Get current study mode status."""
        try:
            manager = get_study_manager()
            return manager.get_status()
        except Exception as e:
            logger.exception(f"Failed to get study status: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/study/cancel")
    async def study_cancel():
        """Cancel an in-progress training session."""
        try:
            manager = get_study_manager()
            cancelled = manager.cancel_training()
            return {
                "ok": cancelled,
                "message": "Training cancelled" if cancelled else "No training in progress"
            }
        except Exception as e:
            logger.exception(f"Failed to cancel training: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/adapters")
    async def adapter_list(tier: Optional[int] = None):
        """List available LoRA adapters."""
        try:
            manager = get_study_manager()
            adapters = manager.list_adapters(tier=tier)
            return {
                "ok": True,
                "adapters": adapters,
                "count": len(adapters)
            }
        except Exception as e:
            logger.exception(f"Failed to list adapters: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/adapters/load")
    async def adapter_load(request: AdapterLoadRequest):
        """Load a LoRA adapter for use in generation."""
        try:
            manager = get_study_manager()
            tier_dir = manager._get_tier_directory(request.tier)
            adapter_path = tier_dir / request.adapter_name

            if not adapter_path.exists():
                raise HTTPException(
                    status_code=404,
                    detail=f"Adapter '{request.adapter_name}' not found in tier {request.tier}"
                )

            # TODO: Actually load into vLLM model pool via gaia-core API
            return {
                "ok": True,
                "adapter_name": request.adapter_name,
                "adapter_path": str(adapter_path),
                "tier": request.tier,
                "message": "Adapter registered for loading (actual loading requires model pool integration)"
            }
        except HTTPException:
            raise
        except Exception as e:
            logger.exception(f"Failed to load adapter: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/adapters/unload")
    async def adapter_unload(request: AdapterLoadRequest):
        """Unload a LoRA adapter."""
        try:
            # TODO: Actually unload from vLLM model pool via gaia-core API
            return {
                "ok": True,
                "adapter_name": request.adapter_name,
                "message": "Adapter unload requested (requires model pool integration)"
            }
        except Exception as e:
            logger.exception(f"Failed to unload adapter: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.delete("/adapters/{adapter_name}")
    async def adapter_delete(adapter_name: str, tier: int = 3):
        """Delete a LoRA adapter."""
        try:
            manager = get_study_manager()
            deleted = manager.delete_adapter(adapter_name, tier)
            return {
                "ok": deleted,
                "adapter_name": adapter_name,
                "tier": tier,
                "message": "Adapter deleted" if deleted else "Adapter not found or protected"
            }
        except Exception as e:
            logger.exception(f"Failed to delete adapter: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/adapters/{adapter_name}")
    async def adapter_info(adapter_name: str, tier: int = 3):
        """Get detailed info about a specific adapter."""
        try:
            import json
            manager = get_study_manager()
            tier_dir = manager._get_tier_directory(tier)
            adapter_path = tier_dir / adapter_name
            metadata_path = adapter_path / "metadata.json"

            if not metadata_path.exists():
                raise HTTPException(
                    status_code=404,
                    detail=f"Adapter '{adapter_name}' not found in tier {tier}"
                )

            with open(metadata_path) as f:
                metadata = json.load(f)

            return {"ok": True, "adapter": metadata}
        except HTTPException:
            raise
        except Exception as e:
            logger.exception(f"Failed to get adapter info: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    return app
