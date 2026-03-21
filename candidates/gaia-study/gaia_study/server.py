"""
GAIA Study Server - FastAPI Application

Background processing API for vector indexing, document management,
and LoRA adapter training (Study Mode).
"""

import os
from typing import Dict, List, Optional

from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field

from gaia_common.utils import get_logger

from .indexer import VectorIndexer
from .study_mode_manager import StudyModeManager, TrainingConfig

logger = get_logger(__name__)


def _registry_base_model_path() -> str:
    """Resolve prime base model path from Config MODEL_REGISTRY."""
    try:
        from gaia_common.config import Config
        return Config.get_instance().model_path("prime", "base") or "/models/Qwen3.5-4B-Abliterated"
    except Exception:
        return "/models/Qwen3.5-4B-Abliterated"


def _registry_lora_dir() -> str:
    """Resolve lora_adapters path from Config MODEL_REGISTRY."""
    try:
        from gaia_common.config import Config
        return Config.get_instance().model_path("lora_adapters") or "/models/lora_adapters"
    except Exception:
        return "/models/lora_adapters"


# Singleton study mode manager (initialized on first use)
_study_manager: Optional[StudyModeManager] = None


def get_study_manager() -> StudyModeManager:
    """Get or create the singleton StudyModeManager instance."""
    global _study_manager
    if _study_manager is None:
        # Load config from environment or defaults
        adapter_dir = os.getenv("LORA_ADAPTER_DIR") or _registry_lora_dir()

        # Build study config from environment
        study_config = {
            "max_training_time_seconds": int(os.getenv("MAX_TRAINING_TIME", "600")),
            "max_training_samples": int(os.getenv("MAX_TRAINING_SAMPLES", "1000")),
            "max_training_content_kb": int(os.getenv("MAX_TRAINING_CONTENT_KB", "200")),
            "use_real_training": os.getenv("USE_REAL_TRAINING", "true").lower() == "true",
            "base_model_path": os.getenv("BASE_MODEL_PATH") or _registry_base_model_path(),
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
    rank: int = Field(default=8, ge=1, le=64, description="LoRA rank")
    alpha: int = Field(default=16, ge=1, le=128, description="LoRA alpha")
    target_modules: Optional[List[str]] = Field(default=None, description="LoRA target modules (default: q_proj, v_proj)")
    max_steps: int = Field(default=100, ge=1, le=1000)
    num_train_epochs: Optional[int] = Field(default=None, ge=1, le=20, description="Epoch-based training (overrides max_steps)")
    target_loss: float = Field(default=0.05, ge=0.0, description="Stop when loss drops below this threshold")
    convergence_patience: int = Field(default=3, ge=1, description="Consecutive checks below target_loss before stopping")
    resume_from: Optional[str] = Field(default=None, description="Path to existing adapter for incremental training")
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

        Kills any training subprocess (which releases all VRAM on exit).
        The parent process never imports torch, so no CUDA cleanup needed.
        """
        import time

        logger.info("GPU release request received from orchestrator")

        manager = get_study_manager()
        status = manager.get_status()

        if status["state"] not in ("idle", "complete", "failed"):
            logger.info(f"Killing training subprocess (state: {status['state']})")
            manager.kill_training_subprocess()
            manager.cancel_training()
        elif status.get("subprocess_alive"):
            logger.info("Killing lingering training subprocess")
            manager.kill_training_subprocess()

        _gpu_available["available"] = False
        _gpu_available["released_at"] = time.strftime(
            "%Y-%m-%dT%H:%M:%SZ", time.gmtime()
        )

        return {
            "ok": True,
            "message": "GPU released, training subprocess killed",
            "gpu_available": False,
        }

    # ═══════════════════════════════════════════════════════════════════════════
    # Training Subprocess Monitoring (for orchestrator)
    # ═══════════════════════════════════════════════════════════════════════════

    @app.get("/study/training/status")
    async def training_subprocess_status():
        """
        Detailed training status including subprocess info and progress file.

        Used by the orchestrator to monitor training progress.
        """
        import json as _json
        from gaia_study.training_subprocess import PROGRESS_FILE

        manager = get_study_manager()
        manager_status = manager.get_status()

        # Read progress file directly
        progress_data = None
        try:
            if PROGRESS_FILE.exists():
                with open(PROGRESS_FILE) as f:
                    progress_data = _json.load(f)
        except Exception:
            pass

        return {
            "ok": True,
            "manager": manager_status,
            "progress_file": progress_data,
        }

    @app.post("/study/training/kill")
    async def training_subprocess_kill():
        """
        Force-kill the training subprocess.

        Last resort for the orchestrator when training is stuck.
        """
        manager = get_study_manager()
        killed = manager.kill_training_subprocess()

        if killed:
            manager.state = manager.state.__class__("failed")
            manager.status_message = "Training subprocess force-killed"
            manager.current_training = None
            logger.warning("Training subprocess force-killed via API")

        return {
            "ok": True,
            "killed": killed,
            "message": "Subprocess killed" if killed else "No subprocess running",
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
                rank=request.rank,
                alpha=request.alpha,
                target_modules=request.target_modules or ["q_proj", "v_proj"],
                max_steps=request.max_steps,
                num_train_epochs=request.num_train_epochs,
                target_loss=request.target_loss,
                convergence_patience=request.convergence_patience,
                resume_from=request.resume_from,
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

    async def _notify_core_adapter_change(adapter_name: str, action: str, tier: int = 3):
        """Notify gaia-core that an adapter has changed."""
        import httpx
        core_url = os.getenv("CORE_ENDPOINT", "http://gaia-core:6415")
        url = f"{core_url}/model/adapters/notify"
        try:
            async with httpx.AsyncClient() as client:
                payload = {
                    "adapter_name": adapter_name,
                    "action": action,
                    "tier": tier
                }
                resp = await client.post(url, json=payload, timeout=5.0)
                if resp.status_code == 200:
                    logger.info(f"Successfully notified core of adapter {action}: {adapter_name}")
                else:
                    logger.warning(f"Failed to notify core of adapter {action}: {resp.status_code} {resp.text}")
        except Exception as e:
            logger.error(f"Error notifying core of adapter change: {e}")

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
    
            # Notify core so the model pool can prepare
            await _notify_core_adapter_change(request.adapter_name, "load", request.tier)
            
            return {
                "ok": True,
                "adapter_name": request.adapter_name,
                "adapter_path": str(adapter_path),
                "tier": request.tier,
                "message": f"Adapter '{request.adapter_name}' loaded and core notified."
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
            # Notify core to unload from model pool
            await _notify_core_adapter_change(request.adapter_name, "unload", request.tier)
            
            return {
                "ok": True,
                "adapter_name": request.adapter_name,
                "message": f"Adapter '{request.adapter_name}' unload requested and core notified."
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
    
    # ── Self-Awareness Pipeline ───────────────────────────────────────────────

    _pipeline_proc = None  # Track running pipeline subprocess

    @app.post("/pipeline/run")
    async def pipeline_run(background_tasks: BackgroundTasks, options: Dict = {}):
        """Launch the self-awareness pipeline as a background subprocess.

        Options:
          dry_run (bool): Run in dry-run mode (no actual training)
          skip_nano (bool): Skip nano model stages
          skip_smoke (bool): Skip cognitive smoke test
          stage (str): Run only a specific stage (e.g. "COGNITIVE_SMOKE")
          resume (bool): Resume from last checkpoint
        """
        nonlocal _pipeline_proc
        import subprocess
        import sys

        if _pipeline_proc is not None and _pipeline_proc.poll() is None:
            return {"ok": False, "error": "Pipeline already running", "pid": _pipeline_proc.pid}

        cmd = [sys.executable, "scripts/self_awareness_pipeline.py"]

        if options.get("dry_run"):
            cmd.append("--dry-run")
        if options.get("skip_nano"):
            cmd.append("--skip-nano")
        if options.get("skip_smoke"):
            cmd.append("--skip-smoke")
        if options.get("resume"):
            cmd.append("--resume")
        if options.get("stage"):
            cmd.extend(["--stage", str(options["stage"])])

        logger.info("Launching pipeline: %s", " ".join(cmd))

        try:
            _pipeline_proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                cwd="/app",
            )
            return {
                "ok": True,
                "status": "started",
                "pid": _pipeline_proc.pid,
                "command": " ".join(cmd),
            }
        except Exception as e:
            logger.exception("Failed to start pipeline")
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/pipeline/status")
    async def pipeline_status():
        """Check if a pipeline subprocess is currently running."""
        nonlocal _pipeline_proc
        import json as _json
        from pathlib import Path

        running = _pipeline_proc is not None and _pipeline_proc.poll() is None
        state_file = Path("/shared/pipeline/self_awareness_state.json")
        state = {}
        if state_file.exists():
            try:
                state = _json.loads(state_file.read_text())
            except Exception:
                pass

        return {
            "running": running,
            "pid": _pipeline_proc.pid if running else None,
            "state": state,
        }

    return app
