"""
Model and Adapter Management Endpoints.
Allows external services (like gaia-study) to notify the core of new models/adapters.
"""

import logging
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from gaia_core.models.model_pool import get_model_pool

logger = logging.getLogger("GAIA.API.Models")
router = APIRouter(prefix="/model", tags=["models"])

class AdapterNotifyRequest(BaseModel):
    adapter_name: str
    action: str  # "load" or "unload"
    tier: int = 3
    model_name: str = "gpu_prime"

@router.post("/adapters/notify")
async def notify_adapter_change(request: AdapterNotifyRequest):
    """
    Notify the core that an adapter has been loaded or unloaded in the study service.
    Triggers a refresh of the model pool's adapter cache if supported.
    """
    logger.info(f"Received adapter notification: {request.action} {request.adapter_name} for {request.model_name}")

    try:
        model_pool = get_model_pool()
        model = model_pool.get_model(request.model_name)

        if not model:
            raise HTTPException(status_code=404, detail=f"Model '{request.model_name}' not found in pool")

        # If it's a vLLM model, it may need to refresh its internal LoRA list
        if hasattr(model, "refresh_adapters"):
            await model.refresh_adapters()

        return {
            "ok": True,
            "message": f"Core notified of adapter {request.action}: {request.adapter_name}"
        }
    except Exception as e:
        logger.exception(f"Failed to process adapter notification: {e}")
        raise HTTPException(status_code=500, detail=str(e))
