"""
Tier Router — automatic GPU handoff for cognitive tier inference.

When a request targets a tier (core/nano/prime), the router:
1. Checks if that tier's model is currently loaded on GPU
2. If not, unloads the current GPU holder (via managed engine /model/unload)
3. Loads the target tier's model (via managed engine /model/load)
4. Proxies the inference request to the loaded tier
5. Returns the response

This eliminates manual handoff orchestration — just ask the tier you want.
"""

import asyncio
import logging
import os
import time
from typing import Optional

import httpx

logger = logging.getLogger("GAIA.Orchestrator.TierRouter")

# Tier definitions: endpoint, model path, device preference
TIER_DEFAULTS = {
    "core": {
        "engine_endpoint": os.environ.get("CORE_INFERENCE_ENDPOINT", "http://gaia-core:8092"),
        "model_path": os.environ.get("CORE_MODEL_PATH", "/models/Qwen3.5-2B-GAIA-Core-v3"),
        "device": os.environ.get("CORE_DEVICE", "cuda"),
        "compile_mode": "reduce-overhead",
    },
    "nano": {
        "engine_endpoint": os.environ.get("NANO_INFERENCE_ENDPOINT", "http://gaia-nano:8080"),
        "model_path": os.environ.get("NANO_MODEL_PATH", "/models/Qwen3.5-0.8B-Abliterated-merged"),
        "device": os.environ.get("NANO_DEVICE", "cuda"),
        "compile_mode": "reduce-overhead",
    },
    "prime": {
        "engine_endpoint": os.environ.get("PRIME_INFERENCE_ENDPOINT", "http://gaia-prime:7777"),
        "model_path": os.environ.get("PRIME_MODEL_PATH", "/models/Huihui-Qwen3-8B-GAIA-Prime-adaptive"),
        "device": os.environ.get("PRIME_DEVICE", "cuda"),
        "compile_mode": "reduce-overhead",
    },
}


class TierRouter:
    """Automatic GPU handoff router for cognitive tiers."""

    def __init__(self, state_manager=None):
        self.state_manager = state_manager
        self._current_gpu_tier: Optional[str] = None
        self._lock = asyncio.Lock()
        self._tiers = {k: dict(v) for k, v in TIER_DEFAULTS.items()}

    async def ensure_tier(self, tier: str, device: str = "cuda") -> dict:
        """Ensure the requested tier's model is loaded on GPU.

        Handles unloading other tiers if needed. Returns status dict.
        """
        if tier not in self._tiers:
            return {"ok": False, "error": f"unknown tier: {tier}. Valid: {list(self._tiers.keys())}"}

        async with self._lock:
            target = self._tiers[tier]
            endpoint = target["engine_endpoint"]

            # Check if already loaded (by tracking OR by probing health)
            health = await self._check_health(endpoint)
            if health.get("model_loaded"):
                self._current_gpu_tier = tier
                logger.info("TIER: %s already loaded on GPU", tier)
                return {"ok": True, "tier": tier, "action": "already_loaded"}

            start = time.time()
            logger.info("TIER: ensuring %s on GPU (current: %s)", tier, self._current_gpu_tier)

            # Step 1: Unload current GPU holder(s)
            unloaded = []
            for other_tier, other_cfg in self._tiers.items():
                if other_tier == tier:
                    continue
                other_health = await self._check_health(other_cfg["engine_endpoint"])
                if other_health.get("model_loaded"):
                    logger.info("TIER: unloading %s to free GPU", other_tier)
                    result = await self._unload_tier(other_cfg["engine_endpoint"])
                    unloaded.append(other_tier)
                    if not result.get("ok"):
                        logger.warning("TIER: %s unload issue: %s", other_tier, result)

            # Brief pause for CUDA contexts to clean up
            if unloaded:
                await asyncio.sleep(1)

            # Step 2: Load the target tier
            logger.info("TIER: loading %s model=%s device=%s", tier, target["model_path"], device)
            load_result = await self._load_tier(
                endpoint, target["model_path"], device, target.get("compile_mode", "reduce-overhead"))

            elapsed = time.time() - start
            if load_result.get("ok") or load_result.get("model_loaded"):
                self._current_gpu_tier = tier
                logger.info("TIER: %s loaded in %.1fs (unloaded: %s)", tier, elapsed, unloaded)
                return {
                    "ok": True,
                    "tier": tier,
                    "action": "loaded",
                    "unloaded": unloaded,
                    "elapsed_s": round(elapsed, 1),
                }
            else:
                logger.error("TIER: %s load failed: %s", tier, load_result)
                return {"ok": False, "tier": tier, "error": load_result.get("error", "load failed")}

    async def infer(self, tier: str, messages: list, max_tokens: int = 512,
                    temperature: float = 0.7, top_p: float = 0.9,
                    device: str = "cuda") -> dict:
        """Ensure tier is loaded, then send inference request.

        This is the "just ask" interface — specify the tier, get the answer.
        """
        # Ensure the tier is loaded
        ensure_result = await self.ensure_tier(tier, device)
        if not ensure_result.get("ok"):
            return {
                "error": f"failed to load tier {tier}: {ensure_result.get('error')}",
                "handoff": ensure_result,
            }

        # Send inference request
        endpoint = self._tiers[tier]["engine_endpoint"]
        try:
            async with httpx.AsyncClient(timeout=120) as client:
                resp = await client.post(
                    f"{endpoint}/v1/chat/completions",
                    json={
                        "messages": messages,
                        "max_tokens": max_tokens,
                        "temperature": temperature,
                        "top_p": top_p,
                    },
                )
                if resp.status_code == 200:
                    result = resp.json()
                    result["_tier"] = tier
                    result["_handoff"] = ensure_result
                    return result
                else:
                    return {
                        "error": f"inference HTTP {resp.status_code}: {resp.text[:500]}",
                        "_tier": tier,
                        "_handoff": ensure_result,
                    }
        except Exception as e:
            return {"error": f"inference failed: {e}", "_tier": tier, "_handoff": ensure_result}

    async def sae_record(self, tier: str, prompts: list = None, tag: str = "handoff_test") -> dict:
        """Trigger SAE atlas recording on a loaded tier."""
        endpoint = self._tiers[tier]["engine_endpoint"]

        health = await self._check_health(endpoint)
        if not health.get("model_loaded"):
            return {"ok": False, "error": f"tier {tier} not loaded"}

        try:
            async with httpx.AsyncClient(timeout=30) as client:
                payload = {"tier": tier, "tag": tag}
                if prompts:
                    payload["prompts"] = prompts
                resp = await client.post(f"{endpoint}/atlas/record", json=payload)
                if resp.status_code == 200:
                    return resp.json()
                return {"ok": False, "error": f"SAE HTTP {resp.status_code}"}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    async def get_loaded_tiers(self) -> dict:
        """Check which tiers currently have models loaded."""
        results = {}
        for tier, cfg in self._tiers.items():
            health = await self._check_health(cfg["engine_endpoint"])
            results[tier] = {
                "model_loaded": health.get("model_loaded", False),
                "mode": health.get("mode", "unreachable"),
                "managed": health.get("managed", False),
                "endpoint": cfg["engine_endpoint"],
            }
        return results

    async def unload_all(self) -> dict:
        """Unload all tiers — zero GPU."""
        results = {}
        for tier, cfg in self._tiers.items():
            results[tier] = await self._unload_tier(cfg["engine_endpoint"])
        self._current_gpu_tier = None
        return results

    # ── Internal helpers ──────────────────────────────────────────────────────

    async def _check_health(self, endpoint: str) -> dict:
        try:
            async with httpx.AsyncClient(timeout=5) as client:
                resp = await client.get(f"{endpoint}/health")
                if resp.status_code == 200:
                    return resp.json()
        except Exception:
            pass
        return {"model_loaded": False, "mode": "unreachable"}

    async def _load_tier(self, endpoint: str, model_path: str,
                         device: str, compile_mode: str) -> dict:
        try:
            async with httpx.AsyncClient(timeout=300) as client:
                resp = await client.post(
                    f"{endpoint}/model/load",
                    json={
                        "model": model_path,
                        "device": device,
                        "compile_mode": compile_mode,
                    },
                )
                if resp.status_code in (200, 409):
                    return resp.json()
                return {"ok": False, "error": f"HTTP {resp.status_code}: {resp.text[:200]}"}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    async def _unload_tier(self, endpoint: str) -> dict:
        try:
            async with httpx.AsyncClient(timeout=30) as client:
                resp = await client.post(f"{endpoint}/model/unload")
                if resp.status_code == 200:
                    return resp.json()
                return {"ok": False, "error": f"HTTP {resp.status_code}"}
        except Exception as e:
            return {"ok": False, "error": str(e)}
