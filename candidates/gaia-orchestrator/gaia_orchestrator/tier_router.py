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


def _build_tier_defaults() -> dict:
    """Build tier defaults from gaia_constants.json with env var overrides."""
    try:
        from gaia_common.config import Config
        cfg = Config.get_instance()
        return {
            "core": {
                "engine_endpoint": os.environ.get("CORE_INFERENCE_ENDPOINT", cfg.get_inference_endpoint("core") or "http://gaia-core:8092"),
                "model_path": os.environ.get("CORE_MODEL_PATH", cfg.model_path("core", "merged") or "/models/core"),
                "device": os.environ.get("CORE_DEVICE", "cuda"),
                "compile_mode": "reduce-overhead",
            },
            "nano": {
                "engine_endpoint": os.environ.get("NANO_INFERENCE_ENDPOINT", cfg.get_inference_endpoint("nano") or "http://gaia-nano:8080"),
                "model_path": os.environ.get("NANO_MODEL_PATH", cfg.model_path("nano", "merged") or "/models/nano"),
                "device": os.environ.get("NANO_DEVICE", "cuda"),
                "compile_mode": "reduce-overhead",
            },
            "prime": {
                "engine_endpoint": os.environ.get("PRIME_INFERENCE_ENDPOINT", cfg.get_inference_endpoint("prime") or "http://gaia-prime:7777"),
                "model_path": os.environ.get("PRIME_MODEL_PATH", cfg.model_path("prime", "merged") or "/models/prime"),
                "device": os.environ.get("PRIME_DEVICE", "cuda"),
                "compile_mode": "reduce-overhead",
            },
        }
    except Exception:
        # Fallback if gaia_common not available
        return {
            "core": {
                "engine_endpoint": os.environ.get("CORE_INFERENCE_ENDPOINT", "http://gaia-core:8092"),
                "model_path": os.environ.get("CORE_MODEL_PATH", "/models/core"),
                "device": os.environ.get("CORE_DEVICE", "cuda"),
                "compile_mode": "reduce-overhead",
            },
            "nano": {
                "engine_endpoint": os.environ.get("NANO_INFERENCE_ENDPOINT", "http://gaia-nano:8080"),
                "model_path": os.environ.get("NANO_MODEL_PATH", "/models/nano"),
                "device": os.environ.get("NANO_DEVICE", "cuda"),
                "compile_mode": "reduce-overhead",
            },
            "prime": {
                "engine_endpoint": os.environ.get("PRIME_INFERENCE_ENDPOINT", "http://gaia-prime:7777"),
                "model_path": os.environ.get("PRIME_MODEL_PATH", "/models/prime"),
                "device": os.environ.get("PRIME_DEVICE", "cuda"),
                "compile_mode": "reduce-overhead",
            },
        }


# Lazy-initialized on first use
TIER_DEFAULTS = None


class TierRouter:
    """Automatic GPU handoff router for cognitive tiers.

    When a lifecycle machine is available, delegates tier transitions to it.
    Otherwise falls back to direct load/unload operations.
    """

    def __init__(self, state_manager=None, lifecycle_machine=None):
        self.state_manager = state_manager
        self.lifecycle_machine = lifecycle_machine
        self._current_gpu_tier: Optional[str] = None
        self._lock = asyncio.Lock()
        global TIER_DEFAULTS
        if TIER_DEFAULTS is None:
            TIER_DEFAULTS = _build_tier_defaults()
        self._tiers = {k: dict(v) for k, v in TIER_DEFAULTS.items()}

    async def ensure_tier(self, tier: str, device: str = "cuda") -> dict:
        """Ensure the requested tier's model is loaded on GPU.

        When lifecycle machine is available, delegates to it for proper state
        transitions. Otherwise falls back to direct load/unload operations.
        """
        if tier not in self._tiers:
            return {"ok": False, "error": f"unknown tier: {tier}. Valid: {list(self._tiers.keys())}"}

        # Use lifecycle machine when available
        if self.lifecycle_machine is not None:
            return await self._ensure_tier_via_lifecycle(tier)

        # Legacy fallback: direct load/unload
        return await self._ensure_tier_direct(tier, device)

    async def _ensure_tier_via_lifecycle(self, tier: str) -> dict:
        """Ensure tier via lifecycle state machine transitions."""
        from gaia_common.lifecycle.states import TransitionTrigger, LifecycleState

        # Map tier to lifecycle state
        tier_to_state = {
            "core": LifecycleState.AWAKE,
            "nano": LifecycleState.AWAKE,
            "prime": LifecycleState.FOCUSING,
        }
        target_state = tier_to_state.get(tier, LifecycleState.AWAKE)

        # Check current lifecycle state
        snapshot = await self.lifecycle_machine.get_snapshot()
        current = LifecycleState(snapshot.state)

        # If we're already in the right state, check if the tier is loaded
        if current == target_state:
            tier_status = snapshot.tiers.get(tier)
            if tier_status and tier_status.model_loaded:
                return {"ok": True, "tier": tier, "action": "already_loaded"}

        # Request transition
        if tier == "prime":
            trigger = TransitionTrigger.ESCALATION_NEEDED
        else:
            trigger = TransitionTrigger.TASK_COMPLETE  # Back to AWAKE

        # If current state matches target, use USER_REQUEST
        if current == target_state:
            trigger = TransitionTrigger.USER_REQUEST

        result = await self.lifecycle_machine.transition(
            trigger, target=target_state, reason=f"ensure_tier:{tier}")

        if result.ok:
            return {
                "ok": True,
                "tier": tier,
                "action": "loaded",
                "unloaded": [a.split(":")[0] for a in result.actions if "unload" in a],
                "elapsed_s": result.elapsed_s,
            }
        return {"ok": False, "tier": tier, "error": result.error}

    async def _ensure_tier_direct(self, tier: str, device: str) -> dict:
        """Legacy: ensure tier via direct load/unload (no lifecycle machine)."""
        async with self._lock:
            target = self._tiers[tier]
            endpoint = target["engine_endpoint"]

            # Check if already loaded
            health = await self._check_health(endpoint)
            if health.get("model_loaded"):
                self._current_gpu_tier = tier
                logger.info("TIER: %s already loaded on GPU", tier)
                return {"ok": True, "tier": tier, "action": "already_loaded"}

            start = time.time()
            logger.info("TIER: ensuring %s on GPU (current: %s)", tier, self._current_gpu_tier)

            # Unload current GPU holders
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

            if unloaded:
                await asyncio.sleep(1)

            # Load the target tier
            logger.info("TIER: loading %s model=%s device=%s", tier, target["model_path"], device)
            load_result = await self._load_tier(
                endpoint, target["model_path"], device, target.get("compile_mode", "reduce-overhead"))

            elapsed = time.time() - start
            if load_result.get("ok") or load_result.get("model_loaded"):
                self._current_gpu_tier = tier
                logger.info("TIER: %s loaded in %.1fs (unloaded: %s)", tier, elapsed, unloaded)
                return {
                    "ok": True, "tier": tier, "action": "loaded",
                    "unloaded": unloaded, "elapsed_s": round(elapsed, 1),
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
                # Fallback: try /slots/idle (llama-server GPU release) then /shutdown
                if resp.status_code == 404:
                    logger.info("TIER: /model/unload not supported, trying llama-server /slots/idle")
                    try:
                        # Free GPU memory by clearing all slots
                        resp2 = await client.post(f"{endpoint}/slots/idle")
                        if resp2.status_code == 200:
                            return {"ok": True, "method": "slots_idle"}
                    except Exception:
                        pass
                return {"ok": False, "error": f"HTTP {resp.status_code}"}
        except Exception as e:
            return {"ok": False, "error": str(e)}
