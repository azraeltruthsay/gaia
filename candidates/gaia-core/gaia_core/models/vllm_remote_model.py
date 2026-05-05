"""
Remote vLLM model backend for GAIA.

HTTP client for a standalone vLLM OpenAI-compatible API server (gaia-prime).
Replaces in-process VLLMChatModel when PRIME_ENDPOINT is set, allowing
gaia-core to offload GPU inference to a separate container.

Environment:
    PRIME_ENDPOINT: Base URL of the vLLM server (e.g. http://gaia-prime-candidate:7777)
    PRIME_MODEL:    Model name registered in the vLLM server (default: /models/Qwen3.5-4B-Abliterated-merged)
"""

import logging
import os
import re
import time
from typing import Any, Dict, Generator, List, Optional

import requests

logger = logging.getLogger("GAIA.VLLMRemote")


class VLLMRemoteModel:
    """
    HTTP client for a remote vLLM OpenAI-compatible API server.

    Provides the same public interface as VLLMChatModel (create_completion,
    create_chat_completion, shutdown) so it can be used as a drop-in
    replacement in the model pool.
    """

    def __init__(self, model_config: dict, global_config=None, **kwargs):
        # Resolve endpoint: config dict → env var → default
        self.endpoint = (
            model_config.get("endpoint")
            or os.getenv("PRIME_ENDPOINT")
            or "http://gaia-prime-candidate:7777"
        )
        self.endpoint = self.endpoint.rstrip("/")

        # Model name as registered by vLLM (appears in /v1/models)
        self.model_name = (
            model_config.get("path")
            or model_config.get("model")
            or os.getenv("PRIME_MODEL")
            or self._registry_prime_path()
        )

        self.timeout = int(model_config.get("timeout", 300))

        # Context window size for auto-clamping max_tokens
        self.max_model_len = int(
            model_config.get("max_model_len")
            or model_config.get("context_length")
            or os.getenv("VLLM_MAX_MODEL_LEN", "8192")
        )

        # LoRA support
        self._lora_config = model_config.get("lora_config") or {}
        self._active_adapter: Optional[str] = None
        # Adapters we've POSTed to /adapter/load on the gaia-managed engine.
        # Skips redundant load calls on subsequent activations of the same
        # adapter within a process lifetime.
        self._loaded_adapters_remote: set = set()
        # Engine kind — probed lazily on first adapter operation. 'gaia' means
        # the managed engine (requires /adapter/load + /adapter/set); 'vllm'
        # means real vLLM (LoRA name in payload model field is sufficient).
        self._engine_kind: Optional[str] = None

        # Stats
        self._request_count = 0
        self._total_tokens = 0

        # Session for connection pooling
        self._session = requests.Session()

        logger.info(
            "VLLMRemoteModel initialised: endpoint=%s model=%s",
            self.endpoint,
            self.model_name,
        )

    @staticmethod
    def _registry_prime_path() -> str:
        """Resolve prime model path from Config singleton MODEL_REGISTRY."""
        try:
            from gaia_common.config import Config
            return Config.get_instance().model_path("prime", "merged") or "/models/Qwen3.5-4B-Abliterated-merged"
        except Exception:
            return "/models/Qwen3.5-4B-Abliterated-merged"

    # ── Token clamping ────────────────────────────────────────────────────────

    def _estimate_prompt_tokens(self, text_or_messages) -> int:
        """Rough token estimate (~3.5 chars/token for English, use 3 to overestimate).

        Multimodal content (list of {type,text|image|image_url} parts) is
        accounted for: text parts via length, images at 256 soft tokens each
        (Gemma 4's per-image expansion).
        """
        if isinstance(text_or_messages, str):
            return len(text_or_messages) // 3 + 4
        total = 0
        for m in text_or_messages:
            content = m.get("content", "")
            if isinstance(content, str):
                total += len(content) // 3
            elif isinstance(content, list):
                for part in content:
                    if not isinstance(part, dict):
                        continue
                    ptype = part.get("type", "")
                    if ptype == "text":
                        total += len(part.get("text", "")) // 3
                    elif ptype in ("image", "image_url"):
                        total += 256
            total += 4  # per-message role overhead
        return total

    def _truncate_messages_to_fit(self, messages: list, reserved_output: int = 256) -> list:
        """Truncate message content to fit within the model's context window.

        Strategy: preserve the system message and last user message intact.
        Trim middle messages (oldest first), then truncate the system message
        content as a last resort. This prevents 400 errors from context overflow.
        """
        budget = self.max_model_len - reserved_output
        est = self._estimate_prompt_tokens(messages)
        if est <= budget:
            return messages

        logger.warning(
            "Prompt (%d est. tokens) exceeds context budget (%d) — truncating messages",
            est, budget,
        )
        # Work on copies
        messages = [dict(m) for m in messages]

        # Phase 1: Drop middle messages (keep system[0] and last user message)
        while len(messages) > 2 and self._estimate_prompt_tokens(messages) > budget:
            # Remove the second message (oldest non-system)
            messages.pop(1)

        # Phase 2: Truncate system message content if still over budget
        if self._estimate_prompt_tokens(messages) > budget and messages and messages[0].get("role") == "system":
            content = messages[0].get("content", "")
            # Estimate how many chars to keep (3 chars ≈ 1 token)
            target_chars = max(200, (budget - reserved_output) * 3)
            if len(content) > target_chars:
                messages[0]["content"] = content[:target_chars] + "\n[...truncated to fit context window]"
                logger.warning("Truncated system message from %d to %d chars", len(content), target_chars)

        return messages

    def _clamp_max_tokens(self, max_tokens: int, estimated_prompt: int) -> int:
        """Clamp max_tokens to fit within context window. Returns clamped value."""
        available = self.max_model_len - estimated_prompt
        if available <= 0:
            logger.warning(
                "Estimated prompt (%d tokens) fills context window (%d) — forcing max_tokens=1",
                estimated_prompt, self.max_model_len,
            )
            return 1
        if max_tokens > available:
            clamped = max(1, available - 32)  # small safety margin
            logger.warning(
                "Clamping max_tokens %d → %d (est. prompt: %d, context: %d)",
                max_tokens, clamped, estimated_prompt, self.max_model_len,
            )
            return clamped
        return max_tokens

    # ── Public interface (matches VLLMChatModel) ─────────────────────────────

    def create_completion(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.95,
        stop: Optional[List[str]] = None,
        stream: bool = False,
        **kwargs,
    ) -> Dict[str, Any] | Generator[Dict[str, Any], None, None]:
        """Text completion via POST /v1/completions."""
        self._request_count += 1
        max_tokens = self._clamp_max_tokens(max_tokens, self._estimate_prompt_tokens(prompt))
        payload = {
            "model": self._resolve_model_field(),
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
        }
        if stop:
            payload["stop"] = stop

        if stream:
            return self._stream_completions(payload)

        start = time.time()
        resp = self._post("/v1/completions", payload)
        duration = time.time() - start

        text = resp["choices"][0].get("text", "")
        self._log_usage(resp, duration)

        return {
            "choices": [{"text": text}],
        }

    # vLLM-specific keys that are forwarded from kwargs into the API payload.
    _VLLM_EXTRA_KEYS = frozenset({
        "guided_json", "guided_regex", "guided_choice",
        "guided_grammar", "guided_decoding_backend",
        "response_format", "stop", "chat_template_kwargs",
        "skip_prefix",  # GAIA Engine: skip KV prefix/awareness/clock injection
    })

    def create_chat_completion(
        self,
        messages: List[Dict[str, Any]],
        max_tokens: int = 1024,
        temperature: float = 0.7,
        top_p: float = 0.95,
        stream: bool = False,
        **kwargs,
    ) -> Dict[str, Any] | Generator[Dict[str, Any], None, None]:
        """Chat completion via POST /v1/chat/completions.

        Extra kwargs matching ``_VLLM_EXTRA_KEYS`` are forwarded into the
        request payload, enabling vLLM features like guided decoding::

            model.create_chat_completion(
                messages=msgs,
                guided_json={"type": "object", ...},
            )
        """
        self._request_count += 1
        clean_messages = self._sanitize_messages(messages)
        clean_messages = self._truncate_messages_to_fit(clean_messages, max_tokens)
        max_tokens = self._clamp_max_tokens(max_tokens, self._estimate_prompt_tokens(clean_messages))

        payload = {
            "model": self._resolve_model_field(),
            "messages": clean_messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
        }

        # Forward vLLM-specific params (guided decoding, stop, etc.)
        for key in self._VLLM_EXTRA_KEYS:
            if key in kwargs:
                payload[key] = kwargs[key]

        # Thinking mode: Qwen3 needs explicit disable to avoid <think> bloat.
        # Gemma 4 does NOT support think suppression — leave thinking enabled.
        # Callers can override by passing chat_template_kwargs explicitly.
        if "chat_template_kwargs" not in payload:
            payload["chat_template_kwargs"] = {"enable_thinking": True}

        if stream:
            return self._stream_chat(payload)

        start = time.time()
        resp = self._post("/v1/chat/completions", payload)
        duration = time.time() - start

        content = resp["choices"][0]["message"]["content"]
        self._log_usage(resp, duration)

        return {
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": content,
                },
                "finish_reason": resp["choices"][0].get("finish_reason", "stop"),
            }],
            "model": self.model_name,
            "provider": "vllm_remote",
        }

    # ── LoRA adapter support ─────────────────────────────────────────────────

    # Standard tier locations under /models/lora_adapters/. Probed in order
    # when resolving an adapter name to a filesystem path.
    _ADAPTER_TIER_ROOTS = (
        "/models/lora_adapters/tier1_global",
        "/models/lora_adapters/tier2_personal",
        "/models/lora_adapters/tier3_session",
        "/models/lora_adapters",
    )

    def _detect_engine_kind(self) -> str:
        """Probe /health to determine whether we're talking to the gaia-managed
        engine or a real vLLM server. Cached after first successful probe.
        """
        if self._engine_kind is not None:
            return self._engine_kind
        try:
            r = self._session.get(f"{self.endpoint}/health", timeout=5)
            if r.status_code == 200:
                data = r.json() if r.content else {}
                # gaia-managed engine identifies itself with engine: "gaia" or
                # engine: "gaia-managed" in /health.
                eng = (data.get("engine") or "").lower()
                if eng.startswith("gaia"):
                    self._engine_kind = "gaia"
                    return "gaia"
        except Exception as e:
            logger.debug("Engine kind probe failed (%s); assuming vllm", e)
        self._engine_kind = "vllm"
        return "vllm"

    def _resolve_adapter_path(self, adapter_name: str) -> Optional[str]:
        """Find an adapter on disk by name, probing standard tier dirs."""
        for root in self._ADAPTER_TIER_ROOTS:
            candidate = os.path.join(root, adapter_name)
            if os.path.isdir(candidate):
                return candidate
        return None

    def set_active_adapter(self, adapter_name: Optional[str]):
        """Select a LoRA adapter by name. Pass None to use the base model.

        For gaia-managed engine: ensures the adapter is loaded server-side
        (POST /adapter/load) and activates it (POST /adapter/set). Loads are
        cached so repeated activations of the same adapter only POST /set.

        For real vLLM: stores the name; create_chat_completion uses it as the
        model field in the payload (vLLM's normal LoRA selection mechanism).
        """
        kind = self._detect_engine_kind()

        # Real vLLM — original behavior, name in model field
        if kind == "vllm":
            self._active_adapter = adapter_name
            logger.info("VLLMRemoteModel(vllm): active adapter set to %s", adapter_name)
            return

        # gaia-managed engine — call /adapter/load + /adapter/set
        if adapter_name is None:
            # Deactivate
            try:
                self._session.post(f"{self.endpoint}/adapter/set",
                                   json={"name": None}, timeout=10)
            except Exception:
                logger.debug("adapter/set None failed", exc_info=True)
            self._active_adapter = None
            logger.info("VLLMRemoteModel(gaia): adapter cleared")
            return

        # Ensure loaded server-side (idempotent on our side via the cache)
        if adapter_name not in self._loaded_adapters_remote:
            adapter_path = self._resolve_adapter_path(adapter_name)
            if adapter_path is None:
                logger.warning(
                    "VLLMRemoteModel(gaia): adapter '%s' not found under tier roots; "
                    "skipping load. Generation will use base model.",
                    adapter_name,
                )
                self._active_adapter = None
                return
            try:
                r = self._session.post(
                    f"{self.endpoint}/adapter/load",
                    json={"name": adapter_name, "path": adapter_path},
                    timeout=120,  # PEFT load can take a while
                )
                resp = r.json() if r.content else {}
                if r.status_code == 200 and resp.get("ok"):
                    self._loaded_adapters_remote.add(adapter_name)
                    logger.info(
                        "VLLMRemoteModel(gaia): adapter '%s' loaded from %s (vram=%s)",
                        adapter_name, adapter_path, resp.get("vram_mb"),
                    )
                else:
                    logger.warning(
                        "VLLMRemoteModel(gaia): /adapter/load failed for '%s': %s",
                        adapter_name, resp,
                    )
                    # Don't set active — fall through with no adapter
                    self._active_adapter = None
                    return
            except Exception as e:
                logger.warning("VLLMRemoteModel(gaia): /adapter/load error: %s", e)
                self._active_adapter = None
                return

        # Activate
        try:
            r = self._session.post(
                f"{self.endpoint}/adapter/set",
                json={"name": adapter_name},
                timeout=10,
            )
            if r.status_code == 200:
                self._active_adapter = adapter_name
                logger.info("VLLMRemoteModel(gaia): active adapter set to %s", adapter_name)
            else:
                logger.warning("VLLMRemoteModel(gaia): /adapter/set HTTP %d", r.status_code)
                self._active_adapter = None
        except Exception as e:
            logger.warning("VLLMRemoteModel(gaia): /adapter/set error: %s", e)
            self._active_adapter = None

    def create_chat_completion_with_adapter(
        self,
        adapter_name: str,
        messages: List[Dict[str, Any]],
        **kwargs,
    ) -> Dict[str, Any]:
        """One-shot chat completion using a specific LoRA adapter."""
        prev = self._active_adapter
        try:
            self.set_active_adapter(adapter_name)
            return self.create_chat_completion(messages=messages, **kwargs)
        finally:
            # Restore previous adapter state. For gaia-managed engine this
            # means re-issuing /adapter/set; for vLLM it's just a name swap.
            if self._engine_kind == "gaia":
                # Only re-set if previous was different to avoid an extra HTTP call
                if prev != adapter_name:
                    self.set_active_adapter(prev)
            else:
                self._active_adapter = prev

    # ── KV Cache Slot API ──────────────────────────────────────────────────

    def save_kv_cache(self, filename: str, slot_id: int = 0) -> bool:
        """Save KV cache state via llama-server slot API. Non-fatal on failure."""
        try:
            url = f"{self.endpoint}/slots/{slot_id}?action=save"
            r = self._session.post(url, json={"filename": filename}, timeout=30)
            if r.status_code == 200:
                logger.info("KV cache saved: slot=%d filename=%s", slot_id, filename)
                return True
            logger.warning("KV cache save failed (HTTP %d): %s", r.status_code, r.text[:200])
            return False
        except Exception as exc:
            logger.warning("KV cache save error: %s", exc)
            return False

    def restore_kv_cache(self, filename: str, slot_id: int = 0) -> bool:
        """Restore KV cache state via llama-server slot API. Non-fatal on failure."""
        try:
            url = f"{self.endpoint}/slots/{slot_id}?action=restore"
            r = self._session.post(url, json={"filename": filename}, timeout=30)
            if r.status_code == 200:
                logger.info("KV cache restored: slot=%d filename=%s", slot_id, filename)
                return True
            logger.warning("KV cache restore failed (HTTP %d): %s", r.status_code, r.text[:200])
            return False
        except Exception as exc:
            logger.warning("KV cache restore error: %s", exc)
            return False

    @property
    def supports_kv_cache(self) -> bool:
        """Check if the remote server supports the slot save/restore API."""
        try:
            r = self._session.get(f"{self.endpoint}/slots", timeout=10)
            return r.status_code == 200
        except Exception:
            return False

    # ── KV Cache Pressure API ─────────────────────────────────────────────

    def get_slot_info(self, slot_id: int = 0) -> Optional[Dict[str, Any]]:
        """Query llama-server /slots endpoint for a specific slot's state.

        Returns the slot dict with keys like n_ctx, n_past, or None on failure.
        """
        try:
            r = self._session.get(f"{self.endpoint}/slots", timeout=10)
            if r.status_code != 200:
                return None
            slots = r.json()
            if isinstance(slots, list):
                for slot in slots:
                    if slot.get("id") == slot_id:
                        return slot
                # If slot_id not found but slots exist, return first
                return slots[0] if slots else None
            return slots if isinstance(slots, dict) else None
        except Exception as exc:
            logger.debug("get_slot_info failed: %s", exc)
            return None

    def erase_slot(self, slot_id: int = 0) -> bool:
        """Erase (clear) a KV cache slot via POST /slots/{id}?action=erase."""
        try:
            url = f"{self.endpoint}/slots/{slot_id}?action=erase"
            r = self._session.post(url, json={}, timeout=30)
            if r.status_code == 200:
                logger.info("KV cache slot %d erased", slot_id)
                return True
            logger.warning("KV cache erase failed (HTTP %d): %s", r.status_code, r.text[:200])
            return False
        except Exception as exc:
            logger.warning("KV cache erase error: %s", exc)
            return False

    def get_cache_pressure(self, slot_id: int = 0) -> float:
        """Return KV cache pressure as n_past / n_ctx ratio (0.0-1.0).

        Returns -1.0 if slot info is unavailable.
        """
        slot = self.get_slot_info(slot_id)
        if slot is None:
            return -1.0
        n_ctx = slot.get("n_ctx", 0)
        n_past = slot.get("n_past", 0)
        if n_ctx <= 0:
            return -1.0
        return min(n_past / n_ctx, 1.0)

    # ── Health / lifecycle ───────────────────────────────────────────────────

    def health_check(self) -> bool:
        """Return True if the remote vLLM server is reachable."""
        try:
            r = self._session.get(
                f"{self.endpoint}/health", timeout=10
            )
            return r.status_code == 200
        except Exception as exc:
            logger.warning("VLLMRemoteModel health_check failed: %s", exc)
            return False

    def shutdown(self):
        """No-op — server lifecycle is managed externally (Docker)."""
        try:
            self._session.close()
        except Exception as _close_exc:
            logger.debug("VLLMRemoteModel: session close failed: %s", _close_exc)

    # ── Self-Healing ──────────────────────────────────────────────────────────

    def _request_prime_load(self) -> None:
        """Ask orchestrator to wake Prime when it's unreachable.

        This is GAIA healing herself — Core detects Prime is down and
        presses the orchestrator button to fix it. Glass stays on.
        """
        import os
        orchestrator = os.environ.get("ORCHESTRATOR_ENDPOINT", "http://gaia-orchestrator:6410")
        try:
            resp = requests.post(
                f"{orchestrator}/gpu/wake",
                json={"reason": "core_self_healing:prime_unreachable"},
                timeout=10,
            )
            if resp.status_code == 200:
                logger.info("Self-healing: requested orchestrator to wake Prime")
            else:
                logger.debug("Self-healing: orchestrator returned %d", resp.status_code)
        except Exception as e:
            logger.debug("Self-healing: could not reach orchestrator: %s", e)

    # ── Internals ────────────────────────────────────────────────────────────

    def _resolve_model_field(self) -> str:
        """Return the model field for the request.

        When a LoRA adapter is active, vLLM expects the adapter name in the
        ``model`` field of the request payload.
        """
        if self._active_adapter:
            return self._active_adapter
        return self.model_name

    # Retry configuration for transient failures
    _MAX_RETRIES = 3
    _RETRY_BASE_DELAY = 1.5  # seconds; delays: 1.5s, 3.0s

    def _post(self, path: str, payload: dict, _allow_clamp_retry: bool = True) -> dict:
        url = f"{self.endpoint}{path}"
        import json as _json
        _payload_bytes = len(_json.dumps(payload))
        _n_msgs = len(payload.get("messages", []))
        logger.warning("vLLM POST %s model=%s payload=%dB msgs=%d max_tokens=%s",
                       url, payload.get("model", "?"), _payload_bytes, _n_msgs,
                       payload.get("max_tokens", "?"))
        last_exc: Optional[Exception] = None

        for attempt in range(1, self._MAX_RETRIES + 1):
            try:
                r = self._session.post(url, json=payload, timeout=self.timeout)
                # Retry on 503 (vLLM model-loading state), not other errors
                if r.status_code == 503 and attempt < self._MAX_RETRIES:
                    # On first 503, request model load from orchestrator
                    if attempt == 1:
                        try:
                            self._request_prime_load()
                        except Exception as _wake_exc:
                            logger.warning("VLLMRemoteModel: prime wake request failed: %s", _wake_exc)
                    # Wait longer after wake request — model takes ~30s to load
                    delay = max(self._RETRY_BASE_DELAY * attempt, 10.0 if attempt == 1 else 5.0)
                    logger.warning(
                        "vLLM returned 503 on attempt %d/%d, retrying in %.1fs (wake requested)...",
                        attempt, self._MAX_RETRIES, delay,
                    )
                    time.sleep(delay)
                    continue
                r.raise_for_status()
                return r.json()
            except requests.exceptions.ConnectionError as exc:
                last_exc = exc
                if attempt < self._MAX_RETRIES:
                    delay = self._RETRY_BASE_DELAY * attempt
                    logger.warning(
                        "vLLM connection failed on attempt %d/%d, retrying in %.1fs...",
                        attempt, self._MAX_RETRIES, delay,
                    )
                    time.sleep(delay)
                    continue
                # Self-healing: ask orchestrator to load Prime before giving up
                try:
                    self._request_prime_load()
                except Exception as _wake_exc:
                    logger.warning("VLLMRemoteModel: prime wake request failed on final attempt: %s", _wake_exc)
                raise RuntimeError(
                    f"Cannot reach vLLM server at {url} after {self._MAX_RETRIES} attempts. "
                    f"Is gaia-prime running?"
                ) from exc
            except requests.exceptions.HTTPError as exc:
                error_text = (r.text or "")[:500]

                # ── LoRA adapter not loaded — graceful fallback to base model ──
                # vLLM returns 400 when the adapter is rejected, or 404 when
                # the adapter model name doesn't exist at all.
                if r.status_code in (400, 404) and self._active_adapter and (
                    "lora" in error_text.lower()
                    or "adapter" in error_text.lower()
                    or self._active_adapter in error_text
                    or "does not exist" in error_text.lower()
                ):
                    logger.warning(
                        "vLLM rejected adapter '%s' (%d: %s). "
                        "Falling back to base model.",
                        self._active_adapter, r.status_code, error_text[:200],
                    )
                    self._active_adapter = None
                    payload["model"] = self.model_name
                    return self._post(path, payload, _allow_clamp_retry=_allow_clamp_retry)

                if r.status_code == 400:

                    # Smart retry for context window overflow: parse exact available tokens
                    if _allow_clamp_retry:
                        m = re.search(r"max_tokens\s+\d+\s*>\s*(\d+)\s*-\s*(\d+)", error_text)
                        if m:
                            available = int(m.group(1)) - int(m.group(2))
                            if available > 0:
                                clamped = max(1, available - 16)
                                logger.warning(
                                    "vLLM context overflow — retrying with max_tokens=%d (was %d)",
                                    clamped, payload.get("max_tokens", 0),
                                )
                                payload["max_tokens"] = clamped
                                return self._post(path, payload, _allow_clamp_retry=False)
                logger.error("vLLM HTTP error %s: %s", r.status_code, r.text[:500])
                raise RuntimeError(f"vLLM request failed ({r.status_code})") from exc

        # Should not reach here, but safety net
        raise RuntimeError(
            f"vLLM request to {url} failed after {self._MAX_RETRIES} attempts"
        ) from last_exc

    def _stream_completions(self, payload: dict) -> Generator[Dict[str, Any], None, None]:
        payload["stream"] = True
        url = f"{self.endpoint}/v1/completions"
        logger.debug("vLLM stream-completions %s", url)
        with self._session.post(url, json=payload, timeout=self.timeout, stream=True) as r:
            r.raise_for_status()
            for line in r.iter_lines(decode_unicode=True):
                if not line or not line.startswith("data: "):
                    continue
                data = line[len("data: "):]
                if data.strip() == "[DONE]":
                    break
                import json
                chunk = json.loads(data)
                text = chunk["choices"][0].get("text", "")
                if text:
                    yield {"choices": [{"text": text}]}

    def _stream_chat(self, payload: dict) -> Generator[Dict[str, Any], None, None]:
        payload["stream"] = True
        url = f"{self.endpoint}/v1/chat/completions"
        logger.debug("vLLM stream-chat %s (model=%s)", url, payload.get("model", "?"))
        content_buffer: list[str] = []
        try:
            r = self._session.post(url, json=payload, timeout=self.timeout, stream=True)
        except Exception:
            raise
        # Adapter fallback: if 404 and adapter is active, retry with base model
        if r.status_code in (400, 404) and self._active_adapter:
            error_text = ""
            try:
                error_text = r.text[:500]
            except Exception as _txt_exc:
                logger.debug("VLLMRemoteModel: could not read error response body: %s", _txt_exc)
            if ("lora" in error_text.lower()
                    or "adapter" in error_text.lower()
                    or self._active_adapter in error_text
                    or "does not exist" in error_text.lower()):
                logger.warning(
                    "vLLM rejected adapter '%s' (%s: %s). Falling back to base model.",
                    self._active_adapter, r.status_code, error_text[:200],
                )
                self._active_adapter = None
                payload["model"] = self.model_name
                r = self._session.post(url, json=payload, timeout=self.timeout, stream=True)
        r.raise_for_status()
        with r:
            content_type = r.headers.get("Content-Type", "")
            # GAIA Engine returns plain JSON (not SSE) — handle both formats
            if "text/event-stream" in content_type or "chunked" in r.headers.get("Transfer-Encoding", ""):
                # SSE streaming (vLLM, llama-server)
                for line in r.iter_lines(decode_unicode=True):
                    if not line or not line.startswith("data: "):
                        continue
                    data = line[len("data: "):]
                    if data.strip() == "[DONE]":
                        break
                    import json
                    chunk = json.loads(data)
                    delta = chunk["choices"][0].get("delta", {})
                    content = delta.get("content", "")
                    if content:
                        content_buffer.append(content)
                        yield {
                            "choices": [{
                                "delta": {"content": content},
                                "finish_reason": None,
                            }]
                        }
            else:
                # Plain JSON response (GAIA Engine managed proxy)
                import json
                try:
                    body = r.content if hasattr(r, 'content') else r.read()
                    if isinstance(body, bytes):
                        body = body.decode("utf-8")
                    # Try SSE parsing first (some proxies don't set content-type)
                    sse_found = False
                    for line in body.split("\n"):
                        if line.startswith("data: "):
                            data = line[len("data: "):]
                            if data.strip() == "[DONE]":
                                break
                            chunk = json.loads(data)
                            delta = chunk.get("choices", [{}])[0].get("delta", {})
                            content = delta.get("content", "")
                            if content:
                                content_buffer.append(content)
                                sse_found = True
                                yield {
                                    "choices": [{
                                        "delta": {"content": content},
                                        "finish_reason": None,
                                    }]
                                }
                    # If no SSE data found, parse as single JSON response
                    if not sse_found:
                        result = json.loads(body)
                        full = result.get("choices", [{}])[0].get("message", {}).get("content", "")
                        if not full:
                            full = result.get("content", result.get("text", ""))
                        if full:
                            content_buffer.append(full)
                            yield {
                                "choices": [{
                                    "delta": {"content": full},
                                    "finish_reason": None,
                                }]
                            }
                except Exception as parse_err:
                    logger.warning("vLLM stream: failed to parse non-SSE response: %s", parse_err)

        full_content = "".join(content_buffer)
        yield {
            "choices": [{
                "message": {"role": "assistant", "content": full_content},
                "finish_reason": "stop",
            }],
            "provider": "vllm_remote",
        }

    @staticmethod
    def _sanitize_messages(messages: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        """Normalise messages for the OpenAI-compatible API.

        Qwen3.5 (and many other models) require that system messages
        appear **only** at position 0.  The prompt builder may produce
        multiple system blocks (persona, summary, sleep context, council,
        session RAG).  We consolidate them into a single system message
        at the front, preserving order.
        """
        system_parts: list[str] = []
        non_system: list[dict[str, str]] = []

        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role not in ("system", "user", "assistant"):
                role = "user"
            if content is None:
                content = ""

            # Multimodal content arrives as a list of {type,text|image|image_url} parts;
            # pass it through untouched so the engine can route to the vision tower.
            if isinstance(content, list):
                if role == "system":
                    text_only = "".join(p.get("text", "") for p in content
                                        if isinstance(p, dict) and p.get("type") == "text")
                    if text_only.strip():
                        system_parts.append(text_only)
                    continue
                non_system.append({"role": role, "content": content})
                continue

            if not isinstance(content, str):
                content = str(content)

            if role == "system":
                if content.strip():
                    system_parts.append(content)
                continue

            if not content.strip():
                continue
            non_system.append({"role": role, "content": content})

        clean: list[dict[str, str]] = []
        if system_parts:
            clean.append({"role": "system", "content": "\n\n".join(system_parts)})
        clean.extend(non_system)

        if not any(m["role"] == "user" for m in clean):
            clean.append({"role": "user", "content": "(continue)"})
        return clean

    def _log_usage(self, resp: dict, duration: float):
        try:
            usage = resp.get("usage", {})
            total = usage.get("total_tokens", 0)
            self._total_tokens += total
            logger.info(
                "VLLMRemote [%s] - Prompt: %s, Completion: %s, Total: %s, "
                "Duration: %.2fs, Session total: %s tokens",
                self.model_name,
                usage.get("prompt_tokens", "?"),
                usage.get("completion_tokens", "?"),
                total,
                duration,
                self._total_tokens,
            )
        except Exception as e:
            logger.debug("Could not log vLLM remote usage: %s", e)
