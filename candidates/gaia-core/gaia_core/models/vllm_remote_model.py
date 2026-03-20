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

        self.timeout = int(model_config.get("timeout", 120))

        # Context window size for auto-clamping max_tokens
        self.max_model_len = int(
            model_config.get("max_model_len")
            or model_config.get("context_length")
            or os.getenv("VLLM_MAX_MODEL_LEN", "8192")
        )

        # LoRA support
        self._lora_config = model_config.get("lora_config") or {}
        self._active_adapter: Optional[str] = None

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
        """Rough token estimate (~3.5 chars/token for English, use 3 to overestimate)."""
        if isinstance(text_or_messages, str):
            return len(text_or_messages) // 3 + 4
        # List of message dicts
        total_chars = sum(len(m.get("content", "")) for m in text_or_messages)
        return total_chars // 3 + len(text_or_messages) * 4  # +4/msg for role overhead

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

        # Disable Qwen3 thinking mode by default to avoid <think> tag bloat.
        # Callers can override by passing chat_template_kwargs explicitly.
        if "chat_template_kwargs" not in payload:
            payload["chat_template_kwargs"] = {"enable_thinking": False}

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

    def set_active_adapter(self, adapter_name: Optional[str]):
        """Select a LoRA adapter by name. Pass None to use the base model."""
        self._active_adapter = adapter_name
        logger.info("VLLMRemoteModel: active adapter set to %s", adapter_name)

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
        except Exception:
            pass

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
        logger.debug("vLLM POST %s (model=%s)", url, payload.get("model", "?"))
        last_exc: Optional[Exception] = None

        for attempt in range(1, self._MAX_RETRIES + 1):
            try:
                r = self._session.post(url, json=payload, timeout=self.timeout)
                # Retry on 503 (vLLM model-loading state), not other errors
                if r.status_code == 503 and attempt < self._MAX_RETRIES:
                    delay = self._RETRY_BASE_DELAY * attempt
                    logger.warning(
                        "vLLM returned 503 on attempt %d/%d, retrying in %.1fs...",
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
            except Exception:
                pass
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
            elif not isinstance(content, str):
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
