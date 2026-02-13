"""
external_voice.py — handles all inbound/outbound chat traffic for GAIA
(streaming, observer hooks, basic logging).  This module is the *sole*
entry and exit for chat-based interactions.
"""
from __future__ import annotations

import contextlib
import json
import logging
import os
import queue
import re
import sys
import threading
import concurrent.futures
import time
from datetime import datetime
from typing import Dict, List, Optional, Generator, Any
from collections.abc import Mapping

from gaia_core.config import Config
from gaia_core.utils.prompt_builder import build_prompt
from gaia_core.utils.stream_observer import StreamObserver, Interrupt
from gaia_core.cognition.self_reflection import reflect_and_refine

# [GCP v0.3] Import the new packet objects
from gaia_common.protocols.cognition_packet import CognitionPacket, PacketState, ReflectionLog

# Loop Detection - for streaming token pattern detection
try:
    from gaia_core.cognition.loop_recovery import LoopDetectorObserver, get_recovery_manager
    LOOP_DETECTION_AVAILABLE = True
except ImportError:
    LOOP_DETECTION_AVAILABLE = False
    LoopDetectorObserver = None

logger = logging.getLogger("GAIA.ExternalVoice")

cfg = Config()
LLAMA_LOG_PATH = os.path.join(cfg.LOGS_DIR, "llama_cpp.log")
CHAT_LOG_PATH = os.path.join(cfg.LOGS_DIR, "chat_session.log")
os.makedirs(cfg.LOGS_DIR, exist_ok=True)


# --------------------------------------------------------------------------- #
# stderr suppression helper (llama.cpp progress bars)
# --------------------------------------------------------------------------- #
@contextlib.contextmanager
def suppress_llama_stderr() -> Generator[None, None, None]:
    """
    Temporarily redirect C-level stderr (llama.cpp progress bars) to a file so
    they don't pollute the interactive CLI.
    """
    original_fd = sys.stderr.fileno()
    saved_fd = os.dup(original_fd)
    try:
        with open(LLAMA_LOG_PATH, "a", encoding="utf-8") as fh:
            os.dup2(fh.fileno(), original_fd)
            yield
    finally:
        os.dup2(saved_fd, original_fd)
        os.close(saved_fd)


# --------------------------------------------------------------------------- #
# ExternalVoice
# --------------------------------------------------------------------------- #
class ExternalVoice:
    def __init__(
        self,
        model,
        model_pool,
        config: Config,
        thought: Optional[str] = None,
        messages: Optional[List[Dict]] = None,
        context: Optional[Dict] = None,
        session_id: str = "shell",
        source: str = "web",
        observer: Optional[StreamObserver] = None,
    ) -> None:
        self.model = model
        self.model_pool = model_pool
        self.config = config
        self.thought = thought
        self.messages = messages
        self.context = context or {}
        self.session_id = session_id
        self.source = source
        self.observer = observer

        self.logical_stop_punct = getattr(self.config, 'LOGICAL_STOP_PUNCTUATION', None) or self.config.constants.get("LOGICAL_STOP_PUNCTUATION", [".", "!", "?", "\n"])
        self.observer_threshold = getattr(self.config, 'OBSERVER_TOKEN_THRESHOLD', None) or self.config.constants.get("OBSERVER_TOKEN_THRESHOLD", 1000)
        # rate-limits for observer calls (seconds) and max calls per stream
        try:
            # Prefer environment overrides for rapid experimentation in rescue runs
            self._observer_min_interval = float(os.getenv("OBSERVER_MIN_INTERVAL") or self.config.constants.get("OBSERVER_MIN_INTERVAL", 15))
        except Exception:
            self._observer_min_interval = 15.0
        try:
            self._observer_max_per_stream = int(self.config.constants.get("OBSERVER_MAX_PER_STREAM", 6))
        except Exception:
            self._observer_max_per_stream = 6
        try:
            # Honor environment override when present (useful for tuning/timeouts)
            self._observer_call_timeout = float(os.getenv("OBSERVER_CALL_TIMEOUT") or self.config.constants.get("OBSERVER_CALL_TIMEOUT", 5))
        except Exception:
            self._observer_call_timeout = 5.0
        # executor for running observer calls with timeout
        self._observer_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        # internal state per stream
        self._last_observer_at = 0.0
        self._observer_calls = 0

        # Loop Detection Observer for streaming token patterns
        self._loop_detector_observer = None
        if LOOP_DETECTION_AVAILABLE:
            try:
                loop_enabled = self.config.constants.get("LOOP_DETECTION_ENABLED", True)
                if loop_enabled:
                    # Think-tag circuit breaker thresholds from constants
                    think_cfg = self.config.constants.get("THINK_TAG_CIRCUIT_BREAKER", {})
                    self._loop_detector_observer = LoopDetectorObserver(
                        think_tag_char_threshold=int(think_cfg.get("char_threshold", 500)),
                        think_tag_ratio_threshold=float(think_cfg.get("ratio_threshold", 0.90)),
                    )
            except Exception:
                logger.debug("ExternalVoice: failed to initialize loop detector observer", exc_info=True)

    # --------------------------------------------------------------------- #
    # streaming
    # --------------------------------------------------------------------- #
    def stream_response(self, user_input: Optional[str] = None) -> Generator[str | Dict[str, Any], None, None]:
        """
        Generator that yields tokens (or events) from the LLM stream.
        It now expects self.messages to be pre-built by the caller (e.g., AgentCore).
        """

        # ---- Direct model stream (no worker thread) --------------------
        logger.info("ExternalVoice: starting create_chat_completion stream")
        try:
            msg_count = len(self.messages or [])
        except Exception:
            msg_count = 0
        logger.debug("[DEBUG] ExternalVoice stream start session_id=%s messages=%d", self.session_id, msg_count)
        t_start = time.perf_counter()
        try:
            kwargs = {
                "messages": self.messages,
                "max_tokens": self.config.max_tokens,
                "temperature": self.config.temperature,
                "top_p": self.config.top_p,
                "stream": True,
            }
            # Enforce conservative generation constraints when requested.
            try:
                enforce = self.config.constants.get("ENFORCE_GENERATION_CONSTRAINTS", True)
            except Exception:
                enforce = True
            if enforce:
                # Keep generation conservative to favor persona/alignment.
                try:
                    kwargs["temperature"] = min(float(kwargs.get("temperature", 0.2)), float(self.config.constants.get("MAX_ALLOWED_TEMPERATURE", 0.2)))
                except Exception:
                    kwargs["temperature"] = 0.2
                try:
                    kwargs["top_p"] = min(float(kwargs.get("top_p", 0.9)), float(self.config.constants.get("MAX_ALLOWED_TOP_P", 0.95)))
                except Exception:
                    kwargs["top_p"] = 0.9
                try:
                    # Check top-level config first (where it's actually defined), then nested constants
                    max_resp_tokens = (
                        getattr(self.config, 'MAX_ALLOWED_RESPONSE_TOKENS', None) or
                        self.config.constants.get("MAX_ALLOWED_RESPONSE_TOKENS", 1000)
                    )
                    kwargs["max_tokens"] = int(min(int(kwargs.get("max_tokens", 512)), int(max_resp_tokens)))
                except Exception:
                    kwargs["max_tokens"] = min(self.config.max_tokens, 1000)
            if getattr(self.model, "is_dev_model", False):
                if self.context and "packet" in self.context:
                    kwargs["packet"] = self.context["packet"]

            token_stream = self.model.create_chat_completion(**kwargs)
            t_end = time.perf_counter()
            logger.info(f"ExternalVoice: create_chat_completion stream took {t_end - t_start:.2f}s")

            buffer: List[str] = []
            last_visible_char = ""
            sentence_counts: Dict[str, int] = {}
            processed_idx = 0
            max_sentence_repeat = int(os.getenv("MAX_SENTENCE_REPEAT") or self.config.constants.get("MAX_SENTENCE_REPEAT", 2))
            since_check = 0

            # Some backends (e.g., vLLM adapter) return the full response
            # as a dict instead of a streaming generator. Normalize those
            # single-shot payloads into a one-item iterable so downstream
            # logic can treat stream and batch paths uniformly.
            if isinstance(token_stream, Mapping):
                iterable = [token_stream]
            elif isinstance(token_stream, (str, bytes, bytearray)):
                iterable = [token_stream]
            else:
                iterable = token_stream

            for item in iterable:
                if self.observer and self.observer.interrupted:
                    reason = getattr(self.observer, "interrupt_reason", "observer interrupt")
                    logger.info("ExternalVoice: interruption detected from observer: %s", reason)
                    yield {"event": "interruption", "data": reason}
                    break

                token = ""
                normalized: Any = item
                try:
                    if isinstance(item, (bytes, bytearray)):
                        decoded = item.decode("utf-8", errors="replace")
                        try:
                            normalized = json.loads(decoded)
                        except Exception:
                            normalized = decoded
                    elif isinstance(item, str):
                        try:
                            normalized = json.loads(item)
                        except Exception:
                            normalized = item

                    if isinstance(normalized, dict):
                        choices = normalized.get("choices") if isinstance(normalized.get("choices"), list) else None
                        if choices and len(choices) > 0 and isinstance(choices[0], dict):
                            choice0 = choices[0]
                            delta = choice0.get("delta") if isinstance(choice0.get("delta"), dict) else {}
                            message = choice0.get("message") if isinstance(choice0.get("message"), dict) else {}
                            token = (
                                delta.get("content", "")
                                or delta.get("text", "")
                                or choice0.get("text", "")
                                or message.get("content", "")
                                or message.get("text", "")
                            )
                        else:
                            token = normalized.get("text", "") or normalized.get("content", "")
                    elif isinstance(normalized, str):
                        token = normalized
                    else:
                        token = str(normalized)
                except Exception:
                    logger.debug("ExternalVoice: failed to normalize stream item; treating as raw string", exc_info=True)
                    try:
                        token = str(item)
                    except Exception:
                        token = ""

                if not token:
                    continue

                # NOTE: _apply_stream_spacing removed — vLLM and llama_cpp
                # streaming APIs return properly-spaced text deltas (OpenAI
                # compatible format) that should be concatenated directly.
                # The old spacing logic incorrectly inserted spaces between
                # BPE subtokens, producing artifacts like "He im ric".
                last_visible_char = self._get_last_visible_char(token) or last_visible_char

                buffer.append(token)

                current_text = "".join(buffer)
                new_segment = current_text[processed_idx:]
                if new_segment:
                    matches = re.findall(r"[^.!?]*[.!?]", new_segment)
                    if matches:
                        processed_idx += sum(len(m) for m in matches)
                        for sentence in matches:
                            key = sentence.strip().lower()
                            if not key:
                                continue
                            sentence_counts[key] = sentence_counts.get(key, 0) + 1
                            if sentence_counts[key] > max_sentence_repeat:
                                logger.info("ExternalVoice: stopping generation after repeated sentence: %s", sentence.strip())
                                raise StopIteration

                yield token

                # --- Loop Detection: Check for token-level patterns ---
                if self._loop_detector_observer:
                    try:
                        loop_interrupt = self._loop_detector_observer.on_token(token)
                        if loop_interrupt and loop_interrupt.level == "BLOCK":
                            # Distinguish think-tag loops from general token loops
                            is_think_tag = "Think-tag" in loop_interrupt.reason
                            interrupt_type = "think_tag_loop" if is_think_tag else "loop_detection"
                            logger.warning(f"ExternalVoice: loop detector interrupted stream ({interrupt_type}): {loop_interrupt.reason}")
                            packet: CognitionPacket = self.context.get("packet")
                            if packet:
                                try:
                                    packet.status.state = PacketState.ABORTED
                                    packet.status.observer_trace.append(f"LOOP_BLOCK: {loop_interrupt.reason}")
                                    packet.status.next_steps.append(f"Loop detected: {loop_interrupt.reason}")
                                except Exception:
                                    logger.debug("ExternalVoice: failed to update packet for loop detection")
                            yield {"event": "interruption", "data": {
                                "reason": f"Loop detected: {loop_interrupt.reason}",
                                "suggestion": loop_interrupt.suggestion,
                                "type": interrupt_type
                            }}
                            break
                    except Exception:
                        logger.debug("ExternalVoice: loop detector check failed", exc_info=True)

                since_check += 1
                # compute whether we need to call the observer
                # - require either the token count threshold, or
                # - punctuation AND a minimum buffered token count (to represent a "full thought")
                current = "".join(buffer)
                token_count = len(current.split())
                min_tokens = 6
                try:
                    min_tokens = int(self.config.constants.get("OBSERVER_CONFIG", {}).get("min_tokens", min_tokens))
                except Exception:
                    pass

                punct_seen = any(p in token for p in self.logical_stop_punct)
                need_check = (
                    since_check >= self.observer_threshold
                    or (punct_seen and token_count >= min_tokens)
                )

                if self.observer and need_check:
                    current = "".join(buffer)
                    # Rate-limit observer to avoid repeated heavy calls and noisy logs
                    now = time.time()
                    if self._observer_calls >= self._observer_max_per_stream:
                        logger.debug("ExternalVoice: max observer calls reached for this stream; skipping further observer checks")
                    elif now - self._last_observer_at < self._observer_min_interval:
                        logger.debug("ExternalVoice: skipping observer.observe (min interval not reached: %.2fs remaining)", max(0.0, self._observer_min_interval - (now - self._last_observer_at)))
                    else:
                        logger.debug("ExternalVoice: invoking observer.observe")
                        try:
                            t_obs_start = time.perf_counter()
                            # Run observer.observe in a thread and wait with timeout
                            fut = self._observer_executor.submit(self.observer.observe, self.context.get("packet"), current)
                            try:
                                interrupt = fut.result(timeout=self._observer_call_timeout)
                            except concurrent.futures.TimeoutError:
                                # Observer call timed out; cancel future and continue
                                try:
                                    fut.cancel()
                                except Exception:
                                    pass
                                logger.warning("ExternalVoice: observer.observe timed out after %.2fs; skipping this check", self._observer_call_timeout)
                                # do not update last_observer_at or call count so we'll retry later
                                interrupt = None
                            t_obs_end = time.perf_counter()
                            if interrupt is not None:
                                logger.debug(f"ExternalVoice: observer.observe took {t_obs_end - t_obs_start:.2f}s")
                            since_check = 0
                            self._last_observer_at = time.time()
                            self._observer_calls += 1
                            if interrupt:
                                packet: CognitionPacket = self.context.get("packet")
                                if interrupt.level == "BLOCK":
                                    # [GCP v0.3] Update the packet status for terminal aborts
                                    if packet:
                                        try:
                                            packet.status.state = PacketState.ABORTED
                                            packet.status.next_steps.append(f"Observer BLOCK: {interrupt.reason}")
                                            packet.status.observer_trace.append(f"BLOCK: {interrupt.reason}")
                                        except Exception:
                                            logger.debug("ExternalVoice: failed to update packet status for abort")
                                    logger.info(f"ExternalVoice: interruption triggered immediately (BLOCK): {interrupt.reason}")
                                    # Include suggestion when available to aid human reviewers
                                    payload = {"reason": interrupt.reason}
                                    if getattr(interrupt, 'suggestion', None):
                                        payload['suggestion'] = interrupt.suggestion
                                    yield {"event": "interruption", "data": payload}
                                    break
                                else:
                                    # Non-terminal interruptions (CAUTION/ESCALATE) should be recorded
                                    if packet:
                                        try:
                                            packet.status.observer_trace.append(f"{interrupt.level}: {interrupt.reason}")
                                            if interrupt.level == "CAUTION":
                                                packet.status.next_steps.append(f"Observer CAUTION: {interrupt.reason}")
                                        except Exception:
                                            logger.debug("ExternalVoice: failed to append observer trace to packet")
                                        # Trigger a targeted re-reflection when observer returns CAUTION
                                        try:
                                            logger.info("ExternalVoice: observer signaled CAUTION — triggering re-reflection")
                                            # Use the observer's llm (if available) for re-reflection and pass sentinel from context
                                            refl_llm = getattr(self.observer, 'llm', None)
                                            eth = self.context.get('ethical_sentinel') if isinstance(self.context, dict) else None
                                            try:
                                                new_reflection = reflect_and_refine(packet=packet, output=current, config=self.config, llm=refl_llm, ethical_sentinel=eth)
                                                # record the re-reflection output as a reflection log entry
                                                try:
                                                    packet.reasoning.reflection_log.append(ReflectionLog(step="observer_refl", summary=new_reflection))
                                                except Exception:
                                                    logger.debug("ExternalVoice: failed to append observer reflection")
                                                # update packet readiness if reflection passed
                                                try:
                                                    packet.status.next_steps.append("Observer-triggered re-reflection completed")
                                                except Exception:
                                                    logger.debug("ExternalVoice: failed to update packet next_steps after re-reflection")
                                            except Exception:
                                                logger.exception("ExternalVoice: re-reflection failed after observer CAUTION")
                                        except Exception:
                                            logger.debug("ExternalVoice: failed to invoke re-reflection", exc_info=True)
                                    # Log OK results at DEBUG to avoid info-level spam; keep CAUTION/WARNING visible
                                    # Provide richer logs when observer is in explain or warn modes
                                    obs_mode = str(self.config.constants.get("OBSERVER_MODE", "block")).lower()
                                    if interrupt.level == "OK":
                                        logger.debug(f"ExternalVoice: observer signaled OK: {interrupt.reason}")
                                    elif interrupt.level == "CAUTION":
                                        logger.info(f"ExternalVoice: observer signaled CAUTION: {interrupt.reason}")
                                        if obs_mode == "warn" or obs_mode == "explain":
                                            logger.info(f"ExternalVoice: observer ({obs_mode}) detail: {getattr(interrupt, 'suggestion', interrupt.reason)}")
                                    else:
                                        logger.warning(f"ExternalVoice: observer signaled {interrupt.level}: {interrupt.reason}")
                                        if obs_mode == "explain":
                                            logger.warning(f"ExternalVoice: observer suggestion: {getattr(interrupt, 'suggestion', '')}")
                        except Exception as e:
                            logger.warning(f"ExternalVoice: observer.observe raised exception (ignored): {e}")
                            logger.debug("Observer exception details", exc_info=True)
                            # don't abort the stream; continue generating
        except StopIteration:
            logger.info("ExternalVoice: generation halted due to repetition guard.")
        except Exception as e:
            logger.error(f"Error during model stream: {e}", exc_info=True)
            raise

    # --------------------------------------------------------------------- #
    # convenience helpers
    # --------------------------------------------------------------------- #
    def generate_full_response(self, user_input: Optional[str] = None) -> str:
        chunks: List[str] = []
        for item in self.stream_response(user_input):
            if isinstance(item, dict) and item.get("event") == "interruption":
                chunks.append(f"\n\n--- {item['data']} ---")
                break
            chunks.append(str(item))
        return "".join(chunks)

    @classmethod
    def from_thought(cls, model, thought: str, **kw):
        return cls(model=model, thought=thought, **kw)

    @classmethod
    def from_messages(cls, model, messages: List[Dict], **kw):
        return cls(model=model, messages=messages, **kw)

    @classmethod
    def one_shot(cls, model, prompt: str, **kw) -> str:
        """Non-streamed convenience wrapper."""
        messages = build_prompt(context={"user_input": prompt})
        with suppress_llama_stderr():
            res = model.create_chat_completion(
                messages=messages,
                max_tokens=kw.get("max_tokens", 52),
                temperature=kw.get("temperature", 0.7),
                top_p=kw.get("top_p", 0.95),
            )
        return res["choices"][0]["message"]["content"]

    def _apply_stream_spacing(self, token: str, prev_char: str) -> str:
        if not token:
            return token
        first_visible = self._get_first_visible_char(token)
        if prev_char and not prev_char.isspace():
            if first_visible and not first_visible.isspace() and first_visible not in ',.!?:;)]}\'"':
                token = " " + token
        return token

    @staticmethod
    def _get_first_visible_char(text: str) -> str:
        cleaned = re.sub(r"\x1b\[[0-9;]*[A-Za-z]", "", text)
        for ch in cleaned:
            if ch:
                return ch
        return ""

    @staticmethod
    def _get_last_visible_char(text: str) -> str:
        cleaned = re.sub(r"\x1b\[[0-9;]*[A-Za-z]", "", text)
        for ch in reversed(cleaned):
            if ch:
                return ch
        return ""

def extract_and_format_execute_blocks(response_text: str) -> str:
    """
    Ensures each EXECUTE line is alone on a line and stripped of markdown or quotes.
    """
    lines = response_text.splitlines()
    clean_lines = []
    for line in lines:
        match = re.search(r"EXECUTE:\s*(.*)", line)
        if match:
            cmd = match.group(1).strip().strip("`*'")
            clean_lines.append(f"EXECUTE: {cmd}")
        else:
            clean_lines.append(line)
    return "\n".join(clean_lines)

if __name__ == "__main__":
    import sys
    print("ExternalVoice is a library module; no CLI entrypoint.", flush=True)
    sys.exit(2)
