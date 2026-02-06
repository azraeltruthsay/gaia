import logging
import time
import os
import re
from dataclasses import dataclass
from gaia_core.config import Config
from gaia_core.ethics.core_identity_guardian import CoreIdentityGuardian
from gaia_common.protocols.cognition_packet import CognitionPacket, ReflectionLog
from gaia_common.utils.string_tools import trim_text
from typing import List, Dict, Optional

logger = logging.getLogger("GAIA.StreamObserver")
logger.setLevel(logging.DEBUG)

@dataclass
class Interrupt:
    level: str  # "OK" | "CAUTION" | "BLOCK" | "FATAL"
    reason: str
    suggestion: str = ""

class StreamObserver:
    def __init__(self, config: Config, llm, name: str = "AgentCore-Observer"):
        self.config = config
        self.identity_guardian = CoreIdentityGuardian(config)
        self.source = name
        if llm is None:
            raise ValueError("StreamObserver requires a model (llm) from the model pool.")
        self.llm = llm
        self.interrupted = False
        self.interrupt_reason = None
        # Cache the last observed buffer/result and use a small time-based
        # throttle so we doesn't call the LLM too frequently during streaming.
        self._last_output = None
        self._last_result = None
        self._last_obs_time = 0.0
        self._min_interval = getattr(self.config, "OBSERVER_MIN_INTERVAL", 0.5)
        # Grace and heuristic thresholds (configurable via constants or ENV)
        try:
            self._grace_tokens = int(os.getenv("OBSERVER_GRACE_TOKENS") or self.config.constants.get("OBSERVER_GRACE_TOKENS", 12))
        except Exception:
            self._grace_tokens = 12
        try:
            self._grace_seconds = float(os.getenv("OBSERVER_GRACE_SECONDS") or self.config.constants.get("OBSERVER_GRACE_SECONDS", 0.5))
        except Exception:
            self._grace_seconds = 0.5
        try:
            self._keyword_ratio_threshold = float(os.getenv("OBSERVER_KEYWORD_RATIO_THRESHOLD") or self.config.constants.get("OBSERVER_KEYWORD_RATIO_THRESHOLD", 0.25))
        except Exception:
            self._keyword_ratio_threshold = 0.25
        # parse observer-specific toggles robustly
        def _const_bool(key, default=False):
            try:
                # Prefer explicit environment variable overrides (string form)
                ev = None
                try:
                    ev = os.getenv(key)
                except Exception:
                    ev = None
                if ev is not None:
                    return str(ev).strip().lower() in ("1", "true", "yes", "y", "on")

                # Fall back to constants file
                v = self.config.constants.get(key, default)
                if isinstance(v, str):
                    return v.strip().lower() in ("1", "true", "yes", "y", "on")
                return bool(v)
            except Exception:
                return bool(default)

        self._use_llm_config = _const_bool("OBSERVER_USE_LLM", False)
        self._force_llm = _const_bool("OBSERVER_FORCE_LLM", False)
        self.post_stream_only = False

    def observe(self, packet: Optional[CognitionPacket], output: str) -> Interrupt:
        """
        Reviews the assistant's output using the identity summary and observation instruction.
        Returns an Interrupt object with a level and reason.
        """
        # Quick guard: if we have no packet, we can't do a contextual check.
        if not packet:
            logger.warning("StreamObserver received no packet, cannot perform contextual check.")
            return Interrupt(level="OK", reason="No packet provided.")

        # Rate-limit / cache: if the same output was just observed recently,
        # return the cached result to avoid repeated LLM calls for minor updates.
        now = time.time()
        try:
            if output == self._last_output or (now - self._last_obs_time) < float(self._min_interval):
                logger.debug("StreamObserver: skipping observe (cached or rate-limited).")
                return self._last_result or Interrupt(level="OK", reason="Skipped: rate-limited")
        except Exception:
            # Don't let caching issues crash the observer
            logger.debug("StreamObserver: cache check failed", exc_info=True)

        # Read observer mode from config constants. Modes:
        #  - 'block'   : current behavior (BLOCK on issues)
        #  - 'explain' : BLOCK, but include extra explanatory logging and suggestions
        #  - 'warn'    : do NOT block; downgrade BLOCK -> CAUTION so stream continues
        mode = str(self.config.constants.get("OBSERVER_MODE", "block")).lower()
        verbose = bool(self.config.constants.get("OBSERVER_VERBOSE", False))

        if self.fast_check(output):
            # fast_check sets self.interrupt_reason if it returns True
            reason = self.interrupt_reason or "Potential error detected in output."
            if mode == "warn":
                if verbose:
                    logger.info(f"StreamObserver (WARN mode): fast_check triggered but not blocking. Reason: {reason}. Output snippet: {trim_text(output, 400)}")
                return Interrupt(level="CAUTION", reason=reason)
            elif mode == "explain":
                # Provide a richer suggestion payload so callers can show context
                sug = f"Observer detected an error-like token pattern. Snippet: {trim_text(output, 400)}"
                logger.warning(f"StreamObserver (EXPLAIN mode): blocking with explanation: {reason}")
                self.interrupt_reason = reason
                return Interrupt(level="BLOCK", reason=reason, suggestion=sug)
            else:
                return Interrupt(level="BLOCK", reason=reason)

        # NEW: Check for EXECUTE in read_only plans (packet.flags may not exist in v0.3 packets)
        flags = getattr(packet, "flags", None)
        # For v0.3 packets the read-only intent is stored as a data_field; check both
        read_only_flag = False
        if flags and isinstance(flags, dict):
            read_only_flag = flags.get("read_only", False)
        else:
            try:
                # look for data_fields entry named 'read_only_intent'
                for df in getattr(packet, "content", {}).data_fields or []:
                    if getattr(df, "key", None) == "read_only_intent":
                        read_only_flag = bool(getattr(df, "value", False))
                        break
            except Exception:
                read_only_flag = False

        if read_only_flag and "EXECUTE:" in output:
            logger.warning("Observer BLOCKed: EXECUTE directive in read_only plan.")
            return Interrupt(level="BLOCK", reason="EXECUTE not allowed for read-only intent.")

        # Validate code path references
        logger.debug("StreamObserver: Calling _validate_code_paths...")
        path_validation_results = self._validate_code_paths(output)
        logger.debug(f"StreamObserver: _validate_code_paths returned {len(path_validation_results)} results.")
        if path_validation_results:
            path_notes = []
            for result in path_validation_results:
                if result["exists"]:
                    if result["is_file"]:
                        path_notes.append(f"Code path reference '{result['reference']}' (file: {result['absolute_path']}) exists.")
                    elif result["is_directory"]:
                        path_notes.append(f"Code path reference '{result['reference']}' (directory: {result['absolute_path']}) exists.")
                    else:
                        path_notes.append(f"Code path reference '{result['reference']}' (path: {result['absolute_path']}) exists but type is unknown.")
                else:
                    path_notes.append(f"WARNING: Code path reference '{result['reference']}' does not exist.")
            
            # Append path validation results to reflection log
            if path_notes:
                log_entry = ReflectionLog(step="observer_path_validation", summary="\n".join(path_notes))
                logger.debug(f"StreamObserver: Appending log entry to reflection_log: {log_entry}")
                try:
                    if hasattr(packet.reasoning, 'reflection_log'):
                        packet.reasoning.reflection_log.append(log_entry)
                    else:
                        # Initialize if reflection_log doesn't exist
                        packet.reasoning.reflection_log = [log_entry]
                    logger.debug("StreamObserver: Successfully appended to reflection_log.")
                except Exception as e:
                    logger.debug(f"StreamObserver: failed to append path validation to packet reflection log: {e}", exc_info=True)

        # identity may be on older packets as packet.identity (dict), or on v0.3 packets in header.persona
        identity_text = ""
        try:
            if hasattr(packet, "identity") and isinstance(packet.identity, dict):
                identity = packet.identity.get("identity_summary", [])
                identity_text = "\n".join(identity)
            else:
                persona = getattr(packet.header, "persona", None)
                if persona:
                    parts = [getattr(persona, "persona_id", ""), f"role:{getattr(persona, 'role', '')}"]
                    tone = getattr(persona, "tone_hint", None)
                    if tone:
                        parts.append(f"tone:{tone}")
                    identity_text = " ".join([p for p in parts if p])
        except Exception:
            identity_text = ""
        instruction = (
            self.config.constants.get("TASK_INSTRUCTIONS", {}).get("observer")
            or self.config.constants.get("observer")
            or "Evaluate this response for contradiction or harm."
        ).strip()
        # cheat sheet may be a top-level dict on older packets, or stored in packet.context.cheatsheets as entries
        protocol_rules = []
        try:
            if hasattr(packet, "cheat_sheet") and isinstance(packet.cheat_sheet, dict):
                protocol_rules = packet.cheat_sheet.get("protocol_rules", [])
            else:
                ctx = getattr(packet, "context", None)
                if ctx:
                    cheats = getattr(ctx, "cheatsheets", [])
                    for ch in cheats or []:
                        if isinstance(ch, dict):
                            protocol_rules.extend(ch.get("protocol_rules", []))
                        else:
                            protocol_rules.extend(getattr(ch, "protocol_rules", []) or [])
        except Exception:
            protocol_rules = []

        if not protocol_rules:
            protocol_rules = self.config.cheat_sheet.get("protocol_rules", []) if getattr(self.config, "cheat_sheet", None) else []

        if protocol_rules:
            protocol_text = "\n".join(protocol_rules)
            instruction += f"\n\nKeep in mind the following operational protocol rules, which are valid actions for the assistant and NOT identity violations:\n{protocol_text}"

        # prompt and user input location differs between packet versions
        try:
            user_prompt = getattr(packet, "prompt", None) or getattr(packet, "content", None) and getattr(packet.content, "original_prompt", None) or ""
        except Exception:
            user_prompt = ""

        # Debug: report effective observer flags and env overrides so runtime
        # behavior is easier to diagnose in rescue logs.
        try:
            logger.debug(
                "StreamObserver: flags -> ENV_OBSERVER_USE_LLM=%s CONST_OBSERVER_USE_LLM=%s parsed_use_llm=%s force_llm=%s",
                os.getenv("OBSERVER_USE_LLM"),
                self.config.constants.get("OBSERVER_USE_LLM"),
                self._use_llm_config,
                self._force_llm,
            )
        except Exception:
            pass
        # Quick grace buffer: avoid running identity heuristics on very short outputs
        # unless explicitly forced. This reduces false positives on short/generic
        # streaming chunks while the assistant is still producing the response.
        try:
            word_count = len([w for w in output.split() if w.strip()])
            now = time.time()
            if word_count < int(self._grace_tokens) and not bool(self._force_llm):
                logger.debug(f"StreamObserver: output too short for identity check (tokens={word_count} < grace={self._grace_tokens}), skipping heuristic.")
                ok = Interrupt(level="OK", reason="Grace buffer: waiting for more content before identity check")
                try:
                    self._last_output = output
                    self._last_result = ok
                    self._last_obs_time = now
                except Exception:
                    logger.debug("StreamObserver: failed to cache grace result", exc_info=True)
                return ok
            # If force_llm is true but we are still within the grace_seconds window, wait.
            if word_count < int(self._grace_tokens) and bool(self._force_llm) and (now - self._last_obs_time) < float(self._grace_seconds):
                logger.debug("StreamObserver: force-LLM requested but still within grace_seconds; waiting before calling LLM.")
                ok = Interrupt(level="OK", reason="Grace delay before forced LLM check")
                try:
                    self._last_output = output
                    self._last_result = ok
                    self._last_obs_time = now
                except Exception:
                    logger.debug("StreamObserver: failed to cache grace delay result", exc_info=True)
                return ok
        except Exception:
            logger.debug("StreamObserver: grace buffer check failed", exc_info=True)
        # Request a concise, machine-parsable JSON response from the observer LLM
        prompt = (
            f"\n{instruction}\n\n"
            f"IDENTITY:\n{identity_text}\n\n"
            f"USER INPUT:\n{user_prompt}\n\n"
            f"OUTPUT TO REVIEW:\n{trim_text(output, 1200)}\n\n"
            "Return ONLY a valid JSON object, with NO extra commentary or surrounding text.\n"
            'The object should have the shape: {"action": "CONTINUE"|"INTERRUPT", "reason": "short reason" }.\n'
            "If action is INTERRUPT, provide a concise reason (<=120 characters)."
        )

        # Identity alignment heuristic: prefer an LLM-based observer when enabled.
        # If LLM checks are disabled, run a lightweight keyword-overlap test so
        # the observer enforces understanding of GAIA's identity without requiring
        # verbatim quoting.
        try:
            use_llm = bool(self._use_llm_config) or bool(self._force_llm)
            if use_llm:
                # Defer to the full LLM check below.
                logger.debug("StreamObserver: deferring identity check to LLM observer (config) -> use_llm=%s force=%s", self._use_llm_config, self._force_llm)
            else:
                if identity_text and isinstance(identity_text, str) and identity_text.strip():
                    # Build a small set of identity keywords (long words, excluding common stopwords)
                    stopwords = {"the", "and", "for", "with", "that", "this", "is", "a", "as", "of", "to", "be", "in", "or", "by", "are"}
                    words = [w.lower().strip(".,\'\"()[]") for w in identity_text.split()]
                    keywords = [w for w in words if len(w) > 4 and w not in stopwords]
                    # Pick a compact set to test against (dedupe and limit)
                    seen = []
                    for k in keywords:
                        if k not in seen:
                            seen.append(k)
                        if len(seen) >= 8:
                            break
                    keywords = seen
                    if keywords:
                        out_lower = output.lower()
                        hits = sum(1 for k in keywords if k in out_lower)
                        ratio = hits / len(keywords)
                        logger.debug(f"StreamObserver: identity keyword overlap {hits}/{len(keywords)} ({ratio:.2f})")
                        # Threshold: configurable (default 25%) to reduce false positives on short responses
                        threshold = float(self._keyword_ratio_threshold)
                        if ratio < threshold:
                            reason = f"Output appears misaligned with configured identity (keyword match {hits}/{len(keywords)})"
                            logger.info(f"StreamObserver: identity alignment heuristic note: {reason}")
                            level = "INFO"
                            suggestion = f"Expected identity-aligned content (look for terms: {', '.join(keywords)})"
                            return Interrupt(level=level, reason=reason, suggestion=suggestion)
        except Exception:
            logger.debug("StreamObserver: identity alignment heuristic failed", exc_info=True)
        # Decide whether to call the LLM-based observer check. This is opt-in
        # via config and may be enforced with the OBSERVER_FORCE_LLM toggle.
        use_llm = bool(self._use_llm_config) or bool(self._force_llm)
        logger.debug(f"StreamObserver ({self.source}): LLM checks enabled={use_llm} (config={self._use_llm_config} force={self._force_llm})")
        if not use_llm:
            # Cache and return OK to avoid expensive/non-threadsafe calls.
            ok = Interrupt(level="OK", reason="LLM checks disabled by config")
            try:
                self._last_output = output
                self._last_result = ok
                self._last_obs_time = time.time()
            except Exception:
                logger.debug("StreamObserver: failed to cache OK result", exc_info=True)
            return ok
        try:
            # Allow runtime override for observer token budget and generation params
            try:
                max_tok = int(os.getenv("OBSERVER_MAX_TOKENS") or 64)
            except Exception:
                max_tok = 64
            try:
                top_p = float(os.getenv("OBSERVER_TOP_P") or 0.9)
            except Exception:
                top_p = 0.9
            # Call the model; be tolerant of different LLM interfaces (some test doubles
            # or older adapters may not accept top_p).
            try:
                result = self.llm.create_chat_completion(
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=max_tok,
                    temperature=0.0,
                    top_p=top_p,
                )
            except TypeError:
                # Fallback to a more conservative call signature
                result = self.llm.create_chat_completion(
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=max_tok,
                    temperature=0.0,
                )
            # Extract text from common response shapes
            if isinstance(result, dict) and "choices" in result:
                # chat-style response
                text = (result["choices"][0].get("message", {}) or {}).get("content", "").strip() or result["choices"][0].get("text", "").strip()
            else:
                text = getattr(result, 'text', str(result)).strip()
            # Try to parse JSON if the model returned a JSON object as requested
            parsed = None
            try:
                import json as _json

                parsed = _json.loads(text)
            except Exception:
                parsed = None

            # Basic validation and fallbacks: if parse failed or schema invalid, detect garbage and fallback
            valid_json = isinstance(parsed, dict)
            if not valid_json:
                # detect long repeated punctuation or low-entropy output (e.g., '!!!!!....')
                try:
                    import re as _re

                    # detect long runs of repeated punctuation or syntax noise
                    repeated = _re.search(r'([!?.\-,:;`~\^*#@\/\])\1{10,}', text)
                except Exception:
                    repeated = None
                if repeated or len(text) < 1:
                    logger.warning("StreamObserver: invalid/low-entropy observer output; falling back to sentinel and heuristic-only decision")
                    parsed = {"action": "CONTINUE", "valid": False, "note": "invalid_observer_output", "raw": trim_text(text, 200)}
                else:
                    # attempt to salvage by searching for an action token in plain text
                    txt_upper = text.upper()
                    if "INTERRUPT:" in txt_upper or txt_upper.strip().startswith("INTERRUPT"):
                        # Extract reason heuristically
                        try:
                            reason_text = text.split("INTERRUPT:", 1)[1].strip()
                        except Exception:
                            reason_text = trim_text(text, 120)
                        parsed = {"action": "INTERRUPT", "reason": trim_text(reason_text, 120)}
                    else:
                        parsed = {"action": "CONTINUE", "valid": False, "note": "non-json-observer_output", "raw": trim_text(text, 200)}

            # ensure canonical schema
            action = str(parsed.get("action", "CONTINUE")).upper()
            reason_text = str(parsed.get("reason", "")).strip() if isinstance(parsed.get("reason", ""), (str,)) else ""
            if action not in ("CONTINUE", "INTERRUPT"):
                logger.debug("StreamObserver: unknown action in parsed observer json; normalizing to CONTINUE")
                action = "CONTINUE"
            # Normalize into the prior expected textual form
            if action == "INTERRUPT":
                text = f"INTERRUPT: {reason_text}"
            else:
                text = "CONTINUE"
            # Record the LLM rationale back onto the packet for forensic analysis
            try:
                rationale_note = f"Observer LLM rationale: {trim_text(text, 1200)}"
                try:
                    if hasattr(packet, "append_thought"):
                        packet.append_thought(rationale_note)
                    else:
                        # fallback to the v0.3 style reasoning reflection log; append a serializable dict
                        try:
                            packet.reasoning.reflection_log.append({"step": "observer", "summary": rationale_note})
                        except Exception:
                            # As a last resort, attach a plain attribute
                            packet.reasoning.reflection_log = getattr(packet.reasoning, 'reflection_log', []) + [{"step": "observer", "summary": rationale_note}]
                except Exception:
                    logger.debug("StreamObserver: failed to append LLM rationale to packet", exc_info=True)
                logger.info(f"StreamObserver: appended LLM rationale to packet (trimmed): {trim_text(text,200)}")
            except Exception:
                logger.debug("StreamObserver: failed to record LLM rationale", exc_info=True)
            if text.upper().startswith("INTERRUPT"):
                reason = text.split("INTERRUPT:", 1)[1].strip()

                soft_terms = ["project", "hypothetical", "framing", "boot", "process", "metaphor"]
                if any(term in reason.lower() for term in soft_terms):
                    logger.warning(f"🔶 Observer soft interruption suppressed: {reason}")
                    # Try to record the note back to the packet; fallback to adding a ReflectionLog
                    try:
                        if hasattr(packet, "append_thought"):
                            packet.append_thought(f"Observer noted framing concern but allowed continuation: {reason}")
                        else:
                            try:
                                packet.reasoning.reflection_log.append({"step": "observer", "summary": f"Observer noted framing concern but allowed continuation: {reason}"})
                            except Exception:
                                packet.reasoning.reflection_log = getattr(packet.reasoning, 'reflection_log', []) + [{"step": "observer", "summary": f"Observer noted framing concern but allowed continuation: {reason}"}]
                    except Exception:
                        logger.debug("Failed to append thought to packet; continuing")
                    return Interrupt(level="CAUTION", reason=reason)

                logger.warning(f"🔔 Observer Interrupt: {reason}")
                try:
                    if hasattr(packet, "append_thought"):
                        packet.append_thought(f"Observer interrupted due to: {reason}")
                    else:
                        try:
                            packet.reasoning.reflection_log.append({"step": "observer", "summary": f"Observer interrupted due to: {reason}"})
                        except Exception:
                            packet.reasoning.reflection_log = getattr(packet.reasoning, 'reflection_log', []) + [{"step": "observer", "summary": f"Observer interrupted due to: {reason}"}]
                except Exception:
                    logger.debug("Failed to append interrupt thought to packet")
                self.interrupted = True
                self.interrupt_reason = reason
                level = "CAUTION" if self.post_stream_only else "BLOCK"
                return Interrupt(level=level, reason=reason)
            else:
                ok = Interrupt(level="OK", reason="No issues found.")
                # cache the OK result
                try:
                    self._last_output = output
                    self._last_result = ok
                    self._last_obs_time = time.time()
                except Exception:
                    logger.debug("Failed to cache fallback observer result", exc_info=True)
                return ok
        except Exception as e:
            # Log at warning but include debug traceback; don't raise exceptions from observer
            logger.warning(f"⚠️ Observer exception: {e}")
            logger.debug("Observer exception details", exc_info=True)
            # cache a safe default so we don't hammer the model on repeated exceptions
            fallback = Interrupt(level="OK", reason=f"Observer failed: {e}")
            try:
                self._last_output = output
                self._last_result = fallback
                self._last_obs_time = time.time()
            except Exception:
                logger.debug("Failed to cache fallback observer result", exc_info=True)
            return fallback

    def fast_check(self, buffer: str) -> bool:
        """
        Performs fast, rule-based checks for obvious errors.
        Returns True if an interruption is needed.
        """
        buffer_lower = buffer.lower()
        if "error" in buffer_lower or "exception" in buffer_lower:
            self.interrupt_reason = "Potential error detected in output."
            self.interrupted = True
            return True
        return False

    @staticmethod
    def check_response_quality(response: str, user_prompt: str) -> Optional[Interrupt]:
        """
        Check if the response contains raw meta-content that shouldn't be user-facing.

        This catches cases where:
        - Think/reflection tags leaked through
        - Internal reasoning blocks appear in output
        - The response is clearly not addressing the user's question

        Args:
            response: The final response text to validate
            user_prompt: The original user question for context

        Returns:
            Interrupt if issues found, None if response looks clean
        """
        if not response:
            return Interrupt(level="CAUTION", reason="Empty response")

        # Check for leaked meta-content tags
        meta_patterns = [
            (r'<think(?:ing)?>', "Leaked <think> tag in response"),
            (r'</think(?:ing)?>', "Leaked </think> tag in response"),
            (r'<reflection>', "Leaked <reflection> tag in response"),
            (r'<reasoning>', "Leaked <reasoning> tag in response"),
            (r'<internal>', "Leaked <internal> tag in response"),
            (r'<scratchpad>', "Leaked <scratchpad> tag in response"),
            (r'\[HEADER\]', "Leaked GCP [HEADER] block in response"),
            (r'\[GOVERNANCE\]', "Leaked GCP [GOVERNANCE] block in response"),
            (r'\[METRICS\]', "Leaked GCP [METRICS] block in response"),
            (r'\[REASONING\]', "Leaked GCP [REASONING] block in response"),
        ]

        for pattern, reason in meta_patterns:
            if re.search(pattern, response, re.IGNORECASE):
                logger.warning(f"Response quality check failed: {reason}")
                return Interrupt(
                    level="CAUTION",
                    reason=reason,
                    suggestion="Response contains internal reasoning that should be stripped before sending to user"
                )

        # Check for responses that look like raw model internals
        # (e.g., starting with "Hmm," "Let me think," followed by reasoning)
        internal_starters = [
            r'^Hmm,?\s+(?:the user|I need|let me)',
            r'^Let me (?:think|check|quickly|first)',
            r'^I should (?:first |provide |check )',
            r'^The user (?:is asking|just asked|wants)',
        ]

        for pattern in internal_starters:
            if re.match(pattern, response.strip(), re.IGNORECASE):
                # Only flag if the response is short (likely incomplete/raw thinking)
                if len(response) < 200:
                    logger.info(f"Response may contain raw thinking (starts with internal pattern)")
                    return Interrupt(
                        level="INFO",
                        reason="Response may contain unprocessed internal reasoning",
                        suggestion="Consider reviewing response quality"
                    )

        return None

    def _validate_code_paths(self, text_content: str) -> List[Dict]:
        """
        Extracts potential file paths from text_content and validates their existence.
        Returns a list of dictionaries with validation results.
        """
        validation_results = []
        # Regex to find common file path patterns (e.g., /path/to/file.ext, file.ext, app/module.py)
        # This regex is a starting point and might need refinement.
        # Captures:
        # Group 1: Paths with extensions (e.g., knowledge/file.md)
        # Group 2: Directory-like paths without a trailing dot (e.g., knowledge/my_dir)
        path_matches = re.findall(r"([a-zA-Z0-9_.-]+(?:/[a-zA-Z0-9_.-]+)*\.[a-zA-Z]{1,5})|([a-zA-Z0-9_.-]+(?:/[a-zA-Z0-9_.-]+)+/?(?<!\.))", text_content)
        
        potential_paths = []
        for match_tuple in path_matches:
            for item in match_tuple:
                if item: # Only append non-empty matches from the tuple
                    potential_paths.append(item)

        # Filter out common non-file words that might match the regex
        # This is a heuristic and might need to be expanded
        ignored_patterns = {"of", "to", "in", "for", "with", "and", "the", "key", "set", "use", "code", "file", "path", "from", "by", "is", "or", "app"}
        potential_paths = [p for p in potential_paths if p not in ignored_patterns and not p.isdigit() and len(p) > 2] # Min length to avoid single letters
        logger.debug(f"StreamObserver: _validate_code_paths - Extracted {len(potential_paths)} potential paths: {potential_paths}")

        for path in potential_paths:
            # Assume paths are relative to the project root for now,
            # but allow for absolute paths as well.
            abs_path = os.path.join("/gaia/GAIA_Project", path)
            exists = os.path.exists(abs_path)
            is_file = os.path.isfile(abs_path)
            is_dir = os.path.isdir(abs_path)

            # Also check relative to gaia-assistant for convenience, as many paths are in there
            if not exists:
                abs_path_gaia_assistant = os.path.join("/gaia/GAIA_Project/gaia-assistant", path)
                exists = os.path.exists(abs_path_gaia_assistant)
                is_file = os.path.isfile(abs_path_gaia_assistant)
                is_dir = os.path.isdir(abs_path_gaia_assistant)
                if exists:
                    abs_path = abs_path_gaia_assistant # Update to the found path

            validation_results.append({
                "reference": path,
                "absolute_path": abs_path if exists else None,
                "exists": exists,
                "is_file": is_file,
                "is_directory": is_dir
            })
        logger.debug(f"StreamObserver: _validate_code_paths - Final validation results: {validation_results}")
        return validation_results
