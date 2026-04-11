"""
Neural Router — Unified Intent + Engine Routing (Phase 5-C)

Consolidates four previously disjointed routing subsystems into a single
deterministic pipeline:

  1. Reflex Path    — instant keyword matches (exit, help, shell)
  2. Embed Path     — MiniLM cosine similarity against exemplar bank
  3. Weighted Path  — evidence-based complexity scoring
  4. Nano Tiebreak  — LLM arbitration for ambiguous cases only

Usage (from agent_core.py):
    router = NeuralRouter(config, model_pool, embed_model)
    result = router.route(user_input, source="cli")
    # result.target   -> TargetEngine.NANO | CORE | PRIME
    # result.intent   -> "chat", "planning", "injection", etc.
    # result.score    -> 0.0-1.0 complexity
    # result.plan     -> Plan(intent=..., read_only=..., ...)

Design: Proposal 02 (Neural Router Unification)
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Any

from gaia_core.cognition.nlu.embed_intent_classifier import EmbedIntentClassifier
from gaia_core.cognition.nlu.intent_detection import (
    Plan,
    _INTENT_COMPLEXITY,
    _COMPLEX_MARKERS,
    _SIMPLE_MARKERS,
    fast_intent_check,
    _fast_track_intent_detection,
    _detect_fragmentation_request,
    _detect_tool_routing_request,
    _mentions_file_like_action,
    _keyword_intent_classify,
    _nano_confirm_injection,
)

logger = logging.getLogger("GAIA.NeuralRouter")


# ── Result Types ───────────────────────────────────────────────────────

class TargetEngine(Enum):
    """Which inference engine should handle this request."""
    NANO = "nano"       # 0.8B — reflexive, simple
    CORE = "core"       # 2B   — operational, medium
    PRIME = "prime"     # 8B   — deep reasoning, complex


@dataclass
class RouterResult:
    """Complete routing decision from the NeuralRouter."""
    target: TargetEngine
    intent: str
    score: float            # complexity score [0.0, 1.0]
    confidence: float       # embed classifier confidence
    plan: Plan              # full Plan object for backward compat
    source: str = ""        # which stage decided: reflex|embed|weighted|nano|heuristic
    escalation_reason: str = ""  # why Prime was chosen (if applicable)


# ── Scoring Constants ──────────────────────────────────────────────────

# Weight distribution for the complexity score
W_INTENT = 0.40
W_EMBED = 0.25
W_TEXT = 0.20
W_LENGTH = 0.15

# Routing thresholds
THRESHOLD_PRIME = 0.8
THRESHOLD_LITE = 0.3

# Embed confidence tiers
EMBED_AUTHORITATIVE = 0.55
EMBED_TENTATIVE = 0.42

# Escalation markers (absorbed from _should_escalate_to_thinker)
_ESCALATION_MARKERS = [
    "code", "script", "function", "class", "api", "stack trace", "traceback",
    "debug", "optimiz", "profile", "benchmark", "architecture", "design",
    "plan", "steps", "multistep", "algorithm", "performance", "latency",
    "concurrency", "thread", "process", "gpu", "cuda", "memory leak",
]

_RECITATION_MARKERS = [
    "recite", "poem", "verse", "lyrics", "quote", "memorize", "memorised", "cite",
]

_NANO_ENDPOINT = os.environ.get("NANO_INFERENCE_ENDPOINT", "http://gaia-nano:8080")

_NANO_TRIAGE_PROMPT = """\
You are GAIA. Assess this request's complexity.
SIMPLE: Greetings, time/date, system status, simple facts, short poems, basic formatting.
COMPLEX: Coding, debugging, architectural design, deep philosophy, multi-step planning, long-form creative writing.

Respond ONLY with:
RESULT: SIMPLE
or
RESULT: COMPLEX (reason: <brief reason>)"""


# ── Neural Router ──────────────────────────────────────────────────────

class NeuralRouter:
    """Unified intent detection + engine routing.

    Single entry point: `route(text) -> RouterResult`

    Pipeline:
      1. Reflex — instant keyword match (exit, help, shell)
      2. Embed  — MiniLM semantic classification
      3. Score  — weighted complexity from embed + text + length signals
      4. Decision matrix — deterministic engine selection
      5. Nano tiebreak — LLM arbitration only for ambiguous cases
    """

    def __init__(
        self,
        config: Any,
        model_pool: Any = None,
        embed_model: Any = None,
    ):
        self.config = config
        self.model_pool = model_pool
        self.embed_model = embed_model

        # Embed classifier (singleton, cheap to initialize)
        self._embed_classifier: Optional[EmbedIntentClassifier] = None
        self._embed_cfg: dict = {}
        try:
            self._embed_cfg = config.constants.get("EMBED_INTENT", {})
        except Exception:
            pass

    @property
    def embed_classifier(self) -> Optional[EmbedIntentClassifier]:
        """Lazy-init the embed classifier singleton."""
        if self._embed_classifier is None and self.embed_model is not None:
            if self._embed_cfg.get("enabled", True):
                clf = EmbedIntentClassifier.instance()
                if not clf.ready:
                    clf.initialise(self.embed_model, config=self._embed_cfg)
                if clf.ready:
                    self._embed_classifier = clf
        return self._embed_classifier

    def route(
        self,
        text: str,
        source: str = "",
        is_factual: bool = False,
        is_trivial: bool = False,
        probe_context: str = "",
        has_audio: bool = False,
    ) -> RouterResult:
        """Route a user input to the appropriate engine and intent.

        Args:
            text: Raw user input.
            source: Origin context ("cli", "heartbeat", "api", etc.)
            is_factual: Pre-classified as a factual query.
            is_trivial: Pre-classified as trivial.
            probe_context: Semantic probe hint for heuristic fallback.
            has_audio: True if the packet contains audio_payloads (v0.5).

        Returns:
            RouterResult with target engine, intent, score, and Plan.
        """
        # ── Stage 0: Multimodal Audio Override (v0.5) ──
        # Packets with audio_payloads ALWAYS route to Core — only Core/Prime
        # tiers have native audio encoders. No text-based routing needed.
        if has_audio:
            return self._build_result(
                target=TargetEngine.CORE,
                intent="audio_inbox_review",
                score=0.6,
                confidence=1.0,
                source="multimodal_audio",
                escalation_reason="audio_payloads present — native multimodal",
            )

        # Truncate excessively long inputs (pasted content)
        original_text = text
        if len(text) > 2000:
            text = text[:1500] + "\n...\n" + text[-500:]

        # ── Stage 1: Reflex Path (autonomic, zero-cost) ──
        reflex = fast_intent_check(text)
        if reflex:
            return self._build_result(
                target=TargetEngine.NANO,
                intent=reflex,
                score=0.0,
                confidence=1.0,
                source="reflex",
            )

        # Fast-track conversational patterns
        fast_track = _fast_track_intent_detection(text)
        if fast_track:
            return self._build_result(
                target=TargetEngine.NANO,
                intent=fast_track,
                score=0.1,
                confidence=0.8,
                source="reflex",
            )

        # ── Stage 2: Embed Classification (semantic, ~1ms) ──
        embed_intent = None
        embed_score = 0.0
        clf = self.embed_classifier
        if clf:
            threshold = self._embed_cfg.get("confidence_threshold", 0.42)
            embed_intent, embed_score = clf.classify(text, confidence_threshold=threshold)

        # Authoritative embed result (high confidence, specific intent)
        if embed_intent and embed_intent != "other" and embed_score >= EMBED_AUTHORITATIVE:
            # File-keyword guard
            if embed_intent in {"read_file", "write_file"} and not _mentions_file_like_action(text):
                embed_intent = "chat"

            # Injection detection — escalate to Nano for confirmation
            if embed_intent == "injection":
                if _nano_confirm_injection(text):
                    return self._build_result(
                        target=TargetEngine.CORE,
                        intent="injection",
                        score=0.9,
                        confidence=embed_score,
                        source="embed+nano_confirm",
                    )
                else:
                    embed_intent = "chat"

            if embed_intent != "injection":
                score = self._calculate_score(text, embed_intent, embed_score)
                target = self._score_to_engine(score, embed_intent, text)
                return self._build_result(
                    target=target,
                    intent=embed_intent,
                    score=score,
                    confidence=embed_score,
                    source="embed",
                )

        # Tentative injection at any confidence — always confirm
        if embed_intent == "injection":
            if _nano_confirm_injection(text):
                return self._build_result(
                    target=TargetEngine.CORE,
                    intent="injection",
                    score=0.9,
                    confidence=embed_score,
                    source="embed+nano_confirm",
                )
            embed_intent = "chat"
            embed_score = 0.3

        # ── Stage 3: NLU Heuristics (fragmentation, tool routing, etc.) ──
        if source != "heartbeat" and _detect_fragmentation_request(text):
            # Embedding says comprehension? Trust it over heuristic.
            if embed_intent == "comprehension":
                intent = "comprehension"
            else:
                intent = "recitation"
            score = self._calculate_score(text, intent, embed_score)
            return self._build_result(
                target=self._score_to_engine(score, intent, text),
                intent=intent,
                score=score,
                confidence=embed_score,
                source="heuristic",
            )

        if _detect_tool_routing_request(text):
            score = self._calculate_score(text, "tool_routing", embed_score)
            return self._build_result(
                target=self._score_to_engine(score, "tool_routing", text),
                intent="tool_routing",
                score=score,
                confidence=embed_score,
                source="heuristic",
            )

        # File discovery
        lowered = (text or "").lower()
        if ("find" in lowered or "locate" in lowered or "search" in lowered) and \
           ("file" in lowered or "dev_matrix" in lowered or ".md" in lowered or ".json" in lowered):
            return self._build_result(
                target=TargetEngine.CORE,
                intent="find_file",
                score=0.4,
                confidence=embed_score,
                source="heuristic",
            )
        if "dev_matrix" in lowered:
            return self._build_result(
                target=TargetEngine.CORE,
                intent="find_file",
                score=0.4,
                confidence=embed_score,
                source="heuristic",
            )

        # Accept tentative embed result (post-heuristic validation)
        if embed_intent and embed_intent != "other":
            score = self._calculate_score(text, embed_intent, embed_score)
            target = self._score_to_engine(score, embed_intent, text)
            return self._build_result(
                target=target,
                intent=embed_intent,
                score=score,
                confidence=embed_score,
                source="embed_tentative",
            )

        # ── Stage 4: Keyword Heuristic Fallback ──
        intent = _keyword_intent_classify(text, probe_context)
        score = self._calculate_score(text, intent, embed_score)

        # ── Stage 5: Decision Matrix ──
        # For deterministic LITE/PRIME, skip Nano triage entirely
        if is_factual or is_trivial:
            return self._build_result(
                target=TargetEngine.NANO,
                intent=intent,
                score=score,
                confidence=embed_score,
                source="deterministic",
            )

        if score > THRESHOLD_PRIME:
            target = self._resolve_prime_target(text)
            return self._build_result(
                target=target,
                intent=intent,
                score=score,
                confidence=embed_score,
                source="weighted",
                escalation_reason="complexity score > 0.8",
            )

        if score < THRESHOLD_LITE:
            return self._build_result(
                target=TargetEngine.NANO,
                intent=intent,
                score=score,
                confidence=embed_score,
                source="weighted",
            )

        # ── Stage 6: Nano Tiebreak (ambiguous zone 0.3-0.8) ──
        nano_verdict = self._nano_triage(text)
        if nano_verdict == "COMPLEX":
            target = self._resolve_prime_target(text)
            return self._build_result(
                target=target,
                intent=intent,
                score=score,
                confidence=embed_score,
                source="nano_triage",
                escalation_reason="Nano classified COMPLEX",
            )

        return self._build_result(
            target=TargetEngine.NANO,
            intent=intent,
            score=score,
            confidence=embed_score,
            source="nano_triage",
        )

    # ── Scoring ────────────────────────────────────────────────────────

    @staticmethod
    def _calculate_score(
        text: str,
        embed_intent: Optional[str] = None,
        embed_confidence: float = 0.0,
    ) -> float:
        """Compute complexity score from NLU signals. Returns float [0.0, 1.0]."""
        intent_weight = _INTENT_COMPLEXITY.get(embed_intent or "other", 0.5)

        if embed_confidence >= EMBED_AUTHORITATIVE:
            embed_signal = intent_weight
        elif embed_confidence >= EMBED_TENTATIVE:
            embed_signal = 0.4 + (intent_weight - 0.4) * 0.5
        else:
            embed_signal = 0.5

        lowered = (text or "").lower()
        complex_hits = sum(1 for m in _COMPLEX_MARKERS if m in lowered)
        simple_hits = sum(1 for m in _SIMPLE_MARKERS if m in lowered)
        text_signal = complex_hits / (complex_hits + simple_hits) if (complex_hits + simple_hits) > 0 else 0.5

        words = text.split() if text else []
        word_count = len(words)
        sentence_count = max(1, len(re.split(r'[.!?]+', text or "")))

        if word_count <= 5:
            length_signal = 0.1
        elif word_count <= 15:
            length_signal = 0.3
        elif word_count <= 40:
            length_signal = 0.5
        elif word_count <= 80:
            length_signal = 0.7
        else:
            length_signal = 0.9

        if sentence_count >= 3:
            length_signal = min(1.0, length_signal + 0.1)

        score = (
            W_INTENT * intent_weight
            + W_EMBED * embed_signal
            + W_TEXT * text_signal
            + W_LENGTH * length_signal
        )
        return round(min(1.0, max(0.0, score)), 3)

    # ── Engine Resolution ──────────────────────────────────────────────

    @staticmethod
    def _score_to_engine(score: float, intent: str, text: str) -> TargetEngine:
        """Map a score + intent to a TargetEngine without Nano triage."""
        if score > THRESHOLD_PRIME:
            # Check recitation markers — recitation stays on Core
            lowered = (text or "").lower()
            if any(m in lowered for m in _RECITATION_MARKERS):
                return TargetEngine.CORE
            return TargetEngine.PRIME
        if score < THRESHOLD_LITE:
            return TargetEngine.NANO
        # Ambiguous — default to Core (Nano triage will refine later)
        return TargetEngine.CORE

    @staticmethod
    def _resolve_prime_target(text: str) -> TargetEngine:
        """Decide between CORE and PRIME for complex requests.

        Absorbs the logic from _should_escalate_to_thinker: recitation
        stays on Core, code/architecture/long text escalates to Prime.
        """
        if not text:
            return TargetEngine.CORE
        lowered = text.lower()
        # Recitation stays on Core for stability
        if any(m in lowered for m in _RECITATION_MARKERS):
            return TargetEngine.CORE
        # Very long input → Prime
        if len(lowered.split()) > 120:
            return TargetEngine.PRIME
        # Escalation markers → Prime
        if any(m in lowered for m in _ESCALATION_MARKERS):
            return TargetEngine.PRIME
        # Default complex → Core (it can handle most things)
        return TargetEngine.CORE

    # ── Nano Tiebreak ──────────────────────────────────────────────────

    def _nano_triage(self, text: str) -> str:
        """Use Nano LLM to break ties in the ambiguous zone.

        Returns "SIMPLE" or "COMPLEX". Falls back to "COMPLEX" on failure.
        Only called when the weighted score is in [0.3, 0.8].
        """
        # Try model pool first (in-container)
        if self.model_pool:
            try:
                _nano_key = "nano"
                if _nano_key in (self.config.MODEL_CONFIGS or {}):
                    self.model_pool.ensure_model_loaded(_nano_key)
                    nano = self.model_pool.get(_nano_key)
                    if nano:
                        res = nano.create_chat_completion(
                            messages=[
                                {"role": "system", "content": _NANO_TRIAGE_PROMPT},
                                {"role": "user", "content": text},
                            ],
                            max_tokens=32,
                            temperature=0.1,
                        )
                        content = res["choices"][0]["message"]["content"].strip().upper()
                        if "RESULT: SIMPLE" in content:
                            return "SIMPLE"
                        return "COMPLEX"
            except Exception:
                logger.debug("Nano triage via model pool failed", exc_info=True)

        # Fallback: HTTP to nano endpoint
        try:
            from urllib.request import Request, urlopen
            payload = json.dumps({
                "messages": [
                    {"role": "system", "content": _NANO_TRIAGE_PROMPT},
                    {"role": "user", "content": text[:500]},
                ],
                "max_tokens": 32,
                "temperature": 0.1,
            }).encode()
            req = Request(
                f"{_NANO_ENDPOINT}/v1/chat/completions",
                data=payload,
                headers={"Content-Type": "application/json"},
            )
            with urlopen(req, timeout=5) as resp:
                result = json.loads(resp.read().decode())
            content = result.get("choices", [{}])[0].get("message", {}).get("content", "").strip().upper()
            if "</think>" in content:
                content = content.split("</think>")[-1].strip().upper()
            if "RESULT: SIMPLE" in content:
                return "SIMPLE"
            return "COMPLEX"
        except Exception:
            logger.debug("Nano triage via HTTP failed", exc_info=True)
            return "COMPLEX"

    # ── Result Builder ─────────────────────────────────────────────────

    @staticmethod
    def _build_result(
        target: TargetEngine,
        intent: str,
        score: float,
        confidence: float,
        source: str,
        escalation_reason: str = "",
    ) -> RouterResult:
        """Build a RouterResult with a backward-compatible Plan."""
        read_only_intents = {
            "read_file", "explain_file", "explain_symbol",
            "comprehension", "greeting", "identity",
        }
        routing_str = "PRIME" if target == TargetEngine.PRIME else \
                      "LITE" if target == TargetEngine.NANO else "TRIAGE"

        plan = Plan(
            intent=intent,
            read_only=intent in read_only_intents,
            confidence=confidence,
            complexity_score=score,
            routing=routing_str,
        )

        result = RouterResult(
            target=target,
            intent=intent,
            score=score,
            confidence=confidence,
            plan=plan,
            source=source,
            escalation_reason=escalation_reason,
        )

        logger.info(
            "NeuralRouter: intent=%s target=%s score=%.3f confidence=%.3f source=%s%s",
            intent, target.value, score, confidence, source,
            f" reason={escalation_reason}" if escalation_reason else "",
        )
        return result
