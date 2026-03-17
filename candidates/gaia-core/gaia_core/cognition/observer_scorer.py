"""observer_scorer.py — Continuous drift detection via Observer scoring.

After each conversation turn, matches user input against a rubric exported
from the cognitive test battery.  When GAIA's response drifts below a
tolerance threshold, the gap is written to a training buffer.  Once the
buffer fills (default 20 entries), gaia-study is triggered automatically.

Non-blocking: runs in a daemon thread so it never delays user responses.
"""

import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import Request, urlopen

log = logging.getLogger("gaia-core.observer-scorer")

# ── Configuration ─────────────────────────────────────────────────────────

TRAINING_BUFFER_PATH = os.environ.get(
    "OBSERVER_BUFFER_PATH", "/shared/observer/training_buffer.jsonl"
)
RUBRIC_PATH = os.environ.get(
    "OBSERVER_RUBRIC_PATH", "/shared/doctor/cognitive_rubric.json"
)
BUFFER_THRESHOLD = int(os.environ.get("OBSERVER_BUFFER_THRESHOLD", "20"))
SIMILARITY_ENDPOINT = os.environ.get(
    "SIMILARITY_ENDPOINT", "http://gaia-core:6415/api/cognitive/similarity"
)
STUDY_ENDPOINT = os.environ.get("STUDY_ENDPOINT", "http://gaia-study:8766")
RUBRIC_CACHE_TTL = int(os.environ.get("OBSERVER_RUBRIC_TTL", "60"))
KEYWORD_MATCH_THRESHOLD = 0.30  # 30% keyword overlap to consider a match

# ── Stopwords for keyword extraction ──────────────────────────────────────

_STOPWORDS = frozenset({
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "to", "of", "in", "for",
    "on", "with", "at", "by", "from", "as", "into", "through", "during",
    "before", "after", "and", "but", "or", "nor", "not", "so", "yet",
    "both", "either", "neither", "each", "every", "all", "any", "few",
    "more", "most", "other", "some", "such", "no", "only", "own", "same",
    "than", "too", "very", "just", "about", "above", "below", "between",
    "how", "what", "when", "where", "which", "who", "whom", "why",
    "i", "me", "my", "you", "your", "it", "its", "we", "our", "they",
    "them", "their", "this", "that", "these", "those", "up", "out",
})


class ObserverScorer:
    """Scores conversation turns against a cognitive rubric."""

    def __init__(self, similarity_endpoint: str = SIMILARITY_ENDPOINT):
        self.similarity_endpoint = similarity_endpoint
        self._rubric_cache: list[dict] | None = None
        self._rubric_loaded_at: float = 0.0

    # ── Public API ────────────────────────────────────────────────────────

    def score_turn(
        self,
        user_input: str,
        response: str,
        packet: object | None = None,
        observer_review: object | None = None,
    ) -> None:
        """Score a turn against the rubric.  Designed to run in a daemon thread."""
        try:
            rubric = self._load_rubric()
            if not rubric:
                return

            matches = self._match_rubric(user_input, rubric)
            if not matches:
                return

            packet_id = ""
            if packet and hasattr(packet, "header"):
                header = packet.header
                if hasattr(header, "packet_id"):
                    packet_id = header.packet_id
                elif isinstance(header, dict):
                    packet_id = header.get("packet_id", "")

            for entry in matches:
                score = self._score_similarity(response, entry["expected"])

                if entry.get("canary", False):
                    log.debug(
                        "Canary %s scored %.2f (observe only)", entry["id"], score
                    )
                    continue

                if score < entry.get("drift_tolerance", 0.4):
                    self._write_buffer_entry(
                        instruction=entry["prompt"],
                        expected=entry["expected"],
                        rubric_id=entry["id"],
                        observed_score=score,
                        weight=entry.get("weight", 1.0),
                        packet_id=packet_id,
                    )
                    log.info(
                        "Drift detected: %s scored %.2f < %.2f — buffered for training",
                        entry["id"], score, entry.get("drift_tolerance", 0.4),
                    )

            self._maybe_trigger_study()

        except Exception:
            log.debug("Observer scorer failed", exc_info=True)

    # ── Rubric Loading ────────────────────────────────────────────────────

    def _load_rubric(self) -> list[dict]:
        now = time.monotonic()
        if self._rubric_cache is not None and (now - self._rubric_loaded_at) < RUBRIC_CACHE_TTL:
            return self._rubric_cache

        try:
            with open(RUBRIC_PATH, "r") as f:
                self._rubric_cache = json.load(f)
            self._rubric_loaded_at = now
            return self._rubric_cache
        except (OSError, json.JSONDecodeError) as e:
            log.debug("Could not load rubric from %s: %s", RUBRIC_PATH, e)
            return []

    # ── Keyword Matching ──────────────────────────────────────────────────

    def _match_rubric(self, user_input: str, rubric: list[dict]) -> list[dict]:
        """Fast keyword overlap matching — no LLM call."""
        input_tokens = set(user_input.lower().split()) - _STOPWORDS
        if not input_tokens:
            return []

        matches = []
        for entry in rubric:
            keywords = set(entry.get("keywords", []))
            if not keywords:
                continue
            overlap = len(input_tokens & keywords) / len(keywords)
            if overlap >= KEYWORD_MATCH_THRESHOLD:
                matches.append(entry)
        return matches

    # ── Similarity Scoring ────────────────────────────────────────────────

    def _score_similarity(self, response: str, expected: str) -> float:
        """Call gaia-core similarity endpoint; fallback to token overlap."""
        try:
            data = json.dumps({"text": response, "reference": expected}).encode()
            req = Request(
                self.similarity_endpoint,
                data=data,
                headers={"Content-Type": "application/json"},
            )
            with urlopen(req, timeout=10) as resp:
                result = json.loads(resp.read().decode())
            return float(result.get("score", 0.0))
        except Exception:
            # Fallback: basic token overlap
            r_tokens = set(response.lower().split())
            ref_tokens = set(expected.lower().split())
            if not ref_tokens:
                return 1.0
            return len(r_tokens & ref_tokens) / len(ref_tokens)

    # ── Buffer Writing ────────────────────────────────────────────────────

    def _write_buffer_entry(
        self,
        instruction: str,
        expected: str,
        rubric_id: str,
        observed_score: float,
        weight: float,
        packet_id: str,
    ) -> None:
        entry = {
            "instruction": instruction,
            "output": expected,
            "metadata": {
                "source": "observer",
                "rubric_id": rubric_id,
                "observed_score": round(observed_score, 4),
                "weight": weight,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "packet_id": packet_id,
            },
        }
        buf_path = Path(TRAINING_BUFFER_PATH)
        buf_path.parent.mkdir(parents=True, exist_ok=True)
        with open(buf_path, "a") as f:
            f.write(json.dumps(entry) + "\n")

    # ── Study Trigger ─────────────────────────────────────────────────────

    def _maybe_trigger_study(self) -> None:
        buf_path = Path(TRAINING_BUFFER_PATH)
        if not buf_path.exists():
            return
        try:
            with open(buf_path, "r") as f:
                count = sum(1 for _ in f)
        except OSError:
            return

        if count >= BUFFER_THRESHOLD:
            log.info(
                "Training buffer has %d entries (threshold=%d) — triggering study",
                count, BUFFER_THRESHOLD,
            )
            self._trigger_study()

    def _find_latest_adapter(self) -> str | None:
        """Find the most recent observer-delta adapter for incremental training."""
        adapter_base = Path("/models/adapters/tier3")
        if not adapter_base.exists():
            return None
        candidates = sorted(
            (d for d in adapter_base.iterdir() if d.is_dir() and d.name.startswith("observer-delta-")),
            key=lambda d: d.stat().st_mtime,
            reverse=True,
        )
        return str(candidates[0]) if candidates else None

    def _trigger_study(self) -> None:
        adapter_name = f"observer-delta-{datetime.now(timezone.utc).strftime('%Y%m%d')}"
        # Incremental: resume from latest observer adapter if one exists
        resume_from = self._find_latest_adapter()
        payload = {
            "adapter_name": adapter_name,
            "documents": [TRAINING_BUFFER_PATH],
            "max_steps": 50,
            "target_loss": 0.1,
            "tags": ["observer", "continuous"],
        }
        if resume_from:
            payload["resume_from"] = resume_from
            log.info("Incremental training: resuming from %s", resume_from)
        try:
            data = json.dumps(payload).encode()
            req = Request(
                f"{STUDY_ENDPOINT}/study/start",
                data=data,
                headers={"Content-Type": "application/json"},
            )
            with urlopen(req, timeout=15) as resp:
                result = json.loads(resp.read().decode())
            log.info("Study triggered: %s — %s", adapter_name, result)

            # Archive the buffer
            archive_path = Path(TRAINING_BUFFER_PATH).with_suffix(
                f".{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.jsonl"
            )
            os.rename(TRAINING_BUFFER_PATH, archive_path)
            log.info("Buffer archived to %s", archive_path)
        except Exception:
            log.warning("Failed to trigger study", exc_info=True)


# ── Module-level singleton ────────────────────────────────────────────────

_scorer: ObserverScorer | None = None


def get_observer_scorer(config=None) -> ObserverScorer:
    """Get or create the singleton ObserverScorer."""
    global _scorer
    if _scorer is None:
        endpoint = SIMILARITY_ENDPOINT
        if config and hasattr(config, "constants"):
            endpoint = config.constants.get("SIMILARITY_ENDPOINT", endpoint)
        _scorer = ObserverScorer(similarity_endpoint=endpoint)
    return _scorer
