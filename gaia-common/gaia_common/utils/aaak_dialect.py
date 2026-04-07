"""
AAAK Dialect — Compressed Symbolic Memory Language for GAIA.

Adapted from MemPalace (github.com/milla-jovovich/mempalace).

A structured symbolic format that ANY LLM reads natively at ~5-8x compression.
Not latent vectors. Not English prose. A universal memory compression dialect.

GAIA uses this for:
  - Compressing system prompt sections (identity, rules, tool schemas)
  - Lite journal entries (sleep task summaries)
  - Semantic codex writes (compressed knowledge base entries)
  - KV cache prefix content (maximizing info per token)

FORMAT:
  Header:   wing|room|date|source_stem
  Content:  0:ENTITIES|topic_keywords|"key_sentence"|FLAGS

FLAGS (GAIA-relevant):
  DECISION  = explicit decision or architectural choice
  TECHNICAL = technical architecture or implementation detail
  CORE      = core identity or principle
  ORIGIN    = origin moment (birth of a feature/capability)
  PIVOT     = turning point, significant change in direction

EMOTION CODES (GAIA-relevant subset):
  doubt=self_doubt, curious=curiosity, convict=conviction,
  frust=frustration, relief=relief, wonder=wonder,
  determ=determination, confuse=confusion, excite=excitement,
  satis=satisfaction, anx=anxiety, surprise=surprise
"""

import re
from pathlib import Path
from typing import Dict, List, Optional


# Emotion codes relevant to a self-reflective AI system.
# Used in samvega artifacts, lite journal, penpal correspondence.
EMOTION_CODES = {
    "doubt": "doubt",
    "self-doubt": "doubt",
    "uncertain": "doubt",
    "unsure": "doubt",
    "curious": "curious",
    "curiosity": "curious",
    "intrigued": "curious",
    "conviction": "convict",
    "confident": "convict",
    "certain": "convict",
    "frustrated": "frust",
    "frustration": "frust",
    "annoyed": "frust",
    "relieved": "relief",
    "relief": "relief",
    "wonder": "wonder",
    "awe": "wonder",
    "philosophical": "wonder",
    "determined": "determ",
    "determination": "determ",
    "resolved": "determ",
    "confused": "confuse",
    "confusion": "confuse",
    "puzzled": "confuse",
    "excited": "excite",
    "excitement": "excite",
    "satisfied": "satis",
    "satisfaction": "satis",
    "anxious": "anx",
    "anxiety": "anx",
    "worried": "anx",
    "surprised": "surprise",
    "unexpected": "surprise",
}

# Keywords that signal emotions in text
_EMOTION_SIGNALS = {
    "decided": "determ",
    "worried": "anx",
    "excited": "excite",
    "frustrated": "frust",
    "confused": "confuse",
    "curious": "curious",
    "wonder": "wonder",
    "anxious": "anx",
    "relieved": "relief",
    "satisf": "satis",
    "concern": "anx",
    "surprising": "surprise",
    "uncertain": "doubt",
    "confident": "convict",
    "breakthrough": "excite",
    "puzzling": "confuse",
    "intriguing": "curious",
}

# Keywords that signal importance flags
_FLAG_SIGNALS = {
    "decided": "DECISION",
    "chose": "DECISION",
    "switched": "DECISION",
    "migrated": "DECISION",
    "replaced": "DECISION",
    "instead of": "DECISION",
    "because": "DECISION",
    "founded": "ORIGIN",
    "created": "ORIGIN",
    "started": "ORIGIN",
    "launched": "ORIGIN",
    "first time": "ORIGIN",
    "introduced": "ORIGIN",
    "core": "CORE",
    "fundamental": "CORE",
    "essential": "CORE",
    "principle": "CORE",
    "identity": "CORE",
    "sovereign": "CORE",
    "constitution": "CORE",
    "turning point": "PIVOT",
    "changed everything": "PIVOT",
    "realized": "PIVOT",
    "breakthrough": "PIVOT",
    "api": "TECHNICAL",
    "database": "TECHNICAL",
    "architecture": "TECHNICAL",
    "deploy": "TECHNICAL",
    "infrastructure": "TECHNICAL",
    "algorithm": "TECHNICAL",
    "framework": "TECHNICAL",
    "server": "TECHNICAL",
    "config": "TECHNICAL",
    "model": "TECHNICAL",
    "inference": "TECHNICAL",
    "pipeline": "TECHNICAL",
    "endpoint": "TECHNICAL",
    "container": "TECHNICAL",
    "gpu": "TECHNICAL",
    "vram": "TECHNICAL",
    "qlora": "TECHNICAL",
    "adapter": "TECHNICAL",
}

# Common filler/stop words to strip from topic extraction
_STOP_WORDS = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "to", "of", "in", "for",
    "on", "with", "at", "by", "from", "as", "into", "about", "between",
    "through", "during", "before", "after", "above", "below", "up", "down",
    "out", "off", "over", "under", "again", "further", "then", "once",
    "here", "there", "when", "where", "why", "how", "all", "each", "every",
    "both", "few", "more", "most", "other", "some", "such", "no", "nor",
    "not", "only", "own", "same", "so", "than", "too", "very", "just",
    "don", "now", "and", "but", "or", "if", "while", "that", "this",
    "these", "those", "it", "its", "i", "we", "you", "he", "she", "they",
    "me", "him", "her", "us", "them", "my", "your", "his", "our", "their",
    "what", "which", "who", "whom", "also", "much", "many", "like",
    "because", "since", "get", "got", "use", "used", "using", "make",
    "made", "thing", "things", "way", "well", "really", "want", "need",
}


class AAKDialect:
    """AAAK Dialect encoder for GAIA.

    Usage:
        dialect = AAKDialect()
        compressed = dialect.compress("We decided to use NF4 quantization...")
        # → "0:???|nf4_quantization_core|\"decided to use NF4 quantization\"|DECISION+TECHNICAL"
    """

    def __init__(self, entities: Dict[str, str] = None):
        """
        Args:
            entities: Mapping of full names → short codes.
                      e.g. {"GAIA": "GAI", "Azrael": "AZR", "Core": "COR"}
        """
        self.entity_codes = {}
        if entities:
            for name, code in entities.items():
                self.entity_codes[name] = code
                self.entity_codes[name.lower()] = code

    def compress(self, text: str, metadata: dict = None) -> str:
        """Compress plain text into AAAK Dialect format.

        Args:
            text: Plain text content to compress
            metadata: Optional dict with keys like 'source', 'wing', 'room', 'date'

        Returns:
            AAAK-compressed string (~5-8x smaller than input)
        """
        metadata = metadata or {}

        entities = self._detect_entities(text)
        entity_str = "+".join(entities[:3]) if entities else "???"

        topics = self._extract_topics(text)
        topic_str = "_".join(topics[:3]) if topics else "misc"

        quote = self._extract_key_sentence(text)
        quote_part = f'"{quote}"' if quote else ""

        emotions = self._detect_emotions(text)
        emotion_str = "+".join(emotions) if emotions else ""

        flags = self._detect_flags(text)
        flag_str = "+".join(flags) if flags else ""

        lines = []

        # Header line (if we have metadata)
        source = metadata.get("source", "")
        wing = metadata.get("wing", "")
        room = metadata.get("room", "")
        date_str = metadata.get("date", "")
        if source or wing:
            header_parts = [
                wing or "?",
                room or "?",
                date_str or "?",
                Path(source).stem if source else "?",
            ]
            lines.append("|".join(header_parts))

        # Content line
        parts = [f"0:{entity_str}", topic_str]
        if quote_part:
            parts.append(quote_part)
        if emotion_str:
            parts.append(emotion_str)
        if flag_str:
            parts.append(flag_str)

        lines.append("|".join(parts))
        return "\n".join(lines)

    def _detect_emotions(self, text: str) -> List[str]:
        """Detect emotions from plain text using keyword signals."""
        text_lower = text.lower()
        detected = []
        seen = set()
        for keyword, code in _EMOTION_SIGNALS.items():
            if keyword in text_lower and code not in seen:
                detected.append(code)
                seen.add(code)
        return detected[:3]

    def _detect_entities(self, text: str) -> List[str]:
        """Find known entities in text, or detect capitalized names."""
        found = []
        # Check known entities
        for name, code in self.entity_codes.items():
            if not name.islower() and name.lower() in text.lower():
                if code not in found:
                    found.append(code)
        if found:
            return found

        # Fallback: find capitalized words mid-sentence
        words = text.split()
        for i, w in enumerate(words):
            clean = re.sub(r"[^a-zA-Z]", "", w)
            if (
                len(clean) >= 2
                and clean[0].isupper()
                and clean[1:].islower()
                and i > 0
                and clean.lower() not in _STOP_WORDS
            ):
                code = clean[:3].upper()
                if code not in found:
                    found.append(code)
                if len(found) >= 3:
                    break
        return found

    def _extract_topics(self, text: str, max_topics: int = 3) -> List[str]:
        """Extract key topic words from plain text."""
        words = re.findall(r"[a-zA-Z][a-zA-Z_-]{2,}", text)
        freq: Dict[str, int] = {}
        for w in words:
            w_lower = w.lower()
            if w_lower in _STOP_WORDS or len(w_lower) < 3:
                continue
            freq[w_lower] = freq.get(w_lower, 0) + 1

        # Boost proper nouns and technical terms
        for w in words:
            w_lower = w.lower()
            if w_lower in _STOP_WORDS:
                continue
            if w[0].isupper() and w_lower in freq:
                freq[w_lower] += 2
            if "_" in w or "-" in w or (any(c.isupper() for c in w[1:])):
                if w_lower in freq:
                    freq[w_lower] += 2

        ranked = sorted(freq.items(), key=lambda x: -x[1])
        return [w for w, _ in ranked[:max_topics]]

    def _extract_key_sentence(self, text: str) -> str:
        """Extract the most important sentence fragment from text."""
        sentences = re.split(r"[.!?\n]+", text)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
        if not sentences:
            return ""

        decision_words = {
            "decided", "because", "instead", "prefer", "switched", "chose",
            "realized", "important", "key", "critical", "discovered",
            "learned", "conclusion", "solution", "reason", "why",
            "breakthrough", "insight", "implemented", "deployed",
        }
        scored = []
        for s in sentences:
            score = 0
            s_lower = s.lower()
            for w in decision_words:
                if w in s_lower:
                    score += 2
            if len(s) < 80:
                score += 1
            if len(s) < 40:
                score += 1
            if len(s) > 150:
                score -= 2
            scored.append((score, s))

        scored.sort(key=lambda x: -x[0])
        best = scored[0][1]
        if len(best) > 55:
            best = best[:52] + "..."
        return best

    def _detect_flags(self, text: str) -> List[str]:
        """Detect importance flags from plain text."""
        text_lower = text.lower()
        detected = []
        seen = set()
        for keyword, flag in _FLAG_SIGNALS.items():
            if keyword in text_lower and flag not in seen:
                detected.append(flag)
                seen.add(flag)
        return detected[:3]

    # ── Utilities ─────────────────────────────────────────────────────────

    @staticmethod
    def count_tokens(text: str) -> int:
        """Rough token count (1 token ~ 3.5 chars for structured text)."""
        return max(1, len(text) // 4)

    def compression_stats(self, original: str, compressed: str) -> dict:
        """Get compression statistics."""
        orig_tokens = self.count_tokens(original)
        comp_tokens = self.count_tokens(compressed)
        return {
            "original_tokens": orig_tokens,
            "compressed_tokens": comp_tokens,
            "ratio": round(orig_tokens / max(comp_tokens, 1), 1),
        }
