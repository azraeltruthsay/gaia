"""
Semantic Probe — Pre-cognition vector lookup for context discovery.

Runs BEFORE intent detection and persona selection. Extracts interesting
phrases from user input, probes all indexed vector collections, and returns
hits that inform downstream routing (persona, intent, RAG, prompt building).

Design doc: /knowledge/Dev_Notebook/2026-02-10_semantic_probe_plan.md
"""

import logging
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger("GAIA.SemanticProbe")


# ---------------------------------------------------------------------------
# Configuration — reads from gaia_constants.json SEMANTIC_PROBE section,
# falls back to hardcoded defaults if config is unavailable.
# ---------------------------------------------------------------------------
def _load_probe_config() -> Dict:
    """Load SEMANTIC_PROBE config from gaia_constants, with safe fallback."""
    try:
        from gaia_common.config.config_manager import get_config
        cfg = get_config()
        return cfg.constants.get("SEMANTIC_PROBE", {})
    except Exception:
        return {}


_PROBE_CFG = _load_probe_config()

# ---------------------------------------------------------------------------
# Common-word filter: skip these when extracting "interesting" phrases.
# Kept deliberately small — we want over-inclusion rather than missed entities.
# ---------------------------------------------------------------------------
_COMMON_WORDS = frozenset({
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "as", "is", "was", "are", "were", "be",
    "been", "being", "have", "has", "had", "do", "does", "did", "will",
    "would", "could", "should", "may", "might", "shall", "can", "need",
    "must", "it", "its", "i", "me", "my", "we", "us", "our", "you",
    "your", "he", "him", "his", "she", "her", "they", "them", "their",
    "this", "that", "these", "those", "what", "which", "who", "whom",
    "how", "when", "where", "why", "if", "then", "so", "not", "no",
    "yes", "all", "each", "every", "any", "some", "just", "about",
    "up", "out", "into", "over", "after", "before", "between", "under",
    "again", "there", "here", "once", "also", "very", "much", "more",
    "most", "only", "than", "too", "now", "get", "got", "go", "went",
    "come", "came", "make", "made", "take", "took", "know", "knew",
    "think", "thought", "say", "said", "tell", "told", "give", "gave",
    "see", "saw", "look", "find", "found", "want", "like", "new",
    "old", "good", "bad", "first", "last", "long", "great", "little",
    "own", "other", "right", "still", "try", "use", "even", "back",
    "way", "well", "because", "thing", "things", "many", "then", "them",
    "same", "different", "around", "help", "through", "while", "such",
    "let", "keep", "end", "does", "set", "put", "kind", "off", "both",
    "down", "ask", "going", "show", "mean", "part", "place", "people",
    "really", "actually", "already", "though", "yet", "during",
    # Chat/command words GAIA sees frequently
    "hey", "hi", "hello", "gaia", "please", "thanks", "thank",
    "okay", "ok", "sure", "yeah", "yep", "nope",
    "what's", "what", "happened", "tell", "me", "about", "did",
    "can", "you", "do", "anything",
    # Words that appear conversational but aren't entities
    "system", "logs", "check", "update", "updated", "today", "yesterday",
    "tomorrow", "time", "work", "working", "start", "stop", "run", "running",
    "read", "write", "send", "change", "changed", "move", "moved",
    "talk", "talking", "context", "information", "question", "answer",
    "character", "sheet", "spell", "spells", "level", "prepared",
    "weather", "status", "error", "message", "problem", "issue",
    "build", "test", "deploy", "server", "file", "folder", "data",
    "config", "setting", "settings", "option", "options", "feature",
    "code", "function", "class", "method", "module", "package",
    "name", "number", "type", "list", "item", "value", "result",
})

# Max phrases to embed per probe (performance cap)
_MAX_PHRASES = _PROBE_CFG.get("max_phrases", 8)

# Min phrase length (skip noise)
_MIN_PHRASE_LEN = _PROBE_CFG.get("min_phrase_len", 3)

# Similarity threshold — below this, a hit is noise
SIMILARITY_THRESHOLD = _PROBE_CFG.get("similarity_threshold", 0.40)

# Short-circuit: skip probe for these reflex commands
_REFLEX_COMMANDS = frozenset({"exit", "quit", "bye", "help", "h", "status", "list_tools", ""})

# Min word count to bother probing
_MIN_WORDS = _PROBE_CFG.get("min_words_to_probe", 3)

# Cache TTL in turns
_CACHE_MAX_AGE = _PROBE_CFG.get("cache_max_age_turns", 10)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class ProbeHit:
    """A single vector match from the probe."""
    phrase: str              # The extracted phrase that matched
    collection: str          # Which knowledge base (e.g. "dnd_campaign")
    chunk_text: str          # Matched chunk (truncated)
    similarity: float        # Cosine similarity score
    filename: str            # Source document filename
    chunk_idx: int = 0       # Position in source document
    confidence_tier: str = ""  # Epistemic confidence tier (e.g. "verified", "curated")

    def to_dict(self) -> dict:
        d = {
            "phrase": self.phrase,
            "collection": self.collection,
            "chunk_text": self.chunk_text[:300],
            "similarity": round(self.similarity, 4),
            "filename": self.filename,
            "chunk_idx": self.chunk_idx,
        }
        if self.confidence_tier:
            d["confidence_tier"] = self.confidence_tier
        return d


@dataclass
class SemanticProbeResult:
    """Aggregated result from probing all collections."""
    hits: List[ProbeHit] = field(default_factory=list)
    primary_collection: Optional[str] = None
    supplemental_collections: List[str] = field(default_factory=list)
    probe_time_ms: float = 0.0
    phrases_tested: List[str] = field(default_factory=list)
    from_cache: int = 0

    def to_dict(self) -> dict:
        return {
            "hits": [h.to_dict() for h in self.hits],
            "primary_collection": self.primary_collection,
            "supplemental_collections": self.supplemental_collections,
            "probe_time_ms": round(self.probe_time_ms, 2),
            "phrases_tested": self.phrases_tested,
            "from_cache": self.from_cache,
        }

    @property
    def has_hits(self) -> bool:
        return len(self.hits) > 0

    def to_metrics_dict(self) -> Dict:
        """Produce a compact metrics summary for CognitionPacket.metrics.semantic_probe."""
        if not self.hits:
            return {
                "skipped": not self.phrases_tested,
                "phrases_extracted": len(self.phrases_tested),
                "total_hits": 0,
                "probe_time_ms": round(self.probe_time_ms, 2),
                "from_cache": self.from_cache,
            }

        sims = [h.similarity for h in self.hits]
        collections_hit = list({h.collection for h in self.hits})
        unique_phrases_matched = len({h.phrase.lower() for h in self.hits})

        return {
            "skipped": False,
            "phrases_extracted": len(self.phrases_tested),
            "phrases_matched": unique_phrases_matched,
            "total_hits": len(self.hits),
            "primary_collection": self.primary_collection,
            "supplemental_collections": self.supplemental_collections,
            "collections_hit": len(collections_hit),
            "similarity_avg": round(sum(sims) / len(sims), 4),
            "similarity_max": round(max(sims), 4),
            "similarity_min": round(min(sims), 4),
            "probe_time_ms": round(self.probe_time_ms, 2),
            "from_cache": self.from_cache,
            "threshold": SIMILARITY_THRESHOLD,
        }


@dataclass
class SessionProbeCache:
    """Per-session cache of phrase → probe hits with turn-based eviction."""
    phrase_hits: Dict[str, List[ProbeHit]] = field(default_factory=dict)
    turn_ages: Dict[str, int] = field(default_factory=dict)
    current_turn: int = 0
    max_age: int = _CACHE_MAX_AGE

    def get(self, phrase: str) -> Optional[List[ProbeHit]]:
        """Return cached hits if phrase is cached and not expired."""
        if phrase in self.phrase_hits:
            age = self.current_turn - self.turn_ages.get(phrase, 0)
            if age <= self.max_age:
                return self.phrase_hits[phrase]
            else:
                # Expired — evict
                del self.phrase_hits[phrase]
                del self.turn_ages[phrase]
        return None

    def put(self, phrase: str, hits: List[ProbeHit]):
        """Cache hits for a phrase at the current turn."""
        self.phrase_hits[phrase] = hits
        self.turn_ages[phrase] = self.current_turn

    def advance_turn(self):
        """Increment turn counter and evict stale entries."""
        self.current_turn += 1
        expired = [p for p, t in self.turn_ages.items()
                   if self.current_turn - t > self.max_age]
        for p in expired:
            self.phrase_hits.pop(p, None)
            self.turn_ages.pop(p, None)


@dataclass
class ProbeSessionStats:
    """Cumulative probe effectiveness stats for a session."""
    total_probes: int = 0
    probes_with_hits: int = 0
    probes_skipped: int = 0
    total_hits: int = 0
    total_phrases_extracted: int = 0
    total_phrases_matched: int = 0
    total_probe_time_ms: float = 0.0
    total_cache_hits: int = 0
    collections_seen: Dict[str, int] = field(default_factory=dict)  # collection → hit count

    def record(self, result: 'SemanticProbeResult', was_skipped: bool = False):
        """Record stats from a single probe invocation."""
        self.total_probes += 1
        if was_skipped:
            self.probes_skipped += 1
            return
        self.total_phrases_extracted += len(result.phrases_tested)
        self.total_probe_time_ms += result.probe_time_ms
        self.total_cache_hits += result.from_cache
        if result.has_hits:
            self.probes_with_hits += 1
            self.total_hits += len(result.hits)
            self.total_phrases_matched += len({h.phrase.lower() for h in result.hits})
            for h in result.hits:
                self.collections_seen[h.collection] = self.collections_seen.get(h.collection, 0) + 1

    @property
    def hit_rate(self) -> float:
        """Fraction of non-skipped probes that found at least one hit."""
        active = self.total_probes - self.probes_skipped
        return self.probes_with_hits / active if active > 0 else 0.0

    @property
    def avg_probe_time_ms(self) -> float:
        active = self.total_probes - self.probes_skipped
        return self.total_probe_time_ms / active if active > 0 else 0.0

    def to_dict(self) -> Dict:
        return {
            "total_probes": self.total_probes,
            "probes_with_hits": self.probes_with_hits,
            "probes_skipped": self.probes_skipped,
            "hit_rate": round(self.hit_rate, 4),
            "total_hits": self.total_hits,
            "total_phrases_extracted": self.total_phrases_extracted,
            "total_phrases_matched": self.total_phrases_matched,
            "avg_probe_time_ms": round(self.avg_probe_time_ms, 2),
            "total_cache_hits": self.total_cache_hits,
            "collections_seen": self.collections_seen,
        }


# Per-session cache instances (session_id → cache)
_session_caches: Dict[str, SessionProbeCache] = {}

# Per-session stats (session_id → stats)
_session_stats: Dict[str, ProbeSessionStats] = {}


def _get_session_cache(session_id: str) -> SessionProbeCache:
    if session_id not in _session_caches:
        _session_caches[session_id] = SessionProbeCache()
    return _session_caches[session_id]


def _get_session_stats(session_id: str) -> ProbeSessionStats:
    if session_id not in _session_stats:
        _session_stats[session_id] = ProbeSessionStats()
    return _session_stats[session_id]


def get_session_probe_stats(session_id: str) -> Optional[Dict]:
    """Public accessor: return cumulative probe stats for a session, or None."""
    stats = _session_stats.get(session_id)
    return stats.to_dict() if stats else None


# ---------------------------------------------------------------------------
# Phase 1a: Phrase extraction
# ---------------------------------------------------------------------------

def extract_candidate_phrases(text: str) -> List[str]:
    """
    Extract interesting phrases from user input for vector probing.

    Strategy (all pure regex/set ops — no model calls):
    1. Quoted strings — "the Maid", 'BlueShot'
    2. Capitalized sequences — Rogue's End, Tower Faction, Jade Phoenix
    3. D&D notation — AC 15, d20, DC 18
    4. Rare words — words not in the common-word filter

    Returns deduplicated list, capped at _MAX_PHRASES.
    """
    if not text or not text.strip():
        return []

    phrases: List[str] = []
    seen_lower: set = set()

    def _add(phrase: str):
        """Add phrase if it meets minimum length and isn't a duplicate."""
        p = phrase.strip()
        if len(p) >= _MIN_PHRASE_LEN and p.lower() not in seen_lower:
            seen_lower.add(p.lower())
            phrases.append(p)

    # 1. Quoted strings: "foo bar" or 'foo bar'
    for m in re.finditer(r'''["']([^"']{3,60})["']''', text):
        _add(m.group(1))

    # 2. Capitalized multi-word sequences (proper nouns / named entities)
    #    Matches: "Rogue's End", "Tower Faction", "Jade Phoenix Order",
    #    "UR Machine", "R.S.S. Alice"
    #    Also handles possessives, all-caps words, and acronyms with periods.
    for m in re.finditer(
        r"\b([A-Z][A-Za-z.]+(?:['']\w+)?(?:\s+(?:of|the|and|in|on|at|de|von|van)\s+)?(?:\s+[A-Z][A-Za-z.]+(?:['']\w+)?)+)\b",
        text
    ):
        _add(m.group(1))

    # 3. Single capitalized words that aren't sentence-initial
    #    (sentence-initial words start after . or at position 0)
    #    Skip very common capitalized words.
    words = text.split()
    for i, word in enumerate(words):
        # Strip trailing punctuation for matching
        clean = re.sub(r"[.,!?;:\"']+$", "", word)
        if not clean or len(clean) < _MIN_PHRASE_LEN:
            continue
        # Must start with uppercase
        if not clean[0].isupper():
            continue
        # Skip sentence-initial (index 0, or preceded by sentence-ending punct)
        if i == 0:
            continue
        prev_word = words[i - 1] if i > 0 else ""
        if prev_word and prev_word[-1] in ".!?":
            continue
        # Skip if it's a common word (case-insensitive)
        if clean.lower() in _COMMON_WORDS:
            continue
        _add(clean)

    # 4. D&D notation patterns: AC 15, DC 18, d20, 2d6+3, etc.
    for m in re.finditer(r"\b(?:AC|DC|HP|XP)\s*\d+\b", text):
        _add(m.group(0))
    for m in re.finditer(r"\b\d*d\d+(?:[+-]\d+)?\b", text):
        _add(m.group(0))

    # 5. Rare standalone words — words not in common set, at least 4 chars,
    #    all lowercase (catches things like "strauthauk", "heimr", "excalibur")
    for word in words:
        clean = re.sub(r"[.,!?;:\"']+$", "", word).strip()
        if (len(clean) >= 4
                and clean.lower() not in _COMMON_WORDS
                and clean.lower() not in seen_lower
                and clean.isalpha()
                and clean[0].islower()):
            _add(clean)

    # Dedup: remove single words that are substrings of already-extracted
    # multi-word phrases (e.g., "Tower" is redundant when "Tower Faction" exists)
    multi_word = {p.lower() for p in phrases if " " in p}
    if multi_word:
        phrases = [
            p for p in phrases
            if " " in p or not any(p.lower() in mw for mw in multi_word)
        ]

    # Cap at max phrases, preferring earlier (higher-signal) extractions
    return phrases[:_MAX_PHRASES]


# ---------------------------------------------------------------------------
# Phase 1b: Multi-collection probing
# ---------------------------------------------------------------------------

# Map collection name → default epistemic confidence tier (used when
# the vector index entry doesn't carry its own confidence_tier field).
_COLLECTION_TIER_MAP = {
    "system": "verified",
    "blueprints": "curated",
    "dnd_campaign": "curated",
    "research": "researched",
}


def _probe_single_collection(
    phrases: List[str],
    collection_name: str,
    top_k: int = 3,
) -> List[ProbeHit]:
    """
    Probe a single vector collection with the given phrases.

    Uses VectorIndexer.instance() for the collection. Each phrase is queried
    independently and results above SIMILARITY_THRESHOLD are kept.
    """
    try:
        from gaia_common.utils.vector_indexer import VectorIndexer
        indexer = VectorIndexer.instance(collection_name)
    except Exception as e:
        logger.debug(
            "SemanticProbe: could not load VectorIndexer for '%s': %s",
            collection_name, e
        )
        return []

    if not indexer.index.get("docs"):
        logger.debug("SemanticProbe: collection '%s' has empty index", collection_name)
        return []

    fallback_tier = _COLLECTION_TIER_MAP.get(collection_name, "curated")

    hits: List[ProbeHit] = []
    for phrase in phrases:
        try:
            results = indexer.query(phrase, top_k=top_k)
            for r in results:
                score = r.get("score", 0.0)
                if score >= SIMILARITY_THRESHOLD:
                    hits.append(ProbeHit(
                        phrase=phrase,
                        collection=collection_name,
                        chunk_text=r.get("text", "")[:300],
                        similarity=score,
                        filename=r.get("filename", ""),
                        chunk_idx=r.get("idx", 0),
                        confidence_tier=r.get("confidence_tier") or fallback_tier,
                    ))
        except Exception as e:
            logger.debug(
                "SemanticProbe: query failed for phrase '%s' in '%s': %s",
                phrase, collection_name, e
            )
    return hits


def _determine_primary_and_supplemental(
    hits: List[ProbeHit],
) -> Tuple[Optional[str], List[str]]:
    """
    Given all hits across collections, determine primary and supplemental.

    Primary = collection with highest aggregate similarity score.
    Supplemental = other collections that had any hits.
    """
    if not hits:
        return None, []

    # Aggregate score per collection
    collection_scores: Dict[str, float] = {}
    collection_hit_count: Dict[str, int] = {}
    for h in hits:
        collection_scores[h.collection] = collection_scores.get(h.collection, 0.0) + h.similarity
        collection_hit_count[h.collection] = collection_hit_count.get(h.collection, 0) + 1

    # Sort by aggregate score (tie-break on hit count)
    ranked = sorted(
        collection_scores.keys(),
        key=lambda c: (collection_scores[c], collection_hit_count[c]),
        reverse=True,
    )

    primary = ranked[0]
    supplemental = [c for c in ranked[1:]]
    return primary, supplemental


def probe_collections(
    phrases: List[str],
    knowledge_bases: Dict[str, dict],
    session_id: str = "",
    top_k_per_phrase: int = 3,
) -> SemanticProbeResult:
    """
    Probe all configured knowledge base collections with extracted phrases.

    Args:
        phrases: Candidate phrases from extract_candidate_phrases()
        knowledge_bases: The KNOWLEDGE_BASES dict from gaia_constants.json
        session_id: For session-level caching
        top_k_per_phrase: Max results per phrase per collection

    Returns:
        SemanticProbeResult with hits, primary/supplemental collections, timing.
    """
    t0 = time.perf_counter()

    cache = _get_session_cache(session_id) if session_id else None
    if cache:
        cache.advance_turn()

    all_hits: List[ProbeHit] = []
    phrases_to_probe: List[str] = []
    from_cache_count = 0

    # Separate cached vs. new phrases
    for phrase in phrases:
        if cache:
            cached = cache.get(phrase)
            if cached is not None:
                all_hits.extend(cached)
                from_cache_count += 1
                continue
        phrases_to_probe.append(phrase)

    # Probe each collection with new phrases
    collection_names = list(knowledge_bases.keys())
    for cname in collection_names:
        if not phrases_to_probe:
            break
        try:
            hits = _probe_single_collection(phrases_to_probe, cname, top_k_per_phrase)
            all_hits.extend(hits)

            # Cache results per phrase
            if cache:
                # Group hits by phrase for this collection
                phrase_hit_map: Dict[str, List[ProbeHit]] = {}
                for h in hits:
                    phrase_hit_map.setdefault(h.phrase, []).append(h)
                for phrase in phrases_to_probe:
                    cached_existing = cache.get(phrase)
                    new_hits = phrase_hit_map.get(phrase, [])
                    if cached_existing:
                        # Merge with any hits already cached from other collections
                        cache.put(phrase, cached_existing + new_hits)
                    else:
                        cache.put(phrase, new_hits)
        except Exception as e:
            logger.warning("SemanticProbe: collection '%s' probe failed: %s", cname, e)

    # Deduplicate hits (same phrase+collection+filename+chunk_idx)
    seen_keys = set()
    deduped: List[ProbeHit] = []
    for h in all_hits:
        key = (h.phrase.lower(), h.collection, h.filename, h.chunk_idx)
        if key not in seen_keys:
            seen_keys.add(key)
            deduped.append(h)

    # Sort by similarity descending
    deduped.sort(key=lambda h: h.similarity, reverse=True)

    primary, supplemental = _determine_primary_and_supplemental(deduped)

    elapsed_ms = (time.perf_counter() - t0) * 1000

    result = SemanticProbeResult(
        hits=deduped,
        primary_collection=primary,
        supplemental_collections=supplemental,
        probe_time_ms=elapsed_ms,
        phrases_tested=phrases,
        from_cache=from_cache_count,
    )

    logger.info(
        "SemanticProbe: %d phrases → %d hits across %d collections "
        "(primary=%s, supplemental=%s, cache=%d, %.1fms)",
        len(phrases), len(deduped), len(collection_names),
        primary, supplemental, from_cache_count, elapsed_ms,
    )

    return result


# ---------------------------------------------------------------------------
# Top-level orchestrator
# ---------------------------------------------------------------------------

def should_skip_probe(user_input: str) -> bool:
    """Check short-circuit rules: skip probe for trivial inputs."""
    stripped = (user_input or "").strip()

    # Empty or reflex command
    if stripped.lower() in _REFLEX_COMMANDS:
        return True

    # Too short to be meaningful
    word_count = len(stripped.split())
    if word_count < _MIN_WORDS:
        return True

    return False


def run_semantic_probe(
    user_input: str,
    knowledge_bases: Dict[str, dict],
    session_id: str = "",
    top_k_per_phrase: int = _PROBE_CFG.get("top_k_per_phrase", 3),
) -> SemanticProbeResult:
    """
    Top-level entry point: extract phrases, probe collections, return results.

    Call this from agent_core.run_turn() before persona selection and intent
    detection. Wire the result into the CognitionPacket as a DataField.

    Args:
        user_input: Raw user message
        knowledge_bases: KNOWLEDGE_BASES dict from config/constants
        session_id: Session identifier for caching
        top_k_per_phrase: Max results per phrase per collection

    Returns:
        SemanticProbeResult (may be empty if short-circuited or no hits)
    """
    stats = _get_session_stats(session_id) if session_id else None

    if should_skip_probe(user_input):
        logger.debug("SemanticProbe: skipped (short-circuit)")
        result = SemanticProbeResult()
        if stats:
            stats.record(result, was_skipped=True)
        return result

    phrases = extract_candidate_phrases(user_input)
    if not phrases:
        logger.debug("SemanticProbe: no interesting phrases extracted")
        result = SemanticProbeResult(phrases_tested=[])
        if stats:
            stats.record(result)
        return result

    logger.info("SemanticProbe: extracted %d phrases: %s", len(phrases), phrases)

    result = probe_collections(
        phrases=phrases,
        knowledge_bases=knowledge_bases,
        session_id=session_id,
        top_k_per_phrase=top_k_per_phrase,
    )

    if stats:
        stats.record(result)
        if stats.total_probes % 10 == 0:
            logger.info(
                "SemanticProbe session stats [%s]: %d probes, hit_rate=%.0f%%, "
                "avg_time=%.1fms, cache_hits=%d",
                session_id[:8] if session_id else "?",
                stats.total_probes, stats.hit_rate * 100,
                stats.avg_probe_time_ms, stats.total_cache_hits,
            )

    return result
