"""
MemPalace — Structured Memory Architecture for GAIA.

Orchestrates the full memory pipeline:
  general_extractor (classify) → AAAK dialect (compress) → KG (entities) → disk (persist)

The palace is organized spatially:
  Palace → Wings → Rooms → Memories (AAAK files)

Each memory flows through:
  1. Classification via general_extractor → memory_type
  2. Routing via type_to_room config → wing/room
  3. Compression via AAAK dialect → symbolic form
  4. Persistence to wing/room directory on disk
  5. Entity extraction → KG triples

Usage:
    from gaia_common.utils.mempalace import MemPalace
    from gaia_common.config import Config

    config = Config()
    palace = MemPalace(config.constants.get("MEMPALACE", {}))
    result = palace.store("We decided to use NF4 quantization for Core.", source="conversation")
    results = palace.recall("quantization")
    layout = palace.navigate()
    stats = palace.status()
"""

import hashlib
import logging
import os
import re
from datetime import date, datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from gaia_common.utils.aaak_dialect import AAKDialect
from gaia_common.utils.general_extractor import extract_memories
from gaia_common.utils.knowledge_graph import KnowledgeGraph

logger = logging.getLogger("GAIA.MemPalace")


class MemPalace:
    """Structured memory palace for GAIA — spatial memory with compression and KG."""

    def __init__(self, config: dict):
        """Initialize from MEMPALACE section of gaia_constants.

        Args:
            config: The MEMPALACE dict from gaia_constants.json.
        """
        self._config = config
        self._root_dir = Path(config.get("root_dir", "/knowledge/mempalace"))
        self._wings = config.get("wings", {})
        self._type_to_room = config.get("type_to_room", {})
        self._entities = config.get("entities", {})

        # Initialize sub-components
        kg_path = config.get("kg_path")
        self._kg = KnowledgeGraph(db_path=kg_path)
        self._dialect = AAKDialect(entities=self._entities)

        # Create palace directory structure
        self._ensure_directories()

        # Seed KG if empty
        try:
            stats = self._kg.stats()
            if stats.get("entities", 0) == 0:
                from gaia_common.config import Config
                constants = Config().constants
                self._kg.seed_from_gaia_constants(constants)
                logger.info("KG seeded from constants on first palace init")
        except Exception as e:
            logger.warning("Could not seed KG: %s", e)

        logger.info(
            "MemPalace initialized: root=%s, wings=%d, rooms=%d",
            self._root_dir,
            len(self._wings),
            sum(len(w.get("rooms", {})) for w in self._wings.values()),
        )

    def _ensure_directories(self):
        """Create wing/room directories if they don't exist."""
        self._root_dir.mkdir(parents=True, exist_ok=True)
        for wing_name, wing_def in self._wings.items():
            wing_dir = self._root_dir / wing_name
            wing_dir.mkdir(exist_ok=True)
            for room_name in wing_def.get("rooms", {}):
                (wing_dir / room_name).mkdir(exist_ok=True)

    # ── Store ────────────────────────────────────────────────────────────

    def store(self, text: str, source: str = "unknown", date_str: str = None) -> dict:
        """Store a memory through the full pipeline.

        1. Classify via general_extractor -> memory_type
        2. Map memory_type -> wing/room via type_to_room config
        3. Compress via AAAK dialect -> symbolic form
        4. Write AAAK file to wing/room directory on disk
        5. Extract basic entities and add to KG
        6. Return {palace_id, wing, room, compressed_size, raw_size}
        """
        if not text or not text.strip():
            return {"ok": False, "error": "Empty text"}

        date_str = date_str or date.today().isoformat()

        # 1. Classify
        memories = extract_memories(text)
        if not memories:
            # If extractor finds nothing, default to technical
            memory_type = "technical"
        else:
            # Use the type of the first (highest-confidence) memory
            memory_type = memories[0].get("memory_type", "technical")

        # 2. Route to wing/room
        room_path = self._type_to_room.get(memory_type, "operational/preferences")
        parts = room_path.split("/", 1)
        wing = parts[0]
        room = parts[1] if len(parts) > 1 else "misc"

        # Validate wing/room exist in config, fallback if not
        if wing not in self._wings:
            wing = "operational"
        if room not in self._wings.get(wing, {}).get("rooms", {}):
            # Use first room in wing as fallback
            rooms = list(self._wings.get(wing, {}).get("rooms", {}).keys())
            room = rooms[0] if rooms else "misc"

        # 3. Compress via AAAK
        metadata = {
            "source": source,
            "wing": wing,
            "room": room,
            "date": date_str,
        }
        compressed = self._dialect.compress(text, metadata=metadata)

        # 4. Write to disk
        palace_id = self._generate_id(text, date_str)
        room_dir = self._root_dir / wing / room
        room_dir.mkdir(parents=True, exist_ok=True)
        file_path = room_dir / f"{palace_id}.aaak"
        file_path.write_text(compressed, encoding="utf-8")

        # 5. Extract entities and add to KG
        triples_added = self._extract_and_store_triples(
            text, memory_type, wing, room, source, date_str, palace_id
        )

        raw_size = len(text.encode("utf-8"))
        compressed_size = len(compressed.encode("utf-8"))
        stats = self._dialect.compression_stats(text, compressed)

        logger.info(
            "Stored memory: %s/%s/%s (type=%s, ratio=%sx, triples=%d)",
            wing, room, palace_id, memory_type, stats["ratio"], triples_added,
        )

        return {
            "ok": True,
            "palace_id": palace_id,
            "wing": wing,
            "room": room,
            "memory_type": memory_type,
            "raw_size": raw_size,
            "compressed_size": compressed_size,
            "compression_ratio": stats["ratio"],
            "triples_added": triples_added,
        }

    def _generate_id(self, text: str, date_str: str) -> str:
        """Generate a deterministic but unique palace ID."""
        content_hash = hashlib.md5(
            f"{text}{date_str}{datetime.now().isoformat()}".encode()
        ).hexdigest()[:10]
        return f"{date_str}_{content_hash}"

    def _extract_and_store_triples(
        self,
        text: str,
        memory_type: str,
        wing: str,
        room: str,
        source: str,
        date_str: str,
        palace_id: str,
    ) -> int:
        """Extract entities from text and add basic KG triples.

        World Model Stage 0 (hrp) — rewrote the entity-detection path
        to stop polluting the KG with token-fragment 'entities' like
        'sup', 'bow', 'new' (truncated 3-char codes from AAAK's
        compression layer). AAAK codes are useful for symbolic
        compression but they are NOT entity names; storing them as KG
        subjects degrades every downstream query.

        New approach:
        - Use proper named-entity extraction (multi-word Title Case
          spans, 4+ char ALL-CAPS acronyms) instead of AAAK codes
        - Stopword filtering rejects sentence-openers ('Between',
          'Speaking', possessive pronouns, etc.)
        - Minimum length 5 chars for single-word entities (raised
          from 2 — the old threshold accepted 'SUP', 'BOW')
        - Drop topic-based 'related_to' triples entirely. Topics are
          frequency-ranked common nouns; they make terrible KG entities.
          (Real topical relationships should come from semantic
          inference, not bag-of-words frequency.)
        - Confidence reflects extraction certainty: 1.0 for known-
          registry matches, 0.7 for fallback-detected names.
        """
        count = 0
        try:
            # Don't store palace_id triples — palace_id is a synthetic
            # opaque ID, not a real entity. Keep palace<->location
            # mapping in the AAAK files themselves, not in the KG.
            # (Removed: stored_in and memory_type triples.)

            entities = self._extract_kg_entities(text)
            seen_subjects: set = set()
            for entity_name, confidence in entities[:5]:
                # Dedupe per-text
                key = entity_name.lower()
                if key in seen_subjects:
                    continue
                seen_subjects.add(key)

                self._kg.add_triple(
                    entity_name, "mentioned_in", palace_id,
                    valid_from=date_str, source=source,
                    confidence=confidence,
                )
                count += 1

        except Exception as e:
            logger.warning("KG triple extraction failed: %s", e)

        return count

    # Stopwords for fallback entity detection — pronouns, sentence-
    # openers, common acronyms that shouldn't be KG entities.
    _ENTITY_STOPWORDS: "frozenset[str]" = frozenset({
        "The", "This", "That", "These", "Those", "Their", "There", "Then",
        "When", "Where", "What", "Which", "Who", "Whom", "Whose", "Why", "How",
        "His", "Her", "Its", "Our", "Your", "My",
        "He", "She", "It", "We", "You", "I", "They",
        "And", "But", "Or", "If", "So", "Also", "Just", "Only", "Both", "Each",
        "Some", "Most", "Many", "Any", "All", "None", "Every", "Other",
        "Such", "Same", "Said", "Note", "Yes", "No", "OK", "Okay",
        "Between", "Among", "Within", "Without", "Through", "Across",
        "Above", "Below", "Beyond", "After", "Before", "During", "Since",
        "Until", "While", "Although", "Though", "However", "Therefore",
        "Indeed", "Sorry", "Thanks", "Maybe", "Perhaps", "Often", "Always",
        "Never", "Sometimes", "Usually", "Likely", "Possibly", "Actually",
        "Especially", "Generally", "Specifically", "Currently",
        "Speaking", "Considering", "Beyond", "Regarding", "Concerning",
        "Including", "Excluding", "Apart", "Aside", "Otherwise",
        "Let", "Make", "Take", "Give", "Find", "Look", "Show", "Tell",
        "Have", "Has", "Had", "Will", "Would", "Could", "Should", "Might",
        "Must", "Need", "Want", "Like", "Love", "Hate", "Feel", "Think",
        "Believe", "Know", "Understand", "Remember", "Forget", "Imagine",
        # NOTE: system identifiers (GAIA, Prime, Core, Nano, AI, Azrael)
        # are NOT in this list — they are legitimate KG entities for
        # mempalace extraction. The consistency_detector has a similar
        # stopword list where these ARE filtered (to prevent self-
        # references from being flagged as confabulation), but that's
        # the opposite goal.
        "User", "Message",
        "API", "URL", "PDT", "UTC", "JSON", "HTTP", "HTTPS", "MCP",
        "GPU", "CPU", "RAM", "PST", "CST", "EST", "TLS", "SSL", "TCP", "UDP",
    })

    def _extract_kg_entities(self, text: str) -> List[Tuple[str, float]]:
        """Extract candidate entities suitable as KG subjects.

        Returns list of (name, confidence) tuples. Two passes:
          1. Known-registry match — high confidence (1.0)
          2. Multi-word Title Case + long acronym fallback — moderate
             confidence (0.7), gated by stopwords and minimum length

        Single Title Case words are NOT accepted. They produce too
        much noise from sentence-initial capitalizations ('Stoic',
        'Roman', 'Egyptian') that are common-knowledge concepts
        rather than discrete named entities.
        """
        if not text or len(text) < 5:
            return []

        out: List[Tuple[str, float]] = []
        seen: "set[str]" = set()

        # Pass 1: known entities from the AAAK registry. These are
        # canonical names the system has explicitly curated.
        for name in self._entities.keys():
            if not name or name.islower():
                continue  # skip lowercase registry codes
            if name.lower() in text.lower():
                key = name.lower()
                if key not in seen:
                    seen.add(key)
                    out.append((name, 1.0))

        # Pass 2: fallback extraction — multi-word Title Case spans.
        # Bounded by sentence to avoid matching "Oregon. Current".
        sentence_split = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")
        multi_title = re.compile(
            r"\b(?:[A-Z][a-zA-Z]{1,}\.?\s+){1,3}[A-Z][a-zA-Z]{1,}\b"
        )
        for sent in sentence_split.split(text):
            for m in multi_title.finditer(sent):
                surface = m.group(0).strip().rstrip(".")
                if len(surface) < 5:
                    continue
                first = surface.split(None, 1)[0]
                if first in self._ENTITY_STOPWORDS:
                    # Drop sentence-opener; if remainder is still
                    # multi-word, try that.
                    remainder = surface.split(None, 1)[1] if " " in surface else ""
                    if remainder and " " in remainder and len(remainder) >= 5:
                        surface = remainder
                    else:
                        continue
                key = surface.lower()
                if key not in seen:
                    seen.add(key)
                    out.append((surface, 0.7))

        # Pass 3: 4+ char ALL-CAPS acronyms (skip stopword set).
        acronym = re.compile(r"\b[A-Z]{4,}(?:[-_][A-Z0-9]+)?\b")
        for m in acronym.finditer(text):
            surface = m.group(0)
            if surface in self._ENTITY_STOPWORDS:
                continue
            key = surface.lower()
            if key not in seen:
                seen.add(key)
                out.append((surface, 0.7))

        return out

    def _reverse_entity(self, code: str) -> Optional[str]:
        """Reverse-lookup: entity code -> entity name.

        Retained for backwards compatibility with callers that still use
        AAAK codes. Returns None for unknown codes rather than passing
        them through (the old "return code if len(code) >= 2" behavior
        is what produced the 'SUP'/'BOW'/'NEW' KG pollution).
        """
        for name, c in self._entities.items():
            if c == code:
                return name
        return None

    # ── Recall ───────────────────────────────────────────────────────────

    def recall(self, query: str, top_k: int = 5) -> dict:
        """Semantic recall with KG enrichment.

        1. Search stored memories for query matches (text search across AAAK files)
        2. For any entities found in results, query KG for related triples
        3. Return {results: [...], kg_context: [...]}
        """
        if not query or not query.strip():
            return {"ok": False, "error": "Empty query"}

        query_lower = query.lower()
        query_words = set(re.findall(r"\b\w{3,}\b", query_lower))
        results = []

        # Search across all AAAK files
        for aaak_file in self._root_dir.rglob("*.aaak"):
            try:
                content = aaak_file.read_text(encoding="utf-8")
                content_lower = content.lower()

                # Score by word overlap
                score = 0
                for word in query_words:
                    if word in content_lower:
                        score += 1

                if score > 0:
                    # Extract relative path for location
                    rel_path = aaak_file.relative_to(self._root_dir)
                    parts = rel_path.parts
                    wing = parts[0] if len(parts) > 1 else "unknown"
                    room = parts[1] if len(parts) > 2 else "unknown"

                    results.append({
                        "palace_id": aaak_file.stem,
                        "wing": wing,
                        "room": room,
                        "content": content,
                        "score": score,
                    })
            except Exception:
                continue

        # Sort by score descending, take top_k
        results.sort(key=lambda x: -x["score"])
        results = results[:top_k]

        # KG enrichment: query entities found in results
        kg_context = []
        seen_entities = set()
        for r in results:
            content = r.get("content", "")
            detected = self._dialect._detect_entities(content)
            for code in detected[:3]:
                entity_name = self._reverse_entity(code)
                if entity_name and entity_name not in seen_entities:
                    seen_entities.add(entity_name)
                    try:
                        facts = self._kg.query_entity(entity_name)
                        for fact in facts[:5]:
                            kg_context.append(fact)
                    except Exception:
                        pass

        return {
            "ok": True,
            "query": query,
            "results": results,
            "result_count": len(results),
            "kg_context": kg_context[:15],
        }

    # ── Navigate ─────────────────────────────────────────────────────────

    def navigate(self, wing: str = None, room: str = None) -> dict:
        """Browse the palace spatially.

        If no args: list all wings with room counts
        If wing: list rooms in that wing with memory counts
        If wing+room: list memories in that room (titles/dates)
        """
        if not wing:
            # List all wings
            wings_info = []
            for wing_name, wing_def in self._wings.items():
                wing_dir = self._root_dir / wing_name
                rooms = list(wing_def.get("rooms", {}).keys())
                memory_count = sum(
                    len(list((wing_dir / r).glob("*.aaak")))
                    for r in rooms
                    if (wing_dir / r).exists()
                )
                wings_info.append({
                    "wing": wing_name,
                    "description": wing_def.get("description", ""),
                    "rooms": len(rooms),
                    "memories": memory_count,
                })
            return {
                "ok": True,
                "location": "palace",
                "contents": wings_info,
            }

        if wing not in self._wings:
            return {"ok": False, "error": f"Wing '{wing}' not found"}

        wing_def = self._wings[wing]
        wing_dir = self._root_dir / wing

        if not room:
            # List rooms in wing
            rooms_info = []
            for room_name, room_desc in wing_def.get("rooms", {}).items():
                room_dir = wing_dir / room_name
                memory_count = len(list(room_dir.glob("*.aaak"))) if room_dir.exists() else 0
                rooms_info.append({
                    "room": room_name,
                    "description": room_desc,
                    "memories": memory_count,
                })
            return {
                "ok": True,
                "location": wing,
                "description": wing_def.get("description", ""),
                "contents": rooms_info,
            }

        # List memories in a room
        rooms = wing_def.get("rooms", {})
        if room not in rooms:
            return {"ok": False, "error": f"Room '{room}' not found in wing '{wing}'"}

        room_dir = wing_dir / room
        memories = []
        if room_dir.exists():
            for aaak_file in sorted(room_dir.glob("*.aaak"), reverse=True):
                try:
                    content = aaak_file.read_text(encoding="utf-8")
                    # Extract first line as title/summary
                    first_line = content.split("\n")[0][:80] if content else "(empty)"
                    memories.append({
                        "palace_id": aaak_file.stem,
                        "title": first_line,
                        "size_bytes": aaak_file.stat().st_size,
                    })
                except Exception:
                    continue

        return {
            "ok": True,
            "location": f"{wing}/{room}",
            "description": rooms.get(room, ""),
            "contents": memories[:50],  # Cap at 50 entries
        }

    # ── Status ───────────────────────────────────────────────────────────

    def status(self) -> dict:
        """Aggregated stats for the entire palace."""
        total_memories = 0
        total_size = 0
        wings_stats = {}

        for wing_name, wing_def in self._wings.items():
            wing_dir = self._root_dir / wing_name
            rooms_count = len(wing_def.get("rooms", {}))
            wing_memories = 0

            for room_name in wing_def.get("rooms", {}):
                room_dir = wing_dir / room_name
                if room_dir.exists():
                    for f in room_dir.glob("*.aaak"):
                        wing_memories += 1
                        total_size += f.stat().st_size

            total_memories += wing_memories
            wings_stats[wing_name] = {
                "rooms": rooms_count,
                "memories": wing_memories,
            }

        # KG stats
        try:
            kg_stats = self._kg.stats()
        except Exception:
            kg_stats = {"entities": 0, "triples": 0}

        return {
            "ok": True,
            "total_memories": total_memories,
            "total_entities": kg_stats.get("entities", 0),
            "total_triples": kg_stats.get("triples", 0),
            "wings": wings_stats,
            "kg_stats": kg_stats,
            "disk_usage_mb": round(total_size / (1024 * 1024), 2),
        }
