"""
Knowledge Graph — Temporal Entity-Relationship Triple Store for GAIA.

Adapted from MemPalace (github.com/milla-jovovich/mempalace).

Provides:
  - Entity nodes (services, concepts, models, people, projects)
  - Typed relationship edges (runs_on, depends_on, trained_with, etc.)
  - Temporal validity (valid_from → valid_to — knows WHEN facts are true)
  - Source attribution (links back to knowledge base files)

Storage: SQLite (local, stdlib only, no external dependencies)
Query: entity-first traversal with time filtering

Usage:
    from gaia_core.memory.knowledge_graph import KnowledgeGraph

    kg = KnowledgeGraph()
    kg.add_triple("Core", "runs_on", "Qwen3.5-4B", valid_from="2026-04-01")
    kg.add_triple("Core", "runs_on", "GPU", valid_from="2026-04-01")
    kg.invalidate("Core", "runs_on", "Qwen3-4B", ended="2026-04-01")

    # Query: everything about Core
    kg.query_entity("Core")

    # Query: what was true about Core in March?
    kg.query_entity("Core", as_of="2026-03-15")
"""

import hashlib
import json
import logging
import os
import sqlite3
from datetime import date, datetime
from pathlib import Path

logger = logging.getLogger("GAIA.KnowledgeGraph")

# Default path inside the shared volume so all containers can access it
DEFAULT_KG_PATH = os.environ.get(
    "GAIA_KG_PATH",
    str(Path(os.environ.get("SHARED_DIR", "/shared")) / "knowledge_graph" / "gaia_kg.sqlite3"),
)


class Contradiction:
    """A detected conflict between an incoming triple and existing facts."""

    def __init__(self, incoming: dict, existing: list, resolution: str = "pending"):
        self.incoming = incoming      # {"subject", "predicate", "object", "valid_from", ...}
        self.existing = existing      # List of conflicting triples from the DB
        self.resolution = resolution  # "update", "reject", "coexist", "pending"
        self.reason = ""

    def __repr__(self):
        return (
            f"Contradiction(incoming={self.incoming['subject']}→{self.incoming['predicate']}→{self.incoming['object']}, "
            f"conflicts={len(self.existing)}, resolution={self.resolution})"
        )


class KnowledgeGraph:
    """SQLite-backed temporal knowledge graph with contradiction detection."""

    def __init__(self, db_path: str = None, contradiction_callback=None):
        """
        Args:
            db_path: Path to SQLite database
            contradiction_callback: Optional callable(Contradiction) → Contradiction
                Called when a conflicting triple is detected. The callback should
                set contradiction.resolution to one of:
                - "update": invalidate old fact, insert new one
                - "reject": discard the incoming triple
                - "coexist": both facts are valid simultaneously
                - "pending": flag for human review (default)
                If no callback is set, contradictions auto-resolve as "update"
                (assume newer information supersedes older).
        """
        self.db_path = db_path or DEFAULT_KG_PATH
        self._contradiction_callback = contradiction_callback
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_db()
        logger.info("KnowledgeGraph initialized: %s", self.db_path)

    def _init_db(self):
        conn = self._conn()
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS entities (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                type TEXT DEFAULT 'unknown',
                properties TEXT DEFAULT '{}',
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS triples (
                id TEXT PRIMARY KEY,
                subject TEXT NOT NULL,
                predicate TEXT NOT NULL,
                object TEXT NOT NULL,
                valid_from TEXT,
                valid_to TEXT,
                confidence REAL DEFAULT 1.0,
                source TEXT,
                extracted_at TEXT DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (subject) REFERENCES entities(id),
                FOREIGN KEY (object) REFERENCES entities(id)
            );

            CREATE INDEX IF NOT EXISTS idx_triples_subject ON triples(subject);
            CREATE INDEX IF NOT EXISTS idx_triples_object ON triples(object);
            CREATE INDEX IF NOT EXISTS idx_triples_predicate ON triples(predicate);
            CREATE INDEX IF NOT EXISTS idx_triples_valid ON triples(valid_from, valid_to);
        """)
        conn.commit()
        conn.close()

    def _conn(self):
        return sqlite3.connect(self.db_path, timeout=10)

    def _entity_id(self, name: str) -> str:
        return name.lower().replace(" ", "_").replace("'", "")

    # ── Write operations ──────────────────────────────────────────────────

    def add_entity(self, name: str, entity_type: str = "unknown", properties: dict = None):
        """Add or update an entity node."""
        eid = self._entity_id(name)
        props = json.dumps(properties or {})
        conn = self._conn()
        conn.execute(
            "INSERT OR REPLACE INTO entities (id, name, type, properties) VALUES (?, ?, ?, ?)",
            (eid, name, entity_type, props),
        )
        conn.commit()
        conn.close()
        return eid

    def add_triple(
        self,
        subject: str,
        predicate: str,
        obj: str,
        valid_from: str = None,
        valid_to: str = None,
        confidence: float = 1.0,
        source: str = None,
    ) -> str:
        """Add a relationship triple: subject → predicate → object.

        Examples:
            add_triple("Core", "runs_on", "Qwen3.5-4B", valid_from="2026-04-01")
            add_triple("gaia-mcp", "exposes", "13 domain tools")
            add_triple("Nano", "handles", "reflex responses", valid_from="2026-03-01")
        """
        sub_id = self._entity_id(subject)
        obj_id = self._entity_id(obj)
        pred = predicate.lower().replace(" ", "_")

        conn = self._conn()
        # Auto-create entities if they don't exist
        conn.execute("INSERT OR IGNORE INTO entities (id, name) VALUES (?, ?)", (sub_id, subject))
        conn.execute("INSERT OR IGNORE INTO entities (id, name) VALUES (?, ?)", (obj_id, obj))

        # Check for existing identical triple (dedup)
        existing = conn.execute(
            "SELECT id FROM triples WHERE subject=? AND predicate=? AND object=? AND valid_to IS NULL",
            (sub_id, pred, obj_id),
        ).fetchone()

        if existing:
            conn.close()
            return existing[0]

        # ── Contradiction detection (Tier 1: deterministic) ──────────
        # Check for same subject+predicate but DIFFERENT object (conflict)
        conflicting = conn.execute(
            "SELECT t.*, e.name as obj_name FROM triples t "
            "JOIN entities e ON t.object = e.id "
            "WHERE t.subject=? AND t.predicate=? AND t.object!=? AND t.valid_to IS NULL",
            (sub_id, pred, obj_id),
        ).fetchall()

        if conflicting:
            conflicts = [
                {
                    "id": row[0], "subject": subject, "predicate": pred,
                    "object": row[9], "valid_from": row[4],
                }
                for row in conflicting
            ]
            incoming = {
                "subject": subject, "predicate": predicate, "object": obj,
                "valid_from": valid_from, "source": source,
            }
            contradiction = Contradiction(incoming=incoming, existing=conflicts)

            # Tier 2: Observer adjudication (if callback registered)
            if self._contradiction_callback:
                try:
                    contradiction = self._contradiction_callback(contradiction)
                except Exception as e:
                    logger.warning("Contradiction callback failed: %s — defaulting to update", e)
                    contradiction.resolution = "update"
            else:
                # No observer — default to update (newer supersedes older)
                contradiction.resolution = "update"
                contradiction.reason = "auto-update (no observer)"

            logger.info(
                "Contradiction detected: %s → %s → %s conflicts with %d existing facts. Resolution: %s",
                subject, predicate, obj, len(conflicts), contradiction.resolution,
            )

            if contradiction.resolution == "reject":
                conn.close()
                return f"REJECTED:{conflicts[0]['id']}"
            elif contradiction.resolution == "coexist":
                pass  # Fall through to insert — both facts are valid
            elif contradiction.resolution in ("update", "pending"):
                # Invalidate old facts before inserting new one
                for c in conflicts:
                    conn.execute(
                        "UPDATE triples SET valid_to=? WHERE id=?",
                        (valid_from or date.today().isoformat(), c["id"]),
                    )

        triple_id = (
            f"t_{sub_id}_{pred}_{obj_id}_"
            f"{hashlib.md5(f'{valid_from}{datetime.now().isoformat()}'.encode()).hexdigest()[:8]}"
        )

        conn.execute(
            """INSERT INTO triples (id, subject, predicate, object, valid_from, valid_to, confidence, source)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (triple_id, sub_id, pred, obj_id, valid_from, valid_to, confidence, source),
        )
        conn.commit()
        conn.close()
        logger.debug("Added triple: %s → %s → %s", subject, predicate, obj)
        return triple_id

    def invalidate(self, subject: str, predicate: str, obj: str, ended: str = None):
        """Mark a relationship as no longer valid (set valid_to date)."""
        sub_id = self._entity_id(subject)
        obj_id = self._entity_id(obj)
        pred = predicate.lower().replace(" ", "_")
        ended = ended or date.today().isoformat()

        conn = self._conn()
        cursor = conn.execute(
            "UPDATE triples SET valid_to=? WHERE subject=? AND predicate=? AND object=? AND valid_to IS NULL",
            (ended, sub_id, pred, obj_id),
        )
        conn.commit()
        rows_affected = cursor.rowcount
        conn.close()
        if rows_affected:
            logger.info("Invalidated: %s → %s → %s (ended %s)", subject, predicate, obj, ended)
        return rows_affected

    # ── Query operations ──────────────────────────────────────────────────

    def query_entity(self, name: str, as_of: str = None, direction: str = "both"):
        """Get all relationships for an entity.

        direction: "outgoing" (entity → ?), "incoming" (? → entity), "both"
        as_of: date string — only return facts valid at that time
        """
        eid = self._entity_id(name)
        conn = self._conn()
        results = []

        if direction in ("outgoing", "both"):
            query = "SELECT t.*, e.name as obj_name FROM triples t JOIN entities e ON t.object = e.id WHERE t.subject = ?"
            params = [eid]
            if as_of:
                query += " AND (t.valid_from IS NULL OR t.valid_from <= ?) AND (t.valid_to IS NULL OR t.valid_to >= ?)"
                params.extend([as_of, as_of])
            for row in conn.execute(query, params).fetchall():
                results.append({
                    "direction": "outgoing",
                    "subject": name,
                    "predicate": row[2],
                    "object": row[9],  # obj_name
                    "valid_from": row[4],
                    "valid_to": row[5],
                    "confidence": row[6],
                    "source": row[7],
                    "current": row[5] is None,
                })

        if direction in ("incoming", "both"):
            query = "SELECT t.*, e.name as sub_name FROM triples t JOIN entities e ON t.subject = e.id WHERE t.object = ?"
            params = [eid]
            if as_of:
                query += " AND (t.valid_from IS NULL OR t.valid_from <= ?) AND (t.valid_to IS NULL OR t.valid_to >= ?)"
                params.extend([as_of, as_of])
            for row in conn.execute(query, params).fetchall():
                results.append({
                    "direction": "incoming",
                    "subject": row[9],  # sub_name
                    "predicate": row[2],
                    "object": name,
                    "valid_from": row[4],
                    "valid_to": row[5],
                    "confidence": row[6],
                    "source": row[7],
                    "current": row[5] is None,
                })

        conn.close()
        return results

    def query_relationship(self, predicate: str, as_of: str = None):
        """Get all triples with a given relationship type."""
        pred = predicate.lower().replace(" ", "_")
        conn = self._conn()
        query = """
            SELECT t.*, s.name as sub_name, o.name as obj_name
            FROM triples t
            JOIN entities s ON t.subject = s.id
            JOIN entities o ON t.object = o.id
            WHERE t.predicate = ?
        """
        params = [pred]
        if as_of:
            query += " AND (t.valid_from IS NULL OR t.valid_from <= ?) AND (t.valid_to IS NULL OR t.valid_to >= ?)"
            params.extend([as_of, as_of])

        results = []
        for row in conn.execute(query, params).fetchall():
            results.append({
                "subject": row[9],
                "predicate": pred,
                "object": row[10],
                "valid_from": row[4],
                "valid_to": row[5],
                "current": row[5] is None,
            })
        conn.close()
        return results

    def timeline(self, entity_name: str = None, limit: int = 100):
        """Get all facts in chronological order, optionally filtered by entity."""
        conn = self._conn()
        if entity_name:
            eid = self._entity_id(entity_name)
            rows = conn.execute("""
                SELECT t.*, s.name as sub_name, o.name as obj_name
                FROM triples t
                JOIN entities s ON t.subject = s.id
                JOIN entities o ON t.object = o.id
                WHERE (t.subject = ? OR t.object = ?)
                ORDER BY t.valid_from ASC NULLS LAST
                LIMIT ?
            """, (eid, eid, limit)).fetchall()
        else:
            rows = conn.execute("""
                SELECT t.*, s.name as sub_name, o.name as obj_name
                FROM triples t
                JOIN entities s ON t.subject = s.id
                JOIN entities o ON t.object = o.id
                ORDER BY t.valid_from ASC NULLS LAST
                LIMIT ?
            """, (limit,)).fetchall()

        conn.close()
        return [
            {
                "subject": r[9],
                "predicate": r[2],
                "object": r[10],
                "valid_from": r[4],
                "valid_to": r[5],
                "current": r[5] is None,
            }
            for r in rows
        ]

    # ── Stats ─────────────────────────────────────────────────────────────

    def stats(self):
        conn = self._conn()
        entities = conn.execute("SELECT COUNT(*) FROM entities").fetchone()[0]
        triples = conn.execute("SELECT COUNT(*) FROM triples").fetchone()[0]
        current = conn.execute("SELECT COUNT(*) FROM triples WHERE valid_to IS NULL").fetchone()[0]
        expired = triples - current
        predicates = [
            r[0] for r in conn.execute(
                "SELECT DISTINCT predicate FROM triples ORDER BY predicate"
            ).fetchall()
        ]
        conn.close()
        return {
            "entities": entities,
            "triples": triples,
            "current_facts": current,
            "expired_facts": expired,
            "relationship_types": predicates,
        }

    # ── Seed from GAIA constants ──────────────────────────────────────────

    def seed_from_gaia_constants(self, constants: dict):
        """Bootstrap the knowledge graph from gaia_constants.json.

        Extracts service configurations, model assignments, and system facts.
        """
        # Service tiers
        for model_name, config in constants.get("MODEL_CONFIGS", {}).items():
            endpoint = config.get("endpoint", "")
            if endpoint:
                self.add_triple(model_name, "endpoint", endpoint, source="gaia_constants")
            model_path = config.get("model_path", "")
            if model_path:
                self.add_triple(model_name, "model_path", model_path, source="gaia_constants")

        # Domain tools
        for domain in constants.get("DOMAIN_TOOLS", {}).keys():
            self.add_triple("gaia-mcp", "exposes_domain", domain, source="gaia_constants")

        logger.info("KG seeded from gaia_constants: %s", self.stats())
