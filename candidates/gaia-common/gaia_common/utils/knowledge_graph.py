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
from typing import Optional

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
        # Step 1: ensure tables exist. CREATE TABLE IF NOT EXISTS is a
        # no-op on pre-existing schemas; the world column may be missing
        # on those — handled by the migration step below.
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
                world TEXT NOT NULL DEFAULT 'actuality',
                FOREIGN KEY (subject) REFERENCES entities(id),
                FOREIGN KEY (object) REFERENCES entities(id)
            );

            CREATE INDEX IF NOT EXISTS idx_triples_subject ON triples(subject);
            CREATE INDEX IF NOT EXISTS idx_triples_object ON triples(object);
            CREATE INDEX IF NOT EXISTS idx_triples_predicate ON triples(predicate);
            CREATE INDEX IF NOT EXISTS idx_triples_valid ON triples(valid_from, valid_to);

            -- World registry (Stage 3, 4da). Worlds are first-class
            -- objects with an opaque ID, a name, and a modality. The
            -- world_edges table forms a DAG describing how worlds relate
            -- to each other (overlays/refines/branches-from). Path
            -- strings like 'actuality > fiction > potterverse' are
            -- rendered traversals of this DAG, not stored keys.
            CREATE TABLE IF NOT EXISTS worlds (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL UNIQUE,
                modality TEXT NOT NULL DEFAULT 'actuality',
                description TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS world_edges (
                parent_id TEXT NOT NULL,
                child_id  TEXT NOT NULL,
                edge_type TEXT NOT NULL CHECK (edge_type IN ('overlays','refines','branches-from')),
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (parent_id, child_id, edge_type),
                FOREIGN KEY (parent_id) REFERENCES worlds(id),
                FOREIGN KEY (child_id)  REFERENCES worlds(id)
            );

            CREATE INDEX IF NOT EXISTS idx_world_edges_parent ON world_edges(parent_id);
            CREATE INDEX IF NOT EXISTS idx_world_edges_child  ON world_edges(child_id);
        """)
        # Step 2: migrate pre-existing databases — add world column if
        # missing. Must happen BEFORE the world-dependent indices below.
        cols = [r[1] for r in conn.execute("PRAGMA table_info(triples)").fetchall()]
        if "world" not in cols:
            conn.execute(
                "ALTER TABLE triples ADD COLUMN world TEXT NOT NULL "
                "DEFAULT 'actuality'"
            )
            logger.info(
                "KG migration: added 'world' column to triples table "
                "(existing rows defaulted to 'actuality')"
            )
        # Step 3: create world-dependent indices now that the column exists.
        conn.executescript("""
            CREATE INDEX IF NOT EXISTS idx_triples_world ON triples(world);
            CREATE INDEX IF NOT EXISTS idx_triples_world_subject ON triples(world, subject);
        """)
        # Step 4: bootstrap the actuality world in the registry. Every
        # KG has actuality as the root world; pre-existing triples that
        # default to world='actuality' need a corresponding registry row
        # so DAG queries don't fail.
        conn.execute(
            "INSERT OR IGNORE INTO worlds (id, name, modality, description) "
            "VALUES (?, ?, ?, ?)",
            ("actuality", "actuality", "actuality",
             "Consensus reality — the default world all triples scope to "
             "unless explicitly overridden."),
        )
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
        world: str = "actuality",
    ) -> str:
        """Add a relationship triple: subject → predicate → object.

        World Model Stage 1 (t2m): the world parameter scopes the triple
        to a context. Default is 'actuality' (consensus reality). Other
        worlds (fiction, counterfactual, etc.) are isolated namespaces —
        the same (subject, predicate, object) triple can exist in both
        actuality and another world without conflicting.

        Examples:
            add_triple("Core", "runs_on", "Qwen3.5-4B", valid_from="2026-04-01")
            add_triple("Hogwarts", "located_in", "Scotland", world="potterverse")
        """
        sub_id = self._entity_id(subject)
        obj_id = self._entity_id(obj)
        pred = predicate.lower().replace(" ", "_")

        conn = self._conn()
        # Auto-create entities if they don't exist
        conn.execute("INSERT OR IGNORE INTO entities (id, name) VALUES (?, ?)", (sub_id, subject))
        conn.execute("INSERT OR IGNORE INTO entities (id, name) VALUES (?, ?)", (obj_id, obj))

        # Dedup check is world-scoped — the same triple in two different
        # worlds is intentionally allowed (Hogwarts being in Scotland is
        # true in potterverse AND in actuality with different sources).
        existing = conn.execute(
            "SELECT id FROM triples WHERE subject=? AND predicate=? "
            "AND object=? AND world=? AND valid_to IS NULL",
            (sub_id, pred, obj_id, world),
        ).fetchone()

        if existing:
            conn.close()
            return existing[0]

        # ── Contradiction detection (Tier 1: deterministic) ──────────
        # Same-world contradictions only. Cross-world facts can disagree
        # by design — that's the whole point of the world dimension.
        conflicting = conn.execute(
            "SELECT t.*, e.name as obj_name FROM triples t "
            "JOIN entities e ON t.object = e.id "
            "WHERE t.subject=? AND t.predicate=? AND t.object!=? "
            "AND t.world=? AND t.valid_to IS NULL",
            (sub_id, pred, obj_id, world),
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
            """INSERT INTO triples
               (id, subject, predicate, object, valid_from, valid_to, confidence, source, world)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (triple_id, sub_id, pred, obj_id, valid_from, valid_to, confidence, source, world),
        )
        conn.commit()
        conn.close()
        logger.debug(
            "Added triple [%s]: %s → %s → %s", world, subject, predicate, obj,
        )
        return triple_id

    def invalidate(
        self,
        subject: str,
        predicate: str,
        obj: str,
        ended: str = None,
        world: str = "actuality",
    ):
        """Mark a relationship as no longer valid (set valid_to date).

        World-scoped: only invalidates the matching triple in the given
        world. To invalidate the same fact across multiple worlds, call
        this once per world.
        """
        sub_id = self._entity_id(subject)
        obj_id = self._entity_id(obj)
        pred = predicate.lower().replace(" ", "_")
        ended = ended or date.today().isoformat()

        conn = self._conn()
        cursor = conn.execute(
            "UPDATE triples SET valid_to=? "
            "WHERE subject=? AND predicate=? AND object=? "
            "AND world=? AND valid_to IS NULL",
            (ended, sub_id, pred, obj_id, world),
        )
        conn.commit()
        rows_affected = cursor.rowcount
        conn.close()
        if rows_affected:
            logger.info(
                "Invalidated [%s]: %s → %s → %s (ended %s)",
                world, subject, predicate, obj, ended,
            )
        return rows_affected

    # ── Query operations ──────────────────────────────────────────────────

    def query_entity(
        self,
        name: str,
        as_of: str = None,
        direction: str = "both",
        world: str = "actuality",
    ):
        """Get all relationships for an entity.

        direction: "outgoing" (entity → ?), "incoming" (? → entity), "both"
        as_of: date string — only return facts valid at that time
        world: scope query to a single world. Pass None to search ALL worlds
               (returns facts from every named graph; each result includes
               its 'world' field so callers can disambiguate).
        """
        eid = self._entity_id(name)
        conn = self._conn()
        results = []

        world_filter = "" if world is None else " AND t.world = ?"
        extra_params = [] if world is None else [world]

        if direction in ("outgoing", "both"):
            query = (
                "SELECT t.*, e.name as obj_name FROM triples t "
                "JOIN entities e ON t.object = e.id WHERE t.subject = ?"
                + world_filter
            )
            params = [eid] + extra_params
            if as_of:
                query += (
                    " AND (t.valid_from IS NULL OR t.valid_from <= ?) "
                    "AND (t.valid_to IS NULL OR t.valid_to >= ?)"
                )
                params.extend([as_of, as_of])
            for row in conn.execute(query, params).fetchall():
                results.append({
                    "direction": "outgoing",
                    "subject": name,
                    "predicate": row[2],
                    "object": row[10],  # obj_name (shifted: world column at index 9)
                    "valid_from": row[4],
                    "valid_to": row[5],
                    "confidence": row[6],
                    "source": row[7],
                    "world": row[9],
                    "current": row[5] is None,
                })

        if direction in ("incoming", "both"):
            query = (
                "SELECT t.*, e.name as sub_name FROM triples t "
                "JOIN entities e ON t.subject = e.id WHERE t.object = ?"
                + world_filter
            )
            params = [eid] + extra_params
            if as_of:
                query += (
                    " AND (t.valid_from IS NULL OR t.valid_from <= ?) "
                    "AND (t.valid_to IS NULL OR t.valid_to >= ?)"
                )
                params.extend([as_of, as_of])
            for row in conn.execute(query, params).fetchall():
                results.append({
                    "direction": "incoming",
                    "subject": row[10],  # sub_name
                    "predicate": row[2],
                    "object": name,
                    "valid_from": row[4],
                    "valid_to": row[5],
                    "confidence": row[6],
                    "source": row[7],
                    "world": row[9],
                    "current": row[5] is None,
                })

        conn.close()
        return results

    def query_relationship(
        self,
        predicate: str,
        as_of: str = None,
        world: str = "actuality",
    ):
        """Get all triples with a given relationship type.

        world: scope to a named world (default 'actuality'). Pass None
        to search across all worlds; each result row includes its world.
        """
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
        if world is not None:
            query += " AND t.world = ?"
            params.append(world)
        if as_of:
            query += " AND (t.valid_from IS NULL OR t.valid_from <= ?) AND (t.valid_to IS NULL OR t.valid_to >= ?)"
            params.extend([as_of, as_of])

        results = []
        for row in conn.execute(query, params).fetchall():
            # Column order: id(0), subject(1), predicate(2), object(3),
            # valid_from(4), valid_to(5), confidence(6), source(7),
            # extracted_at(8), world(9), sub_name(10), obj_name(11).
            results.append({
                "subject": row[10],
                "predicate": pred,
                "object": row[11],
                "valid_from": row[4],
                "valid_to": row[5],
                "world": row[9],
                "current": row[5] is None,
            })
        conn.close()
        return results

    def timeline(
        self,
        entity_name: str = None,
        limit: int = 100,
        world: str = "actuality",
    ):
        """Get all facts in chronological order, optionally filtered by
        entity. World defaults to 'actuality'; pass None for all-worlds.
        """
        conn = self._conn()
        world_filter = "" if world is None else " AND t.world = ?"
        world_params = [] if world is None else [world]
        # When there's no entity filter, world_filter starts with AND;
        # the WHERE clause needs to start cleanly. Build both variants.
        if entity_name:
            eid = self._entity_id(entity_name)
            query = """
                SELECT t.*, s.name as sub_name, o.name as obj_name
                FROM triples t
                JOIN entities s ON t.subject = s.id
                JOIN entities o ON t.object = o.id
                WHERE (t.subject = ? OR t.object = ?)
            """ + world_filter + """
                ORDER BY t.valid_from ASC
                LIMIT ?
            """
            rows = conn.execute(query, [eid, eid, *world_params, limit]).fetchall()
        else:
            where_clause = ""
            if world is not None:
                where_clause = " WHERE t.world = ?"
            query = """
                SELECT t.*, s.name as sub_name, o.name as obj_name
                FROM triples t
                JOIN entities s ON t.subject = s.id
                JOIN entities o ON t.object = o.id
            """ + where_clause + """
                ORDER BY t.valid_from ASC
                LIMIT ?
            """
            rows = conn.execute(query, [*world_params, limit]).fetchall()

        conn.close()
        # Column indices: world is at 9, sub_name at 10, obj_name at 11
        return [
            {
                "subject": r[10],
                "predicate": r[2],
                "object": r[11],
                "valid_from": r[4],
                "valid_to": r[5],
                "world": r[9],
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
        # World breakdown — facts per named graph. Empty 'actuality'-only
        # KG returns {'actuality': N}.
        by_world = {
            r[0]: r[1] for r in conn.execute(
                "SELECT world, COUNT(*) FROM triples GROUP BY world ORDER BY world"
            ).fetchall()
        }
        conn.close()
        return {
            "entities": entities,
            "triples": triples,
            "current_facts": current,
            "expired_facts": expired,
            "relationship_types": predicates,
            "by_world": by_world,
        }

    # ── World registry (Stage 3, 4da) ─────────────────────────────────────

    _VALID_MODALITIES = frozenset({
        "actuality", "fiction", "counterfactual",
        "hypothetical", "projection", "belief_of",
    })
    _VALID_EDGE_TYPES = frozenset({"overlays", "refines", "branches-from"})

    @staticmethod
    def _world_id_for(name: str) -> str:
        """Generate a stable atomic ID from a world name.

        Returns 'actuality' verbatim (the root world keeps its readable
        ID). For all other worlds, a deterministic 'w_<8-hex>' suffix
        based on the name — so callers don't need to manage IDs, but
        renames are still local because the ID never references the
        rendered path.
        """
        if name == "actuality":
            return "actuality"
        return "w_" + hashlib.md5(name.encode()).hexdigest()[:8]

    def create_world(
        self,
        name: str,
        modality: str = "fiction",
        parent: Optional[str] = None,
        edge_type: str = "branches-from",
        description: str = "",
    ) -> str:
        """Register a new world. Returns the opaque world ID.

        name:       human-readable name (e.g. 'potterverse')
        modality:   one of _VALID_MODALITIES — governs query leakage
        parent:     parent world name OR id (e.g. 'actuality'). If None,
                    the new world has no parent edge (a root world).
        edge_type:  relationship to the parent (overlays/refines/branches-from)
        description: optional long-form note
        """
        if modality not in self._VALID_MODALITIES:
            raise ValueError(
                f"Invalid modality {modality!r}; must be one of {sorted(self._VALID_MODALITIES)}"
            )
        if parent and edge_type not in self._VALID_EDGE_TYPES:
            raise ValueError(
                f"Invalid edge_type {edge_type!r}; must be one of {sorted(self._VALID_EDGE_TYPES)}"
            )

        world_id = self._world_id_for(name)
        conn = self._conn()
        conn.execute(
            "INSERT OR IGNORE INTO worlds (id, name, modality, description) "
            "VALUES (?, ?, ?, ?)",
            (world_id, name, modality, description),
        )

        if parent:
            # Resolve parent — accept either id or name
            parent_row = conn.execute(
                "SELECT id FROM worlds WHERE id = ? OR name = ?", (parent, parent)
            ).fetchone()
            if not parent_row:
                conn.close()
                raise ValueError(f"Parent world not found: {parent!r}")
            parent_id = parent_row[0]
            if parent_id == world_id:
                conn.close()
                raise ValueError("A world cannot be its own parent")
            conn.execute(
                "INSERT OR IGNORE INTO world_edges (parent_id, child_id, edge_type) "
                "VALUES (?, ?, ?)",
                (parent_id, world_id, edge_type),
            )

        conn.commit()
        conn.close()
        logger.info(
            "Created world %s (id=%s, modality=%s, parent=%s, edge=%s)",
            name, world_id, modality, parent, edge_type if parent else "—",
        )
        return world_id

    def get_world(self, world: str) -> Optional[dict]:
        """Look up a world by id or name. Returns metadata dict or None."""
        conn = self._conn()
        row = conn.execute(
            "SELECT id, name, modality, description, created_at "
            "FROM worlds WHERE id = ? OR name = ?",
            (world, world),
        ).fetchone()
        conn.close()
        if not row:
            return None
        return {
            "id": row[0],
            "name": row[1],
            "modality": row[2],
            "description": row[3],
            "created_at": row[4],
        }

    def list_worlds(self) -> list:
        """Return all registered worlds with their parent edges."""
        conn = self._conn()
        worlds = {}
        for row in conn.execute(
            "SELECT id, name, modality, description, created_at FROM worlds"
        ):
            worlds[row[0]] = {
                "id": row[0],
                "name": row[1],
                "modality": row[2],
                "description": row[3],
                "created_at": row[4],
                "parents": [],
                "children": [],
            }
        for row in conn.execute(
            "SELECT parent_id, child_id, edge_type FROM world_edges"
        ):
            parent_id, child_id, edge_type = row
            if child_id in worlds:
                worlds[child_id]["parents"].append(
                    {"id": parent_id, "edge_type": edge_type}
                )
            if parent_id in worlds:
                worlds[parent_id]["children"].append(
                    {"id": child_id, "edge_type": edge_type}
                )
        conn.close()
        return list(worlds.values())

    def world_path(self, world: str, separator: str = " > ") -> str:
        """Render the rooted path to this world by walking parent edges.

        e.g. 'actuality > fiction > potterverse > hogwarts-1990s'.
        Picks the first parent at each step (the DAG can have multiple
        parents — `overlays` edges form one parent chain; `branches-
        from` and `refines` are mutually exclusive with that). For
        worlds with multiple parents this is best-effort rendering.

        Returns just the world's name if it has no parents.
        """
        target = self.get_world(world)
        if not target:
            return world
        chain = [target["name"]]
        current_id = target["id"]
        conn = self._conn()
        seen = {current_id}
        for _ in range(32):  # bounded to prevent cycles
            row = conn.execute(
                "SELECT we.parent_id, w.name FROM world_edges we "
                "JOIN worlds w ON we.parent_id = w.id "
                "WHERE we.child_id = ? LIMIT 1",
                (current_id,),
            ).fetchone()
            if not row:
                break
            parent_id, parent_name = row
            if parent_id in seen:
                break  # cycle guard
            seen.add(parent_id)
            chain.insert(0, parent_name)
            current_id = parent_id
        conn.close()
        return separator.join(chain)

    def world_ancestors(self, world: str) -> list:
        """Return the ancestor chain of a world, immediate parent first.

        Used by the inheritance resolver — a query in world W picks up
        triples from W itself plus all ancestors, with shadowing rules
        applied (descendant's facts override ancestor's for the same
        subject+predicate). Does NOT include W itself.

        Bounded to 32 levels to guard against cycles in malformed data.
        """
        target = self.get_world(world)
        if not target:
            return []
        ancestors: list = []
        conn = self._conn()
        seen = {target["id"]}
        current_id = target["id"]
        for _ in range(32):
            row = conn.execute(
                "SELECT parent_id FROM world_edges WHERE child_id = ? LIMIT 1",
                (current_id,),
            ).fetchone()
            if not row:
                break
            parent_id = row[0]
            if parent_id in seen:
                break
            seen.add(parent_id)
            ancestors.append(parent_id)
            current_id = parent_id
        conn.close()
        return ancestors

    def query_entity_inherited(
        self,
        name: str,
        world: str = "actuality",
        as_of: str = None,
        direction: str = "both",
    ) -> list:
        """Query an entity with world inheritance (Stage 4, 80o).

        Walks the world's ancestor chain and returns the union of all
        triples about `name`, with shadowing: a descendant world's
        triple for (subject, predicate) suppresses any ancestor's
        triple for the same (subject, predicate). This implements the
        named-graph inheritance pattern — fiction worlds inherit
        actuality's facts unless they explicitly override them.

        The MODALITY FIREWALL is automatic: inheritance walks UP the
        parent chain only, so a query in 'actuality' never picks up
        fiction/counterfactual/hypothetical descendants. Fiction
        queries do inherit actuality (broomsticks still fall to the
        ground in potterverse, unless potterverse explicitly says
        otherwise).
        """
        # Build the chain: starting world, then ancestors
        worlds_in_order = [world] + self.world_ancestors(world)
        # Resolve names to IDs for the SQL filter
        ids_in_order: list = []
        for w in worlds_in_order:
            meta = self.get_world(w)
            ids_in_order.append(meta["id"] if meta else w)

        seen_sp_keys: set = set()  # (subject, predicate) pairs already covered
        results: list = []
        for w_id in ids_in_order:
            facts = self.query_entity(
                name, as_of=as_of, direction=direction, world=w_id,
            )
            for f in facts:
                key = (f["subject"], f["predicate"], f["direction"])
                if key in seen_sp_keys:
                    continue  # shadowed by descendant
                seen_sp_keys.add(key)
                results.append(f)
        return results

    def world_descendants(self, world: str) -> list:
        """Return all descendant world IDs (recursive children) of the given world.

        Used by inheritance queries — 'show me everything in potterverse
        and its child worlds'. Includes the input world itself in the
        result.
        """
        start = self.get_world(world)
        if not start:
            return []
        conn = self._conn()
        rows = conn.execute(
            """
            WITH RECURSIVE descendants(id) AS (
                SELECT ? AS id
                UNION
                SELECT we.child_id FROM world_edges we
                JOIN descendants d ON we.parent_id = d.id
            )
            SELECT id FROM descendants
            """,
            (start["id"],),
        ).fetchall()
        conn.close()
        return [r[0] for r in rows]

    def delete_world(self, world: str, force: bool = False) -> bool:
        """Remove a world from the registry. Refuses if quads still
        reference it unless force=True.

        Never deletes the 'actuality' world (root).
        """
        target = self.get_world(world)
        if not target:
            return False
        if target["id"] == "actuality":
            raise ValueError("Cannot delete the 'actuality' root world")
        conn = self._conn()
        if not force:
            triple_count = conn.execute(
                "SELECT COUNT(*) FROM triples WHERE world = ?", (target["id"],)
            ).fetchone()[0]
            if triple_count > 0:
                conn.close()
                raise ValueError(
                    f"World {target['name']!r} still has {triple_count} triples; "
                    "pass force=True to delete anyway."
                )
        conn.execute("DELETE FROM world_edges WHERE parent_id = ? OR child_id = ?",
                     (target["id"], target["id"]))
        conn.execute("DELETE FROM worlds WHERE id = ?", (target["id"],))
        if force:
            conn.execute("DELETE FROM triples WHERE world = ?", (target["id"],))
        conn.commit()
        conn.close()
        logger.info("Deleted world %s (id=%s)", target["name"], target["id"])
        return True

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
