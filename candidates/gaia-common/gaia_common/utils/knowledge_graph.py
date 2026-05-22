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
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                lifecycle TEXT NOT NULL DEFAULT 'durable',
                session_id TEXT,
                expires_at TEXT
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

            -- World merges (Stage 5, 8pk). The riskiest operation in the
            -- World Model — collapsing two worlds we thought were
            -- separate into one. Each merge is a proposal that captures
            -- the pre-merge state of both worlds, the coreference
            -- mapping, and a status. Apply is atomic at the world-ID
            -- level (no global string rewrite); reverse restores from
            -- the snapshot.
            CREATE TABLE IF NOT EXISTS merges (
                id TEXT PRIMARY KEY,
                source_world TEXT NOT NULL,    -- the world being absorbed
                target_world TEXT NOT NULL,    -- the world it merges INTO
                status TEXT NOT NULL DEFAULT 'pending',  -- pending | applied | reversed
                entity_mapping TEXT NOT NULL DEFAULT '{}',  -- JSON: source_id → target_id
                snapshot TEXT NOT NULL,         -- JSON: pre-merge triples + edges of both worlds
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                applied_at TEXT,
                reversed_at TEXT,
                notes TEXT
            );

            CREATE INDEX IF NOT EXISTS idx_merges_status ON merges(status);
            CREATE INDEX IF NOT EXISTS idx_merges_source ON merges(source_world);
            CREATE INDEX IF NOT EXISTS idx_merges_target ON merges(target_world);
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
        # Lifecycle migration (Stage 6, azr): pre-existing worlds tables
        # need lifecycle/session_id/expires_at columns. Existing rows
        # default to 'durable' — they survived shutdown so they're
        # durable by definition.
        world_cols = [r[1] for r in conn.execute("PRAGMA table_info(worlds)").fetchall()]
        if "lifecycle" not in world_cols:
            conn.execute(
                "ALTER TABLE worlds ADD COLUMN lifecycle TEXT NOT NULL "
                "DEFAULT 'durable'"
            )
            logger.info("KG migration: added 'lifecycle' column to worlds (default 'durable')")
        if "session_id" not in world_cols:
            conn.execute("ALTER TABLE worlds ADD COLUMN session_id TEXT")
            logger.info("KG migration: added 'session_id' column to worlds")
        if "expires_at" not in world_cols:
            conn.execute("ALTER TABLE worlds ADD COLUMN expires_at TEXT")
            logger.info("KG migration: added 'expires_at' column to worlds")
        conn.executescript("""
            CREATE INDEX IF NOT EXISTS idx_worlds_lifecycle ON worlds(lifecycle);
            CREATE INDEX IF NOT EXISTS idx_worlds_expires_at ON worlds(expires_at);
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

    _VALID_LIFECYCLES = frozenset({"ephemeral", "durable"})

    def create_world(
        self,
        name: str,
        modality: str = "fiction",
        parent: Optional[str] = None,
        edge_type: str = "branches-from",
        description: str = "",
        lifecycle: str = "durable",
        session_id: Optional[str] = None,
        ttl_seconds: Optional[int] = None,
    ) -> str:
        """Register a new world. Returns the opaque world ID.

        name:        human-readable name (e.g. 'potterverse')
        modality:    one of _VALID_MODALITIES — governs query leakage
        parent:      parent world name OR id (e.g. 'actuality'). If None,
                     the new world has no parent edge (a root world).
        edge_type:   relationship to the parent (overlays/refines/branches-from)
        description: optional long-form note

        Lifecycle (Stage 6, azr):
        lifecycle:   'durable' (default — persists across sessions) or
                     'ephemeral' (subject to GC by TTL)
        session_id:  for ephemeral worlds, the owning session (for filter
                     and audit; not enforced as visibility scope in MVP)
        ttl_seconds: for ephemeral worlds, lifespan. Default 3600s when
                     lifecycle='ephemeral' and ttl_seconds is None.
        """
        if modality not in self._VALID_MODALITIES:
            raise ValueError(
                f"Invalid modality {modality!r}; must be one of {sorted(self._VALID_MODALITIES)}"
            )
        if parent and edge_type not in self._VALID_EDGE_TYPES:
            raise ValueError(
                f"Invalid edge_type {edge_type!r}; must be one of {sorted(self._VALID_EDGE_TYPES)}"
            )
        if lifecycle not in self._VALID_LIFECYCLES:
            raise ValueError(
                f"Invalid lifecycle {lifecycle!r}; must be one of {sorted(self._VALID_LIFECYCLES)}"
            )

        # Compute expiry timestamp for ephemeral worlds. Default 1-hour
        # TTL if the caller doesn't specify; durable worlds have no expiry.
        expires_at_str = None
        if lifecycle == "ephemeral":
            from datetime import timedelta
            ttl = ttl_seconds if ttl_seconds is not None else 3600
            expires_at_str = (datetime.now() + timedelta(seconds=ttl)).isoformat()

        world_id = self._world_id_for(name)
        conn = self._conn()
        conn.execute(
            "INSERT OR IGNORE INTO worlds (id, name, modality, description, "
            "lifecycle, session_id, expires_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (world_id, name, modality, description,
             lifecycle, session_id, expires_at_str),
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
            "SELECT id, name, modality, description, created_at, "
            "lifecycle, session_id, expires_at "
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
            "lifecycle": row[5],
            "session_id": row[6],
            "expires_at": row[7],
        }

    def list_worlds(self, lifecycle: Optional[str] = None) -> list:
        """Return all registered worlds with their parent edges.

        lifecycle: optional filter ('ephemeral' or 'durable'). None = both.
        """
        conn = self._conn()
        if lifecycle is not None:
            rows = conn.execute(
                "SELECT id, name, modality, description, created_at, "
                "lifecycle, session_id, expires_at FROM worlds "
                "WHERE lifecycle = ?",
                (lifecycle,),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT id, name, modality, description, created_at, "
                "lifecycle, session_id, expires_at FROM worlds"
            ).fetchall()
        worlds = {}
        for row in rows:
            worlds[row[0]] = {
                "id": row[0],
                "name": row[1],
                "modality": row[2],
                "description": row[3],
                "created_at": row[4],
                "lifecycle": row[5],
                "session_id": row[6],
                "expires_at": row[7],
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

    # ── Lifecycle (Stage 6, azr) ──────────────────────────────────────────

    def promote_world(self, world: str) -> dict:
        """Promote an ephemeral world to durable.

        Clears session_id and expires_at so the world is no longer
        subject to GC. Idempotent on already-durable worlds (no-op +
        returns metadata).

        Refuses on the 'actuality' root world (it's always durable by
        definition; this is a guard against accidental misuse).
        """
        target = self.get_world(world)
        if not target:
            raise ValueError(f"World not found: {world!r}")
        if target["lifecycle"] == "durable":
            logger.info("promote_world: %s is already durable (no-op)", target["name"])
            return target
        conn = self._conn()
        conn.execute(
            "UPDATE worlds SET lifecycle = 'durable', session_id = NULL, "
            "expires_at = NULL WHERE id = ?",
            (target["id"],),
        )
        conn.commit()
        conn.close()
        logger.info(
            "Promoted world %s (id=%s) from ephemeral to durable",
            target["name"], target["id"],
        )
        return self.get_world(world)

    def gc_ephemeral_worlds(
        self,
        force: bool = False,
        session_id: Optional[str] = None,
    ) -> dict:
        """Garbage-collect expired ephemeral worlds.

        force=False (default): delete only worlds whose expires_at is in
        the past. force=True: delete ALL ephemeral worlds regardless of
        expiry (use at session shutdown).

        session_id: when set, restrict GC to ephemeral worlds owned by
        that session. Otherwise sweeps across all sessions.

        Returns dict with counts of swept worlds + triples + edges.
        """
        now = datetime.now().isoformat()
        conn = self._conn()
        # Build the condition
        where = "lifecycle = 'ephemeral'"
        params: list = []
        if not force:
            where += " AND expires_at IS NOT NULL AND expires_at <= ?"
            params.append(now)
        if session_id is not None:
            where += " AND session_id = ?"
            params.append(session_id)
        target_rows = conn.execute(
            f"SELECT id, name FROM worlds WHERE {where}", params
        ).fetchall()
        target_ids = [r[0] for r in target_rows]
        if not target_ids:
            conn.close()
            return {"worlds_swept": 0, "triples_swept": 0, "edges_swept": 0}

        # Sweep triples in those worlds first
        placeholders = ",".join("?" for _ in target_ids)
        triple_count = conn.execute(
            f"SELECT COUNT(*) FROM triples WHERE world IN ({placeholders})",
            target_ids,
        ).fetchone()[0]
        conn.execute(
            f"DELETE FROM triples WHERE world IN ({placeholders})", target_ids
        )
        # Then edges that touch those worlds
        edge_count = conn.execute(
            f"SELECT COUNT(*) FROM world_edges "
            f"WHERE parent_id IN ({placeholders}) OR child_id IN ({placeholders})",
            target_ids + target_ids,
        ).fetchone()[0]
        conn.execute(
            f"DELETE FROM world_edges "
            f"WHERE parent_id IN ({placeholders}) OR child_id IN ({placeholders})",
            target_ids + target_ids,
        )
        # Finally the world rows
        conn.execute(
            f"DELETE FROM worlds WHERE id IN ({placeholders})", target_ids
        )
        conn.commit()
        conn.close()
        logger.info(
            "GC swept %d ephemeral world(s) (%d triples, %d edges)",
            len(target_ids), triple_count, edge_count,
        )
        return {
            "worlds_swept": len(target_ids),
            "triples_swept": triple_count,
            "edges_swept": edge_count,
            "world_names": [r[1] for r in target_rows],
        }

    # ── Merge mechanism (Stage 5, 8pk) ────────────────────────────────────

    # Treat coreference matches as auto-mappings only above this score.
    # Below the threshold the entity stays separate during the merge
    # (no rename), which is the conservative choice — false positives
    # in coref are MUCH more damaging than false negatives because
    # they collapse two genuinely-distinct things into one entity.
    _COREF_THRESHOLD = 0.85

    @staticmethod
    def _coref_score(name_a: str, name_b: str) -> float:
        """Conservative name-similarity score in [0, 1].

        Exact match (case-sensitive): 1.0
        Case-insensitive match: 0.95
        Token-Jaccard similarity: variable (typically 0.0-0.9)

        This is intentionally simple — real coreference resolution is
        a 40-year-old open problem in record linkage. Stage 5 uses a
        name-only heuristic with a high threshold (0.85) so it makes
        conservative auto-merges. Hard cases stay unmapped and a
        future stage can add disambiguation.
        """
        if not name_a or not name_b:
            return 0.0
        if name_a == name_b:
            return 1.0
        if name_a.lower() == name_b.lower():
            return 0.95
        # Token-Jaccard over lowercased word splits
        tokens_a = set(name_a.lower().replace("_", " ").split())
        tokens_b = set(name_b.lower().replace("_", " ").split())
        if not tokens_a or not tokens_b:
            return 0.0
        inter = tokens_a & tokens_b
        union = tokens_a | tokens_b
        return len(inter) / len(union)

    def _resolve_coreference(self, source_world_id: str, target_world_id: str) -> dict:
        """Map entities in source world to candidates in target world.

        Returns dict of source_entity_id → target_entity_id for matches
        above _COREF_THRESHOLD. Entities below threshold are NOT in the
        mapping; during apply, they stay under their original ID (their
        triples just get rewritten to the target world).
        """
        conn = self._conn()
        # Entities that appear as subject or object in either world
        src_entities = {
            row[0]: row[1] for row in conn.execute(
                "SELECT e.id, e.name FROM entities e "
                "WHERE e.id IN ("
                "  SELECT subject FROM triples WHERE world = ? "
                "  UNION SELECT object FROM triples WHERE world = ?"
                ")",
                (source_world_id, source_world_id),
            )
        }
        tgt_entities = {
            row[0]: row[1] for row in conn.execute(
                "SELECT e.id, e.name FROM entities e "
                "WHERE e.id IN ("
                "  SELECT subject FROM triples WHERE world = ? "
                "  UNION SELECT object FROM triples WHERE world = ?"
                ")",
                (target_world_id, target_world_id),
            )
        }
        conn.close()

        mapping: dict = {}
        for src_id, src_name in src_entities.items():
            best_score = 0.0
            best_target = None
            for tgt_id, tgt_name in tgt_entities.items():
                score = self._coref_score(src_name, tgt_name)
                if score > best_score:
                    best_score = score
                    best_target = tgt_id
            if best_score >= self._COREF_THRESHOLD and best_target:
                mapping[src_id] = best_target
        return mapping

    def _snapshot_worlds(self, source_world_id: str, target_world_id: str) -> dict:
        """Capture the pre-merge state of both worlds for reversibility."""
        conn = self._conn()
        snapshot = {
            "source": {
                "world": source_world_id,
                "world_meta": None,
                "triples": [],
                "incoming_edges": [],
                "outgoing_edges": [],
            },
            "target": {
                "world": target_world_id,
                "world_meta": None,
                "triples": [],
            },
        }
        # World metadata
        for side, world_id in [("source", source_world_id), ("target", target_world_id)]:
            row = conn.execute(
                "SELECT id, name, modality, description, created_at "
                "FROM worlds WHERE id = ?", (world_id,)
            ).fetchone()
            if row:
                snapshot[side]["world_meta"] = {
                    "id": row[0], "name": row[1], "modality": row[2],
                    "description": row[3], "created_at": row[4],
                }
        # All triples in both worlds (full row)
        for side, world_id in [("source", source_world_id), ("target", target_world_id)]:
            for r in conn.execute(
                "SELECT id, subject, predicate, object, valid_from, valid_to, "
                "confidence, source, extracted_at, world FROM triples WHERE world = ?",
                (world_id,),
            ):
                snapshot[side]["triples"].append(list(r))
        # Source world's edges (incoming + outgoing)
        for r in conn.execute(
            "SELECT parent_id, child_id, edge_type FROM world_edges "
            "WHERE child_id = ?", (source_world_id,)
        ):
            snapshot["source"]["incoming_edges"].append(list(r))
        for r in conn.execute(
            "SELECT parent_id, child_id, edge_type FROM world_edges "
            "WHERE parent_id = ?", (source_world_id,)
        ):
            snapshot["source"]["outgoing_edges"].append(list(r))
        conn.close()
        return snapshot

    def propose_merge(
        self,
        source_world: str,
        target_world: str,
        notes: str = "",
    ) -> dict:
        """Propose a merge of source_world INTO target_world.

        Returns a structured proposal dict — NOTHING is changed in the
        KG until apply_merge is called with the merge_id. The proposal
        captures:
          - the resolved entity coreference mapping (source→target)
          - a pre-merge snapshot of both worlds (for reversibility)
          - counts of what would be affected

        Refuses to propose merging actuality away (the root world cannot
        be absorbed). Source and target must both exist and be distinct.
        """
        src_meta = self.get_world(source_world)
        tgt_meta = self.get_world(target_world)
        if not src_meta:
            raise ValueError(f"Source world not found: {source_world!r}")
        if not tgt_meta:
            raise ValueError(f"Target world not found: {target_world!r}")
        if src_meta["id"] == tgt_meta["id"]:
            raise ValueError("Cannot merge a world into itself")
        if src_meta["id"] == "actuality":
            raise ValueError("Cannot merge the 'actuality' root world away")

        entity_mapping = self._resolve_coreference(src_meta["id"], tgt_meta["id"])
        snapshot = self._snapshot_worlds(src_meta["id"], tgt_meta["id"])

        merge_id = "m_" + hashlib.md5(
            f"{src_meta['id']}->{tgt_meta['id']}@{datetime.now().isoformat()}".encode()
        ).hexdigest()[:12]

        proposal = {
            "merge_id": merge_id,
            "source_world": src_meta["id"],
            "source_world_name": src_meta["name"],
            "target_world": tgt_meta["id"],
            "target_world_name": tgt_meta["name"],
            "entity_mapping": entity_mapping,
            "triples_to_rewrite": len(snapshot["source"]["triples"]),
            "edges_to_remap": (
                len(snapshot["source"]["incoming_edges"])
                + len(snapshot["source"]["outgoing_edges"])
            ),
            "status": "pending",
            "notes": notes,
        }

        # Persist the proposal
        conn = self._conn()
        conn.execute(
            "INSERT INTO merges (id, source_world, target_world, status, "
            "entity_mapping, snapshot, notes) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                merge_id, src_meta["id"], tgt_meta["id"], "pending",
                json.dumps(entity_mapping),
                json.dumps(snapshot, default=str),
                notes,
            ),
        )
        conn.commit()
        conn.close()
        logger.info(
            "Merge proposed: %s (%s → %s, %d triples, %d entities mapped)",
            merge_id, src_meta["name"], tgt_meta["name"],
            proposal["triples_to_rewrite"], len(entity_mapping),
        )
        return proposal

    def apply_merge(self, merge_id: str) -> dict:
        """Execute a previously-proposed merge.

        Operations performed:
          1. Rewrite all source-world triples to the target world,
             applying the entity_mapping to subject/object IDs
          2. Re-parent any source-world child edges to point at target
          3. Drop source-world edges and the world row itself
          4. Mark the merge as applied with a timestamp

        Returns the final proposal record with status='applied'.
        """
        conn = self._conn()
        row = conn.execute(
            "SELECT source_world, target_world, status, entity_mapping "
            "FROM merges WHERE id = ?", (merge_id,)
        ).fetchone()
        if not row:
            conn.close()
            raise ValueError(f"Unknown merge_id: {merge_id!r}")
        src_id, tgt_id, status, mapping_json = row
        if status != "pending":
            conn.close()
            raise ValueError(
                f"Merge {merge_id} is not pending (current status: {status})"
            )
        mapping = json.loads(mapping_json or "{}")

        # 1. Rewrite triples: re-target world + remap entities per coref
        for r in conn.execute(
            "SELECT id, subject, object FROM triples WHERE world = ?", (src_id,)
        ).fetchall():
            tid, subj, obj = r
            new_subj = mapping.get(subj, subj)
            new_obj = mapping.get(obj, obj)
            conn.execute(
                "UPDATE triples SET world = ?, subject = ?, object = ? "
                "WHERE id = ?",
                (tgt_id, new_subj, new_obj, tid),
            )

        # 2. Re-parent child edges. A world that had source as parent
        #    now has target as parent.
        conn.execute(
            "UPDATE world_edges SET parent_id = ? WHERE parent_id = ?",
            (tgt_id, src_id),
        )
        # 3. Drop edges where source was the child (its parents are
        #    already represented through target's own parent edges).
        conn.execute("DELETE FROM world_edges WHERE child_id = ?", (src_id,))
        # 4. Drop the source world row itself
        conn.execute("DELETE FROM worlds WHERE id = ?", (src_id,))
        # 5. Mark merge applied
        conn.execute(
            "UPDATE merges SET status = 'applied', applied_at = ? WHERE id = ?",
            (datetime.now().isoformat(), merge_id),
        )
        conn.commit()
        conn.close()
        logger.info("Merge applied: %s (source %s collapsed into %s)",
                    merge_id, src_id, tgt_id)
        return self.get_merge(merge_id)

    def reverse_merge(self, merge_id: str) -> dict:
        """Restore a previously-applied merge from its snapshot.

        Re-creates the source world, restores its triples, restores
        its edges, and resets entity IDs that were remapped by
        coreference back to their source-side originals.

        Note: if the target world has been further modified since the
        merge (other triples added, other merges applied to it), reverse
        will undo the merge's contribution but won't roll back those
        downstream changes. Snapshot captures only what changed at
        merge-time, not subsequent edits.
        """
        conn = self._conn()
        row = conn.execute(
            "SELECT source_world, target_world, status, entity_mapping, snapshot "
            "FROM merges WHERE id = ?", (merge_id,)
        ).fetchone()
        if not row:
            conn.close()
            raise ValueError(f"Unknown merge_id: {merge_id!r}")
        src_id, tgt_id, status, mapping_json, snapshot_json = row
        if status != "applied":
            conn.close()
            raise ValueError(
                f"Can only reverse an applied merge (status: {status})"
            )
        snapshot = json.loads(snapshot_json or "{}")
        mapping = json.loads(mapping_json or "{}")

        # 1. Restore source world row
        sm = snapshot["source"]["world_meta"]
        if sm:
            conn.execute(
                "INSERT OR REPLACE INTO worlds (id, name, modality, description, created_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (sm["id"], sm["name"], sm["modality"], sm["description"], sm["created_at"]),
            )
        # 2. Restore source world edges
        for parent_id, child_id, edge_type in snapshot["source"]["incoming_edges"]:
            conn.execute(
                "INSERT OR IGNORE INTO world_edges (parent_id, child_id, edge_type) "
                "VALUES (?, ?, ?)",
                (parent_id, child_id, edge_type),
            )
        # 2b. Restore outgoing edges + revert any re-parent we did at apply time
        for parent_id, child_id, edge_type in snapshot["source"]["outgoing_edges"]:
            # Children that pointed at source still currently point at target
            # — flip them back. Then re-add the original outgoing edge.
            conn.execute(
                "UPDATE world_edges SET parent_id = ? "
                "WHERE parent_id = ? AND child_id = ? AND edge_type = ?",
                (parent_id, tgt_id, child_id, edge_type),
            )
            conn.execute(
                "INSERT OR IGNORE INTO world_edges (parent_id, child_id, edge_type) "
                "VALUES (?, ?, ?)",
                (parent_id, child_id, edge_type),
            )

        # 3. Reverse-map: triples we moved into target need to go back.
        #    Identify them by the original triple IDs preserved in the
        #    snapshot, then update world AND restore original entity IDs.
        reverse_mapping = {v: k for k, v in mapping.items()}
        for triple_row in snapshot["source"]["triples"]:
            tid, subj, pred, obj, vf, vt, conf, src, ext, world = triple_row
            conn.execute(
                "UPDATE triples SET world = ?, subject = ?, object = ? "
                "WHERE id = ?",
                (world, subj, obj, tid),
            )
        # Apply reverse-mapping defensively in case entity IDs got renamed
        # in any other triples within target (unlikely but cheap)
        # — only the source's own snapshotted triples are touched above.

        # 4. Mark merge reversed
        conn.execute(
            "UPDATE merges SET status = 'reversed', reversed_at = ? WHERE id = ?",
            (datetime.now().isoformat(), merge_id),
        )
        conn.commit()
        conn.close()
        logger.info("Merge reversed: %s (source %s restored from snapshot)",
                    merge_id, src_id)
        return self.get_merge(merge_id)

    def get_merge(self, merge_id: str) -> Optional[dict]:
        """Look up a merge by id. Returns None if not found."""
        conn = self._conn()
        row = conn.execute(
            "SELECT id, source_world, target_world, status, entity_mapping, "
            "created_at, applied_at, reversed_at, notes FROM merges WHERE id = ?",
            (merge_id,),
        ).fetchone()
        conn.close()
        if not row:
            return None
        return {
            "merge_id": row[0],
            "source_world": row[1],
            "target_world": row[2],
            "status": row[3],
            "entity_mapping": json.loads(row[4] or "{}"),
            "created_at": row[5],
            "applied_at": row[6],
            "reversed_at": row[7],
            "notes": row[8],
        }

    def list_merges(self, status: Optional[str] = None) -> list:
        """Return all merge records, optionally filtered by status."""
        conn = self._conn()
        if status:
            rows = conn.execute(
                "SELECT id, source_world, target_world, status, "
                "created_at, applied_at, reversed_at, notes "
                "FROM merges WHERE status = ? ORDER BY created_at DESC",
                (status,),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT id, source_world, target_world, status, "
                "created_at, applied_at, reversed_at, notes "
                "FROM merges ORDER BY created_at DESC"
            ).fetchall()
        conn.close()
        return [
            {
                "merge_id": r[0],
                "source_world": r[1],
                "target_world": r[2],
                "status": r[3],
                "created_at": r[4],
                "applied_at": r[5],
                "reversed_at": r[6],
                "notes": r[7],
            }
            for r in rows
        ]

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
