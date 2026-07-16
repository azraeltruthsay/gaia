"""Tests for World Model Stage 1 (t2m): world column on KG triples.

Locks in:
  - Schema migration: pre-existing dbs gain the world column with
    default 'actuality'
  - World scoping: queries default to actuality, never leak fiction
  - Same triple can exist in multiple worlds independently
  - Contradiction detection is world-local
  - All API methods accept world kwarg and propagate it correctly
"""

import pytest


@pytest.fixture
def kg(tmp_path):
    """Fresh KG in a throwaway sqlite file."""
    from gaia_common.utils.knowledge_graph import KnowledgeGraph
    db = tmp_path / "kg.sqlite"
    return KnowledgeGraph(db_path=str(db))


@pytest.fixture
def legacy_kg(tmp_path):
    """KG built on a pre-existing sqlite that lacks the world column.

    Simulates the migration path: create a triples table the old way,
    insert a row, then construct KnowledgeGraph (which runs _init_db
    and should migrate the schema).
    """
    import sqlite3
    db_path = tmp_path / "legacy_kg.sqlite"
    conn = sqlite3.connect(str(db_path))
    conn.executescript("""
        CREATE TABLE entities (
            id TEXT PRIMARY KEY, name TEXT NOT NULL,
            type TEXT DEFAULT 'unknown',
            properties TEXT DEFAULT '{}',
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE triples (
            id TEXT PRIMARY KEY,
            subject TEXT NOT NULL, predicate TEXT NOT NULL, object TEXT NOT NULL,
            valid_from TEXT, valid_to TEXT,
            confidence REAL DEFAULT 1.0, source TEXT,
            extracted_at TEXT DEFAULT CURRENT_TIMESTAMP
        );
        INSERT INTO entities (id, name) VALUES ('core', 'Core');
        INSERT INTO entities (id, name) VALUES ('qwen', 'Qwen');
        INSERT INTO triples (id, subject, predicate, object)
        VALUES ('t1', 'core', 'runs_on', 'qwen');
    """)
    conn.commit()
    conn.close()

    from gaia_common.utils.knowledge_graph import KnowledgeGraph
    return KnowledgeGraph(db_path=str(db_path))


class TestSchemaMigration:
    def test_fresh_kg_has_world_column(self, kg):
        import sqlite3
        conn = sqlite3.connect(kg.db_path)
        cols = [r[1] for r in conn.execute("PRAGMA table_info(triples)")]
        assert "world" in cols

    def test_legacy_kg_gets_migrated(self, legacy_kg):
        """Pre-existing db without world column gains it on KG init."""
        import sqlite3
        conn = sqlite3.connect(legacy_kg.db_path)
        cols = [r[1] for r in conn.execute("PRAGMA table_info(triples)")]
        assert "world" in cols
        # And the pre-existing row got 'actuality' as default
        row = conn.execute(
            "SELECT world FROM triples WHERE id='t1'"
        ).fetchone()
        assert row[0] == "actuality"


class TestWorldScoping:
    def test_default_world_is_actuality(self, kg):
        kg.add_triple("Core", "runs_on", "Qwen3.5-4B")
        results = kg.query_entity("Core")
        assert len(results) == 1
        assert results[0]["world"] == "actuality"

    def test_query_default_filters_to_actuality(self, kg):
        kg.add_triple("Hogwarts", "located_in", "Scotland", world="potterverse")
        kg.add_triple("Hogwarts", "is_a", "school", world="actuality")
        # Default query returns ONLY the actuality fact
        results = kg.query_entity("Hogwarts")
        assert len(results) == 1
        assert results[0]["predicate"] == "is_a"
        assert results[0]["world"] == "actuality"

    def test_query_explicit_world(self, kg):
        kg.add_triple("Hogwarts", "located_in", "Scotland", world="potterverse")
        kg.add_triple("Hogwarts", "is_a", "school", world="actuality")
        results = kg.query_entity("Hogwarts", world="potterverse")
        assert len(results) == 1
        assert results[0]["object"] == "Scotland"
        assert results[0]["world"] == "potterverse"

    def test_query_all_worlds_with_none(self, kg):
        kg.add_triple("Hogwarts", "located_in", "Scotland", world="potterverse")
        kg.add_triple("Hogwarts", "is_a", "school", world="actuality")
        results = kg.query_entity("Hogwarts", world=None)
        worlds = {r["world"] for r in results}
        assert worlds == {"actuality", "potterverse"}


class TestCrossWorldIndependence:
    def test_same_triple_in_two_worlds_is_two_facts(self, kg):
        t1 = kg.add_triple("X", "rel", "Y", world="actuality")
        t2 = kg.add_triple("X", "rel", "Y", world="hypothetical")
        assert t1 != t2
        # Stats by_world should show both
        s = kg.stats()
        assert s["by_world"].get("actuality", 0) >= 1
        assert s["by_world"].get("hypothetical", 0) >= 1

    def test_contradiction_is_world_local(self, kg):
        # Same subject+predicate, different objects, different worlds
        kg.add_triple("Sun", "color", "yellow", world="actuality")
        kg.add_triple("Sun", "color", "purple", world="alien_perspective")
        # actuality result should be yellow, untouched
        actuality = kg.query_entity("Sun", world="actuality")
        assert len(actuality) == 1
        assert actuality[0]["object"] == "yellow"
        assert actuality[0]["current"] is True
        # alien_perspective should be purple
        alien = kg.query_entity("Sun", world="alien_perspective")
        assert len(alien) == 1
        assert alien[0]["object"] == "purple"

    def test_contradiction_within_same_world_supersedes(self, kg):
        """Adding a contradicting triple in the SAME world should
        invalidate the older one."""
        kg.add_triple("Capital", "of", "OldCity", world="alt_history")
        kg.add_triple("Capital", "of", "NewCity", world="alt_history")
        results = kg.query_entity("Capital", world="alt_history")
        currents = [r for r in results if r["current"]]
        assert len(currents) == 1
        assert currents[0]["object"] == "NewCity"


class TestInvalidate:
    def test_invalidate_is_world_scoped(self, kg):
        kg.add_triple("A", "b", "C", world="w1")
        kg.add_triple("A", "b", "C", world="w2")
        # Invalidate only in w1
        kg.invalidate("A", "b", "C", world="w1")
        w1_results = kg.query_entity("A", world="w1")
        w2_results = kg.query_entity("A", world="w2")
        # w1 fact is expired
        assert all(not r["current"] for r in w1_results)
        # w2 fact is unchanged
        assert any(r["current"] for r in w2_results)


class TestStats:
    def test_stats_reports_per_world_breakdown(self, kg):
        kg.add_triple("X", "rel", "Y", world="actuality")
        kg.add_triple("X", "rel", "Y", world="fiction")
        kg.add_triple("X", "rel", "Y", world="fiction")  # dedupe
        s = kg.stats()
        assert "by_world" in s
        assert s["by_world"]["actuality"] == 1
        assert s["by_world"]["fiction"] == 1
