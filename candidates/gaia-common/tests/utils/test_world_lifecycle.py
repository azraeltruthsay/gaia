"""Tests for World Model Stage 6 (azr): ephemeral vs durable lifecycle.

Locks in:
  - Schema migration: legacy worlds get lifecycle='durable'
  - Ephemeral creation with TTL produces expires_at in the future
  - promote_world flips ephemeral → durable, clears session_id + expires_at
  - promote_world is idempotent on durable worlds
  - promote_world refused on unknown world
  - gc_ephemeral_worlds with no force: only expired sweep
  - gc_ephemeral_worlds with force: ALL ephemeral sweep
  - gc with session_id scope: only that session's ephemerals
  - Triples + edges of swept worlds also removed
"""

from datetime import datetime, timedelta

import pytest


@pytest.fixture
def kg(tmp_path):
    from gaia_common.utils.knowledge_graph import KnowledgeGraph
    return KnowledgeGraph(db_path=str(tmp_path / "kg.sqlite"))


@pytest.fixture
def legacy_kg(tmp_path):
    """KG built on a pre-existing worlds table that lacks lifecycle column.

    Simulates the migration path so we can verify legacy worlds get
    'durable' as default.
    """
    import sqlite3
    db_path = tmp_path / "legacy_kg.sqlite"
    conn = sqlite3.connect(str(db_path))
    conn.executescript("""
        CREATE TABLE entities (
            id TEXT PRIMARY KEY, name TEXT NOT NULL,
            type TEXT DEFAULT 'unknown', properties TEXT DEFAULT '{}',
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE triples (
            id TEXT PRIMARY KEY,
            subject TEXT, predicate TEXT, object TEXT,
            valid_from TEXT, valid_to TEXT,
            confidence REAL DEFAULT 1.0, source TEXT,
            extracted_at TEXT DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE worlds (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL UNIQUE,
            modality TEXT NOT NULL DEFAULT 'actuality',
            description TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        );
        INSERT INTO worlds (id, name, modality) VALUES ('legacy_w', 'legacy_world', 'fiction');
    """)
    conn.commit()
    conn.close()

    from gaia_common.utils.knowledge_graph import KnowledgeGraph
    return KnowledgeGraph(db_path=str(db_path))


class TestLifecycleMigration:
    def test_legacy_worlds_get_durable(self, legacy_kg):
        meta = legacy_kg.get_world("legacy_world")
        assert meta is not None
        assert meta["lifecycle"] == "durable"
        assert meta["session_id"] is None
        assert meta["expires_at"] is None

    def test_actuality_bootstrap_is_durable(self, kg):
        meta = kg.get_world("actuality")
        assert meta["lifecycle"] == "durable"


class TestEphemeralCreation:
    def test_create_ephemeral_sets_expires(self, kg):
        kg.create_world("scratch", modality="hypothetical", parent="actuality",
                        lifecycle="ephemeral", ttl_seconds=300)
        meta = kg.get_world("scratch")
        assert meta["lifecycle"] == "ephemeral"
        assert meta["expires_at"] is not None
        # expires_at is in the future
        expiry = datetime.fromisoformat(meta["expires_at"])
        assert expiry > datetime.now()

    def test_create_ephemeral_default_ttl_is_3600(self, kg):
        kg.create_world("dft", modality="hypothetical", parent="actuality",
                        lifecycle="ephemeral")
        meta = kg.get_world("dft")
        expiry = datetime.fromisoformat(meta["expires_at"])
        # Default TTL of 3600s — within 5 minute window of now+3600
        delta = expiry - datetime.now()
        assert 3500 < delta.total_seconds() < 3700

    def test_create_durable_has_no_expires(self, kg):
        kg.create_world("solid", modality="fiction", parent="actuality")
        meta = kg.get_world("solid")
        assert meta["lifecycle"] == "durable"
        assert meta["expires_at"] is None

    def test_session_id_recorded_on_ephemeral(self, kg):
        kg.create_world("sess_scoped", modality="hypothetical",
                        parent="actuality", lifecycle="ephemeral",
                        session_id="sess_abc123", ttl_seconds=60)
        meta = kg.get_world("sess_scoped")
        assert meta["session_id"] == "sess_abc123"

    def test_invalid_lifecycle_rejected(self, kg):
        with pytest.raises(ValueError, match="Invalid lifecycle"):
            kg.create_world("bad", modality="fiction", parent="actuality",
                            lifecycle="permanent")


class TestPromote:
    def test_promote_flips_lifecycle(self, kg):
        kg.create_world("temp", modality="hypothetical", parent="actuality",
                        lifecycle="ephemeral", ttl_seconds=60)
        kg.promote_world("temp")
        meta = kg.get_world("temp")
        assert meta["lifecycle"] == "durable"
        assert meta["session_id"] is None
        assert meta["expires_at"] is None

    def test_promote_idempotent_on_durable(self, kg):
        kg.create_world("already_durable", modality="fiction", parent="actuality")
        before = kg.get_world("already_durable")
        kg.promote_world("already_durable")
        after = kg.get_world("already_durable")
        assert before == after

    def test_promote_unknown_world_raises(self, kg):
        with pytest.raises(ValueError, match="not found"):
            kg.promote_world("never_existed")

    def test_promoted_world_preserves_triples(self, kg):
        kg.create_world("scratch", modality="hypothetical", parent="actuality",
                        lifecycle="ephemeral", ttl_seconds=60)
        scratch_id = kg.get_world("scratch")["id"]
        kg.add_triple("Idea", "is_a", "thought", world=scratch_id)
        kg.promote_world("scratch")
        # Triple should still be there post-promote
        facts = kg.query_entity("Idea", world=scratch_id)
        assert len(facts) == 1
        assert facts[0]["object"] == "thought"


class TestGarbageCollection:
    def test_gc_no_force_sweeps_only_expired(self, kg):
        # Create one expired ephemeral by manipulating expires_at directly
        kg.create_world("expired", modality="hypothetical", parent="actuality",
                        lifecycle="ephemeral", ttl_seconds=60)
        # Create one not-yet-expired
        kg.create_world("fresh", modality="hypothetical", parent="actuality",
                        lifecycle="ephemeral", ttl_seconds=3600)
        # Backdate the expired one
        import sqlite3
        conn = sqlite3.connect(kg.db_path)
        past = (datetime.now() - timedelta(hours=1)).isoformat()
        conn.execute(
            "UPDATE worlds SET expires_at = ? WHERE name = ?",
            (past, "expired"),
        )
        conn.commit()
        conn.close()

        result = kg.gc_ephemeral_worlds()
        assert result["worlds_swept"] == 1
        assert "expired" in result["world_names"]
        # Fresh one survives
        assert kg.get_world("fresh") is not None
        assert kg.get_world("expired") is None

    def test_gc_force_sweeps_all_ephemeral(self, kg):
        kg.create_world("eph1", modality="hypothetical", parent="actuality",
                        lifecycle="ephemeral", ttl_seconds=3600)
        kg.create_world("eph2", modality="hypothetical", parent="actuality",
                        lifecycle="ephemeral", ttl_seconds=3600)
        kg.create_world("durable_one", modality="fiction", parent="actuality")

        result = kg.gc_ephemeral_worlds(force=True)
        assert result["worlds_swept"] == 2
        assert kg.get_world("eph1") is None
        assert kg.get_world("eph2") is None
        # Durable survives
        assert kg.get_world("durable_one") is not None

    def test_gc_session_scoped(self, kg):
        kg.create_world("a", modality="hypothetical", parent="actuality",
                        lifecycle="ephemeral", session_id="sess_A", ttl_seconds=3600)
        kg.create_world("b", modality="hypothetical", parent="actuality",
                        lifecycle="ephemeral", session_id="sess_B", ttl_seconds=3600)

        result = kg.gc_ephemeral_worlds(force=True, session_id="sess_A")
        assert result["worlds_swept"] == 1
        assert kg.get_world("a") is None
        assert kg.get_world("b") is not None

    def test_gc_sweeps_triples_and_edges(self, kg):
        kg.create_world("doomed", modality="hypothetical", parent="actuality",
                        lifecycle="ephemeral", ttl_seconds=3600)
        doomed_id = kg.get_world("doomed")["id"]
        kg.add_triple("X", "rel", "Y", world=doomed_id)
        kg.add_triple("A", "b", "C", world=doomed_id)

        result = kg.gc_ephemeral_worlds(force=True)
        assert result["worlds_swept"] == 1
        assert result["triples_swept"] == 2
        # Edge to actuality also swept
        assert result["edges_swept"] >= 1

    def test_gc_with_nothing_to_sweep_returns_zeros(self, kg):
        # Only durables present
        kg.create_world("solid", modality="fiction", parent="actuality")
        result = kg.gc_ephemeral_worlds()
        assert result["worlds_swept"] == 0
        assert result["triples_swept"] == 0


class TestListFiltering:
    def test_list_worlds_filter_by_lifecycle(self, kg):
        kg.create_world("eph", modality="hypothetical", parent="actuality",
                        lifecycle="ephemeral", ttl_seconds=60)
        kg.create_world("dur", modality="fiction", parent="actuality")
        all_worlds = kg.list_worlds()
        eph_only = kg.list_worlds(lifecycle="ephemeral")
        dur_only = kg.list_worlds(lifecycle="durable")
        # actuality + eph + dur = 3 total
        assert len(all_worlds) == 3
        assert len(eph_only) == 1
        assert eph_only[0]["name"] == "eph"
        # actuality is durable too
        assert len(dur_only) == 2
        assert {w["name"] for w in dur_only} == {"actuality", "dur"}
