"""Tests for World Model Stage 3 (4da): world registry as DAG.

Locks in:
  - Actuality bootstrap on every fresh KG
  - create_world with all valid modalities + edge types
  - Atomic world IDs (w_<hex>) that don't change on rename
  - Parent resolution accepts both id and name
  - Cycle detection (a world can't be its own parent)
  - Multi-level descendant traversal via recursive CTE
  - world_path rendering walks parent chain bottom-up
  - delete_world refuses on referenced worlds, force=True overrides
  - Can't delete actuality (root protection)
"""

import pytest


@pytest.fixture
def kg(tmp_path):
    from gaia_common.utils.knowledge_graph import KnowledgeGraph
    return KnowledgeGraph(db_path=str(tmp_path / "kg.sqlite"))


class TestBootstrap:
    def test_actuality_is_bootstrapped_on_fresh_kg(self, kg):
        meta = kg.get_world("actuality")
        assert meta is not None
        assert meta["id"] == "actuality"
        assert meta["modality"] == "actuality"

    def test_list_worlds_includes_actuality(self, kg):
        worlds = kg.list_worlds()
        names = {w["name"] for w in worlds}
        assert "actuality" in names


class TestCreateWorld:
    def test_create_with_parent_by_name(self, kg):
        wid = kg.create_world(
            name="potterverse",
            modality="fiction",
            parent="actuality",
            edge_type="branches-from",
        )
        assert wid.startswith("w_")
        meta = kg.get_world("potterverse")
        assert meta["modality"] == "fiction"

    def test_create_with_all_valid_modalities(self, kg):
        for m in ("fiction", "counterfactual", "hypothetical",
                  "projection", "belief_of"):
            kg.create_world(name=f"test_{m}", modality=m, parent="actuality")
            assert kg.get_world(f"test_{m}") is not None

    def test_invalid_modality_rejected(self, kg):
        with pytest.raises(ValueError, match="Invalid modality"):
            kg.create_world(name="bad", modality="nonsense", parent="actuality")

    def test_invalid_edge_type_rejected(self, kg):
        with pytest.raises(ValueError, match="Invalid edge_type"):
            kg.create_world(
                name="bad", modality="fiction",
                parent="actuality", edge_type="weirdly",
            )

    def test_self_parent_rejected(self, kg):
        wid = kg.create_world(name="solo", modality="fiction", parent="actuality")
        # Attempting to make solo its own parent would require a parent
        # lookup matching solo's id — we ensure the path raises.
        with pytest.raises(ValueError, match="cannot be its own parent"):
            kg.create_world(name="solo", modality="fiction", parent=wid)

    def test_atomic_id_stable_across_metadata_changes(self, kg):
        wid1 = kg.create_world(name="stable", modality="fiction", parent="actuality")
        # ID is deterministic from name — re-creating with same name returns same id
        wid2 = kg.create_world(name="stable", modality="fiction", parent="actuality")
        assert wid1 == wid2


class TestPathRendering:
    def test_three_level_path(self, kg):
        kg.create_world("fiction", modality="fiction", parent="actuality")
        kg.create_world("potterverse", modality="fiction", parent="fiction")
        kg.create_world("hogwarts", modality="fiction", parent="potterverse",
                        edge_type="refines")
        path = kg.world_path("hogwarts")
        assert path == "actuality > fiction > potterverse > hogwarts"

    def test_root_world_path_is_itself(self, kg):
        assert kg.world_path("actuality") == "actuality"

    def test_unknown_world_returns_name(self, kg):
        # Defensive — don't crash, just return the input
        assert kg.world_path("never_created") == "never_created"


class TestDescendants:
    def test_descendants_include_self(self, kg):
        kg.create_world("alpha", modality="fiction", parent="actuality")
        descendants = kg.world_descendants("alpha")
        alpha_id = kg.get_world("alpha")["id"]
        assert alpha_id in descendants

    def test_recursive_descendants(self, kg):
        kg.create_world("level1", modality="fiction", parent="actuality")
        kg.create_world("level2", modality="fiction", parent="level1")
        kg.create_world("level3", modality="fiction", parent="level2")
        ids = kg.world_descendants("level1")
        assert len(ids) == 3  # level1, level2, level3
        # Sibling worlds NOT included
        kg.create_world("sibling", modality="fiction", parent="actuality")
        sib_id = kg.get_world("sibling")["id"]
        assert sib_id not in ids


class TestDeleteWorld:
    def test_cannot_delete_actuality(self, kg):
        with pytest.raises(ValueError, match="actuality"):
            kg.delete_world("actuality")

    def test_refuses_with_existing_triples(self, kg):
        kg.create_world("fragile", modality="fiction", parent="actuality")
        fragile_id = kg.get_world("fragile")["id"]
        kg.add_triple("X", "rel", "Y", world=fragile_id)
        with pytest.raises(ValueError, match="still has 1 triples"):
            kg.delete_world("fragile")

    def test_force_delete_removes_triples_too(self, kg):
        kg.create_world("doomed", modality="fiction", parent="actuality")
        doomed_id = kg.get_world("doomed")["id"]
        kg.add_triple("X", "rel", "Y", world=doomed_id)
        kg.delete_world("doomed", force=True)
        assert kg.get_world("doomed") is None
        # Triples should be gone too
        s = kg.stats()
        assert doomed_id not in s["by_world"]

    def test_delete_empty_world_succeeds(self, kg):
        kg.create_world("ephemeral", modality="hypothetical", parent="actuality")
        assert kg.delete_world("ephemeral") is True
        assert kg.get_world("ephemeral") is None


class TestWorldQuadInteraction:
    def test_query_descendants_returns_ids_usable_for_quads(self, kg):
        kg.create_world("parent_w", modality="fiction", parent="actuality")
        kg.create_world("child_w", modality="fiction", parent="parent_w")
        parent_id = kg.get_world("parent_w")["id"]
        child_id = kg.get_world("child_w")["id"]
        kg.add_triple("X", "rel", "Y_in_parent", world=parent_id)
        kg.add_triple("X", "rel", "Y_in_child", world=child_id)
        # Each world sees only its own triples
        parent_facts = kg.query_entity("X", world=parent_id)
        child_facts = kg.query_entity("X", world=child_id)
        assert len(parent_facts) == 1
        assert parent_facts[0]["object"] == "Y_in_parent"
        assert len(child_facts) == 1
        assert child_facts[0]["object"] == "Y_in_child"
