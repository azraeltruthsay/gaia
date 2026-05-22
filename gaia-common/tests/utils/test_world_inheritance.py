"""Tests for World Model Stage 4 (80o): inheritance + modality firewall.

Locks in:
  - world_ancestors walks the parent chain and stops at root
  - query_entity_inherited unions parent triples into child queries
  - Shadowing: descendant's (subject, predicate) override ancestor's
  - Firewall direction: actuality queries never see descendants
  - Three-level inheritance (grandchild inherits both grandparent and
    parent, with each level shadowing the next ancestor up)
"""

import pytest


@pytest.fixture
def kg(tmp_path):
    from gaia_common.utils.knowledge_graph import KnowledgeGraph
    return KnowledgeGraph(db_path=str(tmp_path / "kg.sqlite"))


class TestAncestorWalker:
    def test_actuality_has_no_ancestors(self, kg):
        assert kg.world_ancestors("actuality") == []

    def test_single_parent_chain(self, kg):
        kg.create_world("fiction", modality="fiction", parent="actuality")
        ancestors = kg.world_ancestors("fiction")
        assert ancestors == ["actuality"]

    def test_three_level_chain(self, kg):
        kg.create_world("fiction", modality="fiction", parent="actuality")
        kg.create_world("potterverse", modality="fiction", parent="fiction")
        kg.create_world("hogwarts", modality="fiction", parent="potterverse",
                        edge_type="refines")
        ancestors = kg.world_ancestors("hogwarts")
        # Immediate parent first
        assert len(ancestors) == 3
        # Last entry is actuality (the root)
        assert ancestors[-1] == "actuality"

    def test_unknown_world_returns_empty(self, kg):
        assert kg.world_ancestors("never_existed") == []


class TestInheritedQuery:
    def test_inheritance_adds_parent_facts(self, kg):
        kg.create_world("fiction", modality="fiction", parent="actuality")
        kg.add_triple("Broomstick", "falls", "downward", world="actuality")
        f_id = kg.get_world("fiction")["id"]
        kg.add_triple("Broomstick", "flies", "upward", world=f_id)

        # Direct query: only fiction
        direct = kg.query_entity("Broomstick", world=f_id)
        assert len(direct) == 1
        assert direct[0]["object"] == "upward"

        # Inherited: both predicates show up
        inherited = kg.query_entity_inherited("Broomstick", world=f_id)
        objects = {f["object"] for f in inherited}
        assert objects == {"upward", "downward"}

    def test_descendant_shadows_ancestor_on_same_pred(self, kg):
        kg.create_world("alt", modality="counterfactual", parent="actuality")
        kg.add_triple("Sun", "color", "yellow", world="actuality")
        alt_id = kg.get_world("alt")["id"]
        kg.add_triple("Sun", "color", "purple", world=alt_id)

        inherited = kg.query_entity_inherited("Sun", world=alt_id)
        # Only ONE result — alt's purple shadows actuality's yellow
        assert len(inherited) == 1
        assert inherited[0]["object"] == "purple"
        assert inherited[0]["world"] == alt_id

    def test_distinct_predicates_both_returned(self, kg):
        kg.create_world("fic", modality="fiction", parent="actuality")
        kg.add_triple("Char", "lives_in", "London", world="actuality")
        fic_id = kg.get_world("fic")["id"]
        kg.add_triple("Char", "wields", "wand", world=fic_id)

        inherited = kg.query_entity_inherited("Char", world=fic_id)
        # Both facts present — different predicates so no shadowing
        preds = {f["predicate"] for f in inherited}
        assert preds == {"lives_in", "wields"}


class TestModalityFirewall:
    def test_actuality_default_never_sees_descendants(self, kg):
        """The canonical contamination test: fiction triples must NOT
        appear in actuality queries even when they share entity names."""
        kg.create_world("fiction", modality="fiction", parent="actuality")
        f_id = kg.get_world("fiction")["id"]
        # Add a fiction triple about a topic
        kg.add_triple("Spell", "type", "magical", world=f_id)
        # Default actuality query
        results = kg.query_entity("Spell")  # default world='actuality'
        assert results == [], "Fiction triples leaked into actuality"

    def test_actuality_inherited_query_also_clean(self, kg):
        """Even with inherit=True, actuality has no ancestors and so
        cannot pick up fiction facts."""
        kg.create_world("fiction", modality="fiction", parent="actuality")
        f_id = kg.get_world("fiction")["id"]
        kg.add_triple("Spell", "type", "magical", world=f_id)

        results = kg.query_entity_inherited("Spell", world="actuality")
        assert results == []

    def test_inheritance_walks_only_upward(self, kg):
        """A child world inherits parent's facts; parent does NOT
        inherit child's facts (the firewall direction)."""
        kg.create_world("fic", modality="fiction", parent="actuality")
        f_id = kg.get_world("fic")["id"]
        kg.add_triple("Hat", "color", "green", world="actuality")
        kg.add_triple("Hat", "type", "sorting", world=f_id)

        # Child inherits parent: should see both
        child_view = kg.query_entity_inherited("Hat", world=f_id)
        assert len(child_view) == 2

        # Parent's query (inherit=True) sees only its own — fiction
        # is a child, not an ancestor
        parent_view = kg.query_entity_inherited("Hat", world="actuality")
        assert len(parent_view) == 1
        assert parent_view[0]["object"] == "green"


class TestMultiLevelInheritance:
    def test_grandchild_inherits_full_chain(self, kg):
        kg.create_world("fiction", modality="fiction", parent="actuality")
        kg.create_world("potterverse", modality="fiction", parent="fiction")
        kg.create_world("hogwarts", modality="fiction", parent="potterverse",
                        edge_type="refines")

        h_id = kg.get_world("hogwarts")["id"]
        p_id = kg.get_world("potterverse")["id"]
        f_id = kg.get_world("fiction")["id"]

        # One fact at each level
        kg.add_triple("Tower", "in", "Britain", world="actuality")
        kg.add_triple("Tower", "is_a", "fictional_setting", world=f_id)
        kg.add_triple("Tower", "called", "Hogwarts", world=p_id)
        kg.add_triple("Tower", "rooms", "many", world=h_id)

        view = kg.query_entity_inherited("Tower", world=h_id)
        preds = {f["predicate"] for f in view}
        assert preds == {"in", "is_a", "called", "rooms"}

    def test_closest_descendant_wins_shadowing(self, kg):
        """If both actuality and an intermediate world have a fact for
        (X, p), the intermediate's shadows actuality's when querying a
        further descendant."""
        kg.create_world("alt", modality="counterfactual", parent="actuality")
        kg.create_world("deep", modality="counterfactual", parent="alt")
        alt_id = kg.get_world("alt")["id"]
        deep_id = kg.get_world("deep")["id"]

        kg.add_triple("Color", "of", "red", world="actuality")
        kg.add_triple("Color", "of", "blue", world=alt_id)
        # deep doesn't redefine — should inherit blue from alt, NOT red from actuality

        view = kg.query_entity_inherited("Color", world=deep_id)
        assert len(view) == 1
        assert view[0]["object"] == "blue"
        assert view[0]["world"] == alt_id
