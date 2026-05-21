"""Tests for MemPalace._extract_kg_entities (World Model Stage 0 / hrp).

Locks in the entity-extraction quality fixes:
  - No truncated 3-char codes ('SUP', 'BOW', 'NEW') passing as entities
  - No frequency-based topic words as entities
  - Stopwords filter sentence-openers and pronouns
  - Confidence reflects extraction certainty (1.0 for known, 0.7 for fallback)
"""

import pytest


@pytest.fixture
def palace(tmp_path):
    """Build a MemPalace with a throwaway storage root + minimal config."""
    from gaia_common.utils.mempalace import MemPalace

    config = {
        "root_dir": str(tmp_path / "mempalace"),
        # Minimal wing config — the extractor doesn't read wing data,
        # but MemPalace.__init__ touches it.
        "wings": {
            "technical": {
                "rooms": {
                    "infrastructure": {},
                    "preferences": {},
                },
            },
            "operational": {"rooms": {"preferences": {}}},
        },
        "type_to_room": {
            "technical": "technical/infrastructure",
        },
    }
    return MemPalace(config)


class TestEntityExtractor:
    """Direct unit tests on _extract_kg_entities."""

    def test_extracts_multiword_proper_names(self, palace):
        result = palace._extract_kg_entities(
            "Marcus Aurelius was a Roman emperor."
        )
        names = [n for n, _ in result]
        assert "Marcus Aurelius" in names

    def test_extracts_long_acronyms(self, palace):
        result = palace._extract_kg_entities(
            "The game had attendance of LVIII thousand fans."
        )
        names = [n for n, _ in result]
        assert "LVIII" in names

    def test_rejects_three_char_caps_fragments(self, palace):
        """'Super Bowl' must NOT produce 'SUP'/'BOW' entities."""
        result = palace._extract_kg_entities(
            "The Super Bowl is broadcast on cable."
        )
        names = [n for n, _ in result]
        assert "SUP" not in names
        assert "BOW" not in names
        # "Super Bowl" itself is a legit multi-word entity — it should be there.
        assert "Super Bowl" in names

    def test_rejects_sentence_opener_pronouns(self, palace):
        """His/Her/Its at sentence start should not produce two-word entities."""
        result = palace._extract_kg_entities(
            'A man of grave look. His Christian biographer wrote about it.'
        )
        names = [n for n, _ in result]
        assert "His Christian" not in names
        # Should still extract "Christian" alone? No — single Title Case is
        # deliberately disabled.
        assert all(not n.startswith("His ") for n in names)

    def test_rejects_stopword_starters(self, palace):
        """Between, Speaking, Therefore at sentence start should not anchor entities."""
        result = palace._extract_kg_entities(
            'Between the lines I see a theme. Speaking of which, Therefore we conclude.'
        )
        # No multi-word "Between the lines" / "Speaking of" should be flagged
        for name, _ in result:
            for stop in ("Between ", "Speaking ", "Therefore "):
                assert not name.startswith(stop), f"unexpected entity: {name}"

    def test_known_registry_match_is_high_confidence(self, palace):
        # Inject a known entity into the AAAK registry so we can verify
        # the high-confidence path. Real registry contents vary by AAAK
        # version; this isolates the test from that.
        palace._entities["TestSubject"] = "TS_TEST"
        result = palace._extract_kg_entities(
            "TestSubject is a curated reference."
        )
        names = {n: c for n, c in result}
        assert "TestSubject" in names
        assert names["TestSubject"] >= 0.9  # known-registry confidence

    def test_acronym_gaia_extracted_at_fallback_confidence(self, palace):
        """GAIA isn't in AAAK's curated registry but the fallback acronym
        regex should still catch it."""
        result = palace._extract_kg_entities("GAIA runs on Gemma 4 E4B")
        names = {n: c for n, c in result}
        assert "GAIA" in names
        # Fallback confidence range
        assert 0.5 <= names["GAIA"] <= 0.9

    def test_fallback_entities_have_moderate_confidence(self, palace):
        result = palace._extract_kg_entities("Marcus Aurelius wrote a book.")
        names = {n: c for n, c in result}
        assert "Marcus Aurelius" in names
        # Fallback path uses 0.7 — not 1.0
        assert names["Marcus Aurelius"] < 1.0
        assert names["Marcus Aurelius"] >= 0.5

    def test_empty_input_returns_empty(self, palace):
        assert palace._extract_kg_entities("") == []
        assert palace._extract_kg_entities("hi") == []

    def test_no_entities_in_pure_chitchat(self, palace):
        result = palace._extract_kg_entities("just a casual greeting message")
        assert result == []

    def test_doesnt_match_across_sentence_boundary(self, palace):
        """'Oregon. Current' should not be captured as a 2-word entity."""
        result = palace._extract_kg_entities(
            "Portland is in Oregon. Current temp is 50."
        )
        names = [n for n, _ in result]
        assert "Oregon. Current" not in names
        assert "Oregon Current" not in names


class TestExtractAndStoreTriples:
    """Integration: store text via MemPalace and verify triple shape."""

    def test_no_palace_id_subject_triples(self, palace, monkeypatch):
        """The removed stored_in / memory_type triples should not appear."""
        triples_added = []

        def fake_add_triple(subject, predicate, obj, **kwargs):
            triples_added.append((subject, predicate, obj))
            return f"t_{len(triples_added)}"

        monkeypatch.setattr(palace._kg, "add_triple", fake_add_triple)

        palace._extract_and_store_triples(
            text="Marcus Aurelius was a Roman emperor.",
            memory_type="technical",
            wing="technical",
            room="infrastructure",
            source="test",
            date_str="2026-05-21",
            palace_id="2026-05-21_testabc",
        )

        # No 'stored_in' or 'memory_type' triples
        predicates = [p for _, p, _ in triples_added]
        assert "stored_in" not in predicates
        assert "memory_type" not in predicates
        assert "related_to" not in predicates

        # The mentioned_in triples should have the entity as SUBJECT
        # and the palace_id as OBJECT (not the other way around).
        for s, p, o in triples_added:
            if p == "mentioned_in":
                assert o == "2026-05-21_testabc"
                assert s != "2026-05-21_testabc"

    def test_confidence_passed_through(self, palace, monkeypatch):
        captured = []

        def fake_add_triple(subject, predicate, obj, **kwargs):
            captured.append({"s": subject, "p": predicate, "o": obj, **kwargs})
            return "t_1"

        monkeypatch.setattr(palace._kg, "add_triple", fake_add_triple)

        palace._extract_and_store_triples(
            text="Marcus Aurelius was a Roman emperor.",
            memory_type="technical",
            wing="technical",
            room="infrastructure",
            source="test",
            date_str="2026-05-21",
            palace_id="2026-05-21_testabc",
        )

        # Fallback-detected entity should land with 0.7 confidence
        marcus_triples = [t for t in captured if t["s"] == "Marcus Aurelius"]
        assert marcus_triples, f"Marcus Aurelius not captured: {captured}"
        for t in marcus_triples:
            assert t.get("confidence") == 0.7
            assert t.get("source") == "test"
