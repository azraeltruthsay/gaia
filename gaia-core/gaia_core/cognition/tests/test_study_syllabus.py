import json
from gaia_core.cognition import study_syllabus as ss

PAGE = ("The factor of safety expresses how much stronger a system is than it needs "
        "to be for an intended load. Buildings commonly use a factor of safety of 2.0 "
        "for each structural member.")


def test_backed_needs_numbers_and_words_from_source():
    assert ss.backed("Buildings commonly use a factor of safety of 2.0.", PAGE)
    assert not ss.backed("Buildings commonly use a factor of safety of 3.5.", PAGE)
    assert not ss.backed("Bridges require titanium reinforcement against earthquakes.", PAGE)


def test_check_notes_drops_unbacked_lines():
    notes = "- Factor of safety: how much stronger a system is than needed\n- Aircraft use 1.5\n"
    kept, dropped = ss.check_notes(notes, PAGE)
    assert len(kept) == 1 and dropped == ["- Aircraft use 1.5"]


def test_make_pairs_keeps_backed_answers_only():
    notes = "- Buildings commonly use a factor of safety of 2.0 for structural members."
    gen = lambda m: ("Q: What factor of safety do buildings use?\nA: Buildings commonly use 2.0 "
                     "for each structural member.\nQ: What about dams?\nA: Dams always use 7.0 "
                     "because concrete creeps badly.")
    kept, dropped = ss.make_pairs("Factor of safety", notes, gen)
    assert [p["instruction"] for p in kept] == ["What factor of safety do buildings use?"]
    assert len(dropped) == 1


def test_study_writes_notes_pairs_and_report(tmp_path, monkeypatch):
    monkeypatch.setattr(ss, "fetch", lambda url: ("Factor of safety", PAGE * 3))
    def gen(m):
        if "Read the page" in m[0]["content"]:
            return ("- Factor of safety: how much stronger a system is than needed\n"
                    "- Buildings commonly use a factor of safety of 2.0\n"
                    "- Each structural member is designed for an intended load\n- Made up 99")
        return "Q: What factor do buildings use?\nA: Buildings commonly use a factor of safety of 2.0."
    syl = {"subject": "PE", "slug": "pe", "units": {"U": [{"topic": "Factor of safety", "url": "u"}]}}
    rep = ss.study(syl, tmp_path, gen, log=lambda m: None)
    r = rep["factor_of_safety"]
    assert r["notes_kept"] == 3 and r["notes_dropped"] == ["- Made up 99"] and r["pairs_kept"] == 1
    assert json.loads((tmp_path / "pairs.jsonl").read_text())["group"] == "U"
    # a second run skips topics already studied
    ss.study(syl, tmp_path, lambda m: 1 / 0, log=lambda m: None)


def test_section_starts_at_the_anchor_heading():
    page = "Intro text\nChina\nChina text\nUnited States\nUS text"
    assert ss.section(page, "https://x/wiki/P#United_States") == "United States\nUS text"
    assert ss.section(page, "https://x/wiki/P") == page
    assert ss.section(page, "https://x/wiki/P#Nowhere") == page
