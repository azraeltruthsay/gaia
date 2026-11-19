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


def _syl(n):
    return {"subject": "PE", "slug": "pe", "units": {"U": [{"topic": f"T{i}", "url": "u"} for i in range(n)]}}


def test_max_pages_and_rate_limit_stop_the_run(tmp_path, monkeypatch):
    calls = []
    def fetch(url):
        calls.append(url)
        raise RuntimeError("Rate limit exceeded for web_fetch. Try again later.")
    monkeypatch.setattr(ss, "fetch", fetch)
    rep = ss.study(_syl(5), tmp_path, lambda m: "", log=lambda m: None)
    assert len(calls) == 1 and rep == {}          # stopped, nothing marked failed
    monkeypatch.setattr(ss, "fetch", lambda url: ("t", "short"))
    rep = ss.study(_syl(5), tmp_path, lambda m: "", log=lambda m: None, max_pages=2)
    assert len(rep) == 2


def test_split_page_cuts_at_paragraphs_up_to_most(monkeypatch):
    monkeypatch.setattr(ss, "PAGE_CHARS", 40)
    page = "\n".join(f"para {i} " + "x" * 10 for i in range(6))
    parts = ss.split_page(page, most=10)
    assert ss.split_page(page) == parts[:1]
    assert len(parts) == 3 and all(p.startswith("para") for p in parts)
    assert "\n".join(parts) == page


def test_parts_each_get_notes_and_pairs(tmp_path, monkeypatch):
    monkeypatch.setattr(ss, "PAGE_CHARS", 150)
    page = ("Each licensee must keep control of the seal.\nThe seal shows the certificate number.\n"
            "[Statutory Authority: RCW 18.43.035.]\nRenewal of the license is every two years.\n")
    monkeypatch.setattr(ss, "fetch", lambda url: ("WAC", page * 4))
    seen = []
    def gen(m):
        if "Read the page" in m[0]["content"]:
            seen.append(m[1]["content"])
            return m[1]["content"].split("\n\n", 1)[1]
        return "Q: Who controls the seal?\nA: Each licensee must keep control of the seal."
    syl = {"subject": "PE", "slug": "pe", "category": "pe", "units": {"U": [
        {"topic": "Seals", "url": "u", "parts": 3, "drop": r"\[Statutory[^\]]*\]",
         "suggested_by": "Andy"}]}}
    rep = ss.study(syl, tmp_path, gen, log=lambda m: None)
    assert rep["seals"]["parts"] == 3 and len(seen) == 3
    assert not any("Statutory" in s for s in seen)
    rows = [json.loads(l) for l in (tmp_path / "pairs.jsonl").read_text().splitlines()]
    assert rows and all(r["suggested_by"] == "Andy" and r["category"] == "pe" for r in rows)
