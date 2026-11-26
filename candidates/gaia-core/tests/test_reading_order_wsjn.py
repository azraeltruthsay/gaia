"""wsjn: a remembered lookup that only shares words with the question must not
be read ahead of this search's own results ("closest star to Earth" read
Mount Everest, 2026-11-25)."""
from gaia_core.cognition import knowledge_router as kr
from gaia_core.cognition.knowledge_router import GroundingResult


def _r(source, url, qmatch=0.0):
    return GroundingResult(source=source, trust_tier="looked_up", content="x" * 80, url=url,
                           question_match=qmatch)


def test_word_sharing_memory_goes_after_the_search():
    everest = _r("memory", "https://en.wikipedia.org/wiki/Mount_Everest", qmatch=0.0)
    stars = _r("web", "https://en.wikipedia.org/wiki/List_of_nearest_stars")
    proxima = _r("web", "https://en.wikipedia.org/wiki/Proxima_Centauri")
    order = [r.url for r in kr.reading_order([everest, stars, proxima])]
    assert order[0].endswith("List_of_nearest_stars")
    assert order[-1].endswith("Mount_Everest")


def test_memory_looked_up_for_this_question_still_goes_first():
    asked = _r("memory", "https://en.wikipedia.org/wiki/Proxima_Centauri", qmatch=1.0)
    web = _r("web", "https://example.org/stars")
    assert kr.reading_order([web, asked])[0] is asked


def test_other_sources_are_not_read():
    journal = GroundingResult(source="journal", trust_tier="own_record", content="y" * 80, url="j")
    assert kr.reading_order([journal]) == []


def test_router_reads_the_search_result_not_the_word_sharing_memory(monkeypatch):
    everest = _r("memory", "https://en.wikipedia.org/wiki/Mount_Everest", qmatch=0.0)
    stars = _r("web", "https://en.wikipedia.org/wiki/List_of_nearest_stars")
    stars.title = "List of nearest stars"
    stars.content = "Proxima Centauri is the closest star to Earth after the Sun, 4.24 light-years away."
    tried = []

    def fake_read(question, results):
        tried.extend(r.url for r in results)
        return None
    monkeypatch.setattr(kr, "_read_top_result", fake_read)
    monkeypatch.setattr(kr, "_query_lookup_memory", lambda q, **k: [everest])
    monkeypatch.setattr(kr, "_query_web", lambda q, *a, **k: [stars])
    kr.ground_query("What is the closest star to Earth?", intent="chat", max_total_ms=600000)
    assert tried and tried[0].endswith("List_of_nearest_stars")


def test_word_sharing_memory_goes_behind_the_search_in_grounding(monkeypatch):
    trek = _r("memory", "https://en.wikipedia.org/wiki/Star_Trek_Next_Generation", qmatch=0.0)
    asked = _r("memory", "https://en.wikipedia.org/wiki/Andromeda_Galaxy", qmatch=0.8)
    galaxies = _r("web", "https://en.wikipedia.org/wiki/List_of_nearest_galaxies")
    galaxies.title = "List of nearest galaxies"
    galaxies.content = "The closest galaxy to the Milky Way is the Canis Major Dwarf Galaxy."
    monkeypatch.setattr(kr, "_read_top_result", lambda q, r: None)
    monkeypatch.setattr(kr, "_query_lookup_memory", lambda q, **k: [trek, asked])
    monkeypatch.setattr(kr, "_query_web", lambda q, *a, **k: [galaxies])
    ctx = kr.ground_query("What is the closest galaxy to the Milky Way?", intent="chat", max_total_ms=600000)
    urls = [r.url for r in ctx.results]
    assert urls.index(galaxies.url) < urls.index(trek.url)
    assert urls.index(asked.url) < urls.index(trek.url)
