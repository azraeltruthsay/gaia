"""Turns that ask about a search are answered from it (bxzc, 2026-11-06)."""
import time

from gaia_core.cognition import earlier_lookup as el
from gaia_common.utils.tool_call_parser import FRONT_PAGES_NOTE, front_pages_only, tool_result_text

FRONT_PAGES = [
    {"title": "NBC News", "url": "https://www.nbcnews.com/", "snippet": "Go to NBCNews.com for breaking news"},
    {"title": "World news", "url": "https://www.cbsnews.com/world/", "snippet": "Watch CBS News."},
    {"title": "Yahoo News", "url": "https://www.yahoo.com/news/", "snippet": "The latest news"},
    {"title": "Latest", "url": "https://www.bloomberg.com/latest", "snippet": "Get the latest news"},
    {"title": "Richland", "url": "https://www.newsbreak.com/richland-wa", "snippet": "Local news"},
]
STORIES = [
    {"title": "Bible burning", "url": "https://www.nytimes.com/2020/08/11/us/politics/russia-disinformation.html"},
    {"title": "GFCN", "url": "https://tass.com/society/1954155"},
    {"title": "House", "url": "https://www.youtube.com/watch?v=IKJTrTAA_Hc"},
    {"title": "AFP", "url": "https://factcheck.afp.com/doc.afp.com.C2EL7NR"},
    {"title": "Slug", "url": "https://example.com/solar-flare-hits-earth-grid"},
]


class Sessions:
    def __init__(self):
        self.meta = {}

    def get_session_meta(self, sid, key, default=None):
        return self.meta.get((sid, key), default)

    def set_session_meta(self, sid, key, value):
        self.meta[(sid, key)] = value


def test_front_pages_are_told_apart_from_stories():
    assert front_pages_only(FRONT_PAGES)
    assert not front_pages_only(STORIES)
    assert not any(front_pages_only([FRONT_PAGES[0], s]) for s in STORIES)
    # Live, 2026-11-06: an aggregator topic page and "aol.com/?_" among them.
    live = [{"url": "https://apnews.com/"},
            {"url": "https://news.google.com/topics/CAAqKggKIiRDQkFTRlFvSUwyMHZNRFZxYUdjU0JXVnVMVWRDR2dKVFJ5Z0FQAQ"},
            {"url": "https://www.nbcnews.com/"}, {"url": "https://www.yahoo.com/news/"}, {"url": "https://www.aol.com/?_"}]
    assert front_pages_only(live)
    assert front_pages_only(FRONT_PAGES[:2] + STORIES[:1]) and not front_pages_only(FRONT_PAGES[:1] + STORIES[:2])
    assert not front_pages_only(FRONT_PAGES[:1])


def test_she_is_told_when_no_stories_came_back():
    assert tool_result_text({"ok": True, "results": FRONT_PAGES}).startswith(FRONT_PAGES_NOTE)
    assert FRONT_PAGES_NOTE not in tool_result_text({"ok": True, "results": STORIES})


def test_questions_about_what_she_found():
    for q in ("Go ahead and summarize the top 3 stories",
              "What news outlet put out these articles and what date where they published?",
              "Did you find anything?", "Can you summarize them?", "Read me those headlines"):
        assert el.refers_to_results(q), q
    for q in ("Tell me the story of King Arthur", "What's the weather like?",
              "Who wrote The Name of the Rose?", "How are you?"):
        assert not el.refers_to_results(q), q


def test_a_recent_search_answers_and_an_old_one_does_not():
    s = Sessions()
    assert el.recall(s, "a", "summarize the top 3 stories") is None
    el.remember(s, "a", "recent news headlines", FRONT_PAGES)
    got = el.recall(s, "a", "summarize the top 3 stories")
    assert got and got["query"] == "recent news headlines" and len(got["results"]) == 5
    assert el.recall(s, "b", "summarize the top 3 stories") is None
    assert el.recall(s, "a", "Who wrote Hamlet?") is None
    assert el.recall(s, "a", "summarize the top 3 stories", now=time.time() + el.MAX_AGE_S + 1) is None


def test_the_answer_is_shaped_as_a_tool_that_already_ran():
    s = Sessions()
    el.remember(s, "a", "recent news headlines", FRONT_PAGES)
    tr = el.as_tool_result(el.recall(s, "a", "which outlet published these?"))
    assert tr["success"] and "recent news headlines" in tr["tool"]
    text = tool_result_text(tr["output"])
    assert text.startswith(FRONT_PAGES_NOTE) and "nbcnews.com" in text


def test_front_pages_are_read_for_their_headlines():
    asked = []

    def call(method, params):
        asked.append(params["url"])
        if "bbc" in params["url"]:
            return {"ok": True, "title": "BBC News - Breaking news, video", "headlines": [
                {"title": f"Story number {i} about something real", "url": f"https://www.bbc.com/news/articles/c{i}abcdefg"}
                for i in range(5)]}
        return {"ok": False, "error": "not in the allowlist"}

    got = el.headlines_for(FRONT_PAGES, call=call)
    assert got and got["source"] == "BBC News" and len(got["results"]) == 5
    assert got["results"][0]["snippet"].startswith("Story number 0") and got["results"][0]["title"] == "BBC News"
    assert asked[:len(FRONT_PAGES)] == [r["url"] for r in FRONT_PAGES] and asked[-1] == el.NEWS_FRONT_PAGES[0]
    assert el.headlines_for(STORIES, call=call) is None
    assert el.headlines_for(FRONT_PAGES, call=lambda m, p: {"ok": False}) is None


def test_the_answer_says_when_it_was_read():
    s = Sessions()
    el.remember(s, "a", "recent news headlines", FRONT_PAGES, note="Headlines on the front page of BBC News.")
    out = el.as_tool_result(el.recall(s, "a", "when were these published?"))["output"]
    assert out["note"].startswith("Headlines on the front page of BBC News. Read ")
    assert str(time.localtime().tm_year) in out["note"]
