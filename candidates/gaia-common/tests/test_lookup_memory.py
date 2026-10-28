"""0vuz: lookup memory keeps what GAIA looked up, verbatim and with
provenance, recalls it for later questions, and tracks when to look again."""

import json
from datetime import datetime, timedelta, timezone

from gaia_common.utils.lookup_memory import (
    STABLE_DAYS, VOLATILE_DAYS, LookupMemory, content_terms, is_volatile, question_match)

T0 = datetime(2026, 10, 27, 12, 0, tzinfo=timezone.utc)

LADY = {
    "title": "Lady of the Lake - Wikipedia",
    "url": "https://en.wikipedia.org/wiki/Lady_of_the_Lake",
    "snippet": "The Lady of the Lake gives King Arthur the sword Excalibur and later "
               "receives it back when Bedivere throws it into the lake.",
    "domain": "en.wikipedia.org",
    "trust_tier": "trusted",
}
ONIMUSHA = {
    "title": "Onimusha: Way of the Sword",
    "url": "https://example.com/onimusha",
    "snippet": "Onimusha: Way of the Sword is an action game set in Edo-period Kyoto with demons.",
    "domain": "example.com",
}


def _mem(tmp_path):
    return LookupMemory(tmp_path / "lookups")


def test_remembers_verbatim_with_provenance(tmp_path):
    m = _mem(tmp_path)
    ids = m.remember("Where did King Arthur get Excalibur?", [LADY], now=T0)
    rec = json.loads((tmp_path / "lookups" / f"{ids[0]}.json").read_text())
    assert rec["snippet"] == LADY["snippet"]
    assert rec["url"] == LADY["url"] and rec["domain"] == "en.wikipedia.org"
    assert rec["trust_tier"] == "trusted" and rec["kind"] == "search"
    assert rec["queries"] == ["Where did King Arthur get Excalibur?"]
    assert rec["retrieved_at"].startswith("2026-10-27")
    assert rec["revalidate_after"].startswith(str((T0 + timedelta(days=STABLE_DAYS)).date()))


def test_skips_results_without_a_usable_snippet(tmp_path):
    m = _mem(tmp_path)
    assert m.remember("q", [{"title": "x", "url": "u", "snippet": "too short"}, "junk", None]) == []


def test_same_result_again_refreshes_instead_of_duplicating(tmp_path):
    m = _mem(tmp_path)
    first = m.remember("Where did Arthur get the sword?", [LADY], now=T0)
    later = T0 + timedelta(days=30)
    again = m.remember("Who gave Arthur Excalibur?", [LADY], now=later)
    assert first == again
    assert len(list((tmp_path / "lookups").glob("*.json"))) == 1
    rec = json.loads((tmp_path / "lookups" / f"{first[0]}.json").read_text())
    assert rec["queries"] == ["Where did Arthur get the sword?", "Who gave Arthur Excalibur?"]
    assert rec["retrieved_at"].startswith("2026-11-26")
    assert rec["first_retrieved_at"].startswith("2026-10-27")


def test_recall_finds_the_relevant_lookup_first(tmp_path):
    m = _mem(tmp_path)
    m.remember("King Arthur sword", [LADY, ONIMUSHA], now=T0)
    hits = m.recall("Where did he get the sword? King Arthur Excalibur", now=T0)
    assert hits[0]["url"] == LADY["url"]
    assert hits[0]["score"] >= 0.6 and hits[0]["stale"] is False
    assert all(h["url"] != ONIMUSHA["url"] for h in hits)


def test_recall_ignores_unrelated_questions(tmp_path):
    m = _mem(tmp_path)
    m.remember("King Arthur sword", [LADY], now=T0)
    assert m.recall("What is the capital of France?", now=T0) == []
    assert m.recall("how are you?", now=T0) == []  # stopwords only


def test_recall_counts_usage(tmp_path):
    m = _mem(tmp_path)
    rid = m.remember("King Arthur sword", [LADY], now=T0)[0]
    m.recall("King Arthur Excalibur", now=T0)
    m.recall("King Arthur Excalibur", now=T0)
    rec = json.loads((tmp_path / "lookups" / f"{rid}.json").read_text())
    assert rec["times_recalled"] == 2


def test_volatile_lookups_go_stale_quickly(tmp_path):
    m = _mem(tmp_path)
    news = dict(ONIMUSHA, snippet="The latest Onimusha release date was announced today for next spring.")
    m.remember("latest Onimusha release date", [news], now=T0)
    assert m.recall("Onimusha release date", now=T0 + timedelta(days=1))[0]["stale"] is False
    assert m.recall("Onimusha release date", now=T0 + timedelta(days=VOLATILE_DAYS + 1))[0]["stale"] is True
    assert is_volatile("What's the weather today?") and not is_volatile("Who forged Excalibur?")


def test_dispute_marks_stale_until_the_web_confirms_again(tmp_path):
    m = _mem(tmp_path)
    rid = m.remember("King Arthur sword", [LADY], now=T0)[0]
    assert m.mark_disputed([rid, "missing-id"], now=T0) == 1
    assert m.recall("King Arthur Excalibur", now=T0)[0]["stale"] is True
    assert m.stats() == {"lookups": 1, "disputed": 1}
    m.remember("Who gave Arthur Excalibur?", [LADY], now=T0)  # web returned it again
    assert m.recall("King Arthur Excalibur", now=T0)[0]["stale"] is False


def test_unreadable_record_is_skipped_not_fatal(tmp_path):
    m = _mem(tmp_path)
    m.remember("King Arthur sword", [LADY], now=T0)
    (tmp_path / "lookups" / "broken.json").write_text("{not json")
    assert m.recall("King Arthur Excalibur", now=T0)[0]["url"] == LADY["url"]


def test_content_terms_drop_stopwords():
    assert content_terms("Where did he get the sword?") == {"sword"}


def test_question_match_tells_same_subject_questions_apart():
    # Live 0vuz case: results remembered for the sword's name must not
    # answer "where did he get it" without a fresh lookup.
    followup = "Where did he get the sword? name King Arthur's Excalibur"
    assert question_match(followup, ["King Arthur's", "['King Arthur', 'sword']"]) < 0.6
    # ...but the same question asked again in a new session does match.
    assert question_match("Where did King Arthur get Excalibur?", [followup]) >= 0.6


def test_recall_reports_question_match(tmp_path):
    m = _mem(tmp_path)
    m.remember("Where did King Arthur get Excalibur?", [LADY], now=T0)
    hit = m.recall("Where did Arthur get Excalibur?", now=T0)[0]
    assert hit["question_match"] >= 0.6


def test_record_looked_up_for_this_question_outranks_generic_ones(tmp_path):
    # Live 0vuz case: generic "Excalibur is the sword of King Arthur" records
    # tied at relevance 1.0 and pushed the Lady of the Lake out of the top 3.
    m = _mem(tmp_path)
    for i in range(3):
        m.remember("Excalibur", [{"title": f"Excalibur {i}", "url": f"https://x/{i}",
                                  "snippet": f"Excalibur is the legendary sword of King Arthur, version {i}."}], now=T0)
    m.remember("Where did he get the sword? name King Arthur's Excalibur", [LADY], now=T0)
    hits = m.recall("Where did King Arthur get Excalibur?", now=T0)
    assert hits[0]["url"] == LADY["url"] and hits[0]["question_match"] >= 0.6
